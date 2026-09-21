use super::{
    adaptive_rrf_k, collect_edge_cluster_scores_for_seeds, elapsed_ms_and_us,
    extract_named_phrases, weighted_reciprocal_rank_fusion, Feature, HashMap, Instant, Lane,
    QueryIntent, QueryPipelineState, RankedItem, SystemTime, UNIX_EPOCH,
};
use crate::graph::Direction;
use rayon::prelude::*;

fn merge_fused_items(
    fused: impl IntoIterator<Item = (String, u64, f32)>,
) -> HashMap<String, (u64, f32)> {
    let mut fused_map: HashMap<String, (u64, f32)> = HashMap::new();
    for (mid, ts, score) in fused {
        let entry = fused_map.entry(mid).or_insert((ts, 0.0));
        entry.0 = ts;
        entry.1 = entry.1.max(score);
    }
    fused_map
}

// Calibrating by the per-query max made the single most-connected memory
// score exactly 1.0 on *every* query, independent of how strong the
// evidence actually was — a query-independent popularity prior that (at
// `weights.graph` 0.15-0.30, blended at 0.7) outweighed the entire semantic
// contribution of a rank-1 dense hit by 3-6x (R1). A saturating transform
// fixes this: it is bounded in [0, 1), monotonic in the raw score, and
// -- critically -- does not renormalize against whatever the rest of the
// candidate set happened to produce this time. `GRAPH_SCORE_SATURATION_K`
// is the raw score at which the transform returns 0.5; it should be
// calibrated offline as the median top-1 raw graph score over a
// representative query sample. 2.0 is an initial placeholder pending that
// calibration, not a measured value.
const GRAPH_SCORE_SATURATION_K: f32 = 2.0;

/// Lower bound on how many fused candidates survive to hydration. Budgets
/// scale the cap above this (`card_limit` spans 48-180, so the largest
/// Research/Hard budgets hydrate up to 360); the floor keeps the smallest
/// budgets from starving the session/facet coverage selection downstream.
const FUSED_HYDRATION_FLOOR: usize = 200;

fn calibrate_graph_scores(scores: &mut HashMap<String, f32>) {
    for score in scores.values_mut() {
        *score = *score / (*score + GRAPH_SCORE_SATURATION_K);
    }
}

// f32 addition is not associative, and folding contributions straight out
// of a `HashMap` (RandomState, iteration order randomized per process)
// means the same logical set of boosts could sum in a different order on
// two runs over identical input, diverging in the low bits. That breaks the
// byte-identical replay this project's determinism thesis depends on.
// Collecting into `(seed index, memory_id)`-sorted tuples first makes the
// fold order canonical regardless of any HashMap's internal iteration
// order, now or if the upstream traversal ever changes shape.
fn accumulate_seed_scores_deterministically(
    scores_by_seed: Vec<HashMap<String, f32>>,
    graph_scores: &mut HashMap<String, f32>,
    all_linked: &mut Vec<String>,
) {
    let mut contributions: Vec<(usize, String, f32)> = scores_by_seed
        .into_iter()
        .enumerate()
        .flat_map(|(seed_idx, map)| map.into_iter().map(move |(mid, boost)| (seed_idx, mid, boost)))
        .collect();
    contributions.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    for (_seed_idx, linked_mid, boost) in contributions {
        *graph_scores.entry(linked_mid.clone()).or_insert(0.0) += boost;
        all_linked.push(linked_mid);
    }
}

fn sort_fused_items(items: &mut [(String, u64, f32)]) {
    items.sort_by(|a, b| {
        b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))
    });
}

#[allow(clippy::too_many_lines)]
pub(crate) fn fusion_phase(s: &mut QueryPipelineState) {
    let stage_start = Instant::now();
    // Nothing downstream of this phase reads `s.candidates` again, so take
    // these lists instead of cloning them into the fusion input.
    let mut ranked_sources = std::mem::take(&mut s.candidates.semantic_ranked_lists);
    ranked_sources.extend(std::mem::take(&mut s.candidates.fts_ranked_lists));
    if !s.candidates.card_ranked_items.is_empty() {
        let card_weight = if s.plan.needs_decomposition || s.plan.cross_entity {
            s.state.ranking_config.card_boost * s.state.ranking_config.card_boost_strong
        } else if matches!(s.plan.intent, QueryIntent::TemporalAggregation | QueryIntent::Inference)
        {
            s.state.ranking_config.card_boost * s.state.ranking_config.card_boost_medium
        } else {
            s.state.ranking_config.card_boost
        };
        ranked_sources.push((card_weight, std::mem::take(&mut s.candidates.card_ranked_items)));
    }
    if !s.candidates.neural_scores.is_empty() {
        let mut scored_items: Vec<_> =
            std::mem::take(&mut s.candidates.neural_scores).into_iter().collect();
        scored_items.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))
        });
        let mut neural_items = Vec::new();
        let mids: Vec<String> = scored_items.iter().map(|(mid, _)| mid.clone()).collect();
        let lookup = s.tenant.lookup_by_memory_ids_batch(&mids).unwrap_or_default();
        for (mid, _) in scored_items {
            if let Some(&(ts, _)) = lookup.get(&mid) {
                neural_items.push(RankedItem { memory_id: mid, timestamp: ts });
            }
        }
        ranked_sources.push((1.5, neural_items));
    }
    let fused = weighted_reciprocal_rank_fusion(ranked_sources, adaptive_rrf_k(&s.plan));
    (s.diag.fuse_ms, s.diag.fuse_us) = elapsed_ms_and_us(stage_start);

    if s.payload.reference_time_ms.is_none() {
        s.now_ms =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
    }
    let mut fused_map = merge_fused_items(fused);

    // `s.route.memory_scores` was removed: it was only ever assigned
    // `HashMap::new()` in `plan_phase` and never populated, so this lookup
    // loop ran zero iterations on every query while still existing as a
    // read site future changes had to reason about. See also the deleted
    // preference lane immediately below.

    let stage_start = Instant::now();
    // The preference lane below was dead code: `get_observations_batch`
    // returns `embedding: Vec::new()` unconditionally
    // (`storage/repo/memories.rs`), so `obs.embedding.is_empty()` always
    // held and every candidate was skipped before `best_similarity` could
    // ever be computed. It still paid for a `get_preference_memories` call
    // (up to 96 rows) and a batch observation fetch on every Inference
    // query. Making it work requires changing `get_observations_batch` in
    // `storage/repo/memories.rs`, which this pass does not own, so the lane
    // is removed rather than left silently dead. `preference_ms`/`_us`
    // stay wired (near-zero now) since `api/handlers/query.rs`, outside
    // this pass's ownership, still reads them for the response headers.
    (s.diag.preference_ms, s.diag.preference_us) = elapsed_ms_and_us(stage_start);

    let stage_start = Instant::now();
    let mut link_seed_ids: Vec<(String, f32)> =
        fused_map.iter().map(|(mid, (_, score))| (mid.clone(), *score)).collect();
    link_seed_ids.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))
    });

    let mut graph_scores: HashMap<String, f32> = HashMap::new();
    let seed_top: Vec<String> = link_seed_ids
        .into_iter()
        .take(s.state.config.retrieval.graph_seed_count)
        .map(|(mid, _)| mid)
        .collect();

    let intent_for_graph = s.plan.intent;
    let graph_lane = s.state.config.lanes.enabled(Lane::Graph);
    let links_on = graph_lane
        && (s.state.config.features.enabled(Feature::DerivedLinks)
            || s.state.config.features.enabled(Feature::RetrospectiveLinks));
    let edges_on = graph_lane && s.state.config.features.enabled(Feature::GraphEdges);
    let seeds_start = Instant::now();
    let link_start = Instant::now();
    let link_scores: Vec<HashMap<String, f32>> = if links_on {
        seed_top
            .par_iter()
            .map(|seed_mid| {
                s.tenant
                    .get_link_cluster_scores(seed_mid, s.state.config.retrieval.graph_max_depth)
                    .unwrap_or_default()
            })
            .collect()
    } else {
        Vec::new()
    };
    s.diag.graph_links_us = link_start.elapsed().as_micros() as u64;
    let edge_start = Instant::now();
    let (edge_scores, expanded) = if edges_on {
        collect_edge_cluster_scores_for_seeds(
            &s.tenant,
            &seed_top,
            s.state.config.retrieval.graph_max_depth,
            s.state.config.retrieval.graph_max_node_degree,
            None,
            Some(intent_for_graph),
        )
    } else {
        (Vec::new(), 0)
    };
    s.diag.graph_edges_us = edge_start.elapsed().as_micros() as u64;
    s.diag.graph_expanded = expanded;
    s.diag.graph_seeds_wall_us = seeds_start.elapsed().as_micros() as u64;

    let mut all_linked: Vec<String> = Vec::with_capacity(seed_top.len() * 8);
    accumulate_seed_scores_deterministically(link_scores, &mut graph_scores, &mut all_linked);
    accumulate_seed_scores_deterministically(edge_scores, &mut graph_scores, &mut all_linked);
    all_linked.sort();
    all_linked.dedup();
    let lookup_start = Instant::now();
    if !all_linked.is_empty() {
        if let Ok(lookup) = s.tenant.lookup_by_memory_ids_batch(&all_linked) {
            for (linked_mid, (ts, _)) in lookup {
                let entry = fused_map.entry(linked_mid).or_insert((ts, 0.0));
                entry.0 = ts;
            }
        }
    }

    s.diag.graph_lookup_us += lookup_start.elapsed().as_micros() as u64;
    let entities_start = Instant::now();

    // Entity-graph retrieval lane: resolve query entities against registry,
    // then traverse the edges table to find connected memories.
    if let Some(entity_scope) = s.payload.entity_id.as_ref().filter(|_| edges_on) {
        if !entity_scope.is_empty() {
            let mut entity_seeds: Vec<String> = s.plan.subject_entities.clone();
            let query_lines: Vec<String> =
                std::iter::once(s.payload.textual_query.clone()).collect();
            for phrase in extract_named_phrases(&query_lines) {
                if phrase.len() >= 3
                    && !entity_seeds.iter().any(|e| e.eq_ignore_ascii_case(&phrase))
                {
                    entity_seeds.push(phrase);
                }
            }
            entity_seeds.truncate(8);

            let mut batched_edges = Vec::new();
            for seed in &entity_seeds {
                if let Ok(edges) = s.tenant.graph_query_edges(seed, None, Direction::Both, 50) {
                    batched_edges.extend(edges.into_iter().filter(|e| !e.memory_id.is_empty()));
                }
            }
            if !batched_edges.is_empty() {
                let mut edge_mids =
                    batched_edges.iter().map(|e| e.memory_id.clone()).collect::<Vec<_>>();
                edge_mids.sort();
                edge_mids.dedup();

                if let Ok(lookup) = s.tenant.lookup_by_memory_ids_batch(&edge_mids) {
                    for edge in batched_edges {
                        if let Some(&(ts, _)) = lookup.get(&edge.memory_id) {
                            let boost = edge.weight * 0.5;
                            *graph_scores.entry(edge.memory_id.clone()).or_insert(0.0) += boost;
                            let entry =
                                fused_map.entry(edge.memory_id.clone()).or_insert((ts, 0.0));
                            entry.0 = ts;
                        }
                    }
                }
            }
        }
    }

    s.diag.graph_entities_us = entities_start.elapsed().as_micros() as u64;

    // Graph scores are sums over every traversed edge, so memories with many
    // derived records reached values in the hundreds and outranked
    // semantically relevant results regardless of the query. Calibrate with
    // a saturating transform (see `calibrate_graph_scores`) so graph
    // evidence is one bounded, query-independent signal among the others.
    calibrate_graph_scores(&mut graph_scores);
    s.scoring.graph_scores = graph_scores;
    (s.diag.graph_ms, s.diag.graph_us) = elapsed_ms_and_us(stage_start);

    let mut fused_vec: Vec<(String, u64, f32)> =
        fused_map.into_iter().map(|(mid, (ts, score))| (mid, ts, score)).collect();
    sort_fused_items(&mut fused_vec);

    // `score_hydrate` (query/score.rs) does three parallel batch reads over
    // every item here before any relevance filtering runs. For Hard/
    // Research query shapes the fused set can reach the high hundreds to
    // low thousands to answer a query that returns 5-20 results, so
    // hydration was paying for work scoring immediately discards.
    //
    // RECALL RISK: `select_candidates_with_session_head` (respond.rs) does
    // session/facet coverage selection downstream that deliberately wants
    // breadth across sessions, not just the top fused scores — truncating
    // here removes candidates it can never consider. The cap is scaled off
    // the existing retrieval budget and kept generous (2x the card budget,
    // floor 200) to keep that risk small, but it is UNVALIDATED: it needs a
    // dev-split recall benchmark before shipping.
    let hydration_cap = (s.budget.card_limit * 2).max(FUSED_HYDRATION_FLOOR);
    fused_vec.truncate(hydration_cap);

    s.fused.items = fused_vec;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_items_keep_the_best_score_and_latest_timestamp() {
        let merged = merge_fused_items(vec![
            ("memory-a".to_string(), 10, 0.25),
            ("memory-a".to_string(), 20, 0.75),
            ("memory-b".to_string(), 30, 0.50),
        ]);

        assert_eq!(merged.get("memory-a"), Some(&(20, 0.75)));
        assert_eq!(merged.get("memory-b"), Some(&(30, 0.50)));
    }

    #[test]
    fn fused_items_sort_by_score_then_memory_id() {
        let mut items = vec![
            ("memory-b".to_string(), 20, 0.5),
            ("memory-c".to_string(), 30, 0.8),
            ("memory-a".to_string(), 10, 0.5),
        ];

        sort_fused_items(&mut items);

        assert_eq!(
            items,
            vec![
                ("memory-c".to_string(), 30, 0.8),
                ("memory-a".to_string(), 10, 0.5),
                ("memory-b".to_string(), 20, 0.5),
            ]
        );
    }

    #[test]
    fn graph_scores_saturate_instead_of_normalizing_to_the_query_max() {
        // R1 regression: the old max-normalization made the top score
        // exactly 1.0 on every query regardless of evidence strength. The
        // saturating transform must stay strictly below 1.0 and must not
        // depend on what the rest of the candidate set scored.
        let mut scores = HashMap::from([
            ("memory-a".to_string(), 2.0),
            ("memory-b".to_string(), 1.0),
            ("memory-c".to_string(), 0.0),
        ]);

        calibrate_graph_scores(&mut scores);

        let expected_a = 2.0 / (2.0 + GRAPH_SCORE_SATURATION_K);
        let expected_b = 1.0 / (1.0 + GRAPH_SCORE_SATURATION_K);
        assert!((scores["memory-a"] - expected_a).abs() < f32::EPSILON);
        assert!((scores["memory-b"] - expected_b).abs() < f32::EPSILON);
        assert!(scores["memory-c"].abs() < f32::EPSILON);
        assert!(scores["memory-a"] < 1.0);
    }

    #[test]
    fn graph_scores_of_a_single_strong_candidate_are_not_pinned_to_one() {
        // A query with only one graph-connected memory used to make it the
        // argmax at exactly 1.0 under the old normalization no matter how
        // weak its raw evidence was. It must now scale with its own raw
        // magnitude instead.
        let mut weak = HashMap::from([("memory-a".to_string(), 0.1)]);
        let mut strong = HashMap::from([("memory-a".to_string(), 20.0)]);

        calibrate_graph_scores(&mut weak);
        calibrate_graph_scores(&mut strong);

        assert!(weak["memory-a"] < strong["memory-a"]);
        assert!(strong["memory-a"] < 1.0);
    }

    #[test]
    fn seed_score_accumulation_is_independent_of_hashmap_insertion_order() {
        // Two seeds each boost "memory-1"; the per-seed maps are built with
        // different key-insertion orders to stand in for HashMap's
        // process-randomized iteration order. The fold must still produce
        // the exact same bit pattern, because the summation order is fixed
        // by (seed index, memory_id), not by either map's iteration order.
        let seed0_a: HashMap<String, f32> =
            HashMap::from([("memory-1".to_string(), 0.25), ("memory-2".to_string(), 0.1)]);
        let seed0_b: HashMap<String, f32> =
            HashMap::from([("memory-2".to_string(), 0.1), ("memory-1".to_string(), 0.25)]);
        let seed1: HashMap<String, f32> =
            HashMap::from([("memory-1".to_string(), 0.125), ("memory-3".to_string(), 0.5)]);

        let mut scores_a = HashMap::new();
        let mut linked_a = Vec::new();
        accumulate_seed_scores_deterministically(
            vec![seed0_a, seed1.clone()],
            &mut scores_a,
            &mut linked_a,
        );

        let mut scores_b = HashMap::new();
        let mut linked_b = Vec::new();
        accumulate_seed_scores_deterministically(
            vec![seed0_b, seed1],
            &mut scores_b,
            &mut linked_b,
        );

        assert_eq!(
            scores_a["memory-1"].to_bits(),
            scores_b["memory-1"].to_bits(),
            "fold order must be canonical regardless of per-map iteration order"
        );
        assert!((scores_a["memory-2"] - 0.1).abs() < f32::EPSILON);
        assert!((scores_a["memory-3"] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn fused_items_truncate_to_a_budget_scaled_hydration_cap() {
        let cap = |card_limit: usize| (card_limit * 2).max(FUSED_HYDRATION_FLOOR);

        // Small budgets fall back to the floor...
        assert_eq!(cap(48), FUSED_HYDRATION_FLOOR);
        assert_eq!(cap(72), FUSED_HYDRATION_FLOOR);
        // ...and the largest (Research/Hard, card_limit 180) scales past it.
        assert_eq!(cap(180), 360);

        let mut items: Vec<(String, u64, f32)> =
            (0..500).map(|i| (format!("memory-{i}"), i as u64, 1.0)).collect();
        items.truncate(cap(180));
        assert_eq!(items.len(), 360);
    }
}
