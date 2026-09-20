use super::{
    adaptive_rrf_k, collect_edge_cluster_scores_for_seeds, cosine_similarity, elapsed_ms_and_us,
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

fn normalize_graph_scores(scores: &mut HashMap<String, f32>) {
    let max_graph = scores.values().copied().fold(0.0f32, f32::max);
    if max_graph > 0.0 {
        for score in scores.values_mut() {
            *score /= max_graph;
        }
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
    let mut ranked_sources = s.candidates.semantic_ranked_lists.clone();
    ranked_sources.extend(s.candidates.fts_ranked_lists.clone());
    if !s.candidates.card_ranked_items.is_empty() {
        let card_weight = if s.plan.needs_decomposition || s.plan.cross_entity {
            s.state.ranking_config.card_boost * s.state.ranking_config.card_boost_strong
        } else if matches!(s.plan.intent, QueryIntent::TemporalAggregation | QueryIntent::Inference)
        {
            s.state.ranking_config.card_boost * s.state.ranking_config.card_boost_medium
        } else {
            s.state.ranking_config.card_boost
        };
        ranked_sources.push((card_weight, s.candidates.card_ranked_items.clone()));
    }
    if !s.candidates.neural_scores.is_empty() {
        let mut scored_items: Vec<_> = s.candidates.neural_scores.clone().into_iter().collect();
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

    for (mid, boost) in &s.route.memory_scores {
        if let Some((ts, _)) = s.tenant.lookup_by_memory_id(mid).unwrap_or(None) {
            let entry = fused_map.entry(mid.clone()).or_insert((ts, 0.0));
            entry.0 = ts;
            entry.1 += *boost;
        }
    }

    let stage_start = Instant::now();
    if s.plan.intent == QueryIntent::Inference
        && !s.primary_qembed.is_empty()
        && s.state.config.features.enabled(Feature::Preferences)
    {
        if let Some(ref entity_id) = s.payload.entity_id {
            let preference_memories =
                s.tenant.get_preference_memories(entity_id, 96).unwrap_or_default();
            if !preference_memories.is_empty() {
                let option_embeddings = [s.primary_qembed.clone()];
                let memory_ids: Vec<String> =
                    preference_memories.iter().map(|(memory_id, _)| memory_id.clone()).collect();
                let lookup = s.tenant.lookup_by_memory_ids_batch(&memory_ids).unwrap_or_default();
                let observation_keys: Vec<(u64, String)> = memory_ids
                    .iter()
                    .filter_map(|memory_id| {
                        lookup.get(memory_id).map(|(ts, _)| (*ts, memory_id.clone()))
                    })
                    .collect();
                let observations =
                    s.tenant.get_observations_batch(&observation_keys).unwrap_or_default();
                for (memory_id, strength) in preference_memories {
                    let Some(obs) = observations.get(&memory_id) else {
                        continue;
                    };
                    if obs.embedding.len() != s.primary_qembed.len() || obs.embedding.is_empty() {
                        continue;
                    }
                    let best_similarity = option_embeddings
                        .iter()
                        .filter(|candidate| candidate.len() == obs.embedding.len())
                        .map(|candidate| cosine_similarity(candidate, &obs.embedding))
                        .fold(-1.0f32, f32::max);
                    if best_similarity >= 0.35 {
                        if let Some((ts, _)) = lookup.get(&memory_id).copied() {
                            let entry = fused_map.entry(memory_id.clone()).or_insert((ts, 0.0));
                            entry.0 = ts;
                            entry.1 += best_similarity.max(0.0) * 0.12 + strength * 0.04;
                        }
                    }
                }
            }
        }
    }
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
    for link in link_scores {
        for (linked_mid, boost) in link {
            *graph_scores.entry(linked_mid.clone()).or_insert(0.0) += boost;
            all_linked.push(linked_mid);
        }
    }
    for edge in edge_scores {
        for (linked_mid, boost) in edge {
            *graph_scores.entry(linked_mid.clone()).or_insert(0.0) += boost;
            all_linked.push(linked_mid);
        }
    }
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
    // semantically relevant results regardless of the query. Scale to [0, 1]
    // so graph evidence is one bounded signal among the others.
    normalize_graph_scores(&mut graph_scores);
    s.scoring.graph_scores = graph_scores;
    (s.diag.graph_ms, s.diag.graph_us) = elapsed_ms_and_us(stage_start);

    let mut fused_vec: Vec<(String, u64, f32)> =
        fused_map.into_iter().map(|(mid, (ts, score))| (mid, ts, score)).collect();
    sort_fused_items(&mut fused_vec);
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
    fn graph_scores_are_normalized_to_the_unit_interval() {
        let mut scores = HashMap::from([
            ("memory-a".to_string(), 2.0),
            ("memory-b".to_string(), 1.0),
            ("memory-c".to_string(), 0.0),
        ]);

        normalize_graph_scores(&mut scores);

        assert!((scores["memory-a"] - 1.0).abs() < f32::EPSILON);
        assert!((scores["memory-b"] - 0.5).abs() < f32::EPSILON);
        assert!(scores["memory-c"].abs() < f32::EPSILON);
    }
}
