use super::{
    cosine_similarity_from_distance, elapsed_ms_and_us, query_allows_stale_cards,
    query_variant_weight, Feature, HashMap, HashSet, Instant, Lane, MemoryCardSearchInput,
    QueryModality, QueryPipelineState, RankedItem,
};

const NEURAL_TOP: usize = 25;
const MIN_HIT_SIMILARITY: f32 = 0.30;

pub(crate) struct ScopedAnnState {
    pub attempt: usize,
    pub current_top: usize,
    pub max_top: usize,
    pub hit_count: usize,
    pub min_hits: usize,
    pub top_similarity: Option<f32>,
    pub prev_hit_count: Option<usize>,
    pub prev_top_similarity: Option<f32>,
}

type AnnWorkerResult = (usize, Vec<String>, Vec<RankedItem>, Vec<(u64, f32)>, u64, u64);

fn scoped_semantic_start(config: &crate::config::RetrievalConfig, max_top: usize) -> usize {
    config.scoped_semantic_start.min(max_top)
}

fn scoped_semantic_min_hits(
    config: &crate::config::RetrievalConfig,
    limit: usize,
    max_top: usize,
) -> usize {
    config.scoped_min_hits.unwrap_or_else(|| limit.saturating_mul(2).max(24)).min(max_top)
}

fn should_stop_scoped_ann(config: &crate::config::RetrievalConfig, state: &ScopedAnnState) -> bool {
    if state.current_top >= state.max_top || state.attempt >= config.scoped_stop_max_attempts {
        return true;
    }
    if state.hit_count < state.min_hits {
        return false;
    }
    let strong_enough =
        state.top_similarity.is_some_and(|sim| sim >= config.scoped_stop_min_similarity);
    if !strong_enough {
        return false;
    }
    let Some(prev_hits) = state.prev_hit_count else {
        return false;
    };
    let low_hit_gain = state.hit_count.saturating_sub(prev_hits) <= config.scoped_stop_max_hit_gain;
    let low_similarity_gain = match (state.top_similarity, state.prev_top_similarity) {
        (Some(current), Some(previous)) => {
            (current - previous).abs() <= config.scoped_stop_min_similarity_gain
        }
        _ => false,
    };
    low_hit_gain || low_similarity_gain
}

pub(crate) fn retrieval_phase(s: &mut QueryPipelineState) {
    retrieval_ann(s);
    retrieval_fts(s);
    retrieval_cards(s);
}

#[allow(clippy::too_many_lines)]
fn retrieval_ann(s: &mut QueryPipelineState) {
    if !s.state.config.lanes.enabled(Lane::Vector) {
        return;
    }
    let stage_start = Instant::now();
    let embed_dim = s.state.semantic.embedding_dim();
    let semantic_queries = s
        .plan
        .semantic_queries
        .iter()
        .take(s.budget.semantic_query_limit)
        .cloned()
        .collect::<Vec<_>>();
    let primary_qembed_clone = s.primary_qembed.clone();
    // Guard: only proceed if the primary embedding has the right dimension.
    // An empty or wrong-dimension vector causes usearch to exhibit undefined
    // behaviour (the debug assert is stripped in release builds) and the
    // search can spin/hang indefinitely → 30-second timeouts.
    if primary_qembed_clone.len() != embed_dim {
        tracing::warn!(
            got = primary_qembed_clone.len(),
            expected = embed_dim,
            "primary query embedding has wrong dimension; skipping ANN retrieval"
        );
        (s.diag.ann_ms, s.diag.ann_us) = elapsed_ms_and_us(stage_start);
        return;
    }
    let mut embeddings = vec![primary_qembed_clone];
    if semantic_queries.len() > 1 {
        let query_refs: Vec<&str> =
            semantic_queries.iter().skip(1).map(std::string::String::as_str).collect();
        match s.state.semantic.embed_queries(&query_refs) {
            Ok(batch_results) => embeddings.extend(batch_results),
            Err(err) => {
                tracing::warn!(error = ?err, "query variant embedding failed; using primary only");
            }
        }
    }

    let scoped_entity_id = s.payload.entity_id.clone();
    let stage_start = Instant::now();
    let ann_results: Vec<AnnWorkerResult> = {
        let tenant_clone = s.tenant.clone();
        let eid = scoped_entity_id.clone();
        let profile = &s.adaptive_profile;
        let query_limit = s.limit;
        let semantic_top = s.semantic_top;
        let dedup_threshold = s.weights.dedup_similarity_threshold;
        let retrieval_config = s.state.config.retrieval.clone();
        let point_in_time_ms = s.payload.point_in_time_ms;
        let known_as_of_ms = s.payload.known_as_of_ms;
        std::thread::scope(|sc| {
            let mut handles = Vec::with_capacity(embeddings.len());
            for (idx, embedding) in embeddings.iter().enumerate() {
                let tenant = tenant_clone.clone();
                let embedding = embedding.clone();
                let eid = eid.clone();
                let retrieval_config = retrieval_config.clone();
                let handle = sc.spawn(move || {
                    let mut hnsw_hits = Vec::new();
                    let mut variant_rerank_seed_ids = Vec::new();
                    let mut local_cache: HashMap<u64, Option<(u64, String)>> = HashMap::new();
                    let mut search_attempts = 1u64;
                    let mut search_top = semantic_top as u64;
                    let hnsw_raw = if let Some(entity_id) = eid.as_deref() {
                        let scoped_max_top = semantic_top;
                        let scoped_start = scoped_semantic_start(&retrieval_config, scoped_max_top);
                        let scoped_step = retrieval_config.scoped_semantic_step;
                        let scoped_min_hits = scoped_semantic_min_hits(
                            &retrieval_config,
                            query_limit,
                            scoped_max_top,
                        );
                        let mut current_top = scoped_start;
                        let mut attempts = 0usize;
                        let mut prev_hit_count: Option<usize> = None;
                        let mut prev_top_similarity: Option<f32> = None;
                        let mut seen_vector_ids: HashSet<u64> = HashSet::new();
                        let mut cumulative_hnsw_hits: Vec<RankedItem> = Vec::new();
                        let hnsw_raw = loop {
                            attempts += 1;
                            let current_raw = match tenant
                                .vectors()
                                .and_then(|v| v.search(Some(entity_id), &embedding, current_top))
                            {
                                Ok(raw) => raw,
                                Err(err) => {
                                    tracing::warn!(error = ?err, "scoped ANN search failed");
                                    Vec::new()
                                }
                            };
                            let scoped_last_raw = current_raw.clone();
                            let mut scoped_top_similarity = None;
                            let mut new_scoped_hits = 0usize;

                            let unresolved_ids: Vec<u64> = current_raw
                                .iter()
                                .map(|(vid, _)| *vid)
                                .filter(|vid| !local_cache.contains_key(vid))
                                .collect();
                            if !unresolved_ids.is_empty() {
                                if let Ok(looked) = tenant.lookup_by_vector_ids_batch_at(
                                    &unresolved_ids,
                                    point_in_time_ms,
                                    known_as_of_ms,
                                ) {
                                    for (vid, hit) in unresolved_ids.into_iter().zip(looked) {
                                        local_cache.insert(vid, hit);
                                    }
                                }
                            }

                            for (vid, dist) in &current_raw {
                                if !seen_vector_ids.insert(*vid) {
                                    continue;
                                }
                                let lookup = local_cache.get(vid).cloned().unwrap_or(None);
                                let Some((ts, mem_id)) = lookup else {
                                    continue;
                                };
                                // Post-ANN similarity floor: drop zero/near-zero hits
                                // before they pollute cumulative_hnsw_hits and the
                                // downstream RRF lane.
                                let similarity = cosine_similarity_from_distance(*dist);
                                if similarity < MIN_HIT_SIMILARITY {
                                    continue;
                                }
                                new_scoped_hits += 1;
                                if cumulative_hnsw_hits.len() < NEURAL_TOP {
                                    variant_rerank_seed_ids.push(mem_id.clone());
                                }
                                cumulative_hnsw_hits
                                    .push(RankedItem { memory_id: mem_id, timestamp: ts });
                                if scoped_top_similarity.is_none() {
                                    scoped_top_similarity =
                                        Some(cosine_similarity_from_distance(*dist));
                                }
                            }

                            let scoped_state = ScopedAnnState {
                                attempt: attempts,
                                current_top,
                                max_top: scoped_max_top,
                                hit_count: cumulative_hnsw_hits.len(),
                                min_hits: scoped_min_hits,
                                top_similarity: scoped_top_similarity,
                                prev_hit_count,
                                prev_top_similarity,
                            };
                            if should_stop_scoped_ann(&retrieval_config, &scoped_state)
                                || (attempts >= 2
                                    && cumulative_hnsw_hits.len() >= scoped_min_hits
                                    && new_scoped_hits == 0)
                            {
                                break scoped_last_raw;
                            }
                            let next_top =
                                current_top.saturating_add(scoped_step).min(scoped_max_top);
                            if next_top == current_top {
                                break scoped_last_raw;
                            }
                            prev_hit_count = Some(cumulative_hnsw_hits.len());
                            prev_top_similarity = scoped_top_similarity;
                            current_top = next_top;
                        };
                        hnsw_hits = cumulative_hnsw_hits;
                        search_attempts = attempts as u64;
                        search_top = current_top as u64;
                        hnsw_raw
                    } else {
                        let hnsw_raw = match tenant
                            .vectors()
                            .and_then(|v| v.search(None, &embedding, semantic_top))
                        {
                            Ok(raw) => raw,
                            Err(err) => {
                                tracing::warn!(error = ?err, "ANN search failed");
                                Vec::new()
                            }
                        };
                        let vids: Vec<u64> = hnsw_raw.iter().map(|(vid, _)| *vid).collect();
                        if let Ok(looked) = tenant.lookup_by_vector_ids_batch_at(
                            &vids,
                            point_in_time_ms,
                            known_as_of_ms,
                        ) {
                            let looked_ids: Vec<String> =
                                looked.iter().flatten().map(|(_, m)| m.clone()).collect();
                            let identity =
                                tenant.memory_identity_batch(&looked_ids).unwrap_or_default();
                            for (rank, (_vid, dist)) in hnsw_raw.iter().enumerate() {
                                if let Some((ts, mem_id)) = looked[rank].clone() {
                                    // Post-ANN similarity floor: drop zero/near-zero hits
                                    // before they enter hnsw_hits and the downstream RRF lane.
                                    let raw_similarity = cosine_similarity_from_distance(*dist);
                                    if raw_similarity < MIN_HIT_SIMILARITY {
                                        continue;
                                    }
                                    let routed_match =
                                        identity.get(&mem_id).is_some_and(|(session, _)| {
                                            profile.route_sessions.contains(session)
                                        });
                                    if profile.route_strength > 0.0
                                        && !routed_match
                                        && hnsw_hits.len() < (query_limit.saturating_div(2).max(1))
                                        && raw_similarity < dedup_threshold
                                    {
                                        continue;
                                    }
                                    if rank < NEURAL_TOP {
                                        variant_rerank_seed_ids.push(mem_id.clone());
                                    }
                                    hnsw_hits.push(RankedItem { memory_id: mem_id, timestamp: ts });
                                }
                            }
                        }
                        hnsw_raw
                    };
                    (idx, variant_rerank_seed_ids, hnsw_hits, hnsw_raw, search_attempts, search_top)
                });
                handles.push(handle);
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|e| {
                        tracing::error!("ANN search thread panicked: {:?}", e);
                        (usize::MAX, Vec::new(), Vec::new(), Vec::new(), 0, 0)
                    })
                })
                .collect::<Vec<_>>()
        })
    };

    let mut primary_hnsw_raw = Vec::new();
    let mut semantic_ranked_lists = Vec::new();
    for (idx, _seed_ids, hnsw_hits, hnsw_raw, attempts, top) in ann_results {
        if idx == usize::MAX {
            continue;
        }
        if idx == 0 {
            primary_hnsw_raw = hnsw_raw;
            s.diag.scoped_ann_attempts = attempts;
            s.diag.scoped_ann_top = top;
            s.diag.scoped_primary_hits = hnsw_hits.len() as u64;
        }
        semantic_ranked_lists.push((
            query_variant_weight(idx, QueryModality::Semantic, s.plan.intent)
                * s.adaptive_profile.semantic_scale,
            hnsw_hits,
        ));
    }
    s.candidates.primary_hnsw_raw = primary_hnsw_raw;
    s.candidates.semantic_ranked_lists = semantic_ranked_lists;
    (s.diag.ann_ms, s.diag.ann_us) = elapsed_ms_and_us(stage_start);
}

fn retrieval_fts(s: &mut QueryPipelineState) {
    if !s.state.config.lanes.enabled(Lane::Fts) {
        return;
    }
    let stage_start = Instant::now();
    let mut fts_ranked_lists = Vec::new();
    let mut fts_memory_ids_to_lookup = Vec::new();
    let mut fts_results_per_query = Vec::new();

    let fts_queries_to_run: Vec<&String> =
        s.plan.fts_queries.iter().take(s.budget.fts_query_limit).collect();
    {
        let tenant_clone = s.tenant.clone();
        let eid = s.payload.entity_id.clone();
        let fts_top = s.fts_top;
        let point_in_time_ms = s.payload.point_in_time_ms;
        let known_as_of_ms = s.payload.known_as_of_ms;
        std::thread::scope(|sc| {
            let mut handles = Vec::new();
            for (idx, fts_query) in fts_queries_to_run.into_iter().enumerate() {
                let tenant = tenant_clone.clone();
                let eid = eid.clone();
                handles.push(sc.spawn(move || {
                    let hits = tenant
                        .fts_search_at(
                            fts_query.as_str(),
                            fts_top,
                            eid.as_deref(),
                            point_in_time_ms,
                            known_as_of_ms,
                        )
                        .unwrap_or_else(|err| {
                            tracing::warn!(error = %err, query = %fts_query, "FTS lane failed");
                            Vec::new()
                        });
                    (idx, hits)
                }));
            }
            let mut temp_results = Vec::new();
            for handle in handles {
                if let Ok(res) = handle.join() {
                    temp_results.push(res);
                }
            }
            temp_results.sort_by_key(|(idx, _)| *idx);
            for (_, hits) in temp_results {
                fts_results_per_query.push(hits);
            }
        });
    }

    for hits in &fts_results_per_query {
        for (mid, _) in hits {
            fts_memory_ids_to_lookup.push(mid.clone());
        }
    }
    let fts_lookup =
        s.tenant.lookup_by_memory_ids_batch(&fts_memory_ids_to_lookup).unwrap_or_default();
    for (idx, hits) in fts_results_per_query.into_iter().enumerate() {
        let mut ranked_hits = Vec::new();
        for (mid, _) in hits {
            if let Some((ts, _)) = fts_lookup.get(&mid).copied() {
                ranked_hits.push(RankedItem { memory_id: mid, timestamp: ts });
            }
        }
        fts_ranked_lists.push((
            query_variant_weight(idx, QueryModality::Lexical, s.plan.intent)
                * s.adaptive_profile.lexical_scale,
            ranked_hits,
        ));
    }
    s.candidates.fts_ranked_lists = fts_ranked_lists;
    (s.diag.fts_ms, s.diag.fts_us) = elapsed_ms_and_us(stage_start);
}

fn retrieval_cards(s: &mut QueryPipelineState) {
    let stage_start = Instant::now();
    let mut card_ranked_items = Vec::new();
    let entity_scope = s.payload.entity_id.as_ref().filter(|_| {
        s.state.config.features.enabled(Feature::MemoryCards)
            && s.state.config.lanes.enabled(Lane::Cards)
    });
    if let Some(entity_id) = entity_scope {
        let include_stale_cards = query_allows_stale_cards(&s.query_text, &s.plan);
        let card_hits = s
            .tenant
            .search_memory_cards(&MemoryCardSearchInput {
                entity_id,
                lexical_terms: &s.plan.lexical_terms,
                temporal_terms: &s.plan.temporal_terms,
                entities: &s.plan.subject_entities,
                route_sessions: &s.adaptive_profile.route_sessions,
                include_stale: include_stale_cards,
                point_in_time_ms: s.payload.point_in_time_ms,
                known_as_of_ms: s.payload.known_as_of_ms,
                now_ms: s.now_ms,
                limit: s.budget.card_limit,
            })
            .unwrap_or_default();
        s.diag.memory_card_hits = card_hits.len() as u64;
        for hit in card_hits {
            card_ranked_items.push(RankedItem { memory_id: hit.card_id, timestamp: hit.timestamp });
            if !hit.source_session_id.is_empty()
                && hit.lexical_hits + hit.temporal_hits + hit.entity_hits >= 2
            {
                *s.route.session_scores.entry(hit.source_session_id).or_insert(0.0) += hit.score
                    * s.state.ranking_config.session_boost
                    * s.state.ranking_config.session_boost_routed;
            }
        }
    }
    s.candidates.card_ranked_items = card_ranked_items;
    (s.diag.card_ms, s.diag.card_us) = elapsed_ms_and_us(stage_start);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn make_ann_state(
        attempt: usize,
        current_top: usize,
        max_top: usize,
        hit_count: usize,
        min_hits: usize,
        top_similarity: Option<f32>,
        prev_hit_count: Option<usize>,
        prev_top_similarity: Option<f32>,
    ) -> ScopedAnnState {
        ScopedAnnState {
            attempt,
            current_top,
            max_top,
            hit_count,
            min_hits,
            top_similarity,
            prev_hit_count,
            prev_top_similarity,
        }
    }

    #[test]
    fn should_stop_scoped_ann_max_top_reached() {
        let state = make_ann_state(0, 100, 100, 0, 1, None, None, None);
        assert!(should_stop_scoped_ann(&crate::config::RetrievalConfig::default(), &state));
    }

    #[test]
    fn should_stop_scoped_ann_max_attempts_reached() {
        let state = make_ann_state(10, 50, 100, 50, 10, Some(0.9), Some(40), Some(0.8));
        assert!(should_stop_scoped_ann(&crate::config::RetrievalConfig::default(), &state));
    }

    #[test]
    fn should_stop_scoped_ann_not_enough_hits() {
        let state = make_ann_state(0, 50, 100, 5, 10, None, None, None);
        assert!(!should_stop_scoped_ann(&crate::config::RetrievalConfig::default(), &state));
    }

    #[test]
    fn should_stop_scoped_ann_not_strong_enough_similarity() {
        let state = make_ann_state(0, 50, 100, 20, 10, Some(0.6), Some(10), Some(0.5));
        assert!(!should_stop_scoped_ann(&crate::config::RetrievalConfig::default(), &state));
    }

    #[test]
    fn should_stop_scoped_ann_none_similarity_not_strong() {
        let state = make_ann_state(0, 50, 100, 20, 10, None, Some(10), Some(0.5));
        assert!(!should_stop_scoped_ann(&crate::config::RetrievalConfig::default(), &state));
    }

    #[test]
    fn should_stop_scoped_ann_no_prev_hit_count_returns_false() {
        let state = make_ann_state(0, 50, 100, 20, 10, Some(0.8), None, Some(0.79));
        assert!(!should_stop_scoped_ann(&crate::config::RetrievalConfig::default(), &state));
    }

    #[test]
    fn should_stop_scoped_ann_convergence_low_hit_gain() {
        let state = make_ann_state(0, 50, 100, 12, 10, Some(0.8), Some(10), Some(0.7));
        assert!(should_stop_scoped_ann(&crate::config::RetrievalConfig::default(), &state));
    }

    #[test]
    fn should_stop_scoped_ann_convergence_low_similarity_gain() {
        let state = make_ann_state(0, 50, 100, 20, 10, Some(0.71), Some(10), Some(0.70));
        assert!(should_stop_scoped_ann(&crate::config::RetrievalConfig::default(), &state));
    }

    #[test]
    fn should_stop_scoped_ann_no_convergence() {
        let state = make_ann_state(0, 50, 100, 20, 10, Some(0.8), Some(10), Some(0.7));
        assert!(!should_stop_scoped_ann(&crate::config::RetrievalConfig::default(), &state));
    }
}
