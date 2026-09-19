use super::*;

const NEURAL_BATCH: usize = 32;

impl RerankDecision {
    pub(crate) fn applies(self) -> bool {
        matches!(
            self,
            RerankDecision::HeuristicApplied
                | RerankDecision::Always
                | RerankDecision::GateUncertain
                | RerankDecision::Requested
        )
    }
}

pub(crate) fn rerank_policy_name(config: &crate::config::Config) -> &'static str {
    config.rerank.policy.name()
}

/// Relative similarity gap between the top-1 and top-5 ANN hits below which
/// the gate reranks (`TELLODB_RERANK_MARGIN`, default 0.05).
pub(crate) fn rerank_margin(config: &crate::config::Config) -> f32 {
    config.rerank.margin
}

/// Candidates sent to the cross-encoder (`TELLODB_RERANK_TOP`, default 25).
pub(crate) fn rerank_top(config: &crate::config::Config) -> usize {
    config.rerank.top
}

/// True when stage-1 retrieval is uncertain: the top-1 similarity is within
/// `margin` (relative) of the fifth (or last) hit. `hnsw_raw` holds
/// `(vector_id, cosine distance)` sorted by distance.
pub(crate) fn rerank_gate_uncertain(hnsw_raw: &[(u64, f32)], margin: f32) -> bool {
    let kth_index = hnsw_raw.len().min(5).checked_sub(1);
    let (Some(first), Some(kth)) = (hnsw_raw.first(), kth_index.and_then(|i| hnsw_raw.get(i)))
    else {
        return false;
    };
    let top = 1.0 - first.1;
    let other = 1.0 - kth.1;
    if top <= f32::EPSILON {
        return true;
    }
    (top - other) / top < margin
}

pub(crate) fn rerank_phase(s: &mut QueryPipelineState) {
    s.candidates.neural_scores = HashMap::new();
    if !s.state.config.lanes.enabled(Lane::Rerank) {
        s.diag.rerank_reason = RerankDecision::Disabled;
        return;
    }
    if !s.state.semantic.is_rerank_enabled() {
        s.diag.rerank_reason = RerankDecision::Disabled;
        return;
    }
    let retrieval_profile = retrieval_profile(&s.state.config);
    let auto_rerank = auto_rerank_enabled(&s.state.config, retrieval_profile)
        && (s.plan.needs_decomposition
            || s.plan.cross_entity
            || s.plan.ordinal_rank.is_some()
            || matches!(
                s.plan.intent,
                QueryIntent::Inference
                    | QueryIntent::TemporalAggregation
                    | QueryIntent::NumericAggregation
                    | QueryIntent::PeripheralMention
            ));
    let decision = if s.candidates.primary_hnsw_raw.len() < 2 {
        RerankDecision::TooFewCandidates
    } else if s.enable_neural_rerank {
        RerankDecision::Requested
    } else {
        match s.state.config.rerank.policy {
            RerankPolicy::Always => RerankDecision::Always,
            RerankPolicy::Gate
                if rerank_gate_uncertain(
                    &s.candidates.primary_hnsw_raw,
                    s.state.config.rerank.margin,
                ) =>
            {
                RerankDecision::GateUncertain
            }
            RerankPolicy::Gate => RerankDecision::GateConfident,
            RerankPolicy::Heuristic
                if should_apply_neural_rerank(
                    &s.query_text,
                    &s.candidates.primary_hnsw_raw,
                    auto_rerank,
                ) =>
            {
                RerankDecision::HeuristicApplied
            }
            RerankPolicy::Heuristic => RerankDecision::HeuristicSkipped,
        }
    };
    s.diag.rerank_reason = decision;
    if decision.applies() {
        let stage_start = Instant::now();
        let neural_top = s.state.config.rerank.top;
        s.diag.rerank_applied = true;
        let mut rerank_seed_ids = Vec::new();
        for (_weight, list) in &s.candidates.semantic_ranked_lists {
            for item in list.iter().take(neural_top / 2) {
                rerank_seed_ids.push(item.memory_id.clone());
            }
        }
        for (_weight, list) in &s.candidates.fts_ranked_lists {
            for item in list.iter().take(neural_top / 2) {
                rerank_seed_ids.push(item.memory_id.clone());
            }
        }
        let mut rerank_seen = HashSet::new();
        let active_seeds: Vec<String> = rerank_seed_ids
            .into_iter()
            .filter(|mid: &String| rerank_seen.insert(mid.clone()))
            .collect();
        let lookup = s.tenant.lookup_by_memory_ids_batch(&active_seeds).unwrap_or_default();
        let obs_keys: Vec<(u64, String)> = active_seeds
            .iter()
            .filter_map(|mid: &String| lookup.get(mid).map(|(ts, _)| (*ts, mid.clone())))
            .take(neural_top)
            .collect();
        let observations = s.tenant.get_observations_batch(&obs_keys).unwrap_or_default();
        let mut rerank_items = Vec::new();
        let mut rerank_texts = Vec::new();
        for mid in active_seeds {
            if let Some(obs) = observations.get(&mid) {
                rerank_items.push(mid);
                rerank_texts.push(obs.textual_content.clone());
            }
        }
        for (item_chunk, text_chunk) in
            rerank_items.chunks(NEURAL_BATCH).zip(rerank_texts.chunks(NEURAL_BATCH))
        {
            let scores = s
                .state
                .semantic
                .predict_scores_batch(&s.query_text, text_chunk)
                .unwrap_or_else(|_| vec![0.0; item_chunk.len()]);
            for (mid, score) in item_chunk.iter().cloned().zip(scores.into_iter()) {
                s.candidates.neural_scores.insert(mid, score);
            }
        }
        (s.diag.rerank_ms, s.diag.rerank_us) = elapsed_ms_and_us(stage_start);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_active_rerank_decisions_apply() {
        assert!(RerankDecision::Always.applies());
        assert!(RerankDecision::Requested.applies());
        assert!(!RerankDecision::Disabled.applies());
        assert!(!RerankDecision::GateConfident.applies());
    }

    #[test]
    fn gate_marks_close_candidates_uncertain() {
        let hits = [(1, 0.20), (2, 0.205), (3, 0.21), (4, 0.215), (5, 0.22)];
        assert!(rerank_gate_uncertain(&hits, 0.05));
    }

    #[test]
    fn gate_accepts_a_clear_winner() {
        let hits = [(1, 0.10), (2, 0.30), (3, 0.35), (4, 0.38), (5, 0.40)];
        assert!(!rerank_gate_uncertain(&hits, 0.05));
    }

    #[test]
    fn gate_uses_the_last_available_candidate() {
        let hits = [(1, 0.10), (2, 0.50)];
        assert!(!rerank_gate_uncertain(&hits, 0.05));
        assert!(!rerank_gate_uncertain(&[], 0.05));
    }

    #[test]
    fn rerank_config_exposes_positive_limits() {
        let config = crate::config::Config::default();
        assert!(!rerank_policy_name(&config).is_empty());
        assert!(rerank_margin(&config) >= 0.0);
        assert!(rerank_top(&config) > 0);
    }
}
