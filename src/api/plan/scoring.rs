use std::collections::{HashMap, HashSet};

use super::types::{
    FourSignalWeights, QueryAdaptiveProfile, QueryIntent, QueryModality, QueryPlan,
    ScorableObservation, SessionBucket,
};
use crate::api::types::{EvidenceCard, QueryResult, RankedItem};
use crate::api::utils::{
    dedupe_preserve_order, extract_named_phrases, extract_temporal_terms, has_token,
    normalize_alpha_tokens, session_id_from_memory_id, singularize_token,
};
use crate::fts::tokenize_for_similarity;
use crate::ml::QueryIntentClassifier;
use crate::storage::MemoryKind;

pub fn build_query_adaptive_profile(_query: &str, plan: &QueryPlan) -> QueryAdaptiveProfile {
    let lexical_density = (plan.lexical_terms.len() as f32 / 10.0).clamp(0.0, 1.0);
    let has_numeric_cues =
        matches!(plan.intent, QueryIntent::NumericAggregation | QueryIntent::TemporalAggregation);
    let has_entity_like = plan.subject_entities.iter().any(|e| e.len() >= 6);

    let (mut semantic_scale, mut lexical_scale): (f32, f32) = (1.0, 1.0);
    if has_numeric_cues {
        semantic_scale *= 0.92;
        lexical_scale *= 1.18;
    }
    if matches!(plan.intent, QueryIntent::Recommendation) {
        semantic_scale *= 1.18;
        lexical_scale *= 0.90;
    }
    if matches!(plan.intent, QueryIntent::PeripheralMention) {
        semantic_scale *= 0.86;
        lexical_scale *= 1.34;
    }
    if matches!(plan.intent, QueryIntent::Inference) {
        semantic_scale *= 1.12;
        lexical_scale *= 1.06;
    }
    if lexical_density > 0.6 {
        lexical_scale *= 1.08;
    }
    if has_entity_like {
        semantic_scale *= 1.06;
    }

    QueryAdaptiveProfile {
        semantic_scale: semantic_scale.clamp(0.6, 1.6),
        lexical_scale: lexical_scale.clamp(0.6, 1.6),
        route_sessions: HashSet::new(),
        route_strength: 0.0,
    }
}

pub fn fuse_four_signals(
    semantic: f32,
    temporal: f32,
    confidence: f32,
    graph: f32,
    weights: &FourSignalWeights,
) -> f32 {
    semantic * weights.semantic
        + temporal * weights.temporal
        + confidence * weights.confidence
        + graph * weights.graph
}

pub fn compute_evidence_confidence(
    results: &[QueryResult],
    query: &str,
    classifier: Option<&QueryIntentClassifier>,
) -> f32 {
    use super::builder::build_query_plan;
    let lexical = build_query_plan(query, classifier);
    if results.is_empty() {
        return 0.0;
    }
    let top = results[0].similarity.max(0.0);
    let second = results.get(1).map(|r| r.similarity.max(0.0)).unwrap_or(0.0);
    let margin = (top - second).max(0.0);
    let support = results
        .iter()
        .take(6)
        .filter(|r| {
            let scorable = ScorableObservation::new(&r.textual_content);
            lexical_hit_count(&scorable, &lexical) >= 2
        })
        .count() as f32
        / 6.0;
    (0.55 * top + 0.25 * margin + 0.20 * support).clamp(0.0, 1.0)
}

pub fn routed_session_from_memory_id(memory_id: &str) -> Option<String> {
    session_id_from_memory_id(memory_id)
}

pub fn is_query_count_like(query: &str) -> bool {
    let q = query.to_ascii_lowercase();
    q.contains("how many")
        || q.contains("number of")
        || q.contains("count of")
        || q.contains("in total")
        || q.contains("total")
        || q.contains("combined")
        || q.contains("altogether")
        || q.contains("average")
}

pub fn extract_numeric_tokens(text: &str) -> Vec<f32> {
    let mut values = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            current.push(ch);
        } else if !current.is_empty() {
            if let Ok(v) = current.parse::<f32>() {
                values.push(v);
            }
            current.clear();
        }
    }
    if !current.is_empty() {
        if let Ok(v) = current.parse::<f32>() {
            values.push(v);
        }
    }
    values
}

pub fn numeric_signal_bonus(text: &str, lower: &str, intent: QueryIntent) -> f32 {
    if !matches!(intent, QueryIntent::NumericAggregation | QueryIntent::TemporalAggregation) {
        return 0.0;
    }
    let nums = extract_numeric_tokens(text);
    if nums.is_empty() {
        return -0.01;
    }
    let mut bonus = 0.03;
    if lower.contains("total")
        || lower.contains("combined")
        || lower.contains("in total")
        || lower.contains("average")
    {
        bonus += 0.02;
    }
    if nums.iter().any(|v| *v > 1000.0) && lower.contains('$') {
        bonus += 0.01;
    }
    bonus
}

pub fn temporal_consistency_adjustment(
    kind: MemoryKind,
    created_at_ms: u64,
    now_ms: u64,
    intent: QueryIntent,
) -> f32 {
    let age_days = (now_ms.saturating_sub(created_at_ms)) as f32 / (1000.0 * 86_400.0);
    let temporal_query = matches!(intent, QueryIntent::TemporalAggregation);
    let numeric_query = matches!(intent, QueryIntent::NumericAggregation);
    let mut delta = 0.0;
    if temporal_query {
        if age_days <= 45.0 {
            delta += 0.025;
        } else if age_days > 365.0 {
            delta -= 0.03;
        }
    }
    if numeric_query && kind == MemoryKind::Fact {
        if age_days <= 180.0 {
            delta += 0.02;
        } else if age_days > 730.0 {
            delta -= 0.025;
        }
    }
    delta
}

pub fn kind_query_bonus(kind: MemoryKind, plan: &QueryPlan, obs: &ScorableObservation) -> f32 {
    let mut bonus = if plan.prefer_distilled {
        match kind {
            MemoryKind::Fact => 0.12,
            MemoryKind::SessionSummary => 0.08,
            MemoryKind::Lesson => 0.04,
            _ => 0.0,
        }
    } else if plan.prefer_episodic {
        match kind {
            MemoryKind::Conversational => 0.08,
            MemoryKind::Fact => -0.06,
            MemoryKind::SessionSummary => -0.14,
            MemoryKind::Lesson => -0.05,
            MemoryKind::Decision | MemoryKind::Preference => -0.02,
        }
    } else {
        0.0
    };

    if matches!(plan.intent, QueryIntent::Inference) {
        bonus += match kind {
            MemoryKind::Fact => 0.07,
            MemoryKind::SessionSummary => 0.05,
            MemoryKind::Conversational => 0.02,
            MemoryKind::Lesson | MemoryKind::Decision | MemoryKind::Preference => 0.03,
        };
    }

    if !plan.temporal_terms.is_empty()
        && plan.temporal_terms.iter().any(|term| obs.lower.contains(term.as_str())) {
            bonus += 0.05;
        }

    bonus
}

pub fn lexical_hit_count(obs: &ScorableObservation, plan: &QueryPlan) -> usize {
    if plan.lexical_terms.is_empty() {
        return 0;
    }
    plan.lexical_terms.iter().filter(|term| obs.tokens.contains(term)).count()
}

pub fn lexical_overlap_bonus(obs: &ScorableObservation, plan: &QueryPlan) -> f32 {
    if plan.lexical_terms.is_empty() {
        return 0.0;
    }
    let hits = lexical_hit_count(obs, plan) as f32;
    let coverage = hits / plan.lexical_terms.len() as f32;
    coverage.min(1.0) * 0.08
}

pub fn entity_hit_count(obs: &ScorableObservation, plan: &QueryPlan) -> usize {
    if plan.subject_entities.is_empty() {
        return 0;
    }
    plan.subject_entities
        .iter()
        .filter(|entity| {
            let needle = entity.to_ascii_lowercase();
            obs.lower.contains(needle.as_str())
        })
        .count()
}

pub fn entity_coverage_bonus(obs: &ScorableObservation, plan: &QueryPlan) -> f32 {
    if plan.subject_entities.is_empty() {
        return 0.0;
    }
    let hits = entity_hit_count(obs, plan);
    let coverage = (hits as f32 / plan.subject_entities.len() as f32).min(1.0);

    if plan.cross_entity {
        if hits >= 2 {
            0.10 + coverage * 0.04
        } else if hits == 1 {
            -0.02
        } else {
            -0.05
        }
    } else if matches!(plan.intent, QueryIntent::Inference | QueryIntent::PeripheralMention) {
        coverage * 0.06
    } else {
        coverage * 0.03
    }
}

pub fn facet_match_mask(obs: &ScorableObservation, plan: &QueryPlan) -> u64 {
    if plan.coverage_facets.is_empty() {
        return 0;
    }

    let mut mask = 0_u64;
    for (idx, facet) in plan.coverage_facets.iter().take(64).enumerate() {
        let lexical_hits =
            facet.lexical_terms.iter().filter(|term| obs.tokens.contains(term)).count();
        let lexical_ok =
            if facet.lexical_terms.len() <= 2 { lexical_hits >= 1 } else { lexical_hits >= 2 };
        let entity_ok = if facet.entities.is_empty() {
            true
        } else {
            facet.entities.iter().any(|entity| obs.lower.contains(&entity.to_ascii_lowercase()))
        };
        let temporal_ok = if facet.temporal_terms.is_empty() {
            true
        } else {
            facet.temporal_terms.iter().any(|term| obs.lower.contains(term.as_str()))
        };

        if lexical_ok && entity_ok && temporal_ok {
            mask |= 1_u64 << idx;
        }
    }

    mask
}

pub fn temporal_hit_count(obs: &ScorableObservation, plan: &QueryPlan) -> usize {
    if plan.temporal_terms.is_empty() {
        return 0;
    }
    plan.temporal_terms.iter().filter(|term| obs.lower.contains(term.as_str())).count()
}

pub fn ordinal_signal_bonus(kind: MemoryKind, obs: &ScorableObservation, plan: &QueryPlan) -> f32 {
    use super::intent::ordinal_word;

    let Some(rank) = plan.ordinal_rank else {
        return 0.0;
    };

    let mut bonus = 0.0;
    if let Some(word) = ordinal_word(rank) {
        if obs.lower.contains(word) {
            bonus += 0.06;
        }
    }

    let suffix = if rank % 10 == 1 && rank % 100 != 11 {
        "st"
    } else if rank % 10 == 2 && rank % 100 != 12 {
        "nd"
    } else if rank % 10 == 3 && rank % 100 != 13 {
        "rd"
    } else {
        "th"
    };
    if obs.lower.contains(&format!("{rank}{suffix}")) {
        bonus += 0.05;
    }

    if extract_numeric_tokens(obs._text_ref).iter().any(|value| (*value).round() as usize == rank) {
        bonus += 0.02;
    }

    if kind == MemoryKind::Fact {
        bonus += 0.08;
    }

    if bonus == 0.0 {
        -0.01
    } else {
        bonus
    }
}

pub fn query_variant_weight(index: usize, modality: QueryModality, intent: QueryIntent) -> f32 {
    let base = match index {
        0 => 1.0,
        1 => 0.74,
        2 => 0.60,
        _ => 0.50,
    };

    let multiplier = match (intent, modality) {
        (QueryIntent::NumericAggregation, QueryModality::Lexical) => 1.20,
        (QueryIntent::NumericAggregation, QueryModality::Semantic) => 0.88,
        (QueryIntent::TemporalAggregation, QueryModality::Lexical) => 1.10,
        (QueryIntent::TemporalAggregation, QueryModality::Semantic) => 0.94,
        (QueryIntent::Recommendation, QueryModality::Semantic) => 1.15,
        (QueryIntent::Recommendation, QueryModality::Lexical) => 0.86,
        (QueryIntent::Inference, QueryModality::Semantic) => 1.14,
        (QueryIntent::Inference, QueryModality::Lexical) => 1.04,
        (QueryIntent::PeripheralMention, QueryModality::Semantic) => 0.82,
        (QueryIntent::PeripheralMention, QueryModality::Lexical) => 1.35,
        (QueryIntent::General, QueryModality::Semantic) => 1.0,
        (QueryIntent::General, QueryModality::Lexical) => 1.0,
    };

    base * multiplier
}

pub fn adaptive_rrf_k(plan: &QueryPlan) -> usize {
    if plan.cross_entity || plan.ordinal_rank.is_some() {
        45
    } else {
        match plan.intent {
            QueryIntent::PeripheralMention => 40,
            QueryIntent::Inference => 50,
            QueryIntent::TemporalAggregation => 55,
            QueryIntent::NumericAggregation => 55,
            QueryIntent::Recommendation | QueryIntent::General => 60,
        }
    }
}

pub fn session_coverage_bonus(items: &[EvidenceCard], plan: &QueryPlan) -> f32 {
    if items.is_empty() {
        return 0.0;
    }

    let inspect_n = if plan.needs_decomposition { 6 } else { 4 };
    let mut lexical_union = HashSet::new();
    let mut temporal_union = HashSet::new();
    let mut entity_union = HashSet::new();
    let mut kind_union = HashSet::new();
    let mut support_items = 0usize;
    let mut max_entity_hits = 0usize;
    let mut has_distilled = false;
    let mut has_episodic = false;
    let mut facet_union_mask = 0_u64;
    let mut max_facet_hits = 0_u32;

    for item in items.iter().take(inspect_n) {
        let token_set: HashSet<String> = tokenize_for_similarity(&item.claim_text)
            .into_iter()
            .map(|token| singularize_token(&token))
            .collect();
        let lower = item.claim_text.to_ascii_lowercase();

        for term in &plan.lexical_terms {
            if token_set.contains(term.as_str()) {
                lexical_union.insert(term.clone());
            }
        }
        for term in &plan.temporal_terms {
            if lower.contains(term.as_str()) {
                temporal_union.insert(term.clone());
            }
        }
        for entity in &plan.subject_entities {
            if lower.contains(entity.to_ascii_lowercase().as_str()) {
                entity_union.insert(entity.clone());
            }
        }

        if item.lexical_hits > 0 || item.temporal_hits > 0 || item.entity_hits > 0 {
            support_items += 1;
        }
        max_entity_hits = max_entity_hits.max(item.entity_hits);
        facet_union_mask |= item.facet_mask;
        max_facet_hits = max_facet_hits.max(item.facet_mask.count_ones());
        let is_distilled = matches!(
            item.internal_kind,
            MemoryKind::Fact | MemoryKind::SessionSummary | MemoryKind::Lesson
        );
        has_distilled |= is_distilled;
        has_episodic |= item.internal_kind == MemoryKind::Conversational;
        kind_union.insert(format!("{:?}", item.internal_kind));
    }

    let lexical_coverage = if plan.lexical_terms.is_empty() {
        0.0
    } else {
        lexical_union.len() as f32 / plan.lexical_terms.len() as f32
    };
    let temporal_coverage = if plan.temporal_terms.is_empty() {
        0.0
    } else {
        temporal_union.len() as f32 / plan.temporal_terms.len() as f32
    };
    let entity_coverage = if plan.subject_entities.is_empty() {
        0.0
    } else {
        entity_union.len() as f32 / plan.subject_entities.len() as f32
    };
    let facet_coverage = if plan.coverage_facets.is_empty() {
        0.0
    } else {
        facet_union_mask.count_ones() as f32 / plan.coverage_facets.len().min(64) as f32
    };

    let mut bonus = 0.0;
    bonus += lexical_coverage.min(1.0) * if plan.needs_decomposition { 0.05 } else { 0.02 };
    bonus += temporal_coverage.min(1.0)
        * if matches!(plan.intent, QueryIntent::TemporalAggregation) { 0.05 } else { 0.02 };
    bonus += entity_coverage.min(1.0)
        * if plan.cross_entity {
            0.10
        } else if !plan.subject_entities.is_empty() {
            0.03
        } else {
            0.0
        };

    if plan.needs_decomposition && support_items >= 2 {
        bonus += 0.03 + ((support_items - 2).min(2) as f32) * 0.01;
    }
    if facet_coverage > 0.0 {
        bonus += facet_coverage.min(1.0) * if plan.needs_decomposition { 0.10 } else { 0.03 };
    }
    if plan.needs_decomposition
        && facet_union_mask.count_ones() >= 2
        && max_facet_hits < facet_union_mask.count_ones()
    {
        bonus += 0.05;
    }
    if plan.needs_decomposition && has_distilled && has_episodic {
        bonus += 0.04;
    }
    if kind_union.len() >= 2 {
        bonus += (kind_union.len().saturating_sub(1).min(2) as f32) * 0.015;
    }
    if plan.cross_entity
        && entity_union.len() == plan.subject_entities.len()
        && max_entity_hits < plan.subject_entities.len()
    {
        bonus += 0.08;
    } else if plan.cross_entity && entity_coverage < 1.0 {
        bonus -= 0.03;
    }
    if plan.ordinal_rank.is_some() && has_distilled && temporal_coverage > 0.0 {
        bonus += 0.03;
    }

    bonus
}

pub fn weighted_reciprocal_rank_fusion(
    ranked_lists: Vec<(f32, Vec<RankedItem>)>,
    k: usize,
) -> Vec<(String, u64, f32)> {
    let mut rrf_scores: HashMap<String, (u64, f32)> = HashMap::new();

    for (weight, list) in ranked_lists {
        let weight = weight.max(0.0);
        if weight == 0.0 {
            continue;
        }

        for (rank, item) in list.into_iter().enumerate() {
            let score = weight / ((k + rank + 1) as f32);
            let entry = rrf_scores.entry(item.memory_id).or_insert((item.timestamp, 0.0));
            entry.1 += score;
        }
    }

    let mut fused: Vec<(String, u64, f32)> =
        rrf_scores.into_iter().map(|(id, (ts, score))| (id, ts, score)).collect();

    fused.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

pub fn kind_priority(kind: MemoryKind, prefer_distilled: bool) -> u8 {
    if prefer_distilled {
        match kind {
            MemoryKind::Fact => 4,
            MemoryKind::SessionSummary => 3,
            MemoryKind::Lesson => 2,
            MemoryKind::Conversational => 1,
            MemoryKind::Decision | MemoryKind::Preference => 0,
        }
    } else {
        match kind {
            MemoryKind::Conversational => 4,
            MemoryKind::Fact => 3,
            MemoryKind::SessionSummary => 2,
            MemoryKind::Lesson => 1,
            MemoryKind::Decision | MemoryKind::Preference => 0,
        }
    }
}

pub fn session_requirement_mask(items: &[EvidenceCard], plan: &QueryPlan) -> u64 {
    if plan.requirements.is_empty() {
        return 0;
    }

    let inspect_n = if plan.needs_decomposition { 6 } else { 4 };
    let mut lexical_union = HashSet::new();
    let mut temporal_union = HashSet::new();
    let mut entity_union = HashSet::new();

    for item in items.iter().take(inspect_n) {
        let token_set: HashSet<String> = tokenize_for_similarity(&item.claim_text)
            .into_iter()
            .map(|token| singularize_token(&token))
            .collect();
        let lower = item.claim_text.to_ascii_lowercase();

        lexical_union.extend(token_set);
        for term in extract_temporal_terms(&lower) {
            temporal_union.insert(term);
        }
        for entity in &plan.subject_entities {
            if lower.contains(entity.to_ascii_lowercase().as_str()) {
                entity_union.insert(entity.clone());
            }
        }
    }

    let mut mask = 0_u64;
    for (idx, requirement) in plan.requirements.iter().take(64).enumerate() {
        let lexical_hits = requirement
            .lexical_terms
            .iter()
            .filter(|term| lexical_union.contains(term.as_str()))
            .count();
        let lexical_ok = if requirement.lexical_terms.len() <= 2 {
            lexical_hits >= 1
        } else {
            lexical_hits >= 2
        };
        let temporal_ok = if requirement.temporal_terms.is_empty() {
            true
        } else {
            requirement.temporal_terms.iter().any(|term| temporal_union.contains(term))
        };
        let entity_ok = if requirement.entities.is_empty() {
            true
        } else if requirement.require_all_entities {
            requirement.entities.iter().all(|entity| entity_union.contains(entity))
        } else {
            requirement.entities.iter().any(|entity| entity_union.contains(entity))
        };

        if lexical_ok && temporal_ok && entity_ok {
            mask |= 1_u64 << idx;
        }
    }

    mask
}

pub fn reorder_session_items_for_requirements(
    items: &[EvidenceCard],
    plan: &QueryPlan,
) -> Vec<EvidenceCard> {
    if plan.requirements.is_empty() {
        return items.to_vec();
    }

    let mut remaining = items.to_vec();
    let mut ordered = Vec::with_capacity(remaining.len());
    let mut covered_mask = 0_u64;

    while !remaining.is_empty() {
        let mut best_idx = 0usize;
        let mut best_gain = 0u32;
        let mut best_score = f32::MIN;

        for (idx, item) in remaining.iter().enumerate() {
            let gain = (item.facet_mask & !covered_mask).count_ones();
            if gain > best_gain || (gain == best_gain && item.final_score > best_score) {
                best_idx = idx;
                best_gain = gain;
                best_score = item.final_score;
            }
        }

        let item = remaining.remove(best_idx);
        covered_mask |= item.facet_mask;
        ordered.push(item);
    }

    ordered
}

pub fn select_candidates_with_session_head(
    candidates: Vec<EvidenceCard>,
    limit: usize,
    plan: &QueryPlan,
    prefer_distilled: bool,
    prefer_episodic: bool,
) -> Vec<EvidenceCard> {
    if limit == 0 || candidates.is_empty() {
        return Vec::new();
    }

    let mut grouped: HashMap<String, Vec<EvidenceCard>> = HashMap::new();
    for candidate in candidates {
        let session_key = if candidate.source_session_id.is_empty() {
            session_id_from_memory_id(&candidate.source_memory_id)
                .unwrap_or_else(|| candidate.source_memory_id.clone())
        } else {
            candidate.source_session_id.clone()
        };
        grouped.entry(session_key).or_default().push(candidate);
    }

    let mut sessions = grouped.into_values().map(|mut items| {
            items.sort_by(|a, b| {
                b.final_score
                    .partial_cmp(&a.final_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        kind_priority(b.internal_kind, prefer_distilled)
                            .cmp(&kind_priority(a.internal_kind, prefer_distilled))
                    })
            });

            let top1 = items.first().map(|item| item.final_score).unwrap_or(0.0);
            let top2 = items.get(1).map(|item| item.final_score).unwrap_or(0.0);
            let top3 = items.get(2).map(|item| item.final_score).unwrap_or(0.0);
            let max_lexical_hits = items.iter().map(|item| item.lexical_hits).max().unwrap_or(0);
            let max_temporal_hits = items.iter().map(|item| item.temporal_hits).max().unwrap_or(0);
            let max_entity_hits = items.iter().map(|item| item.entity_hits).max().unwrap_or(0);
            let conversational_count = items
                .iter()
                .filter(|item| item.internal_kind == MemoryKind::Conversational)
                .count();
            let distilled_count = items
                .iter()
                .filter(|item| {
                    matches!(
                        item.internal_kind,
                        MemoryKind::Fact | MemoryKind::SessionSummary | MemoryKind::Lesson
                    )
                })
                .count();
            let multi_hit_bonus = ((items.len().saturating_sub(1)).min(3) as f32) * 0.02;
            let lexical_bonus = (max_lexical_hits as f32).min(4.0) * 0.012;
            let temporal_bonus = (max_temporal_hits as f32).min(2.0) * 0.018;
            let entity_bonus = (max_entity_hits as f32).min(3.0) * 0.02;
            let distilled_bonus = if prefer_distilled && distilled_count > 0 { 0.03 } else { 0.0 };
            let episodic_bonus = if prefer_episodic {
                if conversational_count > 0 {
                    0.04
                } else {
                    -0.03
                }
            } else {
                0.0
            };
            let coverage_bonus = session_coverage_bonus(&items, plan);
            let facet_mask = items.iter().fold(0_u64, |mask, item| mask | item.facet_mask);
            let facet_count = facet_mask.count_ones() as f32;
            let coverage_mode_bonus = if plan.coverage_mode {
                facet_count.min(5.0) * 0.038
                    + (max_lexical_hits as f32).min(5.0) * 0.014
                    + (items.len().min(4) as f32) * 0.014
            } else {
                0.0
            };
            let (top1_weight, top2_weight, top3_weight) =
                if plan.coverage_mode { (0.46, 0.31, 0.23) } else { (0.68, 0.22, 0.10) };
            let session_score = top1 * top1_weight
                + top2 * top2_weight
                + top3 * top3_weight
                + multi_hit_bonus
                + lexical_bonus
                + temporal_bonus
                + entity_bonus
                + distilled_bonus
                + episodic_bonus
                + coverage_bonus
                + coverage_mode_bonus;

            SessionBucket {
                score: session_score,
                requirement_mask: session_requirement_mask(&items, plan),
                facet_mask,
                max_entity_hits,
                max_lexical_hits,
                max_temporal_hits,
                items: reorder_session_items_for_requirements(&items, plan),
            }
        })
        .collect::<Vec<_>>();

    let use_requirement_set_selection = (plan.coverage_mode || !plan.requirements.is_empty())
        && (plan.needs_decomposition
            || plan.cross_entity
            || plan.ordinal_rank.is_some()
            || plan.coverage_mode
            || matches!(plan.intent, QueryIntent::Inference));

    if use_requirement_set_selection {
        let mut ordered = Vec::with_capacity(sessions.len());
        let mut chosen = vec![false; sessions.len()];
        let req_count = plan.requirements.len().min(64);
        let mut uncovered_mask = if req_count == 0 {
            0
        } else if req_count >= 64 {
            u64::MAX
        } else {
            (1_u64 << req_count) - 1
        };
        let facet_count = plan.coverage_facets.len().min(64);
        let mut uncovered_facet_mask = if facet_count == 0 {
            0
        } else if facet_count >= 64 {
            u64::MAX
        } else {
            (1_u64 << facet_count) - 1
        };
        let mut selected_entity_hits = 0usize;
        let mut selected_temporal_hit = false;

        while ordered.len() < sessions.len() {
            let mut best_idx = None;
            let mut best_selection_score = f32::MIN;

            for (idx, session) in sessions.iter().enumerate() {
                if chosen[idx] {
                    continue;
                }
                let marginal_mask = session.requirement_mask & uncovered_mask;
                let marginal_gain = marginal_mask.count_ones() as f32;
                let marginal_facet_mask = session.facet_mask & uncovered_facet_mask;
                let marginal_facet_gain = marginal_facet_mask.count_ones() as f32;
                let entity_gain =
                    session.max_entity_hits.saturating_sub(selected_entity_hits).min(2) as f32;
                let temporal_gain =
                    if !selected_temporal_hit && session.max_temporal_hits > 0 { 1.0 } else { 0.0 };
                let full_requirement_hit =
                    if session.requirement_mask != 0 && marginal_mask == session.requirement_mask {
                        0.04
                    } else {
                        0.0
                    };
                let redundancy_penalty = if (uncovered_mask != 0 || uncovered_facet_mask != 0)
                    && marginal_gain == 0.0
                    && marginal_facet_gain == 0.0
                    && entity_gain == 0.0
                    && temporal_gain == 0.0
                {
                    0.08
                } else {
                    0.0
                };
                let lexical_support = (session.max_lexical_hits as f32).min(5.0) * 0.014;
                let hard_selection =
                    plan.needs_decomposition || plan.cross_entity || plan.coverage_mode;
                let base_weight = if hard_selection { 0.62 } else { 1.0 };
                let no_coverage_penalty = if hard_selection
                    && session.requirement_mask == 0
                    && session.facet_mask == 0
                    && session.max_lexical_hits == 0
                {
                    0.14
                } else if hard_selection
                    && (uncovered_mask != 0 || uncovered_facet_mask != 0)
                    && marginal_gain == 0.0
                    && marginal_facet_gain == 0.0
                {
                    0.06
                } else {
                    0.0
                };
                let missing_subject_penalty = if !plan.subject_entities.is_empty()
                    && session.max_entity_hits == 0
                    && hard_selection
                {
                    0.05
                } else {
                    0.0
                };
                let selection_score = session.score * base_weight
                    + marginal_gain * (if hard_selection { 0.36 } else { 0.20 })
                    + marginal_facet_gain * (if hard_selection { 0.18 } else { 0.09 })
                    + entity_gain * (if hard_selection { 0.08 } else { 0.05 })
                    + temporal_gain * (if hard_selection { 0.060 } else { 0.035 })
                    + lexical_support
                    + full_requirement_hit
                    - redundancy_penalty
                    - no_coverage_penalty
                    - missing_subject_penalty;
                if selection_score > best_selection_score {
                    best_selection_score = selection_score;
                    best_idx = Some(idx);
                }
            }

            let Some(best_idx) = best_idx else {
                break;
            };
            chosen[best_idx] = true;
            uncovered_mask &= !sessions[best_idx].requirement_mask;
            uncovered_facet_mask &= !sessions[best_idx].facet_mask;
            selected_entity_hits = selected_entity_hits.max(sessions[best_idx].max_entity_hits);
            selected_temporal_hit |= sessions[best_idx].max_temporal_hits > 0;
            ordered.push(best_idx);
        }

        sessions = ordered.into_iter().filter_map(|idx| sessions.get(idx).cloned()).collect();
    } else {
        sessions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    }

    let mut selected = Vec::with_capacity(limit);
    let mut seen_memory_ids = HashSet::new();
    let mut round = 0usize;

    while selected.len() < limit {
        let mut progressed = false;
        for session in sessions.iter() {
            if let Some(candidate) = session.items.get(round) {
                if seen_memory_ids.insert(candidate.source_memory_id.clone()) {
                    selected.push(candidate.clone());
                    progressed = true;
                    if selected.len() >= limit {
                        break;
                    }
                }
            }
        }
        if !progressed {
            break;
        }
        round += 1;
    }

    if selected.len() < limit {
        let mut leftovers =
            sessions.into_iter().flat_map(|session| session.items.into_iter()).collect::<Vec<_>>();
        leftovers.sort_by(|a, b| {
            b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        for candidate in leftovers {
            if selected.len() >= limit {
                break;
            }
            if seen_memory_ids.insert(candidate.source_memory_id.clone()) {
                selected.push(candidate);
            }
        }
    }

    selected
}

pub fn build_observation_block(
    profile_json: Option<&str>,
    mem_scenes: &[String],
    top_chunks: &[String],
) -> String {
    let mut block = String::from("## [TELLODB MEMORY CONTEXT]\n");

    if let Some(profile) = profile_json {
        block.push_str("### Core State\n");
        block.push_str(profile);
        block.push('\n');
    }

    if !mem_scenes.is_empty() {
        block.push_str("### Knowledge Graph Scenes\n");
        for scene in mem_scenes {
            block.push_str("- ");
            block.push_str(scene);
            block.push('\n');
        }
    }

    if !top_chunks.is_empty() {
        block.push_str("### Relevant Memories\n");
        for chunk in top_chunks {
            block.push_str("- ");
            block.push_str(chunk);
            block.push('\n');
        }
    }

    block
}

pub fn is_coverage_style_query(query: &str) -> bool {
    let lower = query.to_ascii_lowercase();
    let starts = [
        "what items",
        "what activities",
        "what places",
        "what locations",
        "which locations",
        "which places",
        "which cities",
        "which states",
        "which countries",
        "what kinds",
        "what kind of",
        "what types",
        "what type of",
        "what attributes",
        "what hobbies",
        "what problems",
        "what events",
        "what projects",
        "what recipes",
        "what classes",
        "what shared",
        "where has",
        "where have",
        "who are",
        "which of",
    ];
    if starts.iter().any(|prefix| lower.starts_with(prefix)) {
        return true;
    }

    let cues = [
        " both ",
        " and ",
        " shared ",
        " in common",
        " has ",
        " have ",
        " have been ",
        " does ",
        " do ",
        " besides ",
        " other than ",
        " over time",
        " across",
        " different ",
    ];
    let list_nouns = [
        "activities",
        "places",
        "locations",
        "cities",
        "states",
        "countries",
        "items",
        "attributes",
        "hobbies",
        "problems",
        "events",
        "classes",
        "projects",
        "pets",
        "dogs",
        "friends",
        "family members",
        "volunteering",
    ];
    cues.iter().any(|cue| lower.contains(cue)) && list_nouns.iter().any(|noun| lower.contains(noun))
}

pub fn query_prefers_distilled(query: &str) -> bool {
    let lower = query.to_ascii_lowercase();
    let tokens = normalize_alpha_tokens(query);

    lower.contains("in common")
        || ["both", "why", "might", "would", "considered", "future"]
            .iter()
            .any(|needle| has_token(&tokens, needle))
        || [
            "causes",
            "activities",
            "events",
            "areas",
            "items",
            "names",
            "people",
            "focus",
            "many",
            "likely",
            "infer",
            "inference",
            "prefer",
            "preference",
        ]
        .iter()
        .any(|needle| has_token(&tokens, needle))
}

pub fn query_prefers_episodic(query: &str) -> bool {
    let lower = query.to_ascii_lowercase();
    let tokens = normalize_alpha_tokens(query);

    if query_prefers_distilled(query) {
        return false;
    }

    lower.starts_with("when ")
        || lower.starts_with("where ")
        || lower.starts_with("who ")
        || lower.starts_with("what did ")
        || lower.starts_with("what was ")
        || lower.starts_with("what is ")
        || lower.starts_with("how many ")
        || lower.starts_with("which ")
        || ["photo", "shared", "made", "watched", "went", "happened"]
            .iter()
            .any(|needle| has_token(&tokens, needle))
}

pub fn extract_subject_entities(query: &str) -> Vec<String> {
    let stop = [
        "what", "when", "where", "who", "why", "how", "which", "would", "could", "should", "did",
        "does", "is", "are", "was", "were", "can", "will", "might", "likely", "both", "and", "or",
        "the", "a", "an", "of", "in", "on", "for", "to", "with", "from",
    ];

    let query_line = vec![query.to_string()];
    dedupe_preserve_order(
        extract_named_phrases(&query_line)
            .into_iter()
            .map(|phrase| normalize_subject_entity_phrase(&phrase))
            .filter(|phrase| !phrase.is_empty())
            .filter(|phrase| {
                let normalized = phrase.to_ascii_lowercase();
                !stop.contains(&normalized.as_str())
            })
            .filter(|phrase| {
                let tokens = normalize_alpha_tokens(phrase);
                !tokens.is_empty() && tokens.len() <= 2
            })
            .collect(),
    )
}

fn normalize_subject_entity_phrase(phrase: &str) -> String {
    let mut normalized = phrase
        .trim()
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '\'' && ch != ' ')
        .to_string();
    let lower = normalized.to_ascii_lowercase();
    if lower.ends_with("'s") {
        normalized.truncate(normalized.len().saturating_sub(2));
    } else if lower.ends_with("s'") {
        normalized.truncate(normalized.len().saturating_sub(1));
    }
    normalized.trim().to_string()
}
