use super::expansions::{
    build_coverage_facets, build_cross_entity_subqueries, build_expansion_query,
    build_fact_slot_queries, build_hypothetical_semantic_queries, build_inference_semantic_hints,
    build_keyword_query, build_peripheral_fts_query, build_purchase_queries,
    build_query_expansion_terms_with_profile, build_query_requirements,
    infer_query_fact_key_with_profile, is_purchase_query,
};
use super::intent::{
    classify_query_intent, extract_ordinal_rank, ordinal_word, strip_ordinal_tokens,
};
use super::scoring::{
    extract_subject_entities, is_coverage_style_query, query_prefers_distilled,
    query_prefers_episodic,
};
use super::types::{QueryIntent, QueryPlan};
use crate::core::calendar::extract_temporal_terms;
use crate::core::text::{dedupe_preserve_order, is_low_signal_keyword, singularize_token};
use crate::fts::tokenize_for_similarity;
use crate::heuristics::Profile;
use crate::ml::QueryIntentClassifier;

pub fn build_query_plan(query: &str, classifier: Option<&QueryIntentClassifier>) -> QueryPlan {
    build_query_plan_with_profile(query, classifier, Profile::Generic)
}

#[allow(clippy::too_many_lines)]
pub fn build_query_plan_with_profile(
    query: &str,
    classifier: Option<&QueryIntentClassifier>,
    profile: Profile,
) -> QueryPlan {
    let keyword_query = build_keyword_query(query);
    let subject_entities = extract_subject_entities(query);
    let ordinal_rank = extract_ordinal_rank(query);
    let inferred_intent = classify_query_intent(query, classifier);
    let slot_key = infer_query_fact_key_with_profile(query, profile);
    let expansion_terms = build_query_expansion_terms_with_profile(
        query,
        slot_key.as_deref(),
        inferred_intent,
        &subject_entities,
        profile,
    );
    let cross_entity = {
        let lower = query.to_ascii_lowercase();
        let multi_hop_cue = lower.contains(" both ")
            || lower.contains(" and ")
            || lower.contains(" shared ")
            || lower.contains(" share ")
            || lower.contains(" in common")
            || lower.contains(" compare")
            || lower.contains(" similar")
            || lower.contains(" same ");
        subject_entities.len() >= 2 && multi_hop_cue
    };
    let needs_decomposition = {
        let lower = query.to_ascii_lowercase();
        let question_words = lower.contains(" would ")
            || lower.contains(" could ")
            || lower.contains(" why ")
            || lower.contains(" how ");
        cross_entity
            || inferred_intent == QueryIntent::Inference
            || inferred_intent == QueryIntent::TemporalAggregation
            || question_words
    };
    let coverage_mode = is_coverage_style_query(query)
        || cross_entity
        || matches!(
            inferred_intent,
            QueryIntent::NumericAggregation | QueryIntent::TemporalAggregation
        );

    let mut semantic_queries = vec![query.to_string()];
    let mut fts_queries = vec![query.to_string()];

    if let Some(ref keyword_query) = keyword_query {
        semantic_queries.push(keyword_query.clone());
        fts_queries.push(keyword_query.clone());
    }

    if let Some(expansion_query) = build_expansion_query(&subject_entities, &expansion_terms) {
        semantic_queries.push(expansion_query.clone());
        fts_queries.push(expansion_query);
    }

    if let Some(slot_key) = slot_key.as_deref() {
        let (slot_semantic, slot_fts) = build_fact_slot_queries(query, &subject_entities, slot_key);
        semantic_queries.extend(slot_semantic);
        fts_queries.extend(slot_fts);
    }

    semantic_queries.extend(build_hypothetical_semantic_queries(
        query,
        &subject_entities,
        cross_entity,
        inferred_intent,
    ));

    if let Some(rank) = ordinal_rank {
        let stripped = strip_ordinal_tokens(query);
        if !stripped.is_empty() && !stripped.eq_ignore_ascii_case(query) {
            semantic_queries.push(stripped.clone());
            fts_queries.push(stripped);
        }
        if let Some(word) = ordinal_word(rank) {
            fts_queries.push(format!("\"{word}\""));
        }
    }

    if cross_entity {
        let (cross_semantic, cross_fts) = build_cross_entity_subqueries(query, &subject_entities);
        semantic_queries.extend(cross_semantic);
        fts_queries.extend(cross_fts);
    }

    if inferred_intent == QueryIntent::Inference {
        semantic_queries.extend(build_inference_semantic_hints(query, &subject_entities));
    }

    if inferred_intent == QueryIntent::PeripheralMention {
        if let Some(peripheral_query) = build_peripheral_fts_query(query, &subject_entities) {
            fts_queries.push(peripheral_query);
        }
    }

    if is_purchase_query(query) {
        let (purchase_semantic, purchase_fts) = build_purchase_queries(query, &subject_entities);
        semantic_queries.extend(purchase_semantic);
        fts_queries.extend(purchase_fts);
    }

    QueryPlan {
        semantic_queries: dedupe_preserve_order(semantic_queries),
        fts_queries: dedupe_preserve_order(fts_queries),
        coverage_facets: build_coverage_facets(
            query,
            keyword_query.as_deref(),
            &subject_entities,
            cross_entity,
            ordinal_rank,
            slot_key.as_deref(),
            &expansion_terms,
        ),
        requirements: build_query_requirements(
            query,
            keyword_query.as_deref(),
            &subject_entities,
            cross_entity,
            ordinal_rank,
            slot_key.as_deref(),
            &expansion_terms,
        ),
        prefers_latest: query_prefers_latest(query),
        prefer_distilled: query_prefers_distilled(query),
        prefer_episodic: query_prefers_episodic(query),
        temporal_terms: extract_temporal_terms(query),
        lexical_terms: {
            let mut terms = tokenize_for_similarity(query)
                .into_iter()
                .map(|token| singularize_token(&token))
                .filter(|token| !is_low_signal_keyword(token) && token.len() >= 3)
                .collect::<Vec<_>>();
            terms.extend(expansion_terms);
            dedupe_preserve_order(terms)
        },
        intent: inferred_intent,
        subject_entities,
        cross_entity,
        needs_decomposition,
        coverage_mode,
        ordinal_rank,
        fact_key: slot_key,
    }
}

/// Cues that the question asks for the current state rather than history.
/// Explicit past references ("used to", "before", "as of", "back in", "last
/// year") win, so "where did I live before?" is not treated as current.
pub fn query_prefers_latest(query: &str) -> bool {
    const PAST: [&str; 9] = [
        " used to ",
        " before ",
        " previously ",
        " as of ",
        " back in ",
        " last year ",
        " originally ",
        " at first ",
        " formerly ",
    ];
    const CURRENT: [&str; 12] = [
        " currently ",
        " current ",
        " right now ",
        " now ",
        " these days ",
        " at present ",
        " at the moment ",
        " latest ",
        " most recent ",
        " nowadays ",
        " still ",
        " anymore ",
    ];
    let lower = format!(" {} ", query.to_ascii_lowercase().replace(['?', '.', ',', '!'], " "));
    if PAST.iter().any(|cue| lower.contains(cue)) {
        return false;
    }
    CURRENT.iter().any(|cue| lower.contains(cue))
}

#[cfg(test)]
mod prefers_latest_tests {
    use super::query_prefers_latest;

    #[test]
    fn detects_current_value_questions() {
        assert!(query_prefers_latest("Where do I currently live?"));
        assert!(query_prefers_latest("What city do I live in right now?"));
        assert!(query_prefers_latest("Is she still working at Acme?"));
        assert!(!query_prefers_latest("Where did I live before moving?"));
        assert!(!query_prefers_latest("Where was I living as of 2023/05/01?"));
        assert!(!query_prefers_latest("What do I know about hiking?"));
    }
}
