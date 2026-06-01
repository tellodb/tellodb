use super::expansions::*;
use super::intent::*;
use super::scoring::*;
use super::types::*;
use crate::api::utils::{
    dedupe_preserve_order, extract_temporal_terms, is_low_signal_keyword, singularize_token,
};
use crate::fts::tokenize_for_similarity;
use crate::ml::QueryIntentClassifier;

pub fn build_query_plan(query: &str, classifier: Option<&QueryIntentClassifier>) -> QueryPlan {
    let keyword_query = build_keyword_query(query);
    let subject_entities = extract_subject_entities(query);
    let ordinal_rank = extract_ordinal_rank(query);
    let inferred_intent = classify_query_intent(query, classifier);
    let slot_key = infer_query_fact_key(query);
    let expansion_terms =
        build_query_expansion_terms(query, slot_key.as_deref(), inferred_intent, &subject_entities);
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
    }
}
