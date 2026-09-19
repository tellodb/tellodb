use crate::api::ingest::dialogue::extract_bracketed_header_value;
use crate::api::ingest::fact::build_atomic_memory_card_payloads;
use crate::api::ingest::fact::infer_fact_key_with_profile;
use crate::api::ingest::salient::{
    build_keyword_companion_text, extract_named_phrases, extract_salient_terms,
    truncate_for_companion,
};
use crate::api::types::IngestPayload;
use crate::core::calendar::extract_temporal_terms;
use crate::core::memory_id::MemoryId;
use crate::core::text::normalize_fact_text;
use crate::heuristics::Profile;

pub fn build_event_companion_text(payload: &IngestPayload) -> Option<String> {
    let dialogue_lines =
        crate::api::ingest::dialogue::extract_dialogue_messages(&payload.textual_content);
    let source_lines = if dialogue_lines.is_empty() {
        payload
            .textual_content
            .lines()
            .map(|line| {
                normalize_fact_text(crate::api::ingest::dialogue::strip_leading_bracketed_prefixes(
                    line,
                ))
            })
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
    } else {
        dialogue_lines
            .iter()
            .map(|(speaker, line)| format!("{speaker}: {line}"))
            .collect::<Vec<_>>()
    };
    if source_lines.is_empty() {
        return None;
    }

    let time_terms = extract_temporal_terms(&payload.textual_content);
    let entities = extract_named_phrases(&source_lines);
    let salient_terms = extract_salient_terms(&payload.textual_content, 5);
    let anchor = truncate_for_companion(&source_lines.join(" | "), 220);

    if time_terms.is_empty() && entities.is_empty() && salient_terms.is_empty() {
        return None;
    }

    let mut parts = vec!["Canonical event memory".to_string()];
    if !time_terms.is_empty() {
        parts.push(format!("time {}", time_terms.join(", ")));
    }
    if !entities.is_empty() {
        parts.push(format!(
            "participants {}",
            entities.into_iter().take(4).collect::<Vec<_>>().join(", ")
        ));
    }
    if !salient_terms.is_empty() {
        parts.push(format!("topics {}", salient_terms.join(", ")));
    }
    parts.push(anchor);
    Some(parts.join(": "))
}

pub fn build_relation_companion_payloads(payload: &IngestPayload) -> Vec<IngestPayload> {
    payload
        .relations
        .iter()
        .enumerate()
        .map(|(idx, (subject, predicate, object))| {
            let human_predicate = predicate.replace('_', " ");
            IngestPayload {
                entity_id: payload.entity_id.clone(),
                memory_id: MemoryId::derived_from(&payload.memory_id, &format!("rel{idx}")),
                timestamp: payload.timestamp,
                textual_content: format!(
                    "Canonical relation: {subject} {human_predicate} {object}"
                ),
                relations: payload.relations.clone(),
                kind: Some("fact".to_string()),
                fact_key: Some(format!("relation_{}", predicate.replace(' ', "_"))),
                source_memory_id: Some(payload.memory_id.clone()),
                index_semantic: Some(false),
                enable_semantic_dedup: Some(false),
                enable_consolidation: Some(false),
                content_type: payload.content_type.clone(),
                fact_operation: Some("derive".to_string()),
                fact_confidence: Some(0.90),
                fact_subject: Some(subject.clone()),
                fact_predicate: Some(human_predicate),
                fact_object: Some(object.clone()),
                ..Default::default()
            }
        })
        .collect()
}

pub fn build_companion_payloads(payload: &IngestPayload) -> Vec<IngestPayload> {
    build_companion_payloads_with_profile(payload, Profile::Generic)
}

#[allow(clippy::too_many_lines)]
pub fn build_companion_payloads_with_profile(
    payload: &IngestPayload,
    profile: Profile,
) -> Vec<IngestPayload> {
    // Companions are per conversation turn; memories outside a session get none.
    if payload.session_id.as_deref().map_or(true, str::is_empty) {
        return Vec::new();
    }
    let turn_index = payload.turn_index.unwrap_or(0);

    let session_focus = extract_bracketed_header_value(&payload.textual_content, "Session Focus");
    let (fallback_gist, fact_texts) =
        crate::api::ingest::dialogue::extract_companion_texts(&payload.textual_content);
    let mut companions = Vec::new();

    if turn_index == 0 {
        let session_companion_source = session_focus.as_ref().map_or_else(
            || payload.textual_content.clone(),
            |focus| format!("{focus}\n{}", payload.textual_content),
        );
        let gist = session_focus.map(|focus| format!("Session gist: {focus}")).or(fallback_gist);
        let keyword_index = build_keyword_companion_text(&session_companion_source);

        if let Some(gist_text) = gist {
            companions.push(IngestPayload {
                entity_id: payload.entity_id.clone(),
                memory_id: MemoryId::derived_from(&payload.memory_id, "gist"),
                timestamp: payload.timestamp,
                textual_content: gist_text,
                relations: payload.relations.clone(),
                kind: Some("session_summary".to_string()),
                fact_key: None,
                source_memory_id: Some(payload.memory_id.clone()),
                index_semantic: Some(true),
                enable_semantic_dedup: Some(false),
                enable_consolidation: Some(false),
                content_type: payload.content_type.clone(),
                fact_operation: None,
                fact_confidence: None,
                fact_subject: None,
                fact_predicate: None,
                fact_object: None,
                ..Default::default()
            });
        }

        if let Some(keyword_text) = keyword_index {
            companions.push(IngestPayload {
                entity_id: payload.entity_id.clone(),
                memory_id: MemoryId::derived_from(&payload.memory_id, "kw"),
                timestamp: payload.timestamp,
                textual_content: keyword_text,
                relations: payload.relations.clone(),
                kind: Some("session_summary".to_string()),
                fact_key: None,
                source_memory_id: Some(payload.memory_id.clone()),
                index_semantic: Some(false),
                enable_semantic_dedup: Some(false),
                enable_consolidation: Some(false),
                content_type: payload.content_type.clone(),
                fact_operation: None,
                fact_confidence: None,
                fact_subject: None,
                fact_predicate: None,
                fact_object: None,
                ..Default::default()
            });
        }
    }

    for (idx, fact_text) in fact_texts.into_iter().enumerate() {
        let fact_key = infer_fact_key_with_profile(&fact_text, profile);
        let fact_content = if let Some(slot_key) = fact_key.as_deref() {
            format!("Canonical fact about {}: {}", slot_key.replace('_', " "), fact_text)
        } else {
            format!("Canonical fact: {fact_text}")
        };
        companions.push(IngestPayload {
            entity_id: payload.entity_id.clone(),
            memory_id: MemoryId::derived_from(&payload.memory_id, &format!("fact{idx}")),
            timestamp: payload.timestamp,
            textual_content: fact_content,
            relations: payload.relations.clone(),
            kind: Some("fact".to_string()),
            fact_key,
            source_memory_id: Some(payload.memory_id.clone()),
            index_semantic: Some(true),
            enable_semantic_dedup: Some(true),
            enable_consolidation: Some(false),
            enable_mining: Some(false),
            content_type: payload.content_type.clone(),
            fact_operation: Some("derive".to_string()),
            fact_confidence: Some(0.92),
            fact_subject: Some(payload.entity_id.clone()),
            fact_predicate: None,
            fact_object: None,
            visual_description: None,
            visual_query: None,
            session_id: payload.session_id.clone(),
            turn_index: payload.turn_index,
            role: payload.role.clone(),
        });
    }

    companions.extend(build_atomic_memory_card_payloads(payload));

    if let Some(event_text) = build_event_companion_text(payload) {
        companions.push(IngestPayload {
            entity_id: payload.entity_id.clone(),
            memory_id: MemoryId::derived_from(&payload.memory_id, "event"),
            timestamp: payload.timestamp,
            textual_content: event_text,
            relations: payload.relations.clone(),
            kind: Some("session_summary".to_string()),
            fact_key: None,
            source_memory_id: Some(payload.memory_id.clone()),
            index_semantic: Some(true),
            enable_semantic_dedup: Some(false),
            enable_consolidation: Some(false),
            enable_mining: Some(false),
            content_type: payload.content_type.clone(),
            fact_operation: None,
            fact_confidence: None,
            fact_subject: None,
            fact_predicate: None,
            fact_object: None,
            visual_description: None,
            visual_query: None,
            session_id: payload.session_id.clone(),
            turn_index: payload.turn_index,
            role: payload.role.clone(),
        });
    }

    companions.extend(build_relation_companion_payloads(payload));

    companions
}

pub fn build_context_header(prev_text: Option<&str>, next_text: Option<&str>) -> String {
    let mut parts = Vec::new();
    if let Some(prev) = prev_text {
        let entities = extract_named_phrases(&[prev.to_string()]);
        let salient = extract_salient_terms(prev, 3);
        let mut ctx = Vec::new();
        if !entities.is_empty() {
            ctx.push(entities.into_iter().take(2).collect::<Vec<_>>().join(", "));
        }
        if !salient.is_empty() {
            ctx.push(salient.join(" "));
        }
        if !ctx.is_empty() {
            parts.push(format!("Prior: {}", ctx.join(" ")));
        }
    }
    if let Some(next) = next_text {
        let entities = extract_named_phrases(&[next.to_string()]);
        let salient = extract_salient_terms(next, 3);
        let mut ctx = Vec::new();
        if !entities.is_empty() {
            ctx.push(entities.into_iter().take(2).collect::<Vec<_>>().join(", "));
        }
        if !salient.is_empty() {
            ctx.push(salient.join(" "));
        }
        if !ctx.is_empty() {
            parts.push(format!("Next: {}", ctx.join(" ")));
        }
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("[{}] ", parts.join(" | "))
    }
}

#[cfg(test)]
mod identity_tests {
    use super::*;

    #[test]
    fn opaque_memory_ids_get_companions_from_explicit_identity() {
        let text = "I just moved to Denver and started a new job at Acme Corp.";
        let conventional = IngestPayload {
            entity_id: "alice".into(),
            memory_id: "alice::s1::0".into(),
            session_id: Some("s1".into()),
            turn_index: Some(0),
            textual_content: text.into(),
            ..Default::default()
        };
        let opaque = IngestPayload {
            memory_id: "0b7c5e0e-3f1a-4d2b-9c61-2a7d8f0e4b11".into(),
            ..conventional.clone()
        };
        let kinds = |p: &IngestPayload| -> Vec<Option<String>> {
            build_companion_payloads(p).into_iter().map(|c| c.kind).collect()
        };
        assert!(!kinds(&conventional).is_empty());
        assert_eq!(kinds(&opaque), kinds(&conventional));

        let no_session = IngestPayload { session_id: None, ..opaque };
        assert!(build_companion_payloads(&no_session).is_empty());
    }
}
