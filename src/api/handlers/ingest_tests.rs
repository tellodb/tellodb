use super::*;
use crate::config::{EmbedTextConfig, EmbedTextMode};
use crate::features::Features;
use crate::lifecycle::evaluate_lifecycle;
use crate::storage::{AgentObservation, MemoryKind, TenantStore};
use std::time::Instant;
use tempfile::tempdir;

fn payload(text: &str) -> IngestPayload {
    IngestPayload {
        entity_id: "alice".to_string(),
        memory_id: "alice::session::0".to_string(),
        timestamp: 1_700_000_000_000,
        session_id: Some("session".to_string()),
        turn_index: Some(0),
        role: Some("user".to_string()),
        textual_content: text.to_string(),
        index_semantic: Some(false),
        enable_semantic_dedup: Some(false),
        enable_consolidation: Some(false),
        enable_mining: Some(false),
        ..Default::default()
    }
}

fn raw_features() -> Features {
    Features::parse(
        "chunks,gist,keywords,fact_companions,atomic_cards,event_companions,relation_companions",
    )
    .unwrap()
}

fn artifact_features_without_derived() -> Features {
    Features::parse(
        "memory_cards,session_router,preferences,retrospective_links,derived_links,facts,consolidation",
    )
    .unwrap()
}

fn prepared_record(payload: IngestPayload, enable_consolidation: bool) -> PreparedRecord {
    let kind = payload.kind.as_deref().map(MemoryKind::parse).unwrap_or_default();
    let lifecycle = evaluate_lifecycle(
        &payload.textual_content,
        kind,
        payload.timestamp,
        payload.fact_confidence,
        payload
            .fact_operation
            .as_deref()
            .is_some_and(|operation| matches!(operation, "derive" | "infer")),
    );
    let obs = AgentObservation {
        entity_id: payload.entity_id.clone(),
        textual_content: payload.textual_content.clone(),
        embedding: Vec::new(),
        kind,
        content_hash: "hash".to_string(),
        created_at_ms: payload.timestamp,
        session_id: payload.session_id.clone().unwrap_or_default(),
        turn_index: payload.turn_index.unwrap_or_default(),
        role: payload.role.clone().unwrap_or_default(),
        parent_memory_id: payload.source_memory_id.clone(),
    };
    PreparedRecord { payload, obs, lifecycle, enable_consolidation }
}

fn test_tenant() -> (tempfile::TempDir, TenantStore) {
    let directory = tempdir().unwrap();
    let tenant = TenantStore::new(&directory.path().join("tenant.db")).unwrap();
    (directory, tenant)
}

#[test]
fn expand_normalizes_identity_from_memory_id() {
    let mut item = payload("A memory");
    item.memory_id = MemoryId::new("alice", "session-2", 4).as_str().to_string();
    item.session_id = None;
    item.turn_index = None;
    let mut diag = IngestDiagnostics::default();
    let (expanded, _) = expand_and_enrich_payloads(
        vec![item],
        &mut diag,
        Instant::now(),
        raw_features(),
        EmbedTextConfig { mode: EmbedTextMode::Legacy, window: 1 },
        Profile::Generic,
    );
    assert_eq!(expanded[0].session_id.as_deref(), Some("session-2"));
    assert_eq!(expanded[0].turn_index, Some(4));
}

#[test]
fn expand_prepends_visual_context() {
    let mut item = payload("A memory");
    item.visual_description = Some("a cat".to_string());
    item.visual_query = Some("outdoor scene".to_string());
    let mut diag = IngestDiagnostics::default();
    let (expanded, _) = expand_and_enrich_payloads(
        vec![item],
        &mut diag,
        Instant::now(),
        raw_features(),
        EmbedTextConfig { mode: EmbedTextMode::Legacy, window: 1 },
        Profile::Generic,
    );
    assert_eq!(expanded[0].textual_content, "[Visual: a cat] [Context: outdoor scene] A memory");
}

#[test]
fn expand_without_mining_keeps_only_the_source() {
    let mut item = payload("Alice visited Paris on Tuesday.");
    item.enable_mining = Some(false);
    let mut diag = IngestDiagnostics::default();
    let (expanded, _) = expand_and_enrich_payloads(
        vec![item],
        &mut diag,
        Instant::now(),
        Features::default(),
        EmbedTextConfig { mode: EmbedTextMode::Legacy, window: 1 },
        Profile::Generic,
    );
    assert_eq!(expanded.len(), 1);
    assert_eq!(diag.expanded_count, 1);
}

#[test]
fn expand_builds_session_companions_when_mining_is_enabled() {
    let mut item = payload("[Session Focus: travel planning]\nAlice visited Paris on Tuesday.");
    item.enable_mining = Some(true);
    let mut diag = IngestDiagnostics::default();
    let (expanded, _) = expand_and_enrich_payloads(
        vec![item],
        &mut diag,
        Instant::now(),
        Features::default(),
        EmbedTextConfig { mode: EmbedTextMode::Legacy, window: 1 },
        Profile::Generic,
    );
    assert!(expanded.len() > 1);
    assert!(expanded.iter().any(|item| item.memory_id.ends_with("::gist")));
    assert!(expanded.iter().all(|item| {
        item.memory_id == "alice::session::0"
            || item.source_memory_id.as_deref() == Some("alice::session::0")
    }));
}

#[test]
fn expand_legacy_mode_adds_batch_context() {
    let mut first = payload("Alice wrote reports");
    first.memory_id = "alice::session::0".to_string();
    first.turn_index = Some(0);
    let mut second = payload("Bob reviewed documents");
    second.memory_id = "alice::session::1".to_string();
    second.turn_index = Some(1);
    let mut diag = IngestDiagnostics::default();
    let (_, enriched) = expand_and_enrich_payloads(
        vec![first, second],
        &mut diag,
        Instant::now(),
        raw_features(),
        EmbedTextConfig { mode: EmbedTextMode::Legacy, window: 1 },
        Profile::Generic,
    );
    assert!(enriched[0].contains("Next:"));
    assert!(enriched[1].contains("Prior:"));
}

#[test]
fn expand_turn_mode_keeps_text_unchanged() {
    let item = payload("A memory");
    let original = item.textual_content.clone();
    let mut diag = IngestDiagnostics::default();
    let (_, enriched) = expand_and_enrich_payloads(
        vec![item],
        &mut diag,
        Instant::now(),
        raw_features(),
        EmbedTextConfig { mode: EmbedTextMode::Turn, window: 1 },
        Profile::Generic,
    );
    assert_eq!(enriched, vec![original]);
}

#[test]
fn expand_empty_input_is_empty() {
    let mut diag = IngestDiagnostics::default();
    let (expanded, enriched) = expand_and_enrich_payloads(
        Vec::new(),
        &mut diag,
        Instant::now(),
        raw_features(),
        EmbedTextConfig { mode: EmbedTextMode::Legacy, window: 1 },
        Profile::Generic,
    );
    assert!(expanded.is_empty());
    assert!(enriched.is_empty());
    assert_eq!(diag.expanded_count, 0);
}

#[test]
fn build_observations_preserves_identity_and_embedding() {
    let (_directory, tenant) = test_tenant();
    let mut item = payload("I prefer tea");
    item.kind = Some("preference".to_string());
    item.index_semantic = Some(true);
    let mut diag = IngestDiagnostics::default();
    let prepared = build_observations(
        &tenant,
        vec![item],
        vec![vec![0.1, 0.2, 0.3]],
        &mut diag,
        Features::parse("semantic_dedup").unwrap(),
        Profile::Generic,
    )
    .unwrap();
    assert_eq!(prepared.len(), 1);
    assert_eq!(prepared[0].obs.kind, MemoryKind::Preference);
    assert_eq!(prepared[0].obs.session_id, "session");
    assert_eq!(prepared[0].obs.turn_index, 0);
    assert_eq!(prepared[0].obs.embedding, vec![0.1, 0.2, 0.3]);
}

#[test]
fn build_observations_parses_explicit_kind() {
    let (_directory, tenant) = test_tenant();
    let mut item = payload("A session summary");
    item.kind = Some("SessionSummary".to_string());
    let mut diag = IngestDiagnostics::default();
    let prepared = build_observations(
        &tenant,
        vec![item],
        Vec::new(),
        &mut diag,
        Features::parse("semantic_dedup").unwrap(),
        Profile::Generic,
    )
    .unwrap();
    assert_eq!(prepared[0].obs.kind, MemoryKind::SessionSummary);
}

#[test]
fn build_observations_skips_embedding_when_disabled() {
    let (_directory, tenant) = test_tenant();
    let mut item = payload("No vector");
    item.index_semantic = Some(false);
    let mut diag = IngestDiagnostics::default();
    let prepared = build_observations(
        &tenant,
        vec![item],
        Vec::new(),
        &mut diag,
        Features::parse("semantic_dedup").unwrap(),
        Profile::Generic,
    )
    .unwrap();
    assert!(prepared[0].obs.embedding.is_empty());
}

#[test]
fn build_observations_infers_fact_key() {
    let (_directory, tenant) = test_tenant();
    let mut item = payload("I am married");
    item.kind = Some("fact".to_string());
    let mut diag = IngestDiagnostics::default();
    let prepared = build_observations(
        &tenant,
        vec![item],
        Vec::new(),
        &mut diag,
        Features::parse("semantic_dedup").unwrap(),
        Profile::Generic,
    )
    .unwrap();
    assert_eq!(prepared[0].payload.fact_key.as_deref(), Some("relationship_status"));
}

#[test]
fn build_observations_marks_derived_lessons_as_inference() {
    let (_directory, tenant) = test_tenant();
    let mut item = payload("A derived lesson");
    item.kind = Some("lesson".to_string());
    item.fact_operation = Some("derive".to_string());
    let mut diag = IngestDiagnostics::default();
    let prepared = build_observations(
        &tenant,
        vec![item],
        Vec::new(),
        &mut diag,
        Features::parse("semantic_dedup").unwrap(),
        Profile::Generic,
    )
    .unwrap();
    assert!(prepared[0].lifecycle.is_inference);
}

#[test]
fn build_artifacts_indexes_raw_content() {
    let mut item = payload("[Session Date: 2024-01-01]\nAlice writes reports.");
    item.session_id = None;
    let mut diag = IngestDiagnostics::default();
    let batches = build_artifacts(
        vec![prepared_record(item, false)],
        &mut diag,
        artifact_features_without_derived(),
    );
    assert_eq!(batches.fts_batch.len(), 1);
    assert_eq!(batches.fts_batch[0].0, "alice::session::0");
    assert!(!batches.fts_batch[0].2.contains("Session Date"));
}

#[test]
fn build_artifacts_creates_memory_card() {
    let mut item = payload("Alice lives in Paris");
    item.kind = Some("fact".to_string());
    item.fact_key = Some("home_city".to_string());
    item.fact_object = Some("Paris".to_string());
    let mut diag = IngestDiagnostics::default();
    let batches =
        build_artifacts(vec![prepared_record(item, false)], &mut diag, Features::default());
    assert_eq!(batches.memory_card_batch.len(), 1);
    assert_eq!(batches.memory_card_batch[0].card_type, "fact");
}

#[test]
fn build_artifacts_creates_session_router_record() {
    let item = payload("Alice visited Paris.");
    let mut diag = IngestDiagnostics::default();
    let batches =
        build_artifacts(vec![prepared_record(item, false)], &mut diag, Features::default());
    assert_eq!(batches.session_router_updates.len(), 1);
    assert_eq!(batches.session_router_updates[0].session_id, "session");
}

#[test]
fn build_artifacts_records_preferences() {
    let item = payload("I love tea.");
    let mut diag = IngestDiagnostics::default();
    let batches =
        build_artifacts(vec![prepared_record(item, false)], &mut diag, Features::default());
    assert_eq!(batches.preference_batch["alice"].len(), 1);
    assert_eq!(batches.preference_batch["alice"][0].1, 1.0);
}

#[test]
fn build_artifacts_creates_bidirectional_source_links() {
    let mut item = payload("A derived fact");
    item.source_memory_id = Some("alice::session::0".to_string());
    item.memory_id = "alice::session::0::fact0".to_string();
    let mut diag = IngestDiagnostics::default();
    let batches =
        build_artifacts(vec![prepared_record(item, false)], &mut diag, Features::default());
    assert_eq!(batches.memory_links_batch.len(), 2);
    assert!(batches.memory_links_batch.iter().any(|(_, _, kind)| kind == "derived_from"));
    assert!(batches.memory_links_batch.iter().any(|(_, _, kind)| kind == "derived_variant"));
}

#[test]
fn build_artifacts_registers_fact_metadata() {
    let mut item = payload("Alice lives in Paris");
    item.kind = Some("fact".to_string());
    item.fact_key = Some("home_city".to_string());
    item.fact_subject = Some("Alice".to_string());
    item.fact_predicate = Some("lives in".to_string());
    item.fact_object = Some("Paris".to_string());
    let mut diag = IngestDiagnostics::default();
    let batches =
        build_artifacts(vec![prepared_record(item, false)], &mut diag, Features::default());
    assert_eq!(batches.fact_batch.len(), 1);
    assert_eq!(batches.fact_batch[0].fact_key, "home_city");
    assert_eq!(batches.fact_batch[0].object, "Paris");
}

#[test]
fn build_artifacts_creates_consolidation_task() {
    let item = payload("Alice likes hiking.");
    let mut diag = IngestDiagnostics::default();
    let batches =
        build_artifacts(vec![prepared_record(item, true)], &mut diag, Features::default());
    assert_eq!(batches.consolidation_tasks.len(), 1);
    assert_eq!(batches.consolidation_tasks[0].entity_id, "alice");
}

#[test]
fn build_artifacts_extracts_retrospective_candidate() {
    let mut item = payload("Remember when we visited the museum last summer in Boston.");
    item.source_memory_id = Some("alice::session::0".to_string());
    let mut diag = IngestDiagnostics::default();
    let batches =
        build_artifacts(vec![prepared_record(item, false)], &mut diag, Features::default());
    assert_eq!(batches.retrospective_candidates.len(), 1);
    assert!(batches.retrospective_candidates[0].4.contains("we visited the museum"));
}

#[test]
fn build_artifacts_skips_derived_structures_for_synthetic_queries() {
    let mut item = payload("synthetic query");
    item.kind = Some("synthetic_query".to_string());
    let mut diag = IngestDiagnostics::default();
    let batches =
        build_artifacts(vec![prepared_record(item, false)], &mut diag, Features::default());
    assert_eq!(batches.observations.len(), 1);
    assert!(batches.fts_batch.is_empty());
    assert!(batches.memory_card_batch.is_empty());
    assert!(batches.session_router_updates.is_empty());
    assert!(batches.preference_batch.is_empty());
}

#[test]
fn memory_card_builder_rejects_empty_text() {
    let item = payload("   ");
    let lifecycle = evaluate_lifecycle("", MemoryKind::Conversational, item.timestamp, None, false);
    assert!(build_memory_card_from_payload(&item, MemoryKind::Conversational, &lifecycle).is_none());
}

#[test]
fn memory_card_builder_recovers_source_identity() {
    let mut item = payload("Alice lives in Paris.");
    item.memory_id = "alice::chat::4".to_string();
    item.session_id = None;
    item.turn_index = None;
    let lifecycle = evaluate_lifecycle(
        &item.textual_content,
        MemoryKind::Conversational,
        item.timestamp,
        None,
        false,
    );
    let card =
        build_memory_card_from_payload(&item, MemoryKind::Conversational, &lifecycle).unwrap();
    assert_eq!(card.source_session_id, "chat");
    assert_eq!(card.source_turn_index, 4);
}

#[test]
fn memory_card_builder_preserves_fact_metadata() {
    let mut item = payload("Alice lives in Paris.");
    item.kind = Some("fact".to_string());
    item.source_memory_id = Some("source".to_string());
    item.fact_subject = Some("Alice".to_string());
    item.fact_predicate = Some("lives in".to_string());
    item.fact_object = Some("Paris".to_string());
    item.fact_confidence = Some(2.0);
    item.fact_operation = Some("derive".to_string());
    let lifecycle = evaluate_lifecycle(
        &item.textual_content,
        MemoryKind::Fact,
        item.timestamp,
        item.fact_confidence,
        true,
    );
    let card = build_memory_card_from_payload(&item, MemoryKind::Fact, &lifecycle).unwrap();
    assert_eq!(card.source_memory_id, "source");
    assert_eq!(card.subject, "Alice");
    assert_eq!(card.predicate, "lives in");
    assert_eq!(card.object, "Paris");
    assert_eq!(card.confidence, 1.0);
    assert!(card.is_inference);
    assert_eq!(card.root_card_id.as_deref(), Some("source"));
}

#[test]
fn retrospective_links_detect_contradictions() {
    assert_eq!(
        classify_retrospective_link("Actually, the plan was wrong.", "The plan was good.", 2, 1),
        ("contradicts", "contradicted_by")
    );
}

#[test]
fn retrospective_links_detect_clarifications() {
    assert_eq!(
        classify_retrospective_link("On 2024-05-03 I paid 42 dollars.", "I paid for dinner.", 2, 1),
        ("clarifies", "clarified_by")
    );
}

#[test]
fn retrospective_links_detect_extensions() {
    assert_eq!(
        classify_retrospective_link("Alice moved to Paris.", "The move was discussed.", 2, 1),
        ("extends", "extended_by")
    );
}

#[test]
fn retrospective_links_detect_recalls() {
    assert_eq!(
        classify_retrospective_link(
            "We talked about the project.",
            "The project was discussed.",
            1,
            2
        ),
        ("recalls", "recalled_by")
    );
}

#[test]
fn card_type_for_kind_uses_stable_kind_names() {
    assert_eq!(card_type_for_kind(MemoryKind::Preference, "tea"), "preference");
    assert_eq!(card_type_for_kind(MemoryKind::Decision, "tea"), "decision");
    assert_eq!(card_type_for_kind(MemoryKind::Fact, "tea"), "fact");
    assert_eq!(card_type_for_kind(MemoryKind::Lesson, "tea"), "inference");
    assert_eq!(card_type_for_kind(MemoryKind::Conversational, "tea"), "episode");
}

#[test]
fn card_type_for_session_summary_distinguishes_events() {
    assert_eq!(
        card_type_for_kind(MemoryKind::SessionSummary, "Canonical event memory: Paris"),
        "event"
    );
    assert_eq!(
        card_type_for_kind(MemoryKind::SessionSummary, "A planning conversation"),
        "episode"
    );
}
