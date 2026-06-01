use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::lifecycle::LifecycleMetadata;

#[derive(Debug, PartialEq, Clone, Copy)]
#[derive(Default)]
pub enum MemoryKind {
    #[default]
    Conversational,
    Decision,
    Lesson,
    Preference,
    SessionSummary,
    Fact,
}


impl MemoryKind {
    pub fn is_decay_exempt(&self) -> bool {
        matches!(self, MemoryKind::Preference | MemoryKind::Decision)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct AgentObservation {
    pub entity_id: String,
    pub textual_content: String,
    pub embedding: Vec<f32>,
    pub kind: MemoryKind,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryCard {
    pub card_id: String,
    pub entity_id: String,
    pub user_id: String,
    pub source_memory_id: String,
    pub source_session_id: String,
    pub source_turn_index: usize,
    pub document_time: u64,
    pub conversation_time: u64,
    pub event_time: Option<u64>,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub memory_text: String,
    pub card_type: String,
    pub confidence: f32,
    pub is_latest: bool,
    pub is_static: bool,
    pub is_inference: bool,
    pub expires_at: Option<u64>,
    pub root_card_id: Option<String>,
    pub parent_card_id: Option<String>,
    pub lifecycle: Option<LifecycleMetadata>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SessionRouterRecord {
    pub session_id: String,
    pub entity_id: String,
    pub session_date: String,
    pub document_time_ms: u64,
    pub speakers: Vec<String>,
    pub persons: Vec<String>,
    pub session_focus: String,
    pub canonical_facts: Vec<String>,
    pub events: Vec<String>,
    pub objects: Vec<String>,
    pub places: Vec<String>,
    pub activities: Vec<String>,
    pub preference_signals: Vec<String>,
    pub salient_terms: Vec<String>,
    pub source_memory_ids: Vec<String>,
    pub router_text: String,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LedgerTurn {
    pub turn_id: String,
    pub entity_id: String,
    pub session_id: String,
    pub speaker: Option<String>,
    pub turn_index: u32,
    pub raw_text: String,
    pub document_time_ms: u64,
    pub ingest_time_ms: u64,
    pub source_type: String,
    pub source_uri: Option<String>,
    pub raw_sha256: String,
    pub redaction_state: String,
    pub lifecycle: Option<LifecycleMetadata>,
    pub schema_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryArtifact {
    pub artifact_id: String,
    pub artifact_type: String,
    pub entity_id: String,
    pub source_turn_ids: Vec<String>,
    pub source_memory_ids: Vec<String>,
    pub source_session_ids: Vec<String>,
    pub compiler_name: String,
    pub compiler_version: String,
    pub embedding_model: Option<String>,
    pub embedding_dim: Option<usize>,
    pub index_namespace: Option<String>,
    pub lifecycle: Option<LifecycleMetadata>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemporalEvent {
    pub event_id: String,
    pub entity_id: String,
    pub source_session_id: String,
    pub source_memory_id: String,
    pub source_turn_index: usize,
    pub subject: String,
    pub relation: String,
    pub object: Option<String>,
    pub participants: Vec<String>,
    pub place: Option<String>,
    pub document_time_ms: u64,
    pub event_time_ms: Option<u64>,
    pub event_time_range_ms: Option<(u64, u64)>,
    pub event_time_granularity: String,
    pub actor_entities: Vec<String>,
    pub object_entities: Vec<String>,
    pub event_type: String,
    pub is_inferred_time: bool,
    pub event_text: String,
    pub confidence: f32,
    pub lifecycle: Option<LifecycleMetadata>,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShadowQuestion {
    pub shadow_id: String,
    pub entity_id: String,
    pub source_session_id: String,
    pub source_memory_id: String,
    pub source_card_id: Option<String>,
    pub question_text: String,
    pub answer_type: String,
    pub entities: Vec<String>,
    pub facets: Vec<String>,
    pub confidence: f32,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FacetPosting {
    pub entity_id: String,
    pub facet_type: String,
    pub facet_value: String,
    pub target_id: String,
    pub target_type: String,
    pub session_id: String,
    pub memory_id: Option<String>,
    pub card_id: Option<String>,
    pub event_id: Option<String>,
    pub turn_id: Option<String>,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemCell {
    pub cell_id: String,
    pub entity_id: String,
    pub source_session_id: String,
    pub source_turn_ids: Vec<String>,
    pub cell_text: String,
    pub cell_type: String,
    pub subjects: Vec<String>,
    pub objects: Vec<String>,
    pub activities: Vec<String>,
    pub places: Vec<String>,
    pub document_time_ms: u64,
    pub event_time_ms: Option<u64>,
    pub confidence: f32,
    pub saliency: f32,
    pub lifecycle: Option<LifecycleMetadata>,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemSceneRecord {
    pub scene_id: String,
    pub entity_id: String,
    pub scene_title: String,
    pub scene_summary: String,
    pub source_cell_ids: Vec<String>,
    pub source_session_ids: Vec<String>,
    pub entities: Vec<String>,
    pub activities: Vec<String>,
    pub objects: Vec<String>,
    pub places: Vec<String>,
    pub time_range_ms: Option<(u64, u64)>,
    pub scene_type: String,
    pub saliency: f32,
    pub lifecycle: Option<LifecycleMetadata>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProfileFact {
    pub profile_fact_id: String,
    pub entity_id: String,
    pub category: String,
    pub value: String,
    pub source_session_id: String,
    pub source_memory_id: String,
    pub source_card_id: Option<String>,
    pub confidence: f32,
    pub document_time_ms: u64,
    pub is_latest: bool,
    pub lifecycle: Option<LifecycleMetadata>,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionCandidateTrace {
    pub session_id: String,
    pub final_score: f32,
    pub features: HashMap<String, f32>,
    pub source_memory_ids: Vec<String>,
    pub source_card_ids: Vec<String>,
    pub source_event_ids: Vec<String>,
    pub is_gold: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryTrace {
    pub query_trace_id: String,
    pub entity_id: Option<String>,
    pub question: String,
    pub query_plan: String,
    pub candidate_sessions: Vec<SessionCandidateTrace>,
    pub selected_sessions: Vec<String>,
    pub returned_memory_ids: Vec<String>,
    pub latency_ms: u64,
    pub gold_sessions: Vec<String>,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryCardSearchHit {
    pub card_id: String,
    pub source_memory_id: String,
    pub source_session_id: String,
    pub timestamp: u64,
    pub score: f32,
    pub lexical_hits: usize,
    pub temporal_hits: usize,
    pub entity_hits: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionRouterSearchHit {
    pub session_id: String,
    pub score: f32,
    pub lexical_hits: usize,
    pub temporal_hits: usize,
    pub entity_hits: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TemporalEventSearchHit {
    pub event_id: String,
    pub source_memory_id: String,
    pub source_session_id: String,
    pub score: f32,
    pub lexical_hits: usize,
    pub temporal_hits: usize,
    pub entity_hits: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShadowQuestionSearchHit {
    pub shadow_id: String,
    pub source_memory_id: String,
    pub source_session_id: String,
    pub score: f32,
    pub lexical_hits: usize,
    pub entity_hits: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeletedObservation {
    pub vector_id: Option<u64>,
    pub entity_id: String,
    pub tombstone: Option<crate::lifecycle::DeletionTombstone>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum FactVersionStatus {
    Current { superseded: Option<(u64, String)> },
    Stale { current: (u64, String) },
}

pub(crate) fn build_session_router_text(record: &SessionRouterRecord) -> String {
    let mut parts = Vec::new();
    parts.push(format!("session {}", record.session_id));
    if !record.session_date.is_empty() {
        parts.push(format!("date {}", record.session_date));
    }
    if !record.session_focus.is_empty() {
        parts.push(format!("focus {}", record.session_focus));
    }
    if !record.speakers.is_empty() {
        parts.push(format!("speakers {}", record.speakers.join(" ")));
    }
    if !record.persons.is_empty() {
        parts.push(format!("people {}", record.persons.join(" ")));
    }
    if !record.canonical_facts.is_empty() {
        parts.push(format!("facts {}", record.canonical_facts.join(" | ")));
    }
    if !record.events.is_empty() {
        parts.push(format!("events {}", record.events.join(" | ")));
    }
    if !record.objects.is_empty() {
        parts.push(format!("objects {}", record.objects.join(" ")));
    }
    if !record.places.is_empty() {
        parts.push(format!("places {}", record.places.join(" ")));
    }
    if !record.activities.is_empty() {
        parts.push(format!("activities {}", record.activities.join(" ")));
    }
    if !record.preference_signals.is_empty() {
        parts.push(format!("preferences {}", record.preference_signals.join(" | ")));
    }
    if !record.salient_terms.is_empty() {
        parts.push(format!("keywords {}", record.salient_terms.join(" ")));
    }
    parts.join("\n")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CoreClusterStats {
    pub memory_count: usize,
    pub entity_count: usize,
    pub fact_count: usize,
    pub storage_bytes: usize,
    pub request_count: usize,
    pub ingest_count: usize,
    pub query_count: usize,
}

#[derive(Clone)]
pub struct GraphEdgeEntry<'a> {
    pub memory_id: &'a str,
    pub subject: &'a str,
    pub predicate: &'a str,
    pub object: &'a str,
    pub status: &'a str,
    pub ref_info: Option<(&'a str, &'a str)>,
    pub timestamp: u64,
}

#[derive(Clone)]
pub struct CombinedIngestUpsertInput<'a> {
    pub cards: &'a [MemoryCard],
    pub artifacts: &'a [MemoryArtifact],
    pub events: &'a [TemporalEvent],
    pub shadow_questions: &'a [ShadowQuestion],
    pub facet_postings: &'a [FacetPosting],
    pub mem_cells: &'a [MemCell],
    pub mem_scenes: &'a [MemSceneRecord],
    pub profile_facts: &'a [ProfileFact],
}

pub struct MemoryCardSearchInput<'a> {
    pub entity_id: &'a str,
    pub lexical_terms: &'a [String],
    pub temporal_terms: &'a [String],
    pub entities: &'a [String],
    pub route_sessions: &'a HashSet<String>,
    pub include_stale: bool,
    pub limit: usize,
}
