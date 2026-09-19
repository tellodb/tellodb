use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::lifecycle::LifecycleMetadata;

#[derive(Debug, PartialEq, Clone, Copy, Default)]
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
    pub const ALL: [MemoryKind; 6] = [
        MemoryKind::Conversational,
        MemoryKind::Decision,
        MemoryKind::Lesson,
        MemoryKind::Preference,
        MemoryKind::SessionSummary,
        MemoryKind::Fact,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            MemoryKind::Conversational => "conversational",
            MemoryKind::Decision => "decision",
            MemoryKind::Lesson => "lesson",
            MemoryKind::Preference => "preference",
            MemoryKind::SessionSummary => "session_summary",
            MemoryKind::Fact => "fact",
        }
    }

    pub fn parse(s: &str) -> MemoryKind {
        match s {
            "conversational" | "Conversational" => MemoryKind::Conversational,
            "decision" | "Decision" => MemoryKind::Decision,
            "lesson" | "Lesson" => MemoryKind::Lesson,
            "preference" | "Preference" => MemoryKind::Preference,
            "session_summary" | "SessionSummary" => MemoryKind::SessionSummary,
            "fact" | "Fact" => MemoryKind::Fact,
            _ => MemoryKind::Conversational,
        }
    }

    pub fn is_decay_exempt(&self) -> bool {
        matches!(self, MemoryKind::Preference | MemoryKind::Decision)
    }
}

#[derive(Debug, PartialEq, Clone, Default)]
pub struct AgentObservation {
    pub entity_id: String,
    pub textual_content: String,
    pub embedding: Vec<f32>,
    pub kind: MemoryKind,
    pub content_hash: String,
    pub created_at_ms: u64,
    /// Session the memory belongs to ("" if unknown).
    pub session_id: String,
    pub turn_index: u32,
    pub role: String,
    /// Source memory for derived records (chunks, companions, cards); `None`
    /// for memories ingested as sent.
    pub parent_memory_id: Option<String>,
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
pub struct DeletedObservation {
    pub vector_id: Option<u64>,
    /// Vectors of chunk memories removed along with the parent.
    pub chunk_vector_ids: Vec<u64>,
    pub entity_id: String,
    pub tombstone: Option<crate::lifecycle::DeletionTombstone>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum FactVersionStatus {
    Current {
        superseded: Option<(u64, String)>,
    },
    Stale {
        current: (u64, String),
    },
    /// The memory restated a value that an existing version already holds; it
    /// was recorded as evidence for that version instead of starting a new one.
    Confirmed {
        version: (u64, String),
    },
}

/// One version in a fact's history.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactHistoryEntry {
    /// The memory that stated this value (the source turn where there is one).
    pub memory_id: String,
    pub object: String,
    pub is_current: bool,
    pub valid_from_ms: u64,
    /// When the next version took over; `None` while this one holds.
    pub valid_to_ms: Option<u64>,
    /// Memories stating this same value, newest first.
    pub evidence: Vec<String>,
}

/// A fact version as stored, for explaining a result's currency.
#[derive(Debug, Clone, PartialEq)]
pub struct FactVersionRow {
    pub fact_key: String,
    pub entity_id: String,
    pub object: String,
    pub is_current: bool,
    pub valid_from_ms: u64,
    pub valid_to_ms: Option<u64>,
    /// The version that replaced this one, when it is no longer current.
    pub superseded_by: Option<String>,
    pub superseded_at_ms: Option<u64>,
    /// Value of the version that is current now.
    pub current_object: Option<String>,
    /// Memories that state this version's value, newest first.
    pub evidence: Vec<String>,
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

pub struct MemoryCardSearchInput<'a> {
    pub entity_id: &'a str,
    pub lexical_terms: &'a [String],
    pub temporal_terms: &'a [String],
    pub entities: &'a [String],
    pub route_sessions: &'a HashSet<String>,
    pub include_stale: bool,
    pub limit: usize,
}

#[cfg(test)]
mod tests {
    use super::MemoryKind;

    #[test]
    fn memory_kind_canonical_names_round_trip() {
        for kind in MemoryKind::ALL {
            assert_eq!(MemoryKind::parse(kind.as_str()), kind);
        }
    }

    #[test]
    fn memory_kind_legacy_debug_names_parse() {
        let legacy = [
            ("Conversational", MemoryKind::Conversational),
            ("Decision", MemoryKind::Decision),
            ("Lesson", MemoryKind::Lesson),
            ("Preference", MemoryKind::Preference),
            ("SessionSummary", MemoryKind::SessionSummary),
            ("Fact", MemoryKind::Fact),
        ];

        for (value, expected) in legacy {
            assert_eq!(MemoryKind::parse(value), expected);
        }
    }
}
