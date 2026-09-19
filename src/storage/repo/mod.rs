pub mod admin;
pub mod cards;
pub mod entities;
pub mod facts;
pub mod fts;
pub mod graph;
pub mod ingest;
pub mod memories;
pub mod prefs;
pub mod schema;
pub mod sessions;
pub mod traits;

pub const IN_CHUNK: usize = 500;

pub fn in_placeholders(n: usize) -> String {
    (0..n).map(|_| "?").collect::<Vec<_>>().join(",")
}

pub fn padded_in_chunk<T>(chunk: &[T]) -> Vec<&T> {
    let Some(last) = chunk.last() else {
        return Vec::new();
    };
    let mut padded = Vec::with_capacity(IN_CHUNK);
    padded.extend(chunk);
    padded.resize(IN_CHUNK, last);
    padded
}

pub(super) mod prelude {
    pub(super) use anyhow::{Context, Result};
    pub(super) use rusqlite::params;
    pub(super) use std::collections::HashMap;
    pub(super) use std::time::{SystemTime, UNIX_EPOCH};

    pub(super) use super::{in_placeholders, padded_in_chunk, IN_CHUNK};
    pub(super) use crate::storage::tenant::{
        bytes_to_vec_f32, contains_term_count, fts_entity_tok, fts_quote, fts_rowid,
        memory_turn_row, merge_router_records, merge_turn, same_fact_object, unix_timestamp_ms,
        vec_f32_to_bytes, EdgeNeighbour, GraphEdge, GraphEdgeBatch, TenantStore,
        CARD_CONFIDENCE_WEIGHT, CARD_ENTITY_WEIGHT, CARD_LATEST_BOOST, CARD_LEXICAL_WEIGHT,
        CARD_ROUTE_BOOST, CARD_STALE_PENALTY, CARD_TEMPORAL_WEIGHT, DECISION_TYPE_BOOST,
        EVENT_TYPE_BOOST, FACT_TYPE_BOOST, FOCUS_MATCH_MIN_LEN, FTS_MIN_TERM_LEN,
        INFERENCE_TYPE_BOOST, METRICS_DDL, OTHER_TYPE_BOOST, PREFERENCE_TYPE_BOOST, SCHEMA_VERSION,
        SEARCH_MIN_TERM_LEN, SESSION_DEPTH_WEIGHT, SESSION_ENTITY_WEIGHT, SESSION_FOCUS_BONUS,
        SESSION_LEXICAL_WEIGHT, SESSION_TEMPORAL_WEIGHT, SOURCE_DEPTH_DIVISOR,
    };
    pub(super) use crate::storage::types::*;
}
