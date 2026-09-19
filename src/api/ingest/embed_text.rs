//! What text is embedded for a memory.
//!
//! `TELLODB_EMBED_TEXT` selects (default `context`):
//! - `legacy`: the text as sent, prefixed with a context header derived from
//!   neighbouring payloads in the same request (batch-dependent).
//! - `turn`: the turn's own text, prefixed with its role.
//! - `context`: the turn plus up to `TELLODB_CONTEXT_WINDOW` neighbouring
//!   turns of the same session, read from stored turns and the current batch.
//!   When a turn arrives, stored neighbours whose window now includes it are
//!   re-embedded, so the result does not depend on batch size or order.
//!
//! Dates and session ids are never part of embedded text.

use crate::api::types::IngestPayload;
use crate::storage::TenantStore;
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedTextMode {
    Legacy,
    Turn,
    Context,
}

impl EmbedTextMode {
    pub fn as_str(self) -> &'static str {
        match self {
            EmbedTextMode::Legacy => "legacy",
            EmbedTextMode::Turn => "turn",
            EmbedTextMode::Context => "context",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EmbedTextConfig {
    pub mode: EmbedTextMode,
    pub window: u32,
}

pub fn embed_text_config() -> EmbedTextConfig {
    static CONFIG: OnceLock<EmbedTextConfig> = OnceLock::new();
    *CONFIG.get_or_init(|| {
        let mode = match std::env::var("TELLODB_EMBED_TEXT")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "legacy" => EmbedTextMode::Legacy,
            "turn" => EmbedTextMode::Turn,
            _ => EmbedTextMode::Context,
        };
        let window = std::env::var("TELLODB_CONTEXT_WINDOW")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map(|w| w.min(4))
            .unwrap_or(1);
        EmbedTextConfig { mode, window }
    })
}

/// A source turn is a memory as sent (or the first chunk of one), as opposed
/// to derived records such as later chunks, companions and cards.
pub fn is_source_turn(payload: &IngestPayload) -> bool {
    if payload.kind.as_deref() == Some("synthetic_query") {
        return false;
    }
    match payload.source_memory_id.as_deref() {
        None => true,
        Some(parent) => payload.memory_id == format!("{parent}::c0"),
    }
}

pub fn turn_line(role: &str, text: &str) -> String {
    // A caller may prefix `[Session ID: ...]` / `[Session Date: ...]`. Those
    // are read for event time elsewhere and must not be embedded: this module
    // guarantees dates and session ids are not part of embedded text.
    let text = super::dialogue::content_for_index(text).trim();
    if role.is_empty() {
        text.to_string()
    } else {
        format!("{role}: {text}")
    }
}

fn payload_line(payload: &IngestPayload) -> String {
    turn_line(payload.role.as_deref().unwrap_or(""), &payload.textual_content)
}

fn window_text(turns: &BTreeMap<u32, String>, center: u32, window: u32) -> Option<String> {
    let current = turns.get(&center)?;
    let lo = center.saturating_sub(window);
    let hi = center.saturating_add(window);
    let neighbours: Vec<&str> = turns
        .range(lo..=hi)
        .filter(|(turn, _)| **turn != center)
        .map(|(_, line)| line.as_str())
        .collect();
    Some(if neighbours.is_empty() {
        current.clone()
    } else {
        format!("{current}\n[context] {}", neighbours.join(" | "))
    })
}

pub struct ContextTexts {
    /// Embedding text per payload, aligned with the input.
    pub texts: Vec<String>,
    /// Already-stored turns whose context window changed: `(memory_id, text)`.
    pub neighbour_updates: Vec<(String, String)>,
}

/// Embedding texts for `turn` / `context` modes. `payloads` must already be
/// identity-normalized (session and turn filled where known).
pub fn build_embed_texts(
    tenant: &TenantStore,
    payloads: &[IngestPayload],
    config: EmbedTextConfig,
) -> Result<ContextTexts> {
    let mut texts: Vec<String> = payloads
        .iter()
        .map(|p| if is_source_turn(p) { payload_line(p) } else { p.textual_content.clone() })
        .collect();
    let mut neighbour_updates = Vec::new();
    if config.mode != EmbedTextMode::Context || config.window == 0 {
        return Ok(ContextTexts { texts, neighbour_updates });
    }

    // Source turns in this batch, grouped by (entity, session).
    let mut groups: HashMap<(String, String), Vec<(usize, u32)>> = HashMap::new();
    for (idx, payload) in payloads.iter().enumerate() {
        let (Some(session), Some(turn)) = (payload.session_id.as_deref(), payload.turn_index)
        else {
            continue;
        };
        if session.is_empty() || !is_source_turn(payload) {
            continue;
        }
        groups
            .entry((payload.entity_id.clone(), session.to_string()))
            .or_default()
            .push((idx, turn));
    }

    let w = config.window;
    for ((entity, session), members) in groups {
        let lo = members.iter().map(|(_, t)| *t).min().unwrap_or(0);
        let hi = members.iter().map(|(_, t)| *t).max().unwrap_or(0);
        let stored = tenant.session_turn_window(
            &entity,
            &session,
            lo.saturating_sub(2 * w),
            hi.saturating_add(2 * w),
        )?;

        // Turn -> line, stored first so the batch's version wins.
        let mut lines: BTreeMap<u32, String> = BTreeMap::new();
        let mut stored_ids: BTreeMap<u32, String> = BTreeMap::new();
        for (memory_id, turn, role, content) in stored {
            lines.insert(turn, turn_line(&role, &content));
            stored_ids.insert(turn, memory_id);
        }
        let batch_turns: Vec<u32> = members.iter().map(|(_, t)| *t).collect();
        for (idx, turn) in &members {
            lines.insert(*turn, payload_line(&payloads[*idx]));
        }

        for (idx, turn) in &members {
            if let Some(text) = window_text(&lines, *turn, w) {
                texts[*idx] = text;
            }
        }

        // Stored neighbours within `w` of a new turn now see a different window.
        for (turn, memory_id) in stored_ids {
            if batch_turns.contains(&turn) {
                continue;
            }
            if batch_turns.iter().any(|b| turn.abs_diff(*b) <= w) {
                if let Some(text) = window_text(&lines, turn, w) {
                    neighbour_updates.push((memory_id, text));
                }
            }
        }
    }

    Ok(ContextTexts { texts, neighbour_updates })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_text_uses_neighbours_only() {
        let lines: BTreeMap<u32, String> =
            [(0, "user: a".to_string()), (1, "assistant: b".to_string()), (3, "user: d".into())]
                .into_iter()
                .collect();
        assert_eq!(window_text(&lines, 0, 1).unwrap(), "user: a\n[context] assistant: b");
        assert_eq!(window_text(&lines, 3, 1).unwrap(), "user: d");
        assert!(window_text(&lines, 2, 1).is_none());
    }

    #[test]
    fn source_turn_detection() {
        let original = IngestPayload { memory_id: "e::s::0".into(), ..Default::default() };
        let chunk0 = IngestPayload {
            memory_id: "e::s::0::c0".into(),
            source_memory_id: Some("e::s::0".into()),
            ..Default::default()
        };
        let card = IngestPayload {
            memory_id: "e::s::0::card0".into(),
            source_memory_id: Some("e::s::0".into()),
            ..Default::default()
        };
        assert!(is_source_turn(&original) && is_source_turn(&chunk0) && !is_source_turn(&card));
    }

    fn turn(session: &str, t: u32, role: &str, text: &str) -> IngestPayload {
        IngestPayload {
            entity_id: "e".into(),
            memory_id: format!("e::{session}::{t}"),
            session_id: Some(session.into()),
            turn_index: Some(t),
            role: Some(role.into()),
            textual_content: text.into(),
            ..Default::default()
        }
    }

    fn store(tenant: &TenantStore, payloads: &[IngestPayload]) {
        let items: Vec<(u64, String, crate::storage::AgentObservation)> = payloads
            .iter()
            .map(|p| {
                (
                    0,
                    p.memory_id.clone(),
                    crate::storage::AgentObservation {
                        entity_id: p.entity_id.clone(),
                        textual_content: p.textual_content.clone(),
                        session_id: p.session_id.clone().unwrap(),
                        turn_index: p.turn_index.unwrap(),
                        role: p.role.clone().unwrap(),
                        ..Default::default()
                    },
                )
            })
            .collect();
        tenant.insert_observations_batch(&items).unwrap();
    }

    #[test]
    fn context_texts_do_not_depend_on_batching() {
        let config = EmbedTextConfig { mode: EmbedTextMode::Context, window: 1 };
        let turns: Vec<IngestPayload> = (0..4)
            .map(|t| {
                turn("s", t, if t % 2 == 0 { "user" } else { "assistant" }, &format!("msg {t}"))
            })
            .collect();

        // All turns in one batch.
        let temp = tempfile::tempdir().unwrap();
        let one = TenantStore::new(&temp.path().join("a.db")).unwrap();
        let full = build_embed_texts(&one, &turns, config).unwrap();
        let expected: HashMap<String, String> =
            turns.iter().map(|p| p.memory_id.clone()).zip(full.texts).collect();

        // One turn per batch, in reverse order, applying neighbour updates.
        let temp2 = tempfile::tempdir().unwrap();
        let split = TenantStore::new(&temp2.path().join("b.db")).unwrap();
        let mut latest: HashMap<String, String> = HashMap::new();
        for p in turns.iter().rev() {
            let built = build_embed_texts(&split, std::slice::from_ref(p), config).unwrap();
            latest.insert(p.memory_id.clone(), built.texts[0].clone());
            store(&split, std::slice::from_ref(p));
            for (id, text) in built.neighbour_updates {
                latest.insert(id, text);
            }
        }
        assert_eq!(latest, expected);
        assert_eq!(expected["e::s::1"], "assistant: msg 1\n[context] user: msg 0 | user: msg 2");
    }
}
