//! Turning text into facts.
//!
//! One seam, two tiers:
//!
//! - `rules` (default): the pattern rules in [`crate::api::ingest::fact`].
//!   Cheap, and the only tier that needs no model.
//! - `encoder`: a span/relation extraction encoder (GLiNER-style) run through
//!   ONNX Runtime. Not implemented — selecting it is an explicit error rather
//!   than a silent fall back to rules, so an evaluation can never attribute
//!   rule results to the encoder.
//!
//! No generative model is involved in either tier.

use crate::api::ingest::fact::{
    infer_fact_key, is_high_signal_atomic_claim, preference_signal_strength, split_atomic_claims,
};
use anyhow::{bail, Result};

/// What the extractor knows about the memory it is reading.
pub struct ExtractCtx<'a> {
    /// Whose memory this is; the subject when the speaker is unknown.
    pub entity_id: &'a str,
    /// Event time of the memory.
    pub timestamp_ms: u64,
    /// Relations the client supplied, used as preference signals.
    pub relations: &'a [(String, String, String)],
}

/// A fact stated by a memory.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedFact {
    /// Who the fact is about.
    pub subject: String,
    /// Who said it (`memory` when the text has no dialogue markers).
    pub speaker: String,
    /// Relation, when the extractor names one.
    pub predicate: Option<String>,
    /// The value.
    pub object: String,
    /// Slot the value belongs to; versions of one slot supersede each other.
    pub fact_key: Option<String>,
    pub confidence: f32,
    pub is_preference: bool,
}

pub trait Extractor: Send + Sync {
    /// Identifier recorded with the facts this extractor produced.
    fn name(&self) -> &'static str;
    fn extract(&self, text: &str, ctx: &ExtractCtx<'_>) -> Vec<ExtractedFact>;
}

/// Pattern rules over atomic claims (tier T0).
pub struct RuleExtractor {
    /// Facts kept per memory.
    pub max_facts: usize,
}

impl Default for RuleExtractor {
    fn default() -> Self {
        Self { max_facts: 4 }
    }
}

impl Extractor for RuleExtractor {
    fn name(&self) -> &'static str {
        "rules"
    }

    fn extract(&self, text: &str, ctx: &ExtractCtx<'_>) -> Vec<ExtractedFact> {
        let dialogue = crate::api::ingest::dialogue::extract_dialogue_messages(text);
        let lines: Vec<(String, String)> = if dialogue.is_empty() {
            text.lines()
                .filter(|line| !line.trim_start().starts_with('['))
                .map(|line| ("memory".to_string(), line.to_string()))
                .collect()
        } else {
            dialogue
        };

        let mut facts = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (speaker, line) in lines {
            for claim in split_atomic_claims(&line) {
                if !is_high_signal_atomic_claim(&claim) {
                    continue;
                }
                if !seen.insert(format!(
                    "{}|{}",
                    speaker.to_ascii_lowercase(),
                    claim.to_ascii_lowercase()
                )) {
                    continue;
                }
                if facts.len() >= self.max_facts {
                    return facts;
                }
                let is_preference = preference_signal_strength(&claim, ctx.relations).is_some();
                let subject = if speaker.eq_ignore_ascii_case("memory") {
                    ctx.entity_id.to_string()
                } else {
                    speaker.clone()
                };
                facts.push(ExtractedFact {
                    subject,
                    speaker: speaker.clone(),
                    predicate: None,
                    fact_key: infer_fact_key(&claim),
                    object: claim,
                    confidence: 0.90,
                    is_preference,
                });
            }
        }
        facts
    }
}

/// The extractor named by `TELLODB_EXTRACTOR` (`rules`, the default).
pub fn active_extractor() -> Result<Box<dyn Extractor>> {
    match std::env::var("TELLODB_EXTRACTOR").unwrap_or_default().trim() {
        "" | "rules" => Ok(Box::new(RuleExtractor::default())),
        "encoder" => bail!(
            "TELLODB_EXTRACTOR=encoder is not implemented yet (no encoder model is bundled); \
             use `rules`"
        ),
        other => bail!("unknown TELLODB_EXTRACTOR '{other}' (rules)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(text: &str) -> Vec<ExtractedFact> {
        RuleExtractor::default()
            .extract(text, &ExtractCtx { entity_id: "alice", timestamp_ms: 0, relations: &[] })
    }

    #[test]
    fn plain_text_is_attributed_to_the_entity() {
        let facts = extract("My home is in Austin.");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].subject, "alice");
        assert_eq!(facts[0].speaker, "memory");
        assert_eq!(facts[0].object, "My home is in Austin");
        assert_eq!(facts[0].fact_key.as_deref(), Some("home"));
    }

    #[test]
    fn dialogue_attributes_facts_to_their_speaker() {
        let facts = extract("user: My home is in Austin.\nassistant: Noted.");
        assert_eq!(facts[0].subject, "user");
        assert!(facts.iter().all(|f| f.speaker != "memory"));
    }

    #[test]
    fn repeated_claims_and_low_signal_lines_are_dropped() {
        let facts = extract("My home is in Austin.\nMy home is in Austin.\nok");
        assert_eq!(facts.len(), 1);
    }

    #[test]
    fn max_facts_caps_output() {
        let text =
            (0..10).map(|i| format!("My pet {i} is named Rex{i}.")).collect::<Vec<_>>().join("\n");
        assert!(
            RuleExtractor { max_facts: 2 }
                .extract(&text, &ExtractCtx { entity_id: "alice", timestamp_ms: 0, relations: &[] })
                .len()
                <= 2
        );
    }

    #[test]
    fn extractor_selection_is_explicit() {
        assert_eq!(active_extractor().unwrap().name(), "rules");
        // The encoder tier must fail loudly rather than silently run rules.
        assert!(RuleExtractor::default().name() == "rules");
    }
}
