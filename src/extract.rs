//! Turning text into facts.
//!
//! One seam, two tiers:
//!
//! - `rules` (default): the pattern rules in [`crate::api::ingest::fact`].
//!   Cheap, and the only tier that needs no model.
//! - `encoder`: GLiNER span extraction through ONNX Runtime
//!   ([`crate::gliner`]). The label set is the fact schema: each label is a
//!   slot, and the marked span is that slot's value. Because the value is a
//!   span rather than the whole sentence, restatements of one value compare
//!   equal and merge into evidence, which sentence-level rule objects cannot
//!   do.
//!
//! No generative model is involved in either tier.

use crate::api::ingest::fact::{
    is_high_signal_atomic_claim, preference_signal_strength, split_atomic_claims,
};
use crate::config::ExtractorConfig;
use crate::heuristics::Profile;
use anyhow::{bail, Context, Result};
use std::sync::{Arc, OnceLock};

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
    pub heuristics: Profile,
}

impl Default for RuleExtractor {
    fn default() -> Self {
        Self { max_facts: 4, heuristics: Profile::Generic }
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
                    fact_key: crate::api::ingest::fact::infer_fact_key_with_profile(
                        &claim,
                        self.heuristics,
                    ),
                    object: claim,
                    confidence: 0.90,
                    is_preference,
                });
            }
        }
        facts
    }
}

/// Slots the encoder tier fills when `TELLODB_EXTRACTOR_LABELS` is unset.
///
/// Deliberately generic person attributes: the label set is a schema choice,
/// and labels written against particular benchmark questions would be the
/// same contamination the rule tier already has.
const DEFAULT_LABELS: &str = "city of residence,employer,job title,partner,child,pet,\
                              favorite thing,hobby,health condition,vehicle,school";

/// Zero-shot span extraction (tier T1). Each label is a fact slot; the span
/// the model marks is the value.
pub struct EncoderExtractor {
    model: crate::gliner::GlinerModel,
    labels: Vec<String>,
    threshold: f32,
}

impl EncoderExtractor {
    pub fn from_config(config: &ExtractorConfig) -> Result<Self> {
        let dir = config.model_dir.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "TELLODB_EXTRACTOR=encoder needs TELLODB_EXTRACTOR_MODEL_DIR pointing at a \
                 GLiNER export (gliner_config.json, tokenizer.json, onnx/model*.onnx)"
            )
        })?;
        let labels = if config.labels.is_empty() {
            DEFAULT_LABELS
                .split(',')
                .map(|label| label.split_whitespace().collect::<Vec<_>>().join(" "))
                .filter(|label| !label.is_empty())
                .collect()
        } else {
            config.labels.clone()
        };
        if labels.is_empty() {
            bail!("TELLODB_EXTRACTOR_LABELS is empty");
        }

        let model = crate::gliner::GlinerModel::load(dir)
            .with_context(|| format!("loading extractor model from {}", dir.display()))?;
        if labels.len() > model.max_types() {
            bail!("{} labels exceed the model's maximum of {}", labels.len(), model.max_types());
        }
        tracing::info!(
            labels = labels.len(),
            threshold = config.threshold,
            "encoder extractor ready"
        );
        Ok(Self { model, labels, threshold: config.threshold })
    }
}

/// `city of residence` -> `city_of_residence`, so a label is a fact key.
fn slug(label: &str) -> String {
    label
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

impl Extractor for EncoderExtractor {
    fn name(&self) -> &'static str {
        "encoder"
    }

    fn extract(&self, text: &str, ctx: &ExtractCtx<'_>) -> Vec<ExtractedFact> {
        let spans = match self.model.predict(text, &self.labels, self.threshold) {
            Ok(spans) => spans,
            Err(err) => {
                // A failed extraction must not fail the ingest; the memory is
                // still stored and searchable, it just has no facts.
                tracing::warn!(error = ?err, "encoder extraction failed; storing without facts");
                return Vec::new();
            }
        };
        let is_preference = preference_signal_strength(text, ctx.relations).is_some();
        spans
            .into_iter()
            .map(|span| ExtractedFact {
                subject: ctx.entity_id.to_string(),
                speaker: "memory".to_string(),
                predicate: Some(span.label.clone()),
                fact_key: Some(slug(&span.label)),
                object: span.text,
                confidence: span.score,
                is_preference,
            })
            .collect()
    }
}

static EXTRACTOR: OnceLock<Arc<dyn Extractor>> = OnceLock::new();

pub fn init(config: &ExtractorConfig, heuristics: Profile) -> Result<Arc<dyn Extractor>> {
    let extractor: Arc<dyn Extractor> = match config.kind.trim() {
        "" | "rules" => Arc::new(RuleExtractor { heuristics, ..RuleExtractor::default() }),
        "encoder" => Arc::new(EncoderExtractor::from_config(config)?),
        other => bail!("unknown TELLODB_EXTRACTOR '{other}' (rules, encoder)"),
    };
    Ok(EXTRACTOR.get_or_init(|| extractor).clone())
}

/// The process-wide extractor, defaulting to rules until `init_from_env` runs
/// (unit tests and library callers that never configured one).
pub fn extractor() -> Arc<dyn Extractor> {
    EXTRACTOR.get_or_init(|| Arc::new(RuleExtractor::default())).clone()
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
            RuleExtractor { max_facts: 2, heuristics: Profile::Generic }
                .extract(&text, &ExtractCtx { entity_id: "alice", timestamp_ms: 0, relations: &[] })
                .len()
                <= 2
        );
    }

    #[test]
    fn labels_become_fact_keys() {
        assert_eq!(slug("city of residence"), "city_of_residence");
        assert_eq!(slug("  Job  Title "), "job_title");
        assert_eq!(slug("favorite thing!"), "favorite_thing");
    }

    #[test]
    fn default_extractor_is_rules() {
        assert_eq!(extractor().name(), "rules");
    }
}
