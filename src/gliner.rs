//! GLiNER span extraction through ONNX Runtime.
//!
//! GLiNER is a zero-shot span tagger: you hand it a list of labels at call
//! time and it marks the spans of text that fill them. That makes the label
//! set the fact schema, so slots can be declared without training and without
//! writing a pattern per slot.
//!
//! The model is a bi-encoder over a prompt of the form
//! `[CLS] <<ENT>> label₁ <<ENT>> label₂ <<SEP>> word₁ word₂ … [SEP]`.
//! `words_mask` marks the first sub-token of each word (the config's
//! `subtoken_pooling: first`), so the model can pool one vector per word.
//! Every span up to `max_width` words is then scored against every label,
//! giving logits of shape `[batch, words, max_width, labels]`.
//!
//! Only the span (not the relation) variant is implemented; predicates come
//! from the label, which is what the fact chain keys on.

use anyhow::{bail, Context, Result};
use ort::session::Session;
use ort::value::Value;
use parking_lot::Mutex;
use std::path::Path;
use tokenizers::Tokenizer;

/// A span the model marked, with the label it fills.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanPrediction {
    pub label: String,
    /// Span text, sliced from the input.
    pub text: String,
    pub start_word: usize,
    /// Inclusive.
    pub end_word: usize,
    /// Sigmoid of the logit, in (0, 1).
    pub score: f32,
}

pub struct GlinerModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_width: usize,
    max_len: usize,
    /// Labels per call; the model was trained with at most this many.
    max_types: usize,
    ent_token_id: i64,
    sep_token_id: i64,
    cls_id: i64,
    eos_id: i64,
}

/// Fields this code depends on from `gliner_config.json`.
#[derive(serde::Deserialize)]
struct GlinerConfig {
    #[serde(default = "default_max_width")]
    max_width: usize,
    #[serde(default = "default_max_len")]
    max_len: usize,
    #[serde(default = "default_max_types")]
    max_types: usize,
    #[serde(default)]
    span_mode: String,
    #[serde(default)]
    words_splitter_type: String,
    #[serde(default)]
    subtoken_pooling: String,
}

fn default_max_width() -> usize {
    12
}
fn default_max_len() -> usize {
    384
}
fn default_max_types() -> usize {
    25
}

impl GlinerModel {
    /// Loads a GLiNER export: `gliner_config.json`, `tokenizer.json` and an
    /// ONNX file (`onnx/model_int8.onnx`, else `onnx/model.onnx`, else
    /// `model.onnx`). Runs on the device selected by the server configuration.
    pub fn load(dir: &Path) -> Result<Self> {
        let config_path = dir.join("gliner_config.json");
        let config: GlinerConfig = serde_json::from_slice(
            &std::fs::read(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?,
        )
        .with_context(|| format!("parsing {}", config_path.display()))?;

        // Guard the assumptions this implementation makes, rather than
        // silently mis-encoding for a model built another way.
        if !config.span_mode.is_empty() && config.span_mode != "markerV0" {
            bail!(
                "unsupported GLiNER span_mode {:?} (only markerV0 is implemented)",
                config.span_mode
            );
        }
        if !config.words_splitter_type.is_empty() && config.words_splitter_type != "whitespace" {
            bail!(
                "unsupported GLiNER words_splitter_type {:?} (only whitespace is implemented)",
                config.words_splitter_type
            );
        }
        if !config.subtoken_pooling.is_empty() && config.subtoken_pooling != "first" {
            bail!(
                "unsupported GLiNER subtoken_pooling {:?} (only first is implemented)",
                config.subtoken_pooling
            );
        }

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow::anyhow!("loading {}: {err}", tokenizer_path.display()))?;

        let token_id = |token: &str| -> Result<i64> {
            tokenizer
                .token_to_id(token)
                .map(i64::from)
                .with_context(|| format!("tokenizer has no {token} token"))
        };
        let ent_token_id = token_id("<<ENT>>")?;
        let sep_token_id = token_id("<<SEP>>")?;
        let cls_id = token_id("[CLS]")?;
        let eos_id = token_id("[SEP]")?;

        let model_path = ["onnx/model_int8.onnx", "onnx/model.onnx", "model.onnx"]
            .iter()
            .map(|name| dir.join(name))
            .find(|path| path.exists())
            .with_context(|| format!("no ONNX model under {}", dir.display()))?;

        let (use_gpu, use_coreml) = crate::semantic::selected_device();
        let mut builder = Session::builder()
            .map_err(|err| anyhow::anyhow!("creating ONNX session builder: {err}"))?
            .with_execution_providers(crate::semantic::execution_providers_for(use_gpu, use_coreml))
            .map_err(|err| anyhow::anyhow!("selecting execution providers: {err}"))?;
        let session = builder
            .commit_from_file(&model_path)
            .map_err(|err| anyhow::anyhow!("loading {}: {err}", model_path.display()))?;

        tracing::info!(
            model = %model_path.display(),
            max_width = config.max_width,
            max_types = config.max_types,
            "GLiNER extractor loaded"
        );
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_width: config.max_width,
            max_len: config.max_len,
            max_types: config.max_types,
            ent_token_id,
            sep_token_id,
            cls_id,
            eos_id,
        })
    }

    pub fn max_types(&self) -> usize {
        self.max_types
    }

    fn encode_pieces(&self, text: &str) -> Result<Vec<i64>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|err| anyhow::anyhow!("tokenizing {text:?}: {err}"))?
            .get_ids()
            .iter()
            .map(|id| i64::from(*id))
            .collect())
    }

    /// Spans of `text` filling any of `labels`, scored above `threshold`.
    /// Overlapping spans are resolved greedily by score (flat tagging).
    pub fn predict(
        &self,
        text: &str,
        labels: &[String],
        threshold: f32,
    ) -> Result<Vec<SpanPrediction>> {
        if labels.is_empty() || text.trim().is_empty() {
            return Ok(Vec::new());
        }
        if labels.len() > self.max_types {
            bail!("{} labels exceeds the model's maximum of {}", labels.len(), self.max_types);
        }

        // Prompt: [CLS] <<ENT>> label … <<SEP>>, which no word maps onto.
        let mut input_ids = vec![self.cls_id];
        for label in labels {
            input_ids.push(self.ent_token_id);
            input_ids.extend(self.encode_pieces(&label.to_lowercase())?);
        }
        input_ids.push(self.sep_token_id);
        let mut words_mask = vec![0i64; input_ids.len()];

        // Text: one mask entry per word, on its first sub-token (1-indexed;
        // 0 means "not a word start").
        let words: Vec<(usize, &str)> =
            text.split_whitespace().map(|w| (0, w)).enumerate().map(|(i, (_, w))| (i, w)).collect();
        let mut kept_words = 0usize;
        for (index, word) in &words {
            let pieces = self.encode_pieces(word)?;
            if pieces.is_empty() {
                continue;
            }
            // Leave room for the closing [SEP].
            if input_ids.len() + pieces.len() + 1 > self.max_len {
                tracing::debug!(
                    max_len = self.max_len,
                    words = words.len(),
                    kept = kept_words,
                    "GLiNER input truncated"
                );
                break;
            }
            for (offset, piece) in pieces.into_iter().enumerate() {
                input_ids.push(piece);
                words_mask.push(if offset == 0 { *index as i64 + 1 } else { 0 });
            }
            kept_words += 1;
        }
        input_ids.push(self.eos_id);
        words_mask.push(0);

        if kept_words == 0 {
            return Ok(Vec::new());
        }

        let sequence_length = input_ids.len();
        let attention_mask = vec![1i64; sequence_length];

        // Every span of 1..=max_width words, as (start_word, end_word).
        let num_spans = kept_words * self.max_width;
        let mut span_idx = Vec::with_capacity(num_spans * 2);
        let mut span_mask = Vec::with_capacity(num_spans);
        for start in 0..kept_words {
            for width in 0..self.max_width {
                let end = start + width;
                span_idx.push(start as i64);
                span_idx.push(end as i64);
                span_mask.push(end < kept_words);
            }
        }

        let tensor = |name: &str, shape: Vec<usize>, data: Vec<i64>| {
            Value::from_array((shape, data))
                .map_err(|err| anyhow::anyhow!("building {name} tensor: {err}"))
        };
        let (shape, logits) = {
            let mut session = self.session.lock();
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => tensor("input_ids", vec![1, sequence_length], input_ids)?,
                    "attention_mask" =>
                        tensor("attention_mask", vec![1, sequence_length], attention_mask)?,
                    "words_mask" => tensor("words_mask", vec![1, sequence_length], words_mask)?,
                    "text_lengths" => tensor("text_lengths", vec![1, 1], vec![kept_words as i64])?,
                    "span_idx" => tensor("span_idx", vec![1, num_spans, 2], span_idx)?,
                    "span_mask" => Value::from_array((vec![1, num_spans], span_mask))
                        .map_err(|err| anyhow::anyhow!("building span_mask tensor: {err}"))?,
                ])
                .map_err(|err| anyhow::anyhow!("GLiNER inference failed: {err}"))?;
            let (shape, logits) = outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|err| anyhow::anyhow!("reading GLiNER logits: {err}"))?;
            (shape.to_vec(), logits.to_vec())
        };

        // [batch, words, max_width, labels]
        let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        let [_batch, out_words, out_widths, out_labels] = dims[..] else {
            bail!("unexpected GLiNER logits rank {:?}", dims);
        };
        if out_labels != labels.len() {
            bail!("GLiNER returned {out_labels} label scores for {} labels", labels.len());
        }

        // Word byte offsets, to slice spans out of the original text.
        let offsets: Vec<(usize, usize)> = text
            .split_whitespace()
            .map(|word| {
                let start = word.as_ptr() as usize - text.as_ptr() as usize;
                (start, start + word.len())
            })
            .collect();

        let mut candidates: Vec<SpanPrediction> = Vec::new();
        for start in 0..out_words.min(kept_words) {
            for width in 0..out_widths {
                let end = start + width;
                if end >= kept_words {
                    continue;
                }
                for (label_index, label) in labels.iter().enumerate() {
                    let flat = ((start * out_widths) + width) * out_labels + label_index;
                    let score = sigmoid(logits[flat]);
                    if score < threshold {
                        continue;
                    }
                    let span_text = trim_span(&text[offsets[start].0..offsets[end].1]);
                    if span_text.is_empty() {
                        continue;
                    }
                    candidates.push(SpanPrediction {
                        label: label.clone(),
                        text: span_text.to_string(),
                        start_word: start,
                        end_word: end,
                        score,
                    });
                }
            }
        }

        // Flat tagging: highest-scoring span wins its words.
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.start_word.cmp(&b.start_word))
                .then_with(|| a.label.cmp(&b.label))
        });
        let mut taken = vec![false; kept_words];
        let mut selected = Vec::new();
        for candidate in candidates {
            if (candidate.start_word..=candidate.end_word).any(|word| taken[word]) {
                continue;
            }
            taken[candidate.start_word..=candidate.end_word].fill(true);
            selected.push(candidate);
        }
        selected.sort_by_key(|span| span.start_word);
        Ok(selected)
    }
}

/// Drops sentence punctuation and brackets clinging to a span, so the value
/// stored for a fact is `Austin`, not `Austin.`. Internal punctuation stays,
/// keeping `Acme Corp.` -> `Acme Corp` but `U.S.A` intact.
fn trim_span(text: &str) -> &str {
    text.trim().trim_matches(|c: char| ".,;:!?\"'()[]{}".contains(c))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_lose_surrounding_punctuation_only() {
        assert_eq!(trim_span("Austin."), "Austin");
        assert_eq!(trim_span(" \"Acme Corp.\" "), "Acme Corp");
        assert_eq!(trim_span("Seattle, "), "Seattle");
        // Internal punctuation is part of the value.
        assert_eq!(trim_span("Wells Fargo & Co"), "Wells Fargo & Co");
        assert_eq!(trim_span("..."), "");
    }

    #[test]
    fn sigmoid_is_centered_and_monotonic() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(-8.0) < 0.001 && sigmoid(8.0) > 0.999);
        assert!(sigmoid(1.0) > sigmoid(0.5));
    }
}
