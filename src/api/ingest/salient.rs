use std::collections::HashMap;
use std::collections::HashSet;

use crate::api::ingest::dialogue::{extract_dialogue_messages, strip_leading_bracketed_prefixes};
use crate::api::ingest::fact::is_numericish;
use crate::api::utils::{
    is_low_signal_keyword, normalize_fact_text, singularize_token,
};
use crate::fts::tokenize_for_similarity;

pub fn extract_named_phrases(lines: &[String]) -> Vec<String> {
    let mut phrases = Vec::new();

    for line in lines {
        let mut current = Vec::new();
        for raw_word in line.split_whitespace() {
            let word = raw_word
                .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'' && c != '-');
            if word.is_empty() {
                continue;
            }

            let starts_upper = word.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false);
            let has_lower = word.chars().any(|c| c.is_ascii_lowercase());
            if starts_upper && has_lower {
                current.push(word.to_string());
            } else if !current.is_empty() {
                phrases.push(current.join(" "));
                current.clear();
            }
        }
        if !current.is_empty() {
            phrases.push(current.join(" "));
        }
    }

    dedupe_preserve_order(
        phrases
            .into_iter()
            .map(|phrase: String| phrase.trim().to_string())
            .filter(|phrase: &String| !phrase.is_empty())
            .filter(|phrase: &String| phrase.len() > 2)
            .collect(),
    )
}

fn dedupe_preserve_order(values: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    values
        .into_iter()
        .filter(|value| {
            let key = value.trim().to_ascii_lowercase();
            !key.is_empty() && seen.insert(key)
        })
        .collect()
}

pub fn build_keyword_companion_text(text: &str) -> Option<String> {
    let dialogue_lines = extract_dialogue_messages(text);
    let content_lines = if dialogue_lines.is_empty() {
        text.lines()
            .map(|line| normalize_fact_text(strip_leading_bracketed_prefixes(line)))
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
    } else {
        dialogue_lines.into_iter().map(|(_, line)| line).collect::<Vec<_>>()
    };

    if content_lines.is_empty() {
        return None;
    }

    let named_phrases = extract_named_phrases(&content_lines);
    let joined = content_lines.join(" ");
    let tokens = tokenize_for_similarity(&joined)
        .into_iter()
        .map(|token| singularize_token(&token))
        .filter(|token| !is_low_signal_keyword(token))
        .collect::<Vec<_>>();

    let mut phrases = named_phrases;
    let mut seen = phrases.iter().map(|phrase| phrase.to_ascii_lowercase()).collect::<HashSet<_>>();

    let mut bigram_counts = HashMap::new();
    for window in tokens.windows(2) {
        let left = &window[0];
        let right = &window[1];
        if left == right || is_numericish(left) || is_numericish(right) {
            continue;
        }
        *bigram_counts.entry(format!("{left} {right}")).or_insert(0usize) += 1;
    }

    let mut ranked_bigrams = bigram_counts.into_iter().collect::<Vec<_>>();
    ranked_bigrams.sort_by(|a, b| {
        b.1.cmp(&a.1).then_with(|| b.0.len().cmp(&a.0.len())).then_with(|| a.0.cmp(&b.0))
    });
    for (phrase, _) in ranked_bigrams {
        if phrases.len() >= 10 {
            break;
        }
        let lower = phrase.to_ascii_lowercase();
        if seen.insert(lower) {
            phrases.push(phrase);
        }
    }

    let mut unigram_counts = HashMap::new();
    for token in tokens {
        *unigram_counts.entry(token).or_insert(0usize) += 1;
    }
    let mut ranked_unigrams = unigram_counts.into_iter().collect::<Vec<_>>();
    ranked_unigrams.sort_by(|a, b| {
        b.1.cmp(&a.1).then_with(|| b.0.len().cmp(&a.0.len())).then_with(|| a.0.cmp(&b.0))
    });
    for (token, _) in ranked_unigrams {
        if phrases.len() >= 12 {
            break;
        }
        let covered = phrases
            .iter()
            .any(|phrase| phrase.to_ascii_lowercase().split_whitespace().any(|part| part == token));
        if !covered {
            phrases.push(token);
        }
    }

    (!phrases.is_empty()).then(|| format!("Session index: {}", phrases.join(" | ")))
}

pub fn extract_salient_terms(text: &str, limit: usize) -> Vec<String> {
    let mut counts = HashMap::new();
    for token in tokenize_for_similarity(text)
        .into_iter()
        .map(|token| singularize_token(&token))
        .filter(|token| !is_low_signal_keyword(token) && token.len() >= 4)
    {
        *counts.entry(token).or_insert(0usize) += 1;
    }
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.into_iter().take(limit).map(|(token, _)| token).collect()
}

pub fn truncate_for_companion(text: &str, max_chars: usize) -> String {
    let normalized = normalize_fact_text(text);
    if normalized.len() <= max_chars {
        normalized
    } else {
        normalized.chars().take(max_chars).collect::<String>().trim().to_string()
    }
}
