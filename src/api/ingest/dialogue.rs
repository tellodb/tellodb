use std::collections::HashSet;

use crate::api::utils::normalize_fact_text;

pub fn extract_bracketed_header_value(text: &str, label: &str) -> Option<String> {
    let needle = format!("[{}:", label.to_ascii_lowercase());
    for line in text.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with(&needle) && trimmed.ends_with(']') {
            let value = trimmed
                .split_once(':')
                .map(|(_, value)| value.trim().trim_end_matches(']').trim())
                .unwrap_or_default();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

pub fn normalize_speaker_label(label: &str) -> Option<String> {
    let cleaned = label
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '[' | ']' | '-' | '*' | '•' | ' '));
    if cleaned.is_empty() {
        return None;
    }

    let lower = cleaned.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "system" | "session id" | "session date" | "session focus" | "window turns"
    ) || lower.starts_with("session ")
        || lower.starts_with("window ")
    {
        return None;
    }

    Some(cleaned.to_string())
}

pub fn strip_leading_bracketed_prefixes(text: &str) -> &str {
    let mut rest = text.trim();
    loop {
        if !rest.starts_with('[') {
            break;
        }
        let Some(end_idx) = rest.find(']') else {
            break;
        };
        rest = rest[end_idx + 1..].trim_start();
    }
    rest
}

pub fn value_to_text(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => {
            let normalized = normalize_fact_text(text);
            (!normalized.is_empty()).then_some(normalized)
        }
        serde_json::Value::Array(items) => {
            let joined = items.iter().filter_map(value_to_text).collect::<Vec<_>>().join(" ");
            let normalized = normalize_fact_text(joined.as_str());
            (!normalized.is_empty()).then_some(normalized)
        }
        serde_json::Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(value_to_text) {
                return Some(text);
            }
            if let Some(text) = map.get("content").and_then(value_to_text) {
                return Some(text);
            }
            None
        }
        _ => None,
    }
}

pub fn collect_dialogue_messages_from_json(
    value: &serde_json::Value,
    messages: &mut Vec<(String, String)>,
) {
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                collect_dialogue_messages_from_json(item, messages);
            }
        }
        serde_json::Value::Object(map) => {
            if let Some(role) = map.get("role").and_then(|value| value.as_str()) {
                if let Some(speaker) = normalize_speaker_label(role) {
                    if let Some(text) = map.get("content").and_then(value_to_text) {
                        let cleaned = normalize_fact_text(strip_leading_bracketed_prefixes(&text));
                        if !cleaned.is_empty() {
                            messages.push((speaker, cleaned));
                        }
                    }
                }
            }

            for value in map.values() {
                collect_dialogue_messages_from_json(value, messages);
            }
        }
        _ => {}
    }
}

pub fn parse_role_prefixed_line(line: &str) -> Option<(String, String)> {
    let trimmed =
        line.trim().trim_start_matches(['-', '*', '•', '>', ' ']);
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        return None;
    }

    for separator in [":", "=>", " - "] {
        if let Some((role, content)) = trimmed.split_once(separator) {
            let Some(speaker) = normalize_speaker_label(role) else {
                continue;
            };
            let cleaned = normalize_fact_text(strip_leading_bracketed_prefixes(content).trim());
            if !cleaned.is_empty() {
                return Some((speaker, cleaned));
            }
        }
    }
    None
}

pub fn extract_dialogue_messages(text: &str) -> Vec<(String, String)> {
    let mut messages = Vec::new();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
        collect_dialogue_messages_from_json(&value, &mut messages);
    }
    for line in text.lines() {
        if let Some((speaker, content)) = parse_role_prefixed_line(line) {
            messages.push((speaker, content));
        }
    }

    let mut seen = HashSet::new();
    messages
        .into_iter()
        .filter(|(_, message)| !message.is_empty())
        .filter(|(speaker, message)| seen.insert(format!("{speaker}\u{1f}{message}")))
        .collect()
}

pub fn is_fact_like_message(text: &str) -> bool {
    let lower = format!(" {} ", text.to_ascii_lowercase());
    lower.contains(" i ")
        || lower.contains(" my ")
        || lower.contains(" me ")
        || lower.contains(" i'm ")
        || lower.contains(" i'")
        || lower.contains(" we ")
        || lower.contains(" our ")
}

pub fn extract_companion_texts(text: &str) -> (Option<String>, Vec<String>) {
    let dialogue_lines = extract_dialogue_messages(text);

    let gist = if dialogue_lines.is_empty() {
        None
    } else {
        Some(format!(
            "Session gist: {}",
            dialogue_lines
                .iter()
                .take(3)
                .map(|(speaker, line)| format!("{speaker}: {line}"))
                .collect::<Vec<_>>()
                .join(" | ")
        ))
    };

    let fact_like = dialogue_lines
        .into_iter()
        .filter(|(_, line)| is_fact_like_message(line))
        .take(2)
        .map(|(speaker, line)| format!("{speaker}: {line}"))
        .collect::<Vec<_>>();

    (gist, fact_like)
}
