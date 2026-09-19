use std::collections::HashSet;

use crate::core::text::normalize_fact_text;

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

/// The part of a memory that should be embedded and indexed.
///
/// A caller may prefix `[Session ID: ...]` / `[Session Date: ...]`, which the
/// engine reads for event time. Those headers must not reach the embedding or
/// the FTS index: a session id contributes nothing to meaning, and an indexed
/// date lets a lexical query match a document by its metadata rather than by
/// what it says. Derived paths (facts, companions, salient terms) already
/// strip them; embedding and FTS did not.
///
/// A document that is nothing but headers is kept as-is rather than reduced
/// to an empty, unsearchable memory.
pub fn content_for_index(text: &str) -> &str {
    let stripped = strip_leading_bracketed_prefixes(text);
    if stripped.is_empty() {
        text
    } else {
        stripped
    }
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
    let trimmed = line.trim().trim_start_matches(['-', '*', '•', '>', ' ']);
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

#[cfg(test)]
mod moved_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_extract_bracketed_header_value_found() {
        let text = "[Session Focus: cooking]\nSome content";
        assert_eq!(
            extract_bracketed_header_value(text, "Session Focus"),
            Some("cooking".to_string())
        );
    }

    #[test]
    fn test_extract_bracketed_header_value_date() {
        let text = "[Session Date: 2024-01-15]\nContent";
        assert_eq!(
            extract_bracketed_header_value(text, "Session Date"),
            Some("2024-01-15".to_string())
        );
    }

    #[test]
    fn test_extract_bracketed_header_value_missing_label() {
        let text = "[Other: value]";
        assert_eq!(extract_bracketed_header_value(text, "Session Focus"), None);
    }

    #[test]
    fn test_extract_bracketed_header_value_no_brackets() {
        let text = "Session Focus: cooking";
        assert_eq!(extract_bracketed_header_value(text, "Session Focus"), None);
    }

    #[test]
    fn test_extract_bracketed_header_value_case_insensitive_label() {
        let text = "[session focus: cooking]";
        assert_eq!(
            extract_bracketed_header_value(text, "Session Focus"),
            Some("cooking".to_string())
        );
    }

    #[test]
    fn test_extract_bracketed_header_value_empty_value() {
        let text = "[Session Focus: ]";
        assert_eq!(extract_bracketed_header_value(text, "Session Focus"), None);
    }

    #[test]
    fn test_extract_bracketed_header_value_no_closing_bracket() {
        let text = "[Session Focus: cooking";
        assert_eq!(extract_bracketed_header_value(text, "Session Focus"), None);
    }

    #[test]
    fn test_extract_bracketed_header_value_trailing_text_after_bracket() {
        let text = "[Session Focus: cooking] extra";
        assert_eq!(extract_bracketed_header_value(text, "Session Focus"), None);
    }

    #[test]
    fn test_normalize_speaker_label_cleans_quotes() {
        assert_eq!(normalize_speaker_label("\"Alice\""), Some("Alice".to_string()));
    }

    #[test]
    fn test_normalize_speaker_label_cleans_brackets() {
        assert_eq!(normalize_speaker_label("[Bob]"), Some("Bob".to_string()));
    }

    #[test]
    fn test_normalize_speaker_label_cleans_hyphens() {
        assert_eq!(normalize_speaker_label("-Charlie-"), Some("Charlie".to_string()));
    }

    #[test]
    fn test_normalize_speaker_label_cleans_stars() {
        assert_eq!(normalize_speaker_label("*Dave*"), Some("Dave".to_string()));
    }

    #[test]
    fn test_normalize_speaker_label_filters_system() {
        assert_eq!(normalize_speaker_label("system"), None);
    }

    #[test]
    fn test_normalize_speaker_label_filters_session_id() {
        assert_eq!(normalize_speaker_label("Session ID"), None);
    }

    #[test]
    fn test_normalize_speaker_label_filters_date() {
        assert_eq!(normalize_speaker_label("Session Date"), None);
    }

    #[test]
    fn test_normalize_speaker_label_filters_window_turns() {
        assert_eq!(normalize_speaker_label("Window Turns"), None);
    }

    #[test]
    fn test_normalize_speaker_label_filters_session_prefix() {
        assert_eq!(normalize_speaker_label("Session Metadata"), None);
    }

    #[test]
    fn test_normalize_speaker_label_filters_window_prefix() {
        assert_eq!(normalize_speaker_label("Window Context"), None);
    }

    #[test]
    fn test_normalize_speaker_label_empty_after_cleaning() {
        assert_eq!(normalize_speaker_label("---"), None);
    }

    #[test]
    fn test_normalize_speaker_label_case_insensitive_filters() {
        assert_eq!(normalize_speaker_label("SYSTEM"), None);
    }

    #[test]
    fn test_strip_leading_bracketed_prefixes_removes_single() {
        assert_eq!(strip_leading_bracketed_prefixes("[Session Focus: cooking] Hello"), "Hello");
    }

    #[test]
    fn test_strip_leading_bracketed_prefixes_removes_multiple() {
        assert_eq!(strip_leading_bracketed_prefixes("[A][B][C] rest"), "rest");
    }

    #[test]
    fn test_strip_leading_bracketed_prefixes_no_brackets() {
        assert_eq!(strip_leading_bracketed_prefixes("Hello world"), "Hello world");
    }

    #[test]
    fn test_strip_leading_bracketed_prefixes_empty_input() {
        assert_eq!(strip_leading_bracketed_prefixes(""), "");
    }

    #[test]
    fn test_strip_leading_bracketed_prefixes_only_brackets() {
        assert_eq!(strip_leading_bracketed_prefixes("[A][B]"), "");
    }

    #[test]
    fn test_strip_leading_bracketed_prefixes_unclosed_bracket() {
        assert_eq!(strip_leading_bracketed_prefixes("[Unclosed rest"), "[Unclosed rest");
    }

    #[test]
    fn test_value_to_text_string() {
        assert_eq!(value_to_text(&json!("hello world")), Some("hello world".to_string()));
    }

    #[test]
    fn test_value_to_text_array() {
        let v = json!(["hello", "world"]);
        assert_eq!(value_to_text(&v), Some("hello world".to_string()));
    }

    #[test]
    fn test_value_to_text_object_text_key() {
        let v = json!({"text": "hello"});
        assert_eq!(value_to_text(&v), Some("hello".to_string()));
    }

    #[test]
    fn test_value_to_text_object_content_key() {
        let v = json!({"content": "world"});
        assert_eq!(value_to_text(&v), Some("world".to_string()));
    }

    #[test]
    fn test_value_to_text_object_text_preferred_over_content() {
        let v = json!({"text": "chosen", "content": "ignored"});
        assert_eq!(value_to_text(&v), Some("chosen".to_string()));
    }

    #[test]
    fn test_value_to_text_nested_array() {
        let v = json!([{"text": "a"}, {"text": "b"}]);
        assert_eq!(value_to_text(&v), Some("a b".to_string()));
    }

    #[test]
    fn test_value_to_text_number_returns_none() {
        assert_eq!(value_to_text(&json!(42)), None);
    }

    #[test]
    fn test_value_to_text_bool_returns_none() {
        assert_eq!(value_to_text(&json!(true)), None);
    }

    #[test]
    fn test_value_to_text_null_returns_none() {
        assert_eq!(value_to_text(&json!(null)), None);
    }

    #[test]
    fn test_value_to_text_empty_string_returns_none() {
        assert_eq!(value_to_text(&json!("")), None);
    }

    #[test]
    fn test_value_to_text_object_no_text_or_content() {
        let v = json!({"foo": "bar"});
        assert_eq!(value_to_text(&v), None);
    }

    #[test]
    fn test_extract_dialogue_messages_json_role_content() {
        let text = r#"{"role": "user", "content": "hello"}"#;
        let msgs = extract_dialogue_messages(text);
        assert!(!msgs.is_empty());
        assert!(msgs.iter().any(|(r, c)| r == "user" && c == "hello"));
    }

    #[test]
    fn test_extract_dialogue_messages_json_array() {
        let text =
            r#"[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]"#;
        let msgs = extract_dialogue_messages(text);
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn test_extract_dialogue_messages_role_prefixed_line() {
        let text = "User: hello there\nAssistant: how can I help?";
        let msgs = extract_dialogue_messages(text);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0], ("User".to_string(), "hello there".to_string()));
    }

    #[test]
    fn test_extract_dialogue_messages_arrow_separator() {
        let text = "User => hello there";
        let msgs = extract_dialogue_messages(text);
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_extract_dialogue_messages_dash_separator() {
        let text = "User - hello there";
        let msgs = extract_dialogue_messages(text);
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_extract_dialogue_messages_dedup_identical() {
        let text = "User: hello\nUser: hello";
        let msgs = extract_dialogue_messages(text);
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_extract_dialogue_messages_keeps_different() {
        let text = "User: hello\nAssistant: hi";
        let msgs = extract_dialogue_messages(text);
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn test_extract_dialogue_messages_system_filtered() {
        let text = "System: boot\nUser: hello";
        let msgs = extract_dialogue_messages(text);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].0, "User");
    }

    #[test]
    fn test_extract_dialogue_messages_bracketed_line_skipped() {
        let text = "[Session Focus: test]\nUser: hello";
        let msgs = extract_dialogue_messages(text);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].0, "User");
    }

    #[test]
    fn test_extract_dialogue_messages_bullet_prefixed() {
        let text = "- User: hello\n* Assistant: reply";
        let msgs = extract_dialogue_messages(text);
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn test_extract_dialogue_messages_empty_text() {
        let msgs = extract_dialogue_messages("");
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_is_fact_like_message_i_like() {
        assert!(is_fact_like_message("I like pizza"));
    }

    #[test]
    fn test_is_fact_like_message_my_name() {
        assert!(is_fact_like_message("my name is Alice"));
    }

    #[test]
    fn test_is_fact_like_message_me() {
        assert!(is_fact_like_message("tell me about it"));
    }

    #[test]
    fn test_is_fact_like_message_we() {
        assert!(is_fact_like_message("we went to the park"));
    }

    #[test]
    fn test_is_fact_like_message_our() {
        assert!(is_fact_like_message("our house is big"));
    }

    #[test]
    fn test_is_fact_like_message_i_contraction() {
        assert!(is_fact_like_message("I'm tired"));
    }

    #[test]
    fn test_is_fact_like_message_false_third_person() {
        assert!(!is_fact_like_message("He likes pizza"));
    }

    #[test]
    fn test_is_fact_like_message_false_no_pronouns() {
        assert!(!is_fact_like_message("The sky is blue"));
    }

    #[test]
    fn test_is_fact_like_message_false_empty() {
        assert!(!is_fact_like_message(""));
    }

    #[test]
    fn test_is_fact_like_message_false_proper_noun() {
        assert!(!is_fact_like_message("Alice went to the store"));
    }

    #[test]
    fn test_extract_companion_texts_dialogue_produces_gist() {
        let text = "User: hello\nAssistant: hi there";
        let (gist, _facts) = extract_companion_texts(text);
        assert!(gist.is_some());
        assert!(gist.unwrap().contains("Session gist:"));
    }

    #[test]
    fn test_extract_companion_texts_no_dialogue_no_gist() {
        let text = "Just some plain text.";
        let (gist, facts) = extract_companion_texts(text);
        assert!(gist.is_none());
        assert!(facts.is_empty());
    }

    #[test]
    fn test_extract_companion_texts_fact_like_texts_extracted() {
        let text = "User: I love pizza\nAssistant: me too\nUser: my name is Bob";
        let (_, facts) = extract_companion_texts(text);
        assert!(!facts.is_empty());
        assert!(facts.iter().any(|f| f.contains("I love pizza")));
    }

    #[test]
    fn test_extract_companion_texts_gist_first_three_lines() {
        let text = "User: a\nAssistant: b\nUser: c\nAssistant: d";
        let (gist, _) = extract_companion_texts(text);
        let gist = gist.unwrap();
        assert_eq!(gist.matches('|').count(), 2);
    }

    #[test]
    fn test_extract_companion_texts_only_non_fact_returns_empty_facts() {
        let text = "User: hello\nAssistant: hi";
        let (_, facts) = extract_companion_texts(text);
        assert!(facts.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_for_index_drops_a_leading_header_block() {
        let text = "[Session ID: abc]\n[Session Date: 2023/05/20]\nuser: hi\nassistant: hey";
        assert_eq!(content_for_index(text), "user: hi\nassistant: hey");
    }

    #[test]
    fn content_for_index_keeps_a_header_only_document() {
        // Stripping to nothing would store an empty, unsearchable memory.
        let text = "[Session Date: 2023/05/20]";
        assert_eq!(content_for_index(text), text);
    }

    #[test]
    fn content_for_index_leaves_ordinary_text_alone() {
        assert_eq!(content_for_index("user: hi"), "user: hi");
        assert_eq!(content_for_index(""), "");
    }

    #[test]
    fn the_date_header_is_read_before_it_is_stripped() {
        let text = "[Session Date: 2023/05/20]\nuser: hi";
        assert_eq!(
            extract_bracketed_header_value(text, "Session Date").as_deref(),
            Some("2023/05/20")
        );
        assert!(!content_for_index(text).contains("2023/05/20"));
    }
}
