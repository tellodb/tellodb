use crate::api::types::IngestPayload;
use crate::api::utils::split_memory_id;

pub fn chunk_markdown(text: &str) -> Vec<String> {
    let mut sections = Vec::new();
    let mut current = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') && !current.is_empty() {
            sections.push(current.join("\n"));
            current.clear();
        }
        current.push(line.to_string());
    }
    if !current.is_empty() {
        sections.push(current.join("\n"));
    }
    if sections.is_empty() {
        return vec![text.to_string()];
    }
    sections
        .into_iter()
        .flat_map(|section| {
            let lines = section.lines().map(|l| l.to_string()).collect::<Vec<_>>();
            split_by_char_limit(&lines, 1100)
        })
        .collect()
}

pub fn chunk_code(text: &str) -> Vec<String> {
    let boundary_prefixes = ["fn ", "pub fn ", "def ", "class ", "function ", "impl "];
    let mut blocks = Vec::new();
    let mut current = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        let is_boundary = boundary_prefixes.iter().any(|prefix| trimmed.starts_with(prefix));
        if is_boundary && !current.is_empty() {
            blocks.push(current.join("\n"));
            current.clear();
        }
        current.push(line.to_string());
    }
    if !current.is_empty() {
        blocks.push(current.join("\n"));
    }
    if blocks.is_empty() {
        return vec![text.to_string()];
    }
    blocks
        .into_iter()
        .flat_map(|block| {
            let lines = block.lines().map(|l| l.to_string()).collect::<Vec<_>>();
            split_by_char_limit(&lines, 1200)
        })
        .collect()
}

pub fn chunk_email(text: &str) -> Vec<String> {
    let mut messages = Vec::new();
    let mut current = Vec::new();
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        let boundary = lower.starts_with("from:")
            || lower.starts_with("subject:")
            || lower.starts_with("to:")
            || lower.starts_with("date:");
        if boundary && !current.is_empty() {
            messages.push(current.join("\n"));
            current.clear();
        }
        current.push(line.to_string());
    }
    if !current.is_empty() {
        messages.push(current.join("\n"));
    }
    if messages.is_empty() {
        vec![text.to_string()]
    } else {
        messages
    }
}

pub fn chunk_table_like(text: &str) -> Vec<String> {
    let lines = text.lines().map(|l| l.to_string()).collect::<Vec<_>>();
    if lines.len() <= 24 {
        return vec![text.to_string()];
    }
    split_by_char_limit(&lines, 1400)
}

pub fn chunk_plain_or_chat(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    for sentence in text.split(['.', '!', '?', '\n']) {
        let s = sentence.trim();
        if !s.is_empty() {
            sentences.push(s.to_string());
        }
    }
    if sentences.is_empty() {
        return vec![text.to_string()];
    }
    let mut out = Vec::new();
    let mut current = String::new();
    for sentence in sentences {
        let candidate_len =
            if current.is_empty() { sentence.len() } else { current.len() + 2 + sentence.len() };
        if !current.is_empty() && candidate_len > 900 {
            out.push(current.trim().to_string());
            current.clear();
        }
        if !current.is_empty() {
            current.push_str(". ");
        }
        current.push_str(&sentence);
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

pub fn split_by_char_limit(lines: &[String], max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for line in lines {
        let needs_new = !current.is_empty() && current.len() + line.len() + 1 > max_chars;
        if needs_new {
            out.push(current.trim().to_string());
            current.clear();
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

pub fn infer_content_type(payload: &IngestPayload) -> String {
    if let Some(ct) = payload.content_type.as_deref() {
        return ct.trim().to_ascii_lowercase();
    }
    let text = payload.textual_content.as_str();
    let lower = text.to_ascii_lowercase();
    if lower.contains("```") || lower.contains("fn ") || lower.contains("class ") {
        "code".to_string()
    } else if text.lines().any(|line| line.trim_start().starts_with('#')) {
        "markdown".to_string()
    } else if lower.contains("from:") && lower.contains("subject:") {
        "email".to_string()
    } else if text.lines().take(8).any(|line| line.contains('|') || line.contains(',')) {
        "table".to_string()
    } else if lower.contains("user:") || lower.contains("assistant:") {
        "chat".to_string()
    } else {
        "plain".to_string()
    }
}

pub fn build_chunk_memory_id(payload: &IngestPayload, idx: usize) -> String {
    if let Some((entity, session, turn_index)) = split_memory_id(&payload.memory_id) {
        let next_turn = turn_index.saturating_mul(100).saturating_add(idx);
        format!("{entity}::{session}::{next_turn}")
    } else {
        format!("{}::ct{}", payload.memory_id, idx)
    }
}

pub fn expand_payload_for_content_type(payload: &IngestPayload) -> Vec<IngestPayload> {
    let content_type = infer_content_type(payload);
    let chunks = match content_type.as_str() {
        "markdown" => chunk_markdown(&payload.textual_content),
        "code" => chunk_code(&payload.textual_content),
        "email" => chunk_email(&payload.textual_content),
        "table" => chunk_table_like(&payload.textual_content),
        _ => chunk_plain_or_chat(&payload.textual_content),
    };

    if chunks.len() <= 1 {
        return vec![payload.clone()];
    }

    let original_id = payload.memory_id.clone();
    let mut expanded = Vec::with_capacity(chunks.len());
    for (idx, chunk) in chunks.into_iter().enumerate() {
        let mut cloned = payload.clone();
        cloned.textual_content = chunk;
        cloned.source_memory_id = Some(original_id.clone());
        cloned.memory_id = build_chunk_memory_id(payload, idx);
        expanded.push(cloned);
    }
    expanded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_payload_for_content_type_no_split() {
        let payload = IngestPayload {
            entity_id: "user".to_string(),
            memory_id: "user::sess::0".to_string(),
            textual_content: "Short text.".to_string(),
            ..Default::default()
        };
        let expanded = expand_payload_for_content_type(&payload);
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].memory_id, "user::sess::0");
        assert_eq!(expanded[0].source_memory_id, None);
    }

    #[test]
    fn test_expand_payload_for_content_type_with_split() {
        let long_text = "A. ".repeat(600); // 1800 chars
        let payload_long = IngestPayload {
            entity_id: "user".to_string(),
            memory_id: "user::sess::0".to_string(),
            textual_content: long_text,
            content_type: Some("plain".to_string()),
            ..Default::default()
        };
        let expanded = expand_payload_for_content_type(&payload_long);
        assert!(expanded.len() > 1);
        assert_eq!(expanded[0].source_memory_id, Some("user::sess::0".to_string()));
        assert_eq!(expanded[0].memory_id, "user::sess::0");
        assert_eq!(expanded[1].memory_id, "user::sess::1");
    }
}
