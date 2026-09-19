use crate::api::types::IngestPayload;

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
    // Chunks used to be numbered `turn*100 + idx`, which reused real turn ids
    // (chunk 1 of turn 0 overwrote turn 1).
    crate::core::memory_id::MemoryId::derived_from(&payload.memory_id, &format!("c{idx}"))
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
mod moved_tests {
    use super::*;

    fn make_payload(text: &str) -> IngestPayload {
        IngestPayload {
            entity_id: "user".to_string(),
            memory_id: "user::session1::0".to_string(),
            timestamp: 1000000,
            textual_content: text.to_string(),
            relations: vec![],
            kind: None,
            fact_key: None,
            source_memory_id: None,
            index_semantic: None,
            enable_semantic_dedup: None,
            enable_consolidation: None,
            content_type: None,
            fact_operation: None,
            fact_confidence: None,
            fact_subject: None,
            fact_predicate: None,
            fact_object: None,
            ..Default::default()
        }
    }

    #[test]
    fn test_chunk_markdown_splits_on_headers() {
        let text = "# Title\ncontent\n## Subtitle\nmore\n# Another\nlast";
        let chunks = chunk_markdown(text);
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].contains("# Title"));
        assert!(chunks[0].contains("content"));
        assert!(chunks[1].contains("## Subtitle"));
        assert!(chunks[1].contains("more"));
        assert!(chunks[2].contains("# Another"));
        assert!(chunks[2].contains("last"));
    }

    #[test]
    fn test_chunk_markdown_no_headers_returns_full() {
        let text = "just plain text\nwith multiple lines\nbut no markdown headers";
        let chunks = chunk_markdown(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_chunk_markdown_empty_input() {
        let chunks = chunk_markdown("");
        assert_eq!(chunks, vec![""]);
    }

    #[test]
    fn test_chunk_markdown_respects_char_limit() {
        let long_line = "A".repeat(600);
        let text = format!("# H1\n{content}\n# H2\n{content}", content = long_line);
        let chunks = chunk_markdown(&text);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| c.len() <= 1100));
    }

    #[test]
    fn test_chunk_code_splits_on_function_boundaries() {
        let text = "fn foo() {}\nfn bar() {}\nimpl Baz {}\nfn baz() {}";
        let chunks = chunk_code(text);
        assert_eq!(chunks.len(), 4);
        assert!(chunks[0].contains("fn foo()"));
        assert!(chunks[1].contains("fn bar()"));
        assert!(chunks[2].contains("impl Baz"));
        assert!(chunks[3].contains("fn baz()"));
    }

    #[test]
    fn test_chunk_code_pub_fn_boundary() {
        let text = "pub fn foo() {}\npub fn bar() {}";
        let chunks = chunk_code(text);
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_chunk_code_no_boundaries_returns_full() {
        let text = "let x = 1;\nlet y = 2;";
        let chunks = chunk_code(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_chunk_code_respects_char_limit() {
        let long_line = "x".repeat(700);
        let text = format!("fn a() {{ {long_line} }}\nfn b() {{ {long_line} }}");
        let chunks = chunk_code(&text);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| c.len() <= 1200));
    }

    #[test]
    fn test_chunk_code_empty_input() {
        assert_eq!(chunk_code(""), vec![""]);
    }

    #[test]
    fn test_chunk_email_splits_on_from_subject_to_date() {
        let text = "From: alice\nSubject: Hello\n\nBody here";
        let chunks = chunk_email(text);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("From: alice"));
        assert!(chunks[1].contains("Subject: Hello"));
    }

    #[test]
    fn test_chunk_email_case_insensitive() {
        let text = "FROM: alice\nSUBJECT: hi\n\nbody\nfrom: bob\nsubject: re\n\nreply";
        let chunks = chunk_email(text);
        assert_eq!(chunks.len(), 4);
    }

    #[test]
    fn test_chunk_email_no_headers_returns_full() {
        let text = "just a plain message body without headers";
        let chunks = chunk_email(text);
        assert_eq!(chunks, vec![text]);
    }

    #[test]
    fn test_chunk_email_empty_input() {
        assert_eq!(chunk_email(""), vec![""]);
    }

    #[test]
    fn test_chunk_table_like_small_table_no_split() {
        let lines: Vec<String> = (0..20).map(|i| format!("row {i}")).collect();
        let text = lines.join("\n");
        let chunks = chunk_table_like(&text);
        assert_eq!(chunks, vec![text]);
    }

    #[test]
    fn test_chunk_table_like_large_table_splits() {
        let line = "A".repeat(100);
        let lines: Vec<String> = (0..50).map(|i| format!("{line}{i}")).collect();
        let text = lines.join("\n");
        let chunks = chunk_table_like(&text);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_chunk_table_like_respects_char_limit() {
        let long_line = "A".repeat(100);
        let lines: Vec<String> = (0..30).map(|i| format!("{long_line} {i}")).collect();
        let text = lines.join("\n");
        let chunks = chunk_table_like(&text);
        assert!(chunks.iter().all(|c| c.len() <= 1400));
    }

    #[test]
    fn test_chunk_plain_or_chat_splits_on_punctuation() {
        let text = "First sentence. Second sentence! Third? Fourth.";
        let chunks = chunk_plain_or_chat(text);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("First sentence"));
        assert!(chunks[0].contains("Second sentence"));
    }

    #[test]
    fn test_chunk_plain_or_chat_respects_char_limit() {
        let text = format!("{}. {}.", "A".repeat(500), "B".repeat(500));
        let chunks = chunk_plain_or_chat(&text);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_chunk_plain_or_chat_empty_input() {
        let chunks = chunk_plain_or_chat("");
        assert_eq!(chunks, vec![""]);
    }

    #[test]
    fn test_chunk_plain_or_chat_single_sentence() {
        let chunks = chunk_plain_or_chat("Hello world.");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_chunk_plain_or_chat_newline_as_sentence_boundary() {
        let text = "line one\nline two\nline three";
        let chunks = chunk_plain_or_chat(text);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("line one"));
        assert!(chunks[0].contains("line two"));
    }

    #[test]
    fn test_chunk_plain_or_chat_trailing_punctuation_handling() {
        let text = "Hello. World. Test.";
        let chunks = chunk_plain_or_chat(text);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_split_by_char_limit_basic() {
        let lines: Vec<String> = vec!["hello".into(), "world".into()];
        let chunks = split_by_char_limit(&lines, 20);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello\nworld");
    }

    #[test]
    fn test_split_by_char_limit_exceeds_max() {
        let lines: Vec<String> = vec!["A".repeat(100), "B".repeat(100), "C".repeat(100)];
        let chunks = split_by_char_limit(&lines, 150);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| c.len() <= 150));
    }

    #[test]
    fn test_split_by_char_limit_empty_input() {
        let chunks = split_by_char_limit(&[], 100);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_split_by_char_limit_single_line_under() {
        let lines = vec!["short".to_string()];
        let chunks = split_by_char_limit(&lines, 100);
        assert_eq!(chunks, vec!["short"]);
    }

    #[test]
    fn test_split_by_char_limit_exact_fit() {
        let lines = vec!["exact".to_string()];
        let chunks = split_by_char_limit(&lines, 5);
        assert_eq!(chunks, vec!["exact"]);
    }

    #[test]
    fn test_infer_content_type_code_triple_backtick() {
        let p = make_payload("```rust\nfn main() {}\n```");
        assert_eq!(infer_content_type(&p), "code");
    }

    #[test]
    fn test_infer_content_type_code_fn_keyword() {
        let p = make_payload("fn main() {\n  println!();\n}");
        assert_eq!(infer_content_type(&p), "code");
    }

    #[test]
    fn test_infer_content_type_code_class_keyword() {
        let p = make_payload("class Foo {\n  bar() {}\n}");
        assert_eq!(infer_content_type(&p), "code");
    }

    #[test]
    fn test_infer_content_type_markdown() {
        let p = make_payload("# Title\n\nSome content.");
        assert_eq!(infer_content_type(&p), "markdown");
    }

    #[test]
    fn test_infer_content_type_email() {
        let p = make_payload("From: alice\nSubject: hello\n\nBody text.");
        assert_eq!(infer_content_type(&p), "email");
    }

    #[test]
    fn test_infer_content_type_table_pipe() {
        let p = make_payload("| A | B |\n| 1 | 2 |");
        assert_eq!(infer_content_type(&p), "table");
    }

    #[test]
    fn test_infer_content_type_table_comma() {
        let p = make_payload("A,B,C\n1,2,3");
        assert_eq!(infer_content_type(&p), "table");
    }

    #[test]
    fn test_infer_content_type_chat() {
        let p = make_payload("User: hello\nAssistant: hi there");
        assert_eq!(infer_content_type(&p), "chat");
    }

    #[test]
    fn test_infer_content_type_plain() {
        let p = make_payload("Just a regular plain text.");
        assert_eq!(infer_content_type(&p), "plain");
    }

    #[test]
    fn test_infer_content_type_explicit_override() {
        let mut p = make_payload("# Markdown looking text");
        p.content_type = Some("plain".to_string());
        assert_eq!(infer_content_type(&p), "plain");
    }

    #[test]
    fn test_infer_content_type_code_takes_precedence_over_hash() {
        let p = make_payload("# comment\nfn main() {}");
        assert_eq!(infer_content_type(&p), "code");
    }

    #[test]
    fn test_build_chunk_memory_id_with_split() {
        let p = make_payload("");
        let id = build_chunk_memory_id(&p, 0);
        assert_eq!(id, format!("{}::c0", p.memory_id));
        assert_eq!(
            crate::core::memory_id::MemoryId::parse(&id).unwrap().turn(),
            crate::core::memory_id::MemoryId::parse(&p.memory_id).unwrap().turn()
        );
    }

    #[test]
    fn test_build_chunk_memory_id_increments() {
        let p = make_payload("");
        let id = build_chunk_memory_id(&p, 3);
        assert_eq!(id, format!("{}::c3", p.memory_id));
    }

    #[test]
    fn test_build_chunk_memory_id_fallback() {
        let mut p = make_payload("");
        p.memory_id = "invalid".to_string();
        let id = build_chunk_memory_id(&p, 0);
        assert_eq!(id, "invalid::c0");
    }
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
        assert_eq!(expanded[0].memory_id, "user::sess::0::c0");
        assert_eq!(expanded[1].memory_id, "user::sess::0::c1");
    }

    #[test]
    fn chunk_ids_never_reuse_other_turn_ids() {
        let long_text = "A. ".repeat(600);
        let chunk_ids: Vec<String> = (0..3)
            .flat_map(|turn| {
                expand_payload_for_content_type(&IngestPayload {
                    entity_id: "user".to_string(),
                    memory_id: format!("user::sess::{turn}"),
                    textual_content: long_text.clone(),
                    content_type: Some("plain".to_string()),
                    ..Default::default()
                })
            })
            .map(|p| p.memory_id)
            .collect();
        let unique: std::collections::HashSet<_> = chunk_ids.iter().collect();
        assert_eq!(unique.len(), chunk_ids.len());
        assert!(chunk_ids
            .iter()
            .all(|id| !["user::sess::1", "user::sess::2"].contains(&id.as_str())));
    }
}
