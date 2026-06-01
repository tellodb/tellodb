pub use crate::api::ingest::alias::*;
pub use crate::api::ingest::chunking::*;
pub use crate::api::ingest::companion::*;
pub use crate::api::ingest::datetime::*;
pub use crate::api::ingest::dialogue::*;
pub use crate::api::ingest::fact::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ingest::salient::*;
    use crate::api::types::IngestPayload;
    use serde_json::json;

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
        assert!(msgs.len() >= 1);
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
        let (gist, _facts) = extract_companion_texts(text);
        assert!(gist.is_none());
        assert!(_facts.is_empty());
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

    #[test]
    fn test_parse_date_to_epoch_ms_iso_date() {
        let ms = parse_date_to_epoch_ms("2024-01-15", 0).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!(y, 2024);
        assert_eq!(m, 1);
        assert_eq!(d, 15);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_month_day_year() {
        let ms = parse_date_to_epoch_ms("January 15 2024", 0).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!(y, 2024);
        assert_eq!(m, 1);
        assert_eq!(d, 15);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_year_only() {
        let ref_ms = epoch_ms_from_ymd(2024, 6, 15).unwrap();
        let ms = parse_date_to_epoch_ms("2025", ref_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!(y, 2025);
        assert_eq!(m, 1);
        assert_eq!(d, 1);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_may_disambiguation() {
        let ref_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = parse_date_to_epoch_ms("May 2024", ref_ms);
        assert!(ms.is_some(), "May with year should succeed");
        let (y, m, d) = civil_from_epoch_ms(ms.unwrap());
        assert_eq!((y, m, d), (2024, 5, 1));
    }

    #[test]
    fn test_parse_date_to_epoch_ms_may_with_day() {
        let ref_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = parse_date_to_epoch_ms("May 15 2024", ref_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!(y, 2024);
        assert_eq!(m, 5);
        assert_eq!(d, 15);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_day_before_month() {
        let ref_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = parse_date_to_epoch_ms("15 January 2024", ref_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!(y, 2024);
        assert_eq!(m, 1);
        assert_eq!(d, 15);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_invalid_text() {
        assert_eq!(parse_date_to_epoch_ms("not a date", 0), None);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_iso_with_punctuation() {
        let ms = parse_date_to_epoch_ms("date: 2024-03-15!", 0).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!(y, 2024);
        assert_eq!(m, 3);
        assert_eq!(d, 15);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_short_month() {
        let ref_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = parse_date_to_epoch_ms("Jan 15 2024", ref_ms).unwrap();
        let (_y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!(m, 1);
        assert_eq!(d, 15);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_year_out_of_range() {
        assert_eq!(parse_date_to_epoch_ms("1899-01-01", 0), None);
        assert_eq!(parse_date_to_epoch_ms("2201-01-01", 0), None);
    }

    #[test]
    fn test_extract_event_time_ms_tomorrow() {
        let doc_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = extract_event_time_ms("see you tomorrow", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 1, 2));
    }

    #[test]
    fn test_extract_event_time_ms_yesterday() {
        let doc_ms = epoch_ms_from_ymd(2024, 1, 15).unwrap();
        let ms = extract_event_time_ms("it happened yesterday", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 1, 14));
    }

    #[test]
    fn test_extract_event_time_ms_next_week() {
        let doc_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = extract_event_time_ms("next week", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 1, 8));
    }

    #[test]
    fn test_extract_event_time_ms_last_week() {
        let doc_ms = epoch_ms_from_ymd(2024, 1, 15).unwrap();
        let ms = extract_event_time_ms("last week", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 1, 8));
    }

    #[test]
    fn test_extract_event_time_ms_today() {
        let doc_ms = epoch_ms_from_ymd(2024, 6, 15).unwrap();
        let ms = extract_event_time_ms("today", doc_ms).unwrap();
        assert_eq!(ms, doc_ms);
    }

    #[test]
    fn test_extract_event_time_ms_next_month() {
        let doc_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = extract_event_time_ms("next month", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 1, 31));
    }

    #[test]
    fn test_extract_event_time_ms_absolute_date_overrides_relative() {
        let doc_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = extract_event_time_ms("meeting on 2024-06-15", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 6, 15));
    }

    #[test]
    fn test_extract_event_time_ms_no_match() {
        assert_eq!(extract_event_time_ms("no temporal info", 0), None);
    }

    #[test]
    fn test_extract_event_time_ms_past_week() {
        let doc_ms = epoch_ms_from_ymd(2024, 3, 15).unwrap();
        let ms = extract_event_time_ms("past week", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 3, 8));
    }

    #[test]
    fn test_extract_event_time_ms_past_month() {
        let doc_ms = epoch_ms_from_ymd(2024, 3, 15).unwrap();
        let ms = extract_event_time_ms("past month", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 2, 14));
    }

    #[test]
    fn test_infer_fact_key_marriage() {
        assert_eq!(infer_fact_key("I am married"), Some("relationship_status".to_string()));
    }

    #[test]
    fn test_infer_fact_key_children() {
        assert_eq!(infer_fact_key("I have two children"), Some("children_count".to_string()));
    }

    #[test]
    fn test_infer_fact_key_nickname() {
        assert_eq!(infer_fact_key("my nickname is Bob"), Some("nickname".to_string()));
    }

    #[test]
    fn test_infer_fact_key_hobbies() {
        assert_eq!(infer_fact_key("my hobbies include reading"), Some("hobbies".to_string()));
    }

    #[test]
    fn test_infer_fact_key_purchase() {
        assert_eq!(infer_fact_key("I bought a car"), Some("purchase".to_string()));
    }

    #[test]
    fn test_infer_fact_key_recipe() {
        assert_eq!(infer_fact_key("I found a great recipe"), Some("recipe".to_string()));
    }

    #[test]
    fn test_infer_fact_key_research() {
        assert_eq!(infer_fact_key("I research AI"), Some("research_topic".to_string()));
    }

    #[test]
    fn test_infer_fact_key_certificate() {
        assert_eq!(infer_fact_key("I got a certificate"), Some("certificate".to_string()));
    }

    #[test]
    fn test_infer_fact_key_favorite_team() {
        assert_eq!(
            infer_fact_key("my favorite team is the Lakers"),
            Some("favorite_team".to_string())
        );
    }

    #[test]
    fn test_infer_fact_key_degree() {
        assert_eq!(infer_fact_key("graduated with a degree"), Some("degree".to_string()));
    }

    #[test]
    fn test_infer_fact_key_previous_last_name() {
        assert_eq!(infer_fact_key("my old name is Smith"), Some("previous_last_name".to_string()));
    }

    #[test]
    fn test_infer_fact_key_previous_occupation() {
        assert_eq!(
            infer_fact_key("previous occupation was teacher"),
            Some("previous_occupation".to_string())
        );
    }

    #[test]
    fn test_infer_fact_key_commute_duration() {
        assert_eq!(
            infer_fact_key("my commute takes 30 minutes"),
            Some("commute_duration".to_string())
        );
    }

    #[test]
    fn test_infer_fact_key_internet_plan() {
        assert_eq!(
            infer_fact_key("I upgraded my internet plan to 500 mbps"),
            Some("internet_plan_speed".to_string())
        );
    }

    #[test]
    fn test_infer_fact_key_spotify_playlist() {
        assert_eq!(
            infer_fact_key("I created a spotify playlist named vibes"),
            Some("spotify_playlist_name".to_string())
        );
    }

    #[test]
    fn test_infer_fact_key_called_triggers_nickname() {
        assert_eq!(infer_fact_key("people called me Bob"), Some("nickname".to_string()));
    }

    #[test]
    fn test_infer_fact_key_martial_arts() {
        assert_eq!(infer_fact_key("I practice karate"), Some("martial_arts".to_string()));
    }

    #[test]
    fn test_infer_fact_key_martial_arts_phrase() {
        assert_eq!(infer_fact_key("I study martial arts"), Some("martial_arts".to_string()));
    }

    #[test]
    fn test_infer_fact_key_spend_price() {
        let result = infer_fact_key("I spent $50 on books").unwrap();
        assert!(result.ends_with("_price"));
    }

    #[test]
    fn test_infer_fact_key_identity_residence() {
        assert_eq!(infer_fact_key("I live in New York"), Some("residence".to_string()));
    }

    #[test]
    fn test_infer_fact_key_identity_employer() {
        assert_eq!(infer_fact_key("I work at Google"), Some("employer".to_string()));
    }

    #[test]
    fn test_infer_fact_key_identity_occupation() {
        assert_eq!(infer_fact_key("I am a doctor"), Some("occupation".to_string()));
    }

    #[test]
    fn test_infer_fact_key_identity_school() {
        assert_eq!(infer_fact_key("I study at MIT"), Some("school".to_string()));
    }

    #[test]
    fn test_infer_fact_key_prefixed_user_fact_stripped() {
        assert_eq!(
            infer_fact_key("User fact: I am married"),
            Some("relationship_status".to_string())
        );
    }

    #[test]
    fn test_infer_fact_key_irrelevant_returns_none() {
        assert_eq!(infer_fact_key("The sky is blue"), None);
    }

    #[test]
    fn test_infer_fact_key_empty_none() {
        assert_eq!(infer_fact_key(""), None);
    }

    #[test]
    fn test_infer_fact_key_has_new_and_got_returns_purchase() {
        assert_eq!(infer_fact_key("I got a new phone"), Some("purchase".to_string()));
    }

    #[test]
    fn test_infer_fact_key_child_count_no_have_verb() {
        assert_eq!(infer_fact_key("the children are playing"), None);
    }

    #[test]
    fn test_is_high_signal_atomic_claim_temporal_and_relation() {
        assert!(is_high_signal_atomic_claim("I visited New York last year"));
    }

    #[test]
    fn test_is_high_signal_atomic_claim_named_and_relation() {
        assert!(is_high_signal_atomic_claim("Alice works at Google"));
    }

    #[test]
    fn test_is_high_signal_atomic_claim_personal_signal() {
        assert!(is_high_signal_atomic_claim("I work at Google"));
    }

    #[test]
    fn test_is_high_signal_atomic_claim_no_relation_word() {
        assert!(!is_high_signal_atomic_claim("I am fine"));
    }

    #[test]
    fn test_is_high_signal_atomic_claim_empty() {
        assert!(!is_high_signal_atomic_claim(""));
    }

    #[test]
    fn test_is_high_signal_atomic_claim_he_she_personal() {
        assert!(is_high_signal_atomic_claim("He works at Microsoft"));
    }

    #[test]
    fn test_is_high_signal_atomic_claim_missing_relation() {
        assert!(!is_high_signal_atomic_claim("I the ball"));
    }

    #[test]
    fn test_sanitize_key_parts_basic() {
        assert_eq!(
            sanitize_key_parts(&["my", "favorite", "color"]),
            Some("favorite_color".to_string())
        );
    }

    #[test]
    fn test_sanitize_key_parts_filters_stop_words() {
        assert_eq!(sanitize_key_parts(&["the", "a", "an", "of"]), None);
    }

    #[test]
    fn test_sanitize_key_parts_filters_numbers() {
        assert_eq!(sanitize_key_parts(&["hello", "123", "world"]), Some("hello_world".to_string()));
    }

    #[test]
    fn test_sanitize_key_parts_filters_punctuation() {
        assert_eq!(sanitize_key_parts(&["hello!", "world?"]), Some("hello_world".to_string()));
    }

    #[test]
    fn test_sanitize_key_parts_all_filtered_returns_none() {
        assert_eq!(sanitize_key_parts(&["the", "a", "123", "!@#"]), None);
    }

    #[test]
    fn test_sanitize_key_parts_empty_input() {
        assert_eq!(sanitize_key_parts(&[]), None);
    }

    #[test]
    fn test_sanitize_key_parts_owned() {
        let parts = vec!["my".to_string(), "test".to_string()];
        assert_eq!(sanitize_key_parts_owned(&parts), Some("test".to_string()));
    }

    #[test]
    fn test_extract_aliases_from_text_short_name_prefix() {
        let entities = vec!["Melanie".to_string()];
        let text = "I saw Mel at the store";
        let aliases = extract_aliases_from_text(text, &entities);
        assert!(aliases.iter().any(|(a, c)| a == "mel" && c == "Melanie"));
    }

    #[test]
    fn test_extract_aliases_from_text_explicit_nickname() {
        let entities = vec!["Alice".to_string()];
        let text = "people call me Ali and Alice is my friend";
        let aliases = extract_aliases_from_text(text, &entities);
        assert!(aliases.iter().any(|(a, c)| a == "ali" && c == "Alice"));
    }

    #[test]
    fn test_extract_aliases_from_text_no_match() {
        let entities = vec!["Bob".to_string()];
        let text = "I went to the store";
        let aliases = extract_aliases_from_text(text, &entities);
        assert!(aliases.is_empty());
    }

    #[test]
    fn test_extract_aliases_from_text_no_entities() {
        let entities: Vec<String> = vec![];
        let text = "call me Nick";
        let aliases = extract_aliases_from_text(text, &entities);
        assert!(aliases.is_empty());
    }

    #[test]
    fn test_extract_aliases_from_text_short_entity_less_than_five() {
        let entities = vec!["Bob".to_string()];
        let text = "Bob is here";
        let aliases = extract_aliases_from_text(text, &entities);
        assert!(aliases.is_empty(), "entity length < 5 -> no prefix extraction");
    }

    #[test]
    fn test_extract_aliases_from_text_relationship_label() {
        let entities = vec!["Bob".to_string()];
        let text = "hubby Bob is great";
        let aliases = extract_aliases_from_text(text, &entities);
        assert!(aliases.iter().any(|(a, _)| a == "hubby"));
    }

    #[test]
    fn test_extract_aliases_from_text_too_many_prefix_collisions() {
        let entities = vec![
            "Melanie".to_string(),
            "Melissa".to_string(),
            "Melody".to_string(),
            "Melvin".to_string(),
        ];
        let text = "Mel went to the store";
        let aliases = extract_aliases_from_text(text, &entities);
        assert!(aliases.is_empty(), "too many collisions should suppress");
    }

    #[test]
    fn test_is_numericish_digits() {
        assert!(is_numericish("123"));
    }

    #[test]
    fn test_is_numericish_with_dollar() {
        assert!(is_numericish("$50"));
    }

    #[test]
    fn test_is_numericish_with_decimal() {
        assert!(is_numericish("3.14"));
    }

    #[test]
    fn test_is_numericish_with_comma() {
        assert!(is_numericish("1,000"));
    }

    #[test]
    fn test_is_numericish_with_percent() {
        assert!(is_numericish("99%"));
    }

    #[test]
    fn test_is_numericish_text_false() {
        assert!(!is_numericish("hello"));
    }

    #[test]
    fn test_is_numericish_empty_false() {
        assert!(!is_numericish(""));
    }

    #[test]
    fn test_split_atomic_claims_basic() {
        let claims = split_atomic_claims("I like pizza. I have a dog.");
        assert_eq!(claims.len(), 2);
    }

    #[test]
    fn test_split_atomic_claims_filters_short() {
        let claims = split_atomic_claims("Hi. I like pizza. Ok.");
        assert_eq!(claims.len(), 1);
        assert!(claims[0].contains("I like pizza"));
    }

    #[test]
    fn test_split_atomic_claims_strips_bracketed() {
        let claims = split_atomic_claims("[meta] I like pizza.");
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0], "I like pizza");
    }

    #[test]
    fn test_split_atomic_claims_semicolons() {
        let claims = split_atomic_claims("I like pizza; I have a dog");
        assert_eq!(claims.len(), 2);
    }

    #[test]
    fn test_build_chunk_memory_id_with_split() {
        let p = make_payload("");
        let id = build_chunk_memory_id(&p, 0);
        assert_eq!(id, "user::session1::0");
    }

    #[test]
    fn test_build_chunk_memory_id_increments() {
        let p = make_payload("");
        let id = build_chunk_memory_id(&p, 3);
        assert_eq!(id, "user::session1::3");
    }

    #[test]
    fn test_build_chunk_memory_id_fallback() {
        let mut p = make_payload("");
        p.memory_id = "invalid".to_string();
        let id = build_chunk_memory_id(&p, 0);
        assert_eq!(id, "invalid::ct0");
    }

    #[test]
    fn test_extract_document_time_ms_from_header() {
        let text = "[Session Date: 2024-06-15]\ncontent";
        let ms = extract_document_time_ms(text, 0);
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 6, 15));
    }

    #[test]
    fn test_extract_document_time_ms_fallback() {
        let text = "no header here";
        let ms = extract_document_time_ms(text, 5000000);
        assert_eq!(ms, 5000000);
    }

    #[test]
    fn test_extract_named_phrases_basic() {
        let lines = vec!["Alice went to New York".to_string()];
        let phrases = extract_named_phrases(&lines);
        assert!(phrases.contains(&"Alice".to_string()));
        assert!(phrases.contains(&"New York".to_string()));
    }

    #[test]
    fn test_extract_named_phrases_skips_lowercase() {
        let lines = vec!["the cat sat on the mat".to_string()];
        let phrases = extract_named_phrases(&lines);
        assert!(phrases.is_empty());
    }

    #[test]
    fn test_extract_named_phrases_empty_input() {
        assert!(extract_named_phrases(&[]).is_empty());
    }

    #[test]
    fn test_build_contextual_key_basic() {
        let base = vec!["coffee".to_string()];
        let key = build_contextual_key(&[], &base, None);
        assert_eq!(key, Some("coffee".to_string()));
    }

    #[test]
    fn test_build_contextual_key_with_suffix() {
        let base = vec!["coffee".to_string()];
        let key = build_contextual_key(&[], &base, Some("price"));
        assert_eq!(key, Some("coffee_price".to_string()));
    }

    #[test]
    fn test_build_contextual_key_with_context() {
        let ctx = vec!["morning".to_string()];
        let base = vec!["coffee".to_string()];
        let key = build_contextual_key(&ctx, &base, None);
        assert_eq!(key, Some("morning_coffee".to_string()));
    }

    #[test]
    fn test_build_contextual_key_removes_stop_words() {
        let base = vec!["the".to_string(), "coffee".to_string()];
        let key = build_contextual_key(&[], &base, None);
        assert_eq!(key, Some("coffee".to_string()));
    }

    #[test]
    fn test_extract_salient_terms_returns_top_terms() {
        let text = "Alice likes cooking. Alice loves baking. Alice enjoys hiking.";
        let terms = extract_salient_terms(text, 3);
        assert!(!terms.is_empty());
        assert!(terms.len() <= 3);
    }

    #[test]
    fn test_extract_salient_terms_empty_text() {
        let terms = extract_salient_terms("", 5);
        assert!(terms.is_empty());
    }

    #[test]
    fn test_extract_salient_terms_filters_short() {
        let text = "a an the";
        let terms = extract_salient_terms(text, 5);
        assert!(terms.is_empty());
    }

    #[test]
    fn test_truncate_for_companion_under_limit() {
        let result = truncate_for_companion("short text", 100);
        assert_eq!(result, "short text");
    }

    #[test]
    fn test_truncate_for_companion_over_limit() {
        let long = "A".repeat(300);
        let result = truncate_for_companion(&long, 100);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_truncate_for_companion_empty() {
        let result = truncate_for_companion("", 100);
        assert_eq!(result, "");
    }

    #[test]
    fn test_epoch_ms_roundtrip() {
        let ms = epoch_ms_from_ymd(2024, 6, 15).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 6, 15));
    }

    #[test]
    fn test_epoch_ms_from_ymd_invalid_month() {
        assert_eq!(epoch_ms_from_ymd(2024, 0, 1), None);
        assert_eq!(epoch_ms_from_ymd(2024, 13, 1), None);
    }

    #[test]
    fn test_epoch_ms_from_ymd_invalid_day() {
        assert_eq!(epoch_ms_from_ymd(2024, 1, 0), None);
        assert_eq!(epoch_ms_from_ymd(2024, 1, 32), None);
    }

    #[test]
    fn test_parse_iso_date_valid() {
        assert_eq!(parse_iso_date("2024-01-15"), Some((2024, 1, 15)));
    }

    #[test]
    fn test_parse_iso_date_invalid_format() {
        assert_eq!(parse_iso_date("2024/01/15"), None);
    }

    #[test]
    fn test_parse_iso_date_out_of_range() {
        assert_eq!(parse_iso_date("1800-01-01"), None);
    }

    #[test]
    fn test_parse_iso_date_not_enough_parts() {
        assert_eq!(parse_iso_date("2024-01"), None);
    }

    #[test]
    fn test_month_number_full_names() {
        assert_eq!(month_number("january"), Some(1));
        assert_eq!(month_number("february"), Some(2));
        assert_eq!(month_number("december"), Some(12));
    }

    #[test]
    fn test_month_number_abbreviations() {
        assert_eq!(month_number("jan"), Some(1));
        assert_eq!(month_number("feb"), Some(2));
        assert_eq!(month_number("dec"), Some(12));
    }

    #[test]
    fn test_month_number_sept_variant() {
        assert_eq!(month_number("sept"), Some(9));
        assert_eq!(month_number("sep"), Some(9));
    }

    #[test]
    fn test_month_number_invalid() {
        assert_eq!(month_number("xyz"), None);
    }

    #[test]
    fn test_civil_date_roundtrip() {
        let days = days_from_civil(2024, 6, 15);
        let (y, m, d) = civil_from_days(days);
        assert_eq!((y, m, d), (2024, 6, 15));
    }

    #[test]
    fn test_civil_date_epoch() {
        let days = days_from_civil(1970, 1, 1);
        assert_eq!(days, 0);
    }
}
