use std::collections::{HashMap, HashSet};

use crate::fts::tokenize_for_similarity;

const MIN_SALIENT_TOKEN_LEN: usize = 3;
const MIN_PHRASE_LEN: usize = 2;

pub fn normalize_fact_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn normalize_alpha_tokens(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_string())
        .collect()
}

pub fn has_token(tokens: &[String], needle: &str) -> bool {
    tokens.iter().any(|token| token == needle)
}

pub fn singularize_token(token: &str) -> String {
    let token = token.to_ascii_lowercase();
    if token.len() > 3 && token.ends_with("ies") {
        format!("{}y", &token[..token.len() - 3])
    } else if token.len() > 2 && token.ends_with('s') && !token.ends_with("ss") {
        token[..token.len() - 1].to_string()
    } else {
        token.to_string()
    }
}

pub fn dedupe_preserve_order(values: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    values
        .into_iter()
        .filter(|value| {
            let key = value.trim().to_ascii_lowercase();
            !key.is_empty() && seen.insert(key)
        })
        .collect()
}

pub fn is_low_signal_keyword(token: &str) -> bool {
    matches!(
        token,
        "what"
            | "when"
            | "where"
            | "who"
            | "why"
            | "how"
            | "would"
            | "could"
            | "should"
            | "did"
            | "does"
            | "do"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "likely"
            | "might"
            | "will"
            | "can"
            | "have"
            | "has"
            | "had"
            | "the"
            | "this"
            | "that"
            | "these"
            | "those"
            | "their"
            | "there"
            | "them"
            | "they"
            | "his"
            | "her"
            | "him"
            | "she"
            | "you"
            | "your"
            | "our"
            | "for"
            | "from"
            | "with"
            | "than"
            | "then"
            | "kind"
            | "kinds"
            | "type"
            | "types"
            | "really"
            | "still"
            | "just"
            | "some"
            | "many"
            | "more"
            | "very"
            | "also"
            | "about"
            | "around"
            | "into"
            | "over"
            | "under"
            | "after"
            | "before"
            | "today"
            | "tomorrow"
            | "yesterday"
            | "thing"
            | "things"
            | "people"
            | "person"
            | "went"
            | "going"
            | "got"
            | "make"
            | "made"
            | "take"
            | "took"
            | "doing"
            | "done"
            | "want"
            | "wanted"
            | "joined"
            | "started"
            | "looking"
            | "working"
    )
}

pub fn extract_salient_terms(text: &str, limit: usize) -> Vec<String> {
    let mut counts = HashMap::new();
    for token in tokenize_for_similarity(text)
        .into_iter()
        .map(|token| singularize_token(&token))
        .filter(|token| !is_low_signal_keyword(token) && token.len() >= MIN_SALIENT_TOKEN_LEN)
    {
        *counts.entry(token).or_insert(0usize) += 1;
    }
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.into_iter().take(limit).map(|(token, _)| token).collect()
}

pub fn extract_named_phrases(lines: &[String]) -> Vec<String> {
    let mut phrases = Vec::with_capacity(lines.len());

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
        phrases.into_iter().filter(|phrase| phrase.len() > MIN_PHRASE_LEN).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_fact_text_collapses_whitespace() {
        assert_eq!(normalize_fact_text("  hello   world  "), "hello world");
    }

    #[test]
    fn normalize_fact_text_single_word() {
        assert_eq!(normalize_fact_text("hello"), "hello");
    }

    #[test]
    fn normalize_fact_text_empty() {
        assert_eq!(normalize_fact_text(""), "");
    }

    #[test]
    fn normalize_fact_text_tabs_and_newlines() {
        assert_eq!(normalize_fact_text("a\tb\nc"), "a b c");
    }

    #[test]
    fn normalize_alpha_tokens_splits_on_non_alphanumeric() {
        assert_eq!(normalize_alpha_tokens("Hello-World!"), vec!["hello", "world"]);
    }

    #[test]
    fn normalize_alpha_tokens_keeps_alnum_together() {
        assert_eq!(normalize_alpha_tokens("hello123world"), vec!["hello123world"]);
    }

    #[test]
    fn normalize_alpha_tokens_empty() {
        let result: Vec<String> = vec![];
        assert_eq!(normalize_alpha_tokens(""), result);
    }

    #[test]
    fn normalize_alpha_tokens_mixed_punctuation() {
        assert_eq!(normalize_alpha_tokens("a!b@c#"), vec!["a", "b", "c"]);
    }

    #[test]
    fn has_token_exact_match_true() {
        let tokens = vec!["hello".to_string(), "world".to_string()];
        assert!(has_token(&tokens, "hello"));
    }

    #[test]
    fn has_token_no_match_false() {
        let tokens = vec!["hello".to_string(), "world".to_string()];
        assert!(!has_token(&tokens, "hi"));
    }

    #[test]
    fn has_token_case_sensitive() {
        let tokens = vec!["Hello".to_string()];
        assert!(!has_token(&tokens, "hello"));
    }

    #[test]
    fn singularize_token_ies_to_y() {
        assert_eq!(singularize_token("cities"), "city");
        assert_eq!(singularize_token("berries"), "berry");
    }

    #[test]
    fn singularize_token_trailing_s_stripped_not_ss() {
        assert_eq!(singularize_token("dogs"), "dog");
        assert_eq!(singularize_token("cats"), "cat");
    }

    #[test]
    fn singularize_token_ss_unchanged() {
        assert_eq!(singularize_token("class"), "class");
        assert_eq!(singularize_token("grass"), "grass");
    }

    #[test]
    fn singularize_token_already_singular() {
        assert_eq!(singularize_token("cat"), "cat");
        assert_eq!(singularize_token("hello"), "hello");
    }

    #[test]
    fn singularize_token_short_strings() {
        assert_eq!(singularize_token("a"), "a");
        assert_eq!(singularize_token("as"), "as");
    }

    #[test]
    fn dedupe_preserve_order_removes_duplicates() {
        let input = vec![
            "A".to_string(),
            "B".to_string(),
            "a".to_string(),
            "C".to_string(),
            "b".to_string(),
        ];
        let result = dedupe_preserve_order(input);
        assert_eq!(result, vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    }

    #[test]
    fn dedupe_preserve_order_case_insensitive() {
        let input = vec!["Hello".to_string(), "hello".to_string(), "HELLO".to_string()];
        let result = dedupe_preserve_order(input);
        assert_eq!(result, vec!["Hello".to_string()]);
    }

    #[test]
    fn dedupe_preserve_order_filters_empty() {
        let input =
            vec!["".to_string(), "hello".to_string(), "  ".to_string(), "hello".to_string()];
        let result = dedupe_preserve_order(input);
        assert_eq!(result, vec!["hello".to_string()]);
    }

    #[test]
    fn dedupe_preserve_order_empty_input() {
        let result = dedupe_preserve_order(Vec::new());
        assert!(result.is_empty());
    }

    #[test]
    fn is_low_signal_keyword_known_words_return_true() {
        assert!(is_low_signal_keyword("what"));
        assert!(is_low_signal_keyword("when"));
        assert!(is_low_signal_keyword("the"));
        assert!(is_low_signal_keyword("would"));
        assert!(is_low_signal_keyword("yesterday"));
    }

    #[test]
    fn is_low_signal_keyword_meaningful_words_return_false() {
        assert!(!is_low_signal_keyword("python"));
        assert!(!is_low_signal_keyword("database"));
        assert!(!is_low_signal_keyword("algorithm"));
    }

    #[test]
    fn extract_salient_terms_filters_low_signal_and_short_tokens() {
        let result = extract_salient_terms("building planning alice", 10);
        assert_eq!(result.len(), 3);
        assert!(result.contains(&"alice".to_string()));
        assert!(result.contains(&"building".to_string()));
        assert!(result.contains(&"planning".to_string()));
    }

    #[test]
    fn extract_salient_terms_ranks_by_frequency() {
        let result = extract_salient_terms("building planning building alice alice building", 10);
        assert_eq!(result[0], "building");
        assert_eq!(result[1], "alice");
        assert_eq!(result[2], "planning");
    }

    #[test]
    fn extract_salient_terms_respects_limit() {
        let result = extract_salient_terms("building planning alice charlie", 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn extract_named_phrases_capitalized_phrases() {
        let lines = vec!["John went to New York".to_string()];
        let result = extract_named_phrases(&lines);
        assert!(result.contains(&"John".to_string()));
        assert!(result.contains(&"New York".to_string()));
    }

    #[test]
    fn extract_named_phrases_mixed_case_input() {
        let lines = vec!["Hello World test".to_string(), "lowercase".to_string()];
        let result = extract_named_phrases(&lines);
        assert_eq!(result, vec!["Hello World".to_string()]);
    }

    #[test]
    fn extract_named_phrases_empty_input() {
        assert!(extract_named_phrases(&[]).is_empty());
    }

    #[test]
    fn extract_named_phrases_no_capitalized_words() {
        let lines = vec!["all lowercase here".to_string()];
        assert!(extract_named_phrases(&lines).is_empty());
    }
}
