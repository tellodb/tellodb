use axum::http::{HeaderMap, HeaderName, HeaderValue};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::config::RetrievalConfig;
pub use crate::core::calendar::{
    days_in_month, days_since_epoch, extract_temporal_terms, month_to_ms, parse_temporal_window,
    FIRST_WEEK_END_DAY, LAST_WEEK_DAY_OFFSET, MAX_DAY_OF_MONTH, MAX_YEAR, MILLIS_PER_DAY, MIN_YEAR,
    YEAR_DIGITS,
};
use crate::storage::MemoryKind;

pub const RESET_CONFIRM_PHRASE: &str = "delete-all-data";
pub const SEMANTIC_TOP_DEFAULT: usize = 100;
pub const SEMANTIC_TOP_SCOPED_DEFAULT: usize = 3000;
pub const SEMANTIC_TOP_SCOPED_START_DEFAULT: usize = 256;
pub const SEMANTIC_TOP_SCOPED_STEP_DEFAULT: usize = 256;
pub const SEMANTIC_TOP_SCOPED_MIN_HITS_DEFAULT: usize = 24;

pub const SCOPED_ANN_STOP_MAX_ATTEMPTS_DEFAULT: usize = 3;
pub const SCOPED_ANN_STOP_MIN_SIMILARITY_DEFAULT: f32 = 0.70;
pub const SCOPED_ANN_STOP_MAX_HIT_GAIN_DEFAULT: usize = 2;
pub const SCOPED_ANN_STOP_MIN_SIMILARITY_GAIN_DEFAULT: f32 = 0.01;

pub const MIN_TOKENS_FOR_HARD_QUERY: usize = 6;
pub const SHORT_QUERY_TOKEN_MAX: usize = 2;
pub const SIMILARITY_CONVERGENCE_STRICT: f32 = 0.05;
pub const SIMILARITY_CONVERGENCE_LOOSE: f32 = 0.10;
pub const MIN_CONVERGENCE_SAMPLES: usize = 2;
pub const FIFTH_RANK_INDEX: usize = 4;
pub const MIN_SALIENT_TOKEN_LEN: usize = 3;
pub const MIN_PHRASE_LEN: usize = 2;
pub const DECAY_FLOOR: f32 = 0.35;

pub fn scoped_semantic_start(config: &RetrievalConfig, max_top: usize) -> usize {
    config.scoped_semantic_start.min(max_top)
}

pub fn scoped_semantic_min_hits(config: &RetrievalConfig, limit: usize, max_top: usize) -> usize {
    config
        .scoped_min_hits
        .unwrap_or_else(|| limit.saturating_mul(2).max(SEMANTIC_TOP_SCOPED_MIN_HITS_DEFAULT))
        .min(max_top)
}

pub struct ScopedAnnState {
    pub attempt: usize,
    pub current_top: usize,
    pub max_top: usize,
    pub hit_count: usize,
    pub min_hits: usize,
    pub top_similarity: Option<f32>,
    pub prev_hit_count: Option<usize>,
    pub prev_top_similarity: Option<f32>,
}

pub fn should_stop_scoped_ann(config: &RetrievalConfig, state: &ScopedAnnState) -> bool {
    if state.current_top >= state.max_top {
        return true;
    }

    let max_attempts = config.scoped_stop_max_attempts;
    if state.attempt >= max_attempts {
        return true;
    }

    if state.hit_count < state.min_hits {
        return false;
    }

    let strong_enough =
        state.top_similarity.map(|sim| sim >= config.scoped_stop_min_similarity).unwrap_or(false);
    if !strong_enough {
        return false;
    }

    let Some(prev_hits) = state.prev_hit_count else {
        return false;
    };
    let hit_gain = state.hit_count.saturating_sub(prev_hits);
    let low_hit_gain = hit_gain <= config.scoped_stop_max_hit_gain;

    let low_similarity_gain = match (state.top_similarity, state.prev_top_similarity) {
        (Some(current), Some(prev)) => {
            (current - prev).abs() <= config.scoped_stop_min_similarity_gain
        }
        _ => false,
    };

    low_hit_gain || low_similarity_gain
}

pub fn elapsed_ms_and_us(start: Instant) -> (u64, u64) {
    let elapsed = start.elapsed();
    (elapsed.as_millis() as u64, elapsed.as_micros() as u64)
}

pub fn insert_u64_header(headers: &mut HeaderMap, name: &str, value: u64) {
    if let (Ok(header_name), Ok(header_value)) =
        (HeaderName::from_bytes(name.as_bytes()), HeaderValue::from_str(&value.to_string()))
    {
        headers.insert(header_name, header_value);
    }
}

pub fn insert_stage_timing_headers(headers: &mut HeaderMap, base: &str, millis: u64, micros: u64) {
    insert_u64_header(headers, &format!("{base}-ms"), millis);
    insert_u64_header(headers, &format!("{base}-us"), micros);
}

pub fn insert_f32_header(headers: &mut HeaderMap, name: &str, value: f32) {
    if let (Ok(header_name), Ok(header_value)) =
        (HeaderName::from_bytes(name.as_bytes()), HeaderValue::from_str(&format!("{value:.4}")))
    {
        headers.insert(header_name, header_value);
    }
}

/// Cosine similarity from a usearch cosine distance (`1 - cos`).
pub fn cosine_similarity_from_distance(distance: f32) -> f32 {
    (1.0 - distance).clamp(-1.0, 1.0)
}

pub fn parse_kind(s: Option<&str>) -> MemoryKind {
    match s.map(|s| s.to_ascii_lowercase()).as_deref() {
        Some("decision") => MemoryKind::Decision,
        Some("lesson") => MemoryKind::Lesson,
        Some("preference") => MemoryKind::Preference,
        Some("session_summary" | "session-summary" | "sessionsummary") => {
            MemoryKind::SessionSummary
        }
        Some("fact") => MemoryKind::Fact,
        _ => MemoryKind::Conversational,
    }
}

/// Applies exponential time decay to a score.
/// `age_in_days`: Difference between current time and memory timestamp.
/// `half_life_days`: How many days until the memory loses 50% of its weight.
/// `floor`: Minimum multiplier so important old memories aren't lost (e.g., 0.35).
pub fn apply_time_decay(base_score: f32, age_in_days: f32, half_life_days: f32, floor: f32) -> f32 {
    if age_in_days <= 0.0 {
        return base_score;
    }

    let lambda = std::f32::consts::LN_2 / half_life_days;
    let decay_multiplier = (-lambda * age_in_days).exp();
    let final_multiplier = decay_multiplier.max(floor);
    base_score * final_multiplier
}

/// `(half_life_days, floor)` for ranking decay by memory kind.
pub fn decay_policy(kind: MemoryKind) -> (f32, f32) {
    match kind {
        MemoryKind::Conversational => (30.0, DECAY_FLOOR),
        MemoryKind::Lesson => (90.0, DECAY_FLOOR),
        MemoryKind::Fact => (180.0, DECAY_FLOOR),
        MemoryKind::SessionSummary => (14.0, DECAY_FLOOR),
        MemoryKind::Decision | MemoryKind::Preference => (365.0, DECAY_FLOOR),
    }
}

/// Ranks older memories lower, never below the kind's floor. Age never
/// excludes a memory: an imported three-year-old conversation must stay
/// findable. Removing memories is retention's job (`lifecycle`), measured
/// from when they were stored.
pub fn apply_decay_with_policy(
    base_score: f32,
    created_at_ms: u64,
    kind: MemoryKind,
    now_ms: u64,
) -> f32 {
    if kind.is_decay_exempt() {
        return base_score;
    }
    let age_days = (now_ms.saturating_sub(created_at_ms)) as f32 / MILLIS_PER_DAY as f32;
    let (half_life_days, floor) = decay_policy(kind);
    apply_time_decay(base_score, age_days, half_life_days, floor)
}

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

pub fn clip_profile_to_budget(profile_json: &str, max_fields: usize) -> String {
    let Ok(val) = serde_json::from_str::<serde_json::Value>(profile_json) else {
        return profile_json.to_string();
    };
    let Some(obj) = val.as_object() else {
        return profile_json.to_string();
    };
    let clipped: serde_json::Map<String, serde_json::Value> =
        obj.iter().take(max_fields).map(|(k, v)| (k.clone(), v.clone())).collect();
    serde_json::to_string(&clipped).unwrap_or_else(|_| profile_json.to_string())
}

pub fn should_apply_neural_rerank(
    query_text: &str,
    hnsw_raw: &[(u64, f32)],
    requested: bool,
) -> bool {
    if hnsw_raw.len() < MIN_CONVERGENCE_SAMPLES {
        return false;
    }

    let token_count = query_text.split_whitespace().count();
    let lower = query_text.to_ascii_lowercase();
    let implicit_hard_query = token_count >= MIN_TOKENS_FOR_HARD_QUERY
        || lower.starts_with("when ")
        || lower.contains(" before ")
        || lower.contains(" after ")
        || lower.contains(" both ")
        || lower.contains(" and ")
        || lower.contains("would")
        || lower.contains("might")
        || lower.contains("why ");
    if !requested && !implicit_hard_query {
        return false;
    }

    if token_count <= SHORT_QUERY_TOKEN_MAX {
        return true;
    }

    let top = 1.0 - hnsw_raw[0].1;
    let second = 1.0 - hnsw_raw[1].1;
    let fifth = hnsw_raw.get(FIFTH_RANK_INDEX).map(|(_, dist)| 1.0 - dist).unwrap_or(second);

    (top - second).abs() < SIMILARITY_CONVERGENCE_STRICT
        || (top - fifth).abs() < SIMILARITY_CONVERGENCE_LOOSE
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
    use crate::fts::tokenize_for_similarity;
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
    use crate::config::RetrievalConfig;
    use std::time::Instant;

    #[test]
    fn parse_kind_decision_variants() {
        assert_eq!(parse_kind(Some("decision")), MemoryKind::Decision);
        assert_eq!(parse_kind(Some("Decision")), MemoryKind::Decision);
    }

    #[test]
    fn parse_kind_lesson_variants() {
        assert_eq!(parse_kind(Some("lesson")), MemoryKind::Lesson);
        assert_eq!(parse_kind(Some("Lesson")), MemoryKind::Lesson);
    }

    #[test]
    fn parse_kind_preference_variants() {
        assert_eq!(parse_kind(Some("preference")), MemoryKind::Preference);
        assert_eq!(parse_kind(Some("Preference")), MemoryKind::Preference);
    }

    #[test]
    fn parse_kind_session_summary_variants() {
        assert_eq!(parse_kind(Some("session_summary")), MemoryKind::SessionSummary);
        assert_eq!(parse_kind(Some("session-summary")), MemoryKind::SessionSummary);
        assert_eq!(parse_kind(Some("SessionSummary")), MemoryKind::SessionSummary);
    }

    #[test]
    fn parse_kind_fact_variants() {
        assert_eq!(parse_kind(Some("fact")), MemoryKind::Fact);
        assert_eq!(parse_kind(Some("Fact")), MemoryKind::Fact);
    }

    #[test]
    fn parse_kind_defaults_to_conversational() {
        assert_eq!(parse_kind(None), MemoryKind::Conversational);
        assert_eq!(parse_kind(Some("")), MemoryKind::Conversational);
        assert_eq!(parse_kind(Some("DECISIONS")), MemoryKind::Conversational);
        assert_eq!(parse_kind(Some("UNKNOWN")), MemoryKind::Conversational);
    }

    #[test]
    fn apply_time_decay_no_decay_at_age_zero() {
        let result = apply_time_decay(1.0, 0.0, 30.0, 0.35);
        assert!((result - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_time_decay_half_life_halves_score() {
        let result = apply_time_decay(1.0, 30.0, 30.0, 0.35);
        assert!((result - 0.5).abs() < 0.001);
    }

    #[test]
    fn apply_time_decay_floor_clamps_at_minimum() {
        let result = apply_time_decay(1.0, 10000.0, 30.0, 0.35);
        assert!((result - 0.35).abs() < f32::EPSILON);
    }

    #[test]
    fn decay_policy_conversational() {
        let (half_life, floor) = decay_policy(MemoryKind::Conversational);
        assert!((half_life - 30.0).abs() < f32::EPSILON);
        assert!((floor - 0.35).abs() < f32::EPSILON);
    }

    #[test]
    fn decay_policy_lesson() {
        let (half_life, floor) = decay_policy(MemoryKind::Lesson);
        assert!((half_life - 90.0).abs() < f32::EPSILON);
        assert!((floor - 0.35).abs() < f32::EPSILON);
    }

    #[test]
    fn decay_policy_fact() {
        let (half_life, floor) = decay_policy(MemoryKind::Fact);
        assert!((half_life - 180.0).abs() < f32::EPSILON);
        assert!((floor - 0.35).abs() < f32::EPSILON);
    }

    #[test]
    fn decay_policy_session_summary() {
        let (half_life, floor) = decay_policy(MemoryKind::SessionSummary);
        assert!((half_life - 14.0).abs() < f32::EPSILON);
        assert!((floor - 0.35).abs() < f32::EPSILON);
    }

    #[test]
    fn decay_policy_decision() {
        let (half_life, floor) = decay_policy(MemoryKind::Decision);
        assert!((half_life - 365.0).abs() < f32::EPSILON);
        assert!((floor - 0.35).abs() < f32::EPSILON);
    }

    #[test]
    fn decay_policy_preference() {
        let (half_life, floor) = decay_policy(MemoryKind::Preference);
        assert!((half_life - 365.0).abs() < f32::EPSILON);
        assert!((floor - 0.35).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_decay_with_policy_decay_exempt_keeps_score() {
        assert_eq!(apply_decay_with_policy(0.75, 0, MemoryKind::Decision, u64::MAX), 0.75);
        assert_eq!(apply_decay_with_policy(0.75, 0, MemoryKind::Preference, u64::MAX), 0.75);
    }

    #[test]
    fn apply_decay_with_policy_old_memories_decay_to_floor_but_stay() {
        let day = 86_400_000;
        let recent = apply_decay_with_policy(1.0, 0, MemoryKind::Conversational, 30 * day);
        assert!(recent > 0.35 && recent < 1.0);
        let ancient = apply_decay_with_policy(1.0, 0, MemoryKind::Conversational, 3_000 * day);
        assert!((ancient - DECAY_FLOOR).abs() < 1e-6);
    }

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
        let input: Vec<String> = vec![];
        let result = dedupe_preserve_order(input);
        assert!(result.is_empty());
    }

    #[test]
    fn extract_temporal_terms_finds_years() {
        let result = extract_temporal_terms("events in 2023 and 1999");
        assert!(result.contains(&"2023".to_string()));
        assert!(result.contains(&"1999".to_string()));
    }

    #[test]
    fn extract_temporal_terms_finds_months() {
        let result = extract_temporal_terms("meeting in January and March");
        assert!(result.contains(&"january".to_string()));
        assert!(result.contains(&"march".to_string()));
    }

    #[test]
    fn extract_temporal_terms_finds_seasons() {
        let result = extract_temporal_terms("summer vacation 2024");
        assert_eq!(&*result, &["2024".to_string(), "summer".to_string()]);
    }

    #[test]
    fn extract_temporal_terms_special_terms() {
        let result = extract_temporal_terms("what happened yesterday and today");
        assert!(result.contains(&"yesterday".to_string()));
        assert!(result.contains(&"today".to_string()));
    }

    #[test]
    fn extract_temporal_terms_deduplicates() {
        let result = extract_temporal_terms("2024 in January and 2024 also january");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn extract_temporal_terms_no_temporal_terms_returns_empty() {
        let result = extract_temporal_terms("hello world");
        assert!(result.is_empty());
    }

    #[test]
    fn parse_temporal_window_month_year() {
        let result = parse_temporal_window("October 2023", None).unwrap();
        let expected = (month_to_ms(2023, 10, 1), month_to_ms(2023, 10, 31) + 86_400_000);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_season_year() {
        let result = parse_temporal_window("summer 2022", None).unwrap();
        let expected = (month_to_ms(2022, 6, 1), month_to_ms(2022, 8, 31) + 86_400_000);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_winter_wraps_year() {
        let result = parse_temporal_window("winter 2024", None).unwrap();
        let expected =
            (month_to_ms(2024, 12, 1), month_to_ms(2025, 2, days_in_month(2025, 2)) + 86_400_000);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_last_week_of_month() {
        let result = parse_temporal_window("last week of October 2023", None).unwrap();
        let expected = (month_to_ms(2023, 10, 25), month_to_ms(2023, 10, 31) + 86_400_000);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_specific_day() {
        let result = parse_temporal_window("May 1 2022", None).unwrap();
        let expected = (month_to_ms(2022, 5, 1), month_to_ms(2022, 5, 1) + 86_400_000);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_iso_formatted_date() {
        let result = parse_temporal_window("Where was I living as of 2024/05/12?", None).unwrap();
        let expected_start = month_to_ms(2024, 5, 12);
        assert_eq!(result, (expected_start, expected_start + 86_400_000));

        let result2 = parse_temporal_window("events on 2023-11-05", None).unwrap();
        let expected_start2 = month_to_ms(2023, 11, 5);
        assert_eq!(result2, (expected_start2, expected_start2 + 86_400_000));
    }

    #[test]
    fn parse_temporal_window_no_temporal_info_returns_none() {
        assert_eq!(parse_temporal_window("no time mentioned here", None), None);
    }

    #[test]
    fn parse_temporal_window_first_week_of_month() {
        let result = parse_temporal_window("first week of March 2023", None).unwrap();
        let expected = (month_to_ms(2023, 3, 1), month_to_ms(2023, 3, 7) + 86_400_000);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_relative_yesterday() {
        let ref_ms = 1_700_000_000_000_u64;
        let day_ms = 86_400_000_u64;
        let ref_day = (ref_ms / day_ms) * day_ms;
        let result = parse_temporal_window("what did I do yesterday?", Some(ref_ms)).unwrap();
        assert_eq!(result, (ref_day - day_ms, ref_day));
    }

    #[test]
    fn parse_temporal_window_relative_last_week() {
        let ref_ms = 1_700_000_000_000_u64;
        let day_ms = 86_400_000_u64;
        let result = parse_temporal_window("what happened last week?", Some(ref_ms)).unwrap();
        assert_eq!(result, (ref_ms - 7 * day_ms, ref_ms));
    }

    #[test]
    fn parse_temporal_window_relative_diff_ref_times() {
        let t1 = 1_650_000_000_000_u64;
        let t2 = 1_720_000_000_000_u64;
        let query = "show activities from last week";
        let win1 = parse_temporal_window(query, Some(t1)).unwrap();
        let win2 = parse_temporal_window(query, Some(t2)).unwrap();
        assert_ne!(win1, win2);
        assert_eq!(win1.1, t1);
        assert_eq!(win2.1, t2);
    }

    #[test]
    fn parse_temporal_window_past_n_days() {
        let ref_ms = 1_700_000_000_000_u64;
        let day_ms = 86_400_000_u64;
        let result = parse_temporal_window("updates in the past 5 days", Some(ref_ms)).unwrap();
        assert_eq!(result, (ref_ms - 5 * day_ms, ref_ms));
    }

    #[test]
    fn days_in_month_all_months() {
        assert_eq!(days_in_month(2023, 1), 31);
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(2023, 3), 31);
        assert_eq!(days_in_month(2023, 4), 30);
        assert_eq!(days_in_month(2023, 5), 31);
        assert_eq!(days_in_month(2023, 6), 30);
        assert_eq!(days_in_month(2023, 7), 31);
        assert_eq!(days_in_month(2023, 8), 31);
        assert_eq!(days_in_month(2023, 9), 30);
        assert_eq!(days_in_month(2023, 10), 31);
        assert_eq!(days_in_month(2023, 11), 30);
        assert_eq!(days_in_month(2023, 12), 31);
    }

    #[test]
    fn days_in_month_leap_year_feb() {
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2000, 2), 29);
    }

    #[test]
    fn days_in_month_non_leap_feb() {
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(1900, 2), 28);
    }

    #[test]
    fn clip_profile_to_budget_valid_json_clipped() {
        let input = r#"{"a":1,"b":2,"c":3}"#;
        let result = clip_profile_to_budget(input, 2);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.len(), 2);
    }

    #[test]
    fn clip_profile_to_budget_invalid_json_unchanged() {
        let input = "not valid json";
        let result = clip_profile_to_budget(input, 5);
        assert_eq!(result, input);
    }

    #[test]
    fn clip_profile_to_budget_non_object_json_unchanged() {
        let result = clip_profile_to_budget("[1,2,3]", 2);
        assert_eq!(result, "[1,2,3]");
    }

    #[test]
    fn should_apply_neural_rerank_empty_results() {
        assert!(!should_apply_neural_rerank("test", &[], false));
    }

    #[test]
    fn should_apply_neural_rerank_single_result() {
        assert!(!should_apply_neural_rerank("test", &[(1, 0.5)], false));
    }

    #[test]
    fn should_apply_neural_rerank_not_requested_not_hard() {
        assert!(!should_apply_neural_rerank("hello world", &[(1, 0.1), (2, 0.2)], false));
    }

    #[test]
    fn should_apply_neural_rerank_requested_short_query() {
        assert!(should_apply_neural_rerank("hello world", &[(1, 0.1), (2, 0.2)], true));
    }

    #[test]
    fn should_apply_neural_rerank_implicit_hard_when_prefix() {
        assert!(should_apply_neural_rerank("when did this", &[(1, 0.1), (2, 0.2)], false));
    }

    #[test]
    fn should_apply_neural_rerank_convergence_close_similarities() {
        assert!(should_apply_neural_rerank(
            "when did this happen here now",
            &[(1, 0.10), (2, 0.11)],
            false,
        ));
    }

    #[test]
    fn should_apply_neural_rerank_no_convergence_distant_similarities() {
        assert!(!should_apply_neural_rerank(
            "when did this happen here",
            &[(1, 0.10), (2, 0.30)],
            false,
        ));
    }

    #[test]
    fn should_apply_neural_rerank_six_or_more_tokens_triggers_hard() {
        assert!(should_apply_neural_rerank(
            "this is a six word query string",
            &[(1, 0.10), (2, 0.12)],
            false,
        ));
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
        let lines: Vec<String> = vec![];
        let result = extract_named_phrases(&lines);
        assert!(result.is_empty());
    }

    #[test]
    fn extract_named_phrases_no_capitalized_words() {
        let lines = vec!["all lowercase here".to_string()];
        let result = extract_named_phrases(&lines);
        assert!(result.is_empty());
    }

    #[allow(clippy::too_many_arguments)]
    fn make_ann_state(
        attempt: usize,
        current_top: usize,
        max_top: usize,
        hit_count: usize,
        min_hits: usize,
        top_similarity: Option<f32>,
        prev_hit_count: Option<usize>,
        prev_top_similarity: Option<f32>,
    ) -> ScopedAnnState {
        ScopedAnnState {
            attempt,
            current_top,
            max_top,
            hit_count,
            min_hits,
            top_similarity,
            prev_hit_count,
            prev_top_similarity,
        }
    }

    #[test]
    fn should_stop_scoped_ann_max_top_reached() {
        let s = make_ann_state(0, 100, 100, 0, 1, None, None, None);
        assert!(should_stop_scoped_ann(&RetrievalConfig::default(), &s));
    }

    #[test]
    fn should_stop_scoped_ann_max_attempts_reached() {
        let s = make_ann_state(10, 50, 100, 50, 10, Some(0.9), Some(40), Some(0.8));
        assert!(should_stop_scoped_ann(&RetrievalConfig::default(), &s));
    }

    #[test]
    fn should_stop_scoped_ann_not_enough_hits() {
        let s = make_ann_state(0, 50, 100, 5, 10, None, None, None);
        assert!(!should_stop_scoped_ann(&RetrievalConfig::default(), &s));
    }

    #[test]
    fn should_stop_scoped_ann_not_strong_enough_similarity() {
        let s = make_ann_state(0, 50, 100, 20, 10, Some(0.6), Some(10), Some(0.5));
        assert!(!should_stop_scoped_ann(&RetrievalConfig::default(), &s));
    }

    #[test]
    fn should_stop_scoped_ann_none_similarity_not_strong() {
        let s = make_ann_state(0, 50, 100, 20, 10, None, Some(10), Some(0.5));
        assert!(!should_stop_scoped_ann(&RetrievalConfig::default(), &s));
    }

    #[test]
    fn should_stop_scoped_ann_no_prev_hit_count_returns_false() {
        let s = make_ann_state(0, 50, 100, 20, 10, Some(0.8), None, Some(0.79));
        assert!(!should_stop_scoped_ann(&RetrievalConfig::default(), &s));
    }

    #[test]
    fn should_stop_scoped_ann_convergence_low_hit_gain() {
        let s = make_ann_state(0, 50, 100, 12, 10, Some(0.8), Some(10), Some(0.7));
        assert!(should_stop_scoped_ann(&RetrievalConfig::default(), &s));
    }

    #[test]
    fn should_stop_scoped_ann_convergence_low_similarity_gain() {
        let s = make_ann_state(0, 50, 100, 20, 10, Some(0.71), Some(10), Some(0.70));
        assert!(should_stop_scoped_ann(&RetrievalConfig::default(), &s));
    }

    #[test]
    fn should_stop_scoped_ann_no_convergence() {
        let s = make_ann_state(0, 50, 100, 20, 10, Some(0.8), Some(10), Some(0.7));
        assert!(!should_stop_scoped_ann(&RetrievalConfig::default(), &s));
    }

    #[test]
    fn elapsed_ms_and_us_returns_positive_values() {
        let start = Instant::now();
        let (ms, us) = elapsed_ms_and_us(start);
        assert!(us >= ms * 1000);
    }
}
