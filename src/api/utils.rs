use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use std::collections::{HashMap, HashSet};
use std::env;
use std::time::Instant;

use crate::storage::MemoryKind;

pub const RESET_CONFIRM_PHRASE: &str = "delete-all-data";
pub const SEMANTIC_TOP_DEFAULT: usize = 100;
pub const SEMANTIC_TOP_SCOPED_DEFAULT: usize = 3000;
pub const SEMANTIC_TOP_SCOPED_START_DEFAULT: usize = 512;
pub const SEMANTIC_TOP_SCOPED_STEP_DEFAULT: usize = 512;
pub const SEMANTIC_TOP_SCOPED_MIN_HITS_DEFAULT: usize = 24;

pub const SCOPED_ANN_STOP_MAX_ATTEMPTS_DEFAULT: usize = 10;
pub const SCOPED_ANN_STOP_MIN_SIMILARITY_DEFAULT: f32 = 0.70;
pub const SCOPED_ANN_STOP_MAX_HIT_GAIN_DEFAULT: usize = 2;
pub const SCOPED_ANN_STOP_MIN_SIMILARITY_GAIN_DEFAULT: f32 = 0.01;

pub const MILLIS_PER_DAY: u64 = 86_400_000;
pub const MIN_YEAR: i32 = 1990;
pub const MAX_YEAR: i32 = 2100;
pub const MAX_DAY_OF_MONTH: u32 = 31;
pub const FIRST_WEEK_END_DAY: u32 = 7;
pub const YEAR_DIGITS: usize = 4;
pub const MIN_TOKENS_FOR_HARD_QUERY: usize = 6;
pub const SHORT_QUERY_TOKEN_MAX: usize = 2;
pub const SIMILARITY_CONVERGENCE_STRICT: f32 = 0.05;
pub const SIMILARITY_CONVERGENCE_LOOSE: f32 = 0.10;
pub const MIN_CONVERGENCE_SAMPLES: usize = 2;
pub const FIFTH_RANK_INDEX: usize = 4;
pub const MIN_SALIENT_TOKEN_LEN: usize = 3;
pub const MIN_PHRASE_LEN: usize = 2;
pub const DECAY_FLOOR: f32 = 0.35;
pub const LAST_WEEK_DAY_OFFSET: u32 = 6;

pub fn env_bool(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

pub fn temporal_recency_scoring_enabled() -> bool {
    env_bool("TEMPORAL_MEMORY_ENABLE_TEMPORAL_RECENCY_SCORING", false)
}

pub fn scoped_semantic_top() -> usize {
    env::var("TEMPORAL_MEMORY_SCOPED_SEMANTIC_TOP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v >= SEMANTIC_TOP_DEFAULT)
        .unwrap_or(SEMANTIC_TOP_SCOPED_DEFAULT)
}

pub fn scoped_semantic_start(max_top: usize) -> usize {
    env::var("TEMPORAL_MEMORY_SCOPED_SEMANTIC_START")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v >= SEMANTIC_TOP_DEFAULT)
        .unwrap_or(SEMANTIC_TOP_SCOPED_START_DEFAULT)
        .min(max_top)
}

pub fn scoped_semantic_step() -> usize {
    env::var("TEMPORAL_MEMORY_SCOPED_SEMANTIC_STEP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(SEMANTIC_TOP_SCOPED_STEP_DEFAULT)
}

pub fn scoped_semantic_min_hits(limit: usize, max_top: usize) -> usize {
    env::var("TEMPORAL_MEMORY_SCOPED_MIN_HITS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or_else(|| limit.saturating_mul(2).max(SEMANTIC_TOP_SCOPED_MIN_HITS_DEFAULT))
        .min(max_top)
}

pub fn scoped_ann_stop_max_attempts() -> usize {
    env::var("TEMPORAL_MEMORY_SCOPED_STOP_MAX_ATTEMPTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(SCOPED_ANN_STOP_MAX_ATTEMPTS_DEFAULT)
}

pub fn scoped_ann_stop_min_similarity() -> f32 {
    env::var("TEMPORAL_MEMORY_SCOPED_STOP_MIN_SIM")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= -1.0 && *v <= 1.0)
        .unwrap_or(SCOPED_ANN_STOP_MIN_SIMILARITY_DEFAULT)
}

pub fn scoped_ann_stop_max_hit_gain() -> usize {
    env::var("TEMPORAL_MEMORY_SCOPED_STOP_MAX_HIT_GAIN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(SCOPED_ANN_STOP_MAX_HIT_GAIN_DEFAULT)
}

pub fn scoped_ann_stop_min_similarity_gain() -> f32 {
    env::var("TEMPORAL_MEMORY_SCOPED_STOP_MIN_SIM_GAIN")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(SCOPED_ANN_STOP_MIN_SIMILARITY_GAIN_DEFAULT)
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

pub fn should_stop_scoped_ann(state: &ScopedAnnState) -> bool {
    if state.current_top >= state.max_top {
        return true;
    }

    let max_attempts = scoped_ann_stop_max_attempts();
    if state.attempt >= max_attempts {
        return true;
    }

    if state.hit_count < state.min_hits {
        return false;
    }

    let strong_enough = state
        .top_similarity
        .map(|sim| sim >= scoped_ann_stop_min_similarity())
        .unwrap_or(false);
    if !strong_enough {
        return false;
    }

    let Some(prev_hits) = state.prev_hit_count else {
        return false;
    };
    let hit_gain = state.hit_count.saturating_sub(prev_hits);
    let low_hit_gain = hit_gain <= scoped_ann_stop_max_hit_gain();

    let low_similarity_gain = match (state.top_similarity, state.prev_top_similarity) {
        (Some(current), Some(prev)) => {
            (current - prev).abs() <= scoped_ann_stop_min_similarity_gain()
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

pub fn parse_kind(s: Option<&str>) -> MemoryKind {
    match s.map(|s| s.to_ascii_lowercase()).as_deref() {
        Some("decision") => MemoryKind::Decision,
        Some("lesson") => MemoryKind::Lesson,
        Some("preference") => MemoryKind::Preference,
        Some("session_summary" | "session-summary") => MemoryKind::SessionSummary,
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

pub fn decay_policy(kind: MemoryKind) -> (f32, f32, Option<f32>) {
    match kind {
        MemoryKind::Conversational => (30.0, DECAY_FLOOR, Some(90.0)),
        MemoryKind::Lesson => (90.0, DECAY_FLOOR, Some(365.0)),
        MemoryKind::Fact => (180.0, DECAY_FLOOR, Some(730.0)),
        MemoryKind::SessionSummary => (14.0, DECAY_FLOOR, Some(60.0)),
        MemoryKind::Decision | MemoryKind::Preference => (365.0, DECAY_FLOOR, None),
    }
}

pub fn apply_decay_with_policy(
    base_score: f32,
    created_at_ms: u64,
    kind: MemoryKind,
    now_ms: u64,
) -> Option<f32> {
    if kind.is_decay_exempt() {
        return Some(base_score);
    }

    let age_days = (now_ms.saturating_sub(created_at_ms)) as f32 / MILLIS_PER_DAY as f32;
    let (half_life_days, floor, ttl_days) = decay_policy(kind);

    if let Some(ttl) = ttl_days {
        if age_days > ttl {
            return None;
        }
    }

    Some(apply_time_decay(base_score, age_days, half_life_days, floor))
}

pub fn session_id_from_memory_id(memory_id: &str) -> Option<String> {
    let mut parts = memory_id.split("::");
    let _entity = parts.next()?;
    let session = parts.next()?;
    Some(session.to_string())
}

pub fn turn_index_from_memory_id(memory_id: &str) -> usize {
    memory_id.rsplit("::").next().and_then(|part| part.parse::<usize>().ok()).unwrap_or(0)
}

pub fn split_memory_id(memory_id: &str) -> Option<(&str, &str, usize)> {
    let mut parts = memory_id.split("::");
    let entity_id = parts.next()?;
    let session_id = parts.next()?;
    let turn_index = parts.next()?.parse::<usize>().ok()?;
    Some((entity_id, session_id, turn_index))
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

pub fn extract_temporal_terms(query: &str) -> Vec<String> {
    let lower = query.to_ascii_lowercase();
    let mut terms = Vec::with_capacity(4);

    for token in normalize_alpha_tokens(query) {
        if token.len() == YEAR_DIGITS && token.chars().all(|c| c.is_ascii_digit()) {
            terms.push(token);
        }
    }

    let months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "spring",
        "summer",
        "fall",
        "autumn",
        "winter",
        "weekend",
        "week",
        "month",
        "year",
        "yesterday",
        "today",
        "tomorrow",
        "recently",
        "latest",
        "last",
        "recent",
    ];
    for term in &months {
        if lower.contains(term) {
            terms.push(term.to_string());
        }
    }

    dedupe_preserve_order(terms)
}

/// Parse a (start_ms, end_ms) temporal window from natural language query text.
/// Returns `None` if no specific temporal window can be extracted.
/// Examples that parse: "October 2023", "May 1 2022", "last week of October 2023",
/// "March 2023", "summer 2022".
pub fn parse_temporal_window(query: &str) -> Option<(u64, u64)> {
    use std::collections::HashMap;

    let month_map: HashMap<&str, u32> = [
        ("january", 1),
        ("jan", 1),
        ("february", 2),
        ("feb", 2),
        ("march", 3),
        ("mar", 3),
        ("april", 4),
        ("apr", 4),
        ("may", 5),
        ("june", 6),
        ("jun", 6),
        ("july", 7),
        ("jul", 7),
        ("august", 8),
        ("aug", 8),
        ("september", 9),
        ("sep", 9),
        ("sept", 9),
        ("october", 10),
        ("oct", 10),
        ("november", 11),
        ("nov", 11),
        ("december", 12),
        ("dec", 12),
    ]
    .into_iter()
    .collect();

    let season_map: HashMap<&str, (u32, u32)> = [
        ("spring", (3, 5)),
        ("summer", (6, 8)),
        ("fall", (9, 11)),
        ("autumn", (9, 11)),
        ("winter", (12, 2)),
    ]
    .into_iter()
    .collect();

    let lower = query.to_ascii_lowercase();
    let tokens: Vec<&str> = lower.split_whitespace().collect();

    // Extract year (4-digit)
    let year: Option<i32> = tokens.iter().find_map(|t| {
        let digits: String = t.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() == 4 {
            digits.parse().ok().filter(|&y: &i32| (MIN_YEAR..=MAX_YEAR).contains(&y))
        } else {
            None
        }
    });

    let year = year?; // No year → can't build a reliable window

    // Check for season first
    for (season, (start_month, end_month)) in &season_map {
        if lower.contains(season) {
            let start_ms = month_to_ms(year, *start_month, 1);
            let end_ms = if *end_month < *start_month {
                // winter wraps: Dec-Feb
                month_to_ms(year + 1, *end_month, days_in_month(year + 1, *end_month))
            } else {
                month_to_ms(year, *end_month, days_in_month(year, *end_month))
            };
            return Some((start_ms, end_ms + MILLIS_PER_DAY));
        }
    }

    // Look for a month name
    let mut found_month: Option<u32> = None;
    for (name, month_num) in &month_map {
        if lower.contains(name) {
            found_month = Some(*month_num);
            break;
        }
    }

    let month = found_month?;

    // Check for "last week of <month> <year>"
    let is_last_week = lower.contains("last week");
    // Check for "first week of <month> <year>"
    let is_first_week = lower.contains("first week");
    // Check for a specific day number
    let day: Option<u32> = tokens.iter().find_map(|t| {
        let digits: String = t.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() <= 2 {
            digits.parse::<u32>().ok().filter(|&d| (1..=MAX_DAY_OF_MONTH).contains(&d))
        } else {
            None
        }
    });

    let dom = days_in_month(year, month);

    let (start_ms, end_ms) = if is_last_week {
        let last_day = dom;
        let first_day = last_day.saturating_sub(LAST_WEEK_DAY_OFFSET).max(1);
        (month_to_ms(year, month, first_day), month_to_ms(year, month, last_day) + MILLIS_PER_DAY)
    } else if is_first_week {
        (month_to_ms(year, month, 1), month_to_ms(year, month, FIRST_WEEK_END_DAY) + MILLIS_PER_DAY)
    } else if let Some(day) = day {
        let d = day.min(dom);
        (month_to_ms(year, month, d), month_to_ms(year, month, d) + MILLIS_PER_DAY)
    } else {
        // Whole month
        (month_to_ms(year, month, 1), month_to_ms(year, month, dom) + MILLIS_PER_DAY)
    };

    Some((start_ms, end_ms))
}

fn month_to_ms(year: i32, month: u32, day: u32) -> u64 {
    // Days since Unix epoch (1970-01-01)
    let days = days_since_epoch(year, month, day);
    days as u64 * MILLIS_PER_DAY
}

fn days_since_epoch(year: i32, month: u32, day: u32) -> i64 {
    // Compute Julian Day Number and subtract epoch JDN
    let a = (14 - month as i64) / 12;
    let y = year as i64 + 4800 - a;
    let m = month as i64 + 12 * a - 3;
    let jdn = day as i64 + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
    const UNIX_EPOCH_JDN: i64 = 2_440_588;
    jdn - UNIX_EPOCH_JDN
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
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

pub fn ok_or_500<T, E: std::fmt::Debug>(r: Result<T, E>) -> Result<T, StatusCode> {
    r.map_err(|e| {
        tracing::warn!("Internal error: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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
        assert_eq!(parse_kind(Some("DECISION")), MemoryKind::Conversational);
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
        let (half_life, floor, ttl) = decay_policy(MemoryKind::Conversational);
        assert!((half_life - 30.0).abs() < f32::EPSILON);
        assert!((floor - 0.35).abs() < f32::EPSILON);
        assert_eq!(ttl, Some(90.0));
    }

    #[test]
    fn decay_policy_lesson() {
        let (half_life, floor, ttl) = decay_policy(MemoryKind::Lesson);
        assert!((half_life - 90.0).abs() < f32::EPSILON);
        assert!((floor - 0.35).abs() < f32::EPSILON);
        assert_eq!(ttl, Some(365.0));
    }

    #[test]
    fn decay_policy_fact() {
        let (half_life, floor, ttl) = decay_policy(MemoryKind::Fact);
        assert!((half_life - 180.0).abs() < f32::EPSILON);
        assert!((floor - 0.35).abs() < f32::EPSILON);
        assert_eq!(ttl, Some(730.0));
    }

    #[test]
    fn decay_policy_session_summary() {
        let (half_life, floor, ttl) = decay_policy(MemoryKind::SessionSummary);
        assert!((half_life - 14.0).abs() < f32::EPSILON);
        assert!((floor - 0.35).abs() < f32::EPSILON);
        assert_eq!(ttl, Some(60.0));
    }

    #[test]
    fn decay_policy_decision() {
        let (half_life, floor, ttl) = decay_policy(MemoryKind::Decision);
        assert!((half_life - 365.0).abs() < f32::EPSILON);
        assert!((floor - 0.35).abs() < f32::EPSILON);
        assert_eq!(ttl, None);
    }

    #[test]
    fn decay_policy_preference() {
        let (half_life, floor, ttl) = decay_policy(MemoryKind::Preference);
        assert!((half_life - 365.0).abs() < f32::EPSILON);
        assert!((floor - 0.35).abs() < f32::EPSILON);
        assert_eq!(ttl, None);
    }

    #[test]
    fn apply_decay_with_policy_decay_exempt_returns_some() {
        let result = apply_decay_with_policy(0.75, 0, MemoryKind::Decision, u64::MAX);
        assert_eq!(result, Some(0.75));

        let result = apply_decay_with_policy(0.75, 0, MemoryKind::Preference, u64::MAX);
        assert_eq!(result, Some(0.75));
    }

    #[test]
    fn apply_decay_with_policy_expired_ttl_returns_none() {
        let created_ms = 0;
        let now_ms = (100 * 86_400) * 1000;
        let result = apply_decay_with_policy(1.0, created_ms, MemoryKind::Conversational, now_ms);
        assert_eq!(result, None);
    }

    #[test]
    fn apply_decay_with_policy_within_ttl_returns_decayed() {
        let created_ms = 0;
        let now_ms = (30 * 86_400) * 1000;
        let result = apply_decay_with_policy(1.0, created_ms, MemoryKind::Conversational, now_ms);
        assert!(result.is_some());
        let score = result.unwrap();
        assert!(score > 0.35 && score < 1.0);
    }

    #[test]
    fn session_id_from_memory_id_valid() {
        assert_eq!(
            session_id_from_memory_id("entity::session123::42"),
            Some("session123".to_string())
        );
    }

    #[test]
    fn session_id_from_memory_id_no_turn() {
        assert_eq!(session_id_from_memory_id("entity::session456"), Some("session456".to_string()));
    }

    #[test]
    fn session_id_from_memory_id_invalid_returns_none() {
        assert_eq!(session_id_from_memory_id("only_one_part"), None);
    }

    #[test]
    fn session_id_from_memory_id_empty_returns_none() {
        assert_eq!(session_id_from_memory_id(""), None);
    }

    #[test]
    fn turn_index_from_memory_id_valid() {
        assert_eq!(turn_index_from_memory_id("entity::session::42"), 42);
    }

    #[test]
    fn turn_index_from_memory_id_missing_returns_zero() {
        assert_eq!(turn_index_from_memory_id("entity::session"), 0);
    }

    #[test]
    fn turn_index_from_memory_id_non_numeric_returns_zero() {
        assert_eq!(turn_index_from_memory_id("entity::session::abc"), 0);
    }

    #[test]
    fn turn_index_from_memory_id_empty_returns_zero() {
        assert_eq!(turn_index_from_memory_id(""), 0);
    }

    #[test]
    fn split_memory_id_valid() {
        assert_eq!(split_memory_id("entity1::session1::7"), Some(("entity1", "session1", 7)));
    }

    #[test]
    fn split_memory_id_missing_turn_returns_none() {
        assert_eq!(split_memory_id("entity1::session1"), None);
    }

    #[test]
    fn split_memory_id_single_part_returns_none() {
        assert_eq!(split_memory_id("entity1"), None);
    }

    #[test]
    fn split_memory_id_empty_returns_none() {
        assert_eq!(split_memory_id(""), None);
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
        let result = parse_temporal_window("October 2023").unwrap();
        let expected = (month_to_ms(2023, 10, 1), month_to_ms(2023, 10, 31) + 86_400_000);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_season_year() {
        let result = parse_temporal_window("summer 2022").unwrap();
        let expected = (month_to_ms(2022, 6, 1), month_to_ms(2022, 8, 31) + 86_400_000);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_winter_wraps_year() {
        let result = parse_temporal_window("winter 2024").unwrap();
        let expected =
            (month_to_ms(2024, 12, 1), month_to_ms(2025, 2, days_in_month(2025, 2)) + 86_400_000);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_last_week_of_month() {
        let result = parse_temporal_window("last week of October 2023").unwrap();
        let expected = (month_to_ms(2023, 10, 25), month_to_ms(2023, 10, 31) + 86_400_000);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_specific_day() {
        let result = parse_temporal_window("May 1 2022").unwrap();
        let expected = (month_to_ms(2022, 5, 1), month_to_ms(2022, 5, 1) + 86_400_000);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_no_year_returns_none() {
        assert_eq!(parse_temporal_window("no year here"), None);
        assert_eq!(parse_temporal_window("January"), None);
    }

    #[test]
    fn parse_temporal_window_first_week_of_month() {
        let result = parse_temporal_window("first week of March 2023").unwrap();
        let expected = (month_to_ms(2023, 3, 1), month_to_ms(2023, 3, 7) + 86_400_000);
        assert_eq!(result, expected);
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
    fn cosine_similarity_identical_vectors() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < f32::EPSILON);
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

    fn make_ann_state(
        attempt: usize, current_top: usize, max_top: usize,
        hit_count: usize, min_hits: usize,
        top_similarity: Option<f32>, prev_hit_count: Option<usize>, prev_top_similarity: Option<f32>,
    ) -> ScopedAnnState {
        ScopedAnnState { attempt, current_top, max_top, hit_count, min_hits, top_similarity, prev_hit_count, prev_top_similarity }
    }

    #[test]
    fn should_stop_scoped_ann_max_top_reached() {
        let s = make_ann_state(0, 100, 100, 0, 1, None, None, None);
        assert!(should_stop_scoped_ann(&s));
    }

    #[test]
    fn should_stop_scoped_ann_max_attempts_reached() {
        let s = make_ann_state(10, 50, 100, 50, 10, Some(0.9), Some(40), Some(0.8));
        assert!(should_stop_scoped_ann(&s));
    }

    #[test]
    fn should_stop_scoped_ann_not_enough_hits() {
        let s = make_ann_state(0, 50, 100, 5, 10, None, None, None);
        assert!(!should_stop_scoped_ann(&s));
    }

    #[test]
    fn should_stop_scoped_ann_not_strong_enough_similarity() {
        let s = make_ann_state(0, 50, 100, 20, 10, Some(0.6), Some(10), Some(0.5));
        assert!(!should_stop_scoped_ann(&s));
    }

    #[test]
    fn should_stop_scoped_ann_none_similarity_not_strong() {
        let s = make_ann_state(0, 50, 100, 20, 10, None, Some(10), Some(0.5));
        assert!(!should_stop_scoped_ann(&s));
    }

    #[test]
    fn should_stop_scoped_ann_no_prev_hit_count_returns_false() {
        let s = make_ann_state(0, 50, 100, 20, 10, Some(0.8), None, Some(0.79));
        assert!(!should_stop_scoped_ann(&s));
    }

    #[test]
    fn should_stop_scoped_ann_convergence_low_hit_gain() {
        let s = make_ann_state(0, 50, 100, 12, 10, Some(0.8), Some(10), Some(0.7));
        assert!(should_stop_scoped_ann(&s));
    }

    #[test]
    fn should_stop_scoped_ann_convergence_low_similarity_gain() {
        let s = make_ann_state(0, 50, 100, 20, 10, Some(0.71), Some(10), Some(0.70));
        assert!(should_stop_scoped_ann(&s));
    }

    #[test]
    fn should_stop_scoped_ann_no_convergence() {
        let s = make_ann_state(0, 50, 100, 20, 10, Some(0.8), Some(10), Some(0.7));
        assert!(!should_stop_scoped_ann(&s));
    }

    #[test]
    fn elapsed_ms_and_us_returns_positive_values() {
        let start = Instant::now();
        let (ms, us) = elapsed_ms_and_us(start);
        assert!(us >= ms * 1000);
    }
}
