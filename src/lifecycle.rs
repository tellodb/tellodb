use serde::{Deserialize, Serialize};

use crate::storage::MemoryKind;

pub const LIFECYCLE_POLICY_VERSION: &str = "lifecycle-v1-deterministic";

// ---------------------------------------------------------------------------
// Admission-score weights
// ---------------------------------------------------------------------------
const ADMISSION_WEIGHT_SALIENCE: f32 = 0.22;
const ADMISSION_WEIGHT_NOVELTY: f32 = 0.12;
const ADMISSION_WEIGHT_CONFIDENCE: f32 = 0.18;
const ADMISSION_WEIGHT_SPECIFICITY: f32 = 0.16;
const ADMISSION_WEIGHT_TEMPORAL: f32 = 0.10;
const ADMISSION_WEIGHT_UTILITY: f32 = 0.14;
const ADMISSION_WEIGHT_STABILITY: f32 = 0.08;

const SENSITIVITY_RISK_PUBLIC: f32 = 0.00;
const SENSITIVITY_RISK_PERSONAL: f32 = 0.04;
const SENSITIVITY_RISK_SENSITIVE: f32 = 0.10;
const SENSITIVITY_RISK_RESTRICTED: f32 = 0.18;

const INFERENCE_PENALTY: f32 = 0.10;

const NOVELTY_TOKEN_THRESHOLD: usize = 8;
const NOVELTY_SCORE_LONG: f32 = 0.62;
const NOVELTY_SCORE_SHORT: f32 = 0.35;

const ADMISSION_SCORE_INDEXED: f32 = 0.50;
const ADMISSION_SCORE_ADMITTED: f32 = 0.30;

const PROMOTE_TO_PROFILE_MIN_STABILITY: f32 = 0.55;
const PROMOTE_TO_PROFILE_MIN_CONFIDENCE: f32 = 0.70;

const INDEX_VECTOR_MIN_ADMISSION: f32 = 0.34;

// ---------------------------------------------------------------------------
// Salience scoring
// ---------------------------------------------------------------------------
const SALIENCE_DECISION: f32 = 0.88;
const SALIENCE_PREFERENCE: f32 = 0.78;
const SALIENCE_FACT: f32 = 0.72;
const SALIENCE_LESSON: f32 = 0.70;
const SALIENCE_SESSION_SUMMARY: f32 = 0.56;
const SALIENCE_CONVERSATIONAL: f32 = 0.42;
const SALIENCE_KEYWORD_BOOST: f32 = 0.035;
const SALIENCE_KEYWORD_BOOST_MAX: f32 = 0.18;

// ---------------------------------------------------------------------------
// Specificity scoring
// ---------------------------------------------------------------------------
const SPECIFICITY_BASE: f32 = 0.28;
const SPECIFICITY_NUMERIC_BOOST: f32 = 0.18;
const SPECIFICITY_NAMED_ENTITY_BOOST: f32 = 0.22;
const SPECIFICITY_WORD_BOOST: f32 = 0.05;
const SPECIFICITY_WORD_BOOST_MAX: f32 = 0.20;

// ---------------------------------------------------------------------------
// Temporal scoring
// ---------------------------------------------------------------------------
const TEMPORAL_SCORE_WITH_SIGNAL: f32 = 0.82;
const TEMPORAL_SCORE_NO_SIGNAL: f32 = 0.20;

// ---------------------------------------------------------------------------
// Confidence scoring
// ---------------------------------------------------------------------------
const CONFIDENCE_INFERENCE: f32 = 0.52;
const CONFIDENCE_HIGH: f32 = 0.88;
const CONFIDENCE_LESSON: f32 = 0.72;
const CONFIDENCE_SESSION_SUMMARY: f32 = 0.68;
const CONFIDENCE_CONVERSATIONAL: f32 = 0.64;

// ---------------------------------------------------------------------------
// Stability scoring
// ---------------------------------------------------------------------------
const STABILITY_INFERENCE: f32 = 0.32;
const STABILITY_TEMPORARY: f32 = 0.34;
const STABILITY_PERMANENT: f32 = 0.78;
const STABILITY_LESSON: f32 = 0.62;
const STABILITY_SESSION_SUMMARY: f32 = 0.50;
const STABILITY_CONVERSATIONAL: f32 = 0.44;

// ---------------------------------------------------------------------------
// Retention-class thresholds
// ---------------------------------------------------------------------------
const RETENTION_INFERENCE_STABILITY_MAX: f32 = 0.45;
const RETENTION_SALIENCE_MIN: f32 = 0.75;
const RETENTION_STABILITY_MIN: f32 = 0.72;

// ---------------------------------------------------------------------------
// Utility scoring
// ---------------------------------------------------------------------------
const UTILITY_HIGH: f32 = 0.72;
const UTILITY_FACT: f32 = 0.66;
const UTILITY_SESSION_SUMMARY: f32 = 0.48;
const UTILITY_CONVERSATIONAL: f32 = 0.38;
const UTILITY_KEYWORD_BOOST: f32 = 0.04;
const UTILITY_KEYWORD_BOOST_MAX: f32 = 0.18;

// ---------------------------------------------------------------------------
// Retention durations (in days)
// ---------------------------------------------------------------------------
const EPHEMERAL_INFERENCE_DAYS: u64 = 30;
const EPHEMERAL_DIRECT_DAYS: u64 = 14;
const WORKING_DAYS: u64 = 90;
const EPISODIC_DAYS: u64 = 730;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum SensitivityClass {
    Public,
    Personal,
    Sensitive,
    Restricted,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetentionClass {
    Ephemeral,
    Working,
    Episodic,
    LongTerm,
    Archive,
    ComplianceSensitive,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum LifecycleState {
    Observed,
    Admitted,
    Indexed,
    Expired,
    Invalidated,
    Tombstoned,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LifecycleMetadata {
    pub policy_version: String,
    pub lifecycle_state: LifecycleState,
    pub admission_score: f32,
    pub salience_score: f32,
    pub novelty_score: f32,
    pub confidence_score: f32,
    pub stability_score: f32,
    pub specificity_score: f32,
    pub temporal_score: f32,
    pub utility_score: f32,
    pub sensitivity_class: SensitivityClass,
    pub retention_class: RetentionClass,
    pub index_fts: bool,
    pub index_vector: bool,
    pub promote_to_profile: bool,
    pub is_inference: bool,
    pub expires_at_ms: Option<u64>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArtifactVersionRecord {
    pub artifact_id: String,
    pub version_id: String,
    pub operation: String,
    pub previous_version_id: Option<String>,
    pub compiler_version: String,
    pub reason: String,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeletionTombstone {
    pub tombstone_id: String,
    pub scope: String,
    pub target_id: String,
    pub deleted_at_ms: u64,
    pub reason: String,
    pub cascade_count: usize,
    pub proof_hash: String,
}

pub fn evaluate_lifecycle(
    text: &str,
    kind: MemoryKind,
    timestamp_ms: u64,
    confidence_hint: Option<f32>,
    is_inference: bool,
) -> LifecycleMetadata {
    let lower = text.to_ascii_lowercase();
    let token_count = lower.split_whitespace().count();
    let mut reasons = Vec::new();

    let salience_score = salience_score(&lower, kind, &mut reasons);
    let specificity_score = specificity_score(text, &lower, &mut reasons);
    let temporal_score = temporal_score(&lower, &mut reasons);
    let sensitivity_class = sensitivity_class(&lower, &mut reasons);
    let confidence_score =
        confidence_hint.unwrap_or_else(|| confidence_for_kind(kind, is_inference)).clamp(0.0, 1.0);
    let stability_score = stability_score(&lower, kind, is_inference, &mut reasons);
    let novelty_score = if token_count >= NOVELTY_TOKEN_THRESHOLD {
        NOVELTY_SCORE_LONG
    } else {
        NOVELTY_SCORE_SHORT
    };
    let utility_score = utility_score(&lower, kind, &mut reasons);
    let sensitivity_risk = match sensitivity_class {
        SensitivityClass::Public => SENSITIVITY_RISK_PUBLIC,
        SensitivityClass::Personal => SENSITIVITY_RISK_PERSONAL,
        SensitivityClass::Sensitive => SENSITIVITY_RISK_SENSITIVE,
        SensitivityClass::Restricted => SENSITIVITY_RISK_RESTRICTED,
    };
    let inference_penalty = if is_inference { INFERENCE_PENALTY } else { 0.0 };
    let admission_score = (ADMISSION_WEIGHT_SALIENCE * salience_score
        + ADMISSION_WEIGHT_NOVELTY * novelty_score
        + ADMISSION_WEIGHT_CONFIDENCE * confidence_score
        + ADMISSION_WEIGHT_SPECIFICITY * specificity_score
        + ADMISSION_WEIGHT_TEMPORAL * temporal_score
        + ADMISSION_WEIGHT_UTILITY * utility_score
        + ADMISSION_WEIGHT_STABILITY * stability_score
        - sensitivity_risk
        - inference_penalty)
        .clamp(0.0, 1.0);

    let retention_class =
        retention_class(kind, sensitivity_class, salience_score, stability_score, is_inference);
    let expires_at_ms = expires_at_for(retention_class, timestamp_ms, is_inference);
    let promote_to_profile =
        matches!(kind, MemoryKind::Preference | MemoryKind::Decision | MemoryKind::Fact)
            && stability_score >= PROMOTE_TO_PROFILE_MIN_STABILITY
            && confidence_score >= PROMOTE_TO_PROFILE_MIN_CONFIDENCE
            && !matches!(sensitivity_class, SensitivityClass::Restricted);
    let index_vector = admission_score >= INDEX_VECTOR_MIN_ADMISSION
        && !matches!(retention_class, RetentionClass::Ephemeral);
    let lifecycle_state = if admission_score >= ADMISSION_SCORE_INDEXED {
        LifecycleState::Indexed
    } else if admission_score >= ADMISSION_SCORE_ADMITTED {
        LifecycleState::Admitted
    } else {
        LifecycleState::Observed
    };

    LifecycleMetadata {
        policy_version: LIFECYCLE_POLICY_VERSION.to_string(),
        lifecycle_state,
        admission_score,
        salience_score,
        novelty_score,
        confidence_score,
        stability_score,
        specificity_score,
        temporal_score,
        utility_score,
        sensitivity_class,
        retention_class,
        index_fts: true,
        index_vector,
        promote_to_profile,
        is_inference,
        expires_at_ms,
        reasons,
    }
}

fn salience_score(lower: &str, kind: MemoryKind, reasons: &mut Vec<String>) -> f32 {
    let mut score: f32 = match kind {
        MemoryKind::Decision => SALIENCE_DECISION,
        MemoryKind::Preference => SALIENCE_PREFERENCE,
        MemoryKind::Fact => SALIENCE_FACT,
        MemoryKind::Lesson => SALIENCE_LESSON,
        MemoryKind::SessionSummary => SALIENCE_SESSION_SUMMARY,
        MemoryKind::Conversational => SALIENCE_CONVERSATIONAL,
    };
    let salient = [
        "adopted",
        "married",
        "moved",
        "graduated",
        "started",
        "joined",
        "won",
        "lost",
        "diagnosed",
        "allergy",
        "health",
        "family",
        "children",
        "dog",
        "pet",
        "job",
        "career",
        "degree",
        "favorite",
        "prefers",
        "decided",
        "plan",
        "goal",
        "trip",
        "visited",
    ];
    let hits = salient.iter().filter(|needle| lower.contains(**needle)).count();
    if hits > 0 {
        reasons.push("salient_keyword".to_string());
        score += (hits as f32 * SALIENCE_KEYWORD_BOOST).min(SALIENCE_KEYWORD_BOOST_MAX);
    }
    score.clamp(0.0, 1.0)
}

fn specificity_score(text: &str, lower: &str, reasons: &mut Vec<String>) -> f32 {
    let mut score: f32 = SPECIFICITY_BASE;
    if text.chars().any(|c| c.is_ascii_digit()) {
        score += SPECIFICITY_NUMERIC_BOOST;
        reasons.push("numeric_or_date_signal".to_string());
    }
    let has_named_entity = text
        .split_whitespace()
        .any(|w| w.chars().next().is_some_and(|c| c.is_uppercase()) && w.len() > 2);
    if has_named_entity {
        score += SPECIFICITY_NAMED_ENTITY_BOOST;
        reasons.push("named_entity_signal".to_string());
    }
    let specific =
        ["because", "when", "where", "with", "at", "on", "in", "named", "called", "from"];
    let hits = specific.iter().filter(|needle| lower.contains(&format!(" {needle} "))).count();
    score += (hits as f32 * SPECIFICITY_WORD_BOOST).min(SPECIFICITY_WORD_BOOST_MAX);
    score.clamp(0.0, 1.0)
}

fn temporal_score(lower: &str, reasons: &mut Vec<String>) -> f32 {
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
        "yesterday",
        "today",
        "tomorrow",
        "last ",
        "next ",
        "before",
        "after",
        "recently",
        "year",
        "month",
        "week",
    ];
    if months.iter().any(|needle| lower.contains(needle)) {
        reasons.push("temporal_signal".to_string());
        TEMPORAL_SCORE_WITH_SIGNAL
    } else {
        TEMPORAL_SCORE_NO_SIGNAL
    }
}

fn confidence_for_kind(kind: MemoryKind, is_inference: bool) -> f32 {
    if is_inference {
        return CONFIDENCE_INFERENCE;
    }
    match kind {
        MemoryKind::Fact | MemoryKind::Preference | MemoryKind::Decision => CONFIDENCE_HIGH,
        MemoryKind::Lesson => CONFIDENCE_LESSON,
        MemoryKind::SessionSummary => CONFIDENCE_SESSION_SUMMARY,
        MemoryKind::Conversational => CONFIDENCE_CONVERSATIONAL,
    }
}

fn stability_score(
    lower: &str,
    kind: MemoryKind,
    is_inference: bool,
    reasons: &mut Vec<String>,
) -> f32 {
    if is_inference {
        reasons.push("inference_lower_stability".to_string());
        return STABILITY_INFERENCE;
    }
    let temporary = [
        "today",
        "currently",
        "right now",
        "lately",
        "recently",
        "this week",
        "this month",
        "feeling",
        "mood",
    ];
    if temporary.iter().any(|needle| lower.contains(needle)) {
        reasons.push("temporary_state".to_string());
        return STABILITY_TEMPORARY;
    }
    match kind {
        MemoryKind::Preference | MemoryKind::Decision | MemoryKind::Fact => STABILITY_PERMANENT,
        MemoryKind::Lesson => STABILITY_LESSON,
        MemoryKind::SessionSummary => STABILITY_SESSION_SUMMARY,
        MemoryKind::Conversational => STABILITY_CONVERSATIONAL,
    }
}

fn utility_score(lower: &str, kind: MemoryKind, reasons: &mut Vec<String>) -> f32 {
    let useful = [
        "favorite",
        "prefers",
        "doesn't like",
        "allergy",
        "health",
        "birthday",
        "family",
        "job",
        "career",
        "goal",
        "plan",
        "remember",
        "important",
        "decided",
        "lesson",
    ];
    let hits = useful.iter().filter(|needle| lower.contains(**needle)).count();
    if hits > 0 {
        reasons.push("future_utility_signal".to_string());
    }
    (match kind {
        MemoryKind::Preference | MemoryKind::Decision | MemoryKind::Lesson => UTILITY_HIGH,
        MemoryKind::Fact => UTILITY_FACT,
        MemoryKind::SessionSummary => UTILITY_SESSION_SUMMARY,
        MemoryKind::Conversational => UTILITY_CONVERSATIONAL,
    } + (hits as f32 * UTILITY_KEYWORD_BOOST).min(UTILITY_KEYWORD_BOOST_MAX))
    .clamp(0.0, 1.0)
}

fn sensitivity_class(lower: &str, reasons: &mut Vec<String>) -> SensitivityClass {
    let restricted = [
        "password",
        "secret key",
        "api key",
        "token",
        "private key",
        "ssn",
        "social security",
        "credit card",
    ];
    if restricted.iter().any(|needle| lower.contains(needle)) {
        reasons.push("restricted_sensitive_signal".to_string());
        return SensitivityClass::Restricted;
    }
    let sensitive = [
        "health",
        "diagnosed",
        "allergy",
        "medical",
        "therapy",
        "salary",
        "bank",
        "financial",
        "religion",
        "political",
        "pregnant",
    ];
    if sensitive.iter().any(|needle| lower.contains(needle)) {
        reasons.push("sensitive_signal".to_string());
        return SensitivityClass::Sensitive;
    }
    let personal = ["family", "spouse", "children", "address", "lives", "birthday", "relationship"];
    if personal.iter().any(|needle| lower.contains(needle)) {
        reasons.push("personal_signal".to_string());
        return SensitivityClass::Personal;
    }
    SensitivityClass::Public
}

fn retention_class(
    kind: MemoryKind,
    sensitivity: SensitivityClass,
    salience: f32,
    stability: f32,
    is_inference: bool,
) -> RetentionClass {
    if matches!(sensitivity, SensitivityClass::Restricted) {
        return RetentionClass::ComplianceSensitive;
    }
    if is_inference && stability < RETENTION_INFERENCE_STABILITY_MAX {
        return RetentionClass::Ephemeral;
    }
    if salience >= RETENTION_SALIENCE_MIN || stability >= RETENTION_STABILITY_MIN {
        return RetentionClass::LongTerm;
    }
    match kind {
        MemoryKind::Preference | MemoryKind::Decision | MemoryKind::Fact => {
            RetentionClass::LongTerm
        }
        MemoryKind::Lesson => RetentionClass::Archive,
        MemoryKind::SessionSummary | MemoryKind::Conversational => RetentionClass::Episodic,
    }
}

fn expires_at_for(retention: RetentionClass, timestamp_ms: u64, is_inference: bool) -> Option<u64> {
    const DAY_MS: u64 = 86_400_000;
    match retention {
        RetentionClass::Ephemeral => Some(timestamp_ms.saturating_add(if is_inference {
            EPHEMERAL_INFERENCE_DAYS * DAY_MS
        } else {
            EPHEMERAL_DIRECT_DAYS * DAY_MS
        })),
        RetentionClass::Working => Some(timestamp_ms.saturating_add(WORKING_DAYS * DAY_MS)),
        RetentionClass::Episodic => Some(timestamp_ms.saturating_add(EPISODIC_DAYS * DAY_MS)),
        RetentionClass::LongTerm
        | RetentionClass::Archive
        | RetentionClass::ComplianceSensitive => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_score_differs_by_kind() {
        let scores: Vec<f32> = [
            MemoryKind::Conversational,
            MemoryKind::Decision,
            MemoryKind::Lesson,
            MemoryKind::Preference,
            MemoryKind::SessionSummary,
            MemoryKind::Fact,
        ]
        .iter()
        .map(|k| evaluate_lifecycle("some text", *k, 1000, None, false).admission_score)
        .collect();

        for i in 1..scores.len() {
            assert!(
                (scores[i] - scores[i - 1]).abs() > f32::EPSILON,
                "kinds {} and {} produced the same score {}",
                i - 1,
                i,
                scores[i]
            );
        }
    }

    #[test]
    fn inference_flag_reduces_score() {
        let direct = evaluate_lifecycle("hello world", MemoryKind::Fact, 1000, None, false);
        let inferred = evaluate_lifecycle("hello world", MemoryKind::Fact, 1000, None, true);
        assert!(direct.admission_score > inferred.admission_score);
    }

    #[test]
    fn salient_keywords_boost_salience_score() {
        let base = evaluate_lifecycle("hello world", MemoryKind::Fact, 1000, None, false);
        let boosted =
            evaluate_lifecycle("married moved graduated won", MemoryKind::Fact, 1000, None, false);
        assert!(boosted.salience_score > base.salience_score);
        assert!(boosted.reasons.contains(&"salient_keyword".to_string()));
    }

    #[test]
    fn temporal_signal_increases_temporal_score() {
        let no_signal = evaluate_lifecycle("hello world", MemoryKind::Fact, 1000, None, false);
        let with_signal =
            evaluate_lifecycle("back in january", MemoryKind::Fact, 1000, None, false);
        assert!((no_signal.temporal_score - 0.20).abs() < f32::EPSILON);
        assert!((with_signal.temporal_score - 0.82).abs() < f32::EPSILON);
        assert!(with_signal.reasons.contains(&"temporal_signal".to_string()));
    }

    #[test]
    fn sensitivity_classification_restricted() {
        let meta =
            evaluate_lifecycle("my password is secret123", MemoryKind::Fact, 1000, None, false);
        assert_eq!(meta.sensitivity_class, SensitivityClass::Restricted);
        assert!(meta.reasons.contains(&"restricted_sensitive_signal".to_string()));
    }

    #[test]
    fn sensitivity_classification_sensitive() {
        let meta = evaluate_lifecycle(
            "my health diagnosis came back",
            MemoryKind::Fact,
            1000,
            None,
            false,
        );
        assert_eq!(meta.sensitivity_class, SensitivityClass::Sensitive);
        assert!(meta.reasons.contains(&"sensitive_signal".to_string()));
    }

    #[test]
    fn sensitivity_classification_personal() {
        let meta =
            evaluate_lifecycle("my family lives in Chicago", MemoryKind::Fact, 1000, None, false);
        assert_eq!(meta.sensitivity_class, SensitivityClass::Personal);
        assert!(meta.reasons.contains(&"personal_signal".to_string()));
    }

    #[test]
    fn sensitivity_classification_public() {
        let meta = evaluate_lifecycle("the sky is blue", MemoryKind::Fact, 1000, None, false);
        assert_eq!(meta.sensitivity_class, SensitivityClass::Public);
    }

    #[test]
    fn retention_class_high_salience_long_term() {
        let meta =
            evaluate_lifecycle("married won diagnosed", MemoryKind::Lesson, 1000, None, false);
        assert_eq!(meta.retention_class, RetentionClass::LongTerm);
        assert!(meta.salience_score >= 0.75);
    }

    #[test]
    fn retention_class_inference_ephemeral() {
        let meta =
            evaluate_lifecycle("today I feel happy", MemoryKind::Conversational, 1000, None, true);
        assert_eq!(meta.retention_class, RetentionClass::Ephemeral);
    }

    #[test]
    fn retention_class_compliance_sensitive() {
        let meta = evaluate_lifecycle(
            "my api key is abc123",
            MemoryKind::Conversational,
            1000,
            None,
            false,
        );
        assert_eq!(meta.retention_class, RetentionClass::ComplianceSensitive);
    }

    #[test]
    fn retention_class_preference_long_term() {
        let meta = evaluate_lifecycle("hello", MemoryKind::Preference, 1000, None, false);
        assert_eq!(meta.retention_class, RetentionClass::LongTerm);
    }

    #[test]
    fn retention_class_fact_long_term() {
        let meta = evaluate_lifecycle("hello", MemoryKind::Fact, 1000, None, false);
        assert_eq!(meta.retention_class, RetentionClass::LongTerm);
    }

    #[test]
    fn retention_class_lesson_archive() {
        let meta = evaluate_lifecycle("hello", MemoryKind::Lesson, 1000, None, false);
        assert_eq!(meta.retention_class, RetentionClass::Archive);
    }

    #[test]
    fn retention_class_episodic_for_conversational() {
        let meta = evaluate_lifecycle("hello", MemoryKind::Conversational, 1000, None, false);
        assert_eq!(meta.retention_class, RetentionClass::Episodic);
    }

    #[test]
    fn expires_at_ephemeral_inference_is_30_days() {
        let meta = evaluate_lifecycle("hello", MemoryKind::Conversational, 1_000_000, None, true);
        assert_eq!(meta.retention_class, RetentionClass::Ephemeral);
        assert!(meta.expires_at_ms.is_some());
        assert_eq!(meta.expires_at_ms.unwrap(), 1_000_000 + 30 * 86_400_000);
    }

    #[test]
    fn expires_at_working_is_90_days() {
        let result = expires_at_for(RetentionClass::Working, 1_000_000, false);
        assert_eq!(result, Some(1_000_000 + 90 * 86_400_000));
    }

    #[test]
    fn expires_at_long_term_archive_is_none() {
        let pref = evaluate_lifecycle("hello", MemoryKind::Preference, 1_000_000, None, false);
        let lesson = evaluate_lifecycle("hello", MemoryKind::Lesson, 1_000_000, None, false);
        assert_eq!(pref.retention_class, RetentionClass::LongTerm);
        assert!(pref.expires_at_ms.is_none());
        assert_eq!(lesson.retention_class, RetentionClass::Archive);
        assert!(lesson.expires_at_ms.is_none());
    }

    #[test]
    fn specificity_scoring_with_numbers() {
        let meta = evaluate_lifecycle("I have 3 cats", MemoryKind::Fact, 1000, None, false);
        assert!(meta.specificity_score > 0.28);
        assert!(meta.reasons.contains(&"numeric_or_date_signal".to_string()));
    }

    #[test]
    fn specificity_scoring_with_named_entities() {
        let meta = evaluate_lifecycle("Alice went to Paris", MemoryKind::Fact, 1000, None, false);
        assert!(meta.specificity_score > 0.28);
        assert!(meta.reasons.contains(&"named_entity_signal".to_string()));
    }

    #[test]
    fn specificity_scoring_with_specific_words() {
        let meta =
            evaluate_lifecycle("because when where with", MemoryKind::Fact, 1000, None, false);
        assert!(meta.specificity_score > 0.28);
    }

    #[test]
    fn stability_score_temporary_keywords() {
        let meta = evaluate_lifecycle("today I feel happy", MemoryKind::Fact, 1000, None, false);
        assert!((meta.stability_score - 0.34).abs() < f32::EPSILON);
        assert!(meta.reasons.contains(&"temporary_state".to_string()));
    }

    #[test]
    fn stability_score_inference() {
        let meta = evaluate_lifecycle("hello world", MemoryKind::Fact, 1000, None, true);
        assert!((meta.stability_score - 0.32).abs() < f32::EPSILON);
        assert!(meta.reasons.contains(&"inference_lower_stability".to_string()));
    }

    #[test]
    fn stability_score_permanent_kinds() {
        let pref = evaluate_lifecycle("hello", MemoryKind::Preference, 1000, None, false);
        let conv = evaluate_lifecycle("hello", MemoryKind::Conversational, 1000, None, false);
        assert!((pref.stability_score - 0.78).abs() < f32::EPSILON);
        assert!((conv.stability_score - 0.44).abs() < f32::EPSILON);
    }

    #[test]
    fn utility_score_with_keywords() {
        let base = evaluate_lifecycle("hello", MemoryKind::Fact, 1000, None, false);
        let boosted =
            evaluate_lifecycle("favorite family job career", MemoryKind::Fact, 1000, None, false);
        assert!(boosted.utility_score > base.utility_score);
        assert!(boosted.reasons.contains(&"future_utility_signal".to_string()));
    }

    #[test]
    fn empty_text_handled() {
        let meta = evaluate_lifecycle("", MemoryKind::Fact, 1000, None, false);
        assert_eq!(meta.sensitivity_class, SensitivityClass::Public);
        assert!((meta.specificity_score - 0.28).abs() < f32::EPSILON);
    }

    #[test]
    fn promote_to_profile_conditions_met() {
        let meta =
            evaluate_lifecycle("prefer blue", MemoryKind::Preference, 1000, Some(0.80), false);
        assert!(meta.promote_to_profile);
    }

    #[test]
    fn promote_to_profile_blocked_by_restricted() {
        let meta = evaluate_lifecycle(
            "prefer password abc",
            MemoryKind::Preference,
            1000,
            Some(0.80),
            false,
        );
        assert!(!meta.promote_to_profile);
    }

    #[test]
    fn promote_to_profile_blocked_by_low_confidence() {
        let meta =
            evaluate_lifecycle("prefer blue", MemoryKind::Preference, 1000, Some(0.50), false);
        assert!(!meta.promote_to_profile);
    }

    #[test]
    fn promote_to_profile_not_for_conversational() {
        let meta =
            evaluate_lifecycle("prefer blue", MemoryKind::Conversational, 1000, Some(0.80), false);
        assert!(!meta.promote_to_profile);
    }

    #[test]
    fn all_sensitivity_markers_restricted_wins() {
        let meta = evaluate_lifecycle(
            "my family has a health condition and my password is secret",
            MemoryKind::Fact,
            1000,
            None,
            false,
        );
        assert_eq!(meta.sensitivity_class, SensitivityClass::Restricted);
    }
}
