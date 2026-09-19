use crate::core::calendar::MILLIS_PER_DAY;
use crate::storage::MemoryKind;

pub const DECAY_FLOOR: f32 = 0.35;

pub fn cosine_similarity_from_distance(distance: f32) -> f32 {
    (1.0 - distance).clamp(-1.0, 1.0)
}

pub fn apply_time_decay(base_score: f32, age_in_days: f32, half_life_days: f32, floor: f32) -> f32 {
    if age_in_days <= 0.0 {
        return base_score;
    }

    let lambda = std::f32::consts::LN_2 / half_life_days;
    let decay_multiplier = (-lambda * age_in_days).exp();
    let final_multiplier = decay_multiplier.max(floor);
    base_score * final_multiplier
}

pub fn decay_policy(kind: MemoryKind) -> (f32, f32) {
    match kind {
        MemoryKind::Conversational => (30.0, DECAY_FLOOR),
        MemoryKind::Lesson => (90.0, DECAY_FLOOR),
        MemoryKind::Fact => (180.0, DECAY_FLOOR),
        MemoryKind::SessionSummary => (14.0, DECAY_FLOOR),
        MemoryKind::Decision | MemoryKind::Preference => (365.0, DECAY_FLOOR),
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(
            (apply_decay_with_policy(0.75, 0, MemoryKind::Decision, u64::MAX) - 0.75).abs()
                < f32::EPSILON
        );
        assert!(
            (apply_decay_with_policy(0.75, 0, MemoryKind::Preference, u64::MAX) - 0.75).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn apply_decay_with_policy_old_memories_decay_to_floor_but_stay() {
        let day = 86_400_000;
        let recent = apply_decay_with_policy(1.0, 0, MemoryKind::Conversational, 30 * day);
        assert!(recent > 0.35 && recent < 1.0);
        let ancient = apply_decay_with_policy(1.0, 0, MemoryKind::Conversational, 3_000 * day);
        assert!((ancient - DECAY_FLOOR).abs() < 1e-6);
    }
}
