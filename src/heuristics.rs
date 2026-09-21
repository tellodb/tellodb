//! Which hand-written lexical rules are active.
//!
//! Some rules in the query planner and the fact-key rules were written while
//! looking at particular benchmark questions: brand names (`spotify`), proper
//! nouns (`dr seuss`) and phrases lifted from LoCoMo. Tuning on the
//! evaluation set is not a defect that can be measured away by rerunning it —
//! the numbers are simply not evidence about unseen data.
//!
//! Rather than delete the rules, the profile selects them:
//!
//! - `generic` (default): no rule keyed on a proper noun, a brand, or a
//!   phrase taken from a benchmark question. Topic-class expansions stay,
//!   because they are not about any particular question.
//! - `legacy-tuned`: everything, including the rules above.
//!
//! Reporting both quantifies what the hand-tuning bought, on the benchmark it
//! was tuned on and on one it was not.

use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    Generic,
    LegacyTuned,
}

impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Profile::Generic => "generic",
            Profile::LegacyTuned => "legacy-tuned",
        }
    }

    pub fn parse(spec: &str) -> Result<Self> {
        match spec.trim().to_ascii_lowercase().as_str() {
            "" | "generic" => Ok(Profile::Generic),
            "legacy-tuned" | "legacy_tuned" | "legacy" => Ok(Profile::LegacyTuned),
            other => bail!("unknown TELLODB_HEURISTICS '{other}' (generic, legacy-tuned)"),
        }
    }
}

pub fn benchmark_tuned_rules(profile: Profile) -> bool {
    profile == Profile::LegacyTuned
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_is_generic() {
        assert_eq!(Profile::parse("").unwrap(), Profile::Generic);
        assert_eq!(Profile::parse("generic").unwrap(), Profile::Generic);
    }

    #[test]
    fn legacy_profile_has_aliases() {
        for spec in ["legacy-tuned", "legacy_tuned", "LEGACY", " legacy-tuned "] {
            assert_eq!(Profile::parse(spec).unwrap(), Profile::LegacyTuned, "{spec:?}");
        }
    }

    #[test]
    fn the_override_scopes_to_one_call() {
        assert!(!benchmark_tuned_rules(Profile::Generic));
        assert!(benchmark_tuned_rules(Profile::LegacyTuned));
    }

    #[test]
    fn typos_are_errors() {
        assert!(Profile::parse("genric").is_err());
        assert!(Profile::parse("tuned").is_err());
    }
}
