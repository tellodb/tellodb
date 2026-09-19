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
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

static PROFILE: OnceLock<Profile> = OnceLock::new();

#[cfg(test)]
thread_local! {
    /// Per-test override. The process-wide `OnceLock` is set once, so without
    /// this only one profile would ever be reachable from unit tests, and the
    /// profile we do not test is the one that ships.
    static TEST_PROFILE: std::cell::Cell<Option<Profile>> = const { std::cell::Cell::new(None) };
}

/// Runs `f` with `profile` active on this thread.
#[cfg(test)]
pub fn with_profile<T>(profile: Profile, f: impl FnOnce() -> T) -> T {
    let previous = TEST_PROFILE.with(|slot| slot.replace(Some(profile)));
    let out = f();
    TEST_PROFILE.with(|slot| slot.set(previous));
    out
}

/// Parses `TELLODB_HEURISTICS`. Call once at startup to fail fast on typos.
pub fn init_from_env() -> Result<Profile> {
    let parsed = Profile::parse(&std::env::var("TELLODB_HEURISTICS").unwrap_or_default())?;
    Ok(*PROFILE.get_or_init(|| parsed))
}

pub fn profile() -> Profile {
    #[cfg(test)]
    if let Some(profile) = TEST_PROFILE.with(|slot| slot.get()) {
        return profile;
    }
    *PROFILE.get_or_init(|| {
        Profile::parse(&std::env::var("TELLODB_HEURISTICS").unwrap_or_default()).unwrap_or_else(
            |err| {
                tracing::error!(error = %err, "ignoring invalid TELLODB_HEURISTICS");
                Profile::Generic
            },
        )
    })
}

/// True when benchmark-derived rules are allowed to fire.
///
/// Guard every rule that keys on a proper noun, a brand, or a phrase lifted
/// from a benchmark question with this.
pub fn benchmark_tuned_rules() -> bool {
    profile() == Profile::LegacyTuned
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
        assert_eq!(profile(), Profile::Generic);
        with_profile(Profile::LegacyTuned, || {
            assert!(benchmark_tuned_rules());
        });
        assert_eq!(profile(), Profile::Generic, "override leaked past the call");
    }

    #[test]
    fn typos_are_errors() {
        assert!(Profile::parse("genric").is_err());
        assert!(Profile::parse("tuned").is_err());
    }
}
