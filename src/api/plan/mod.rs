pub mod builder;
pub mod expansions;
pub mod intent;
pub mod rewrite;
pub mod scoring;
pub mod types;

pub use builder::*;
pub use expansions::*;
pub use intent::*;
pub use rewrite::*;
pub use scoring::*;
pub use types::*;
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn purchase_queries_are_slot_routed() {
        assert_eq!(
            infer_query_fact_key("What new item did Dave buy recently?"),
            Some("purchase".to_string())
        );
        assert!(is_purchase_query("What did Calvin recently get?"));
    }

    #[test]
    fn purchase_plan_adds_buy_synonyms() {
        let plan = build_query_plan("What new item did Dave buy recently?", None);
        assert!(plan.fts_queries.iter().any(|query| query.contains("\"Dave\" \"bought\"")));
        assert!(plan
            .semantic_queries
            .iter()
            .any(|query| query.contains("bought purchased acquired")));
    }

    #[test]
    fn inference_plan_adds_archetype_expansion_terms() {
        // Synthetic phrasing: the LoCoMo wording this once used made the test
        // pass for the wrong reason, since a rule was written against it.
        let plan = build_query_plan("What fields would Robin pursue in her education?", None);
        assert!(plan.lexical_terms.iter().any(|term| term == "career"));
        assert!(plan.lexical_terms.iter().any(|term| term == "education"));
        assert!(plan
            .semantic_queries
            .iter()
            .any(|query| query.contains("career") && query.contains("training")));
    }

    #[test]
    fn park_preference_bridges_to_outdoor_evidence_only_when_tuned() {
        use crate::heuristics::Profile;
        let question = "Would Robin prefer a national park or a theme park?";

        // "national park" and "theme park" are literal phrases taken from a
        // benchmark question, so only the tuned profile bridges them.
        let plan = build_query_plan_with_profile(question, None, Profile::LegacyTuned);
        assert!(plan.lexical_terms.iter().any(|term| term == "camping"));
        assert!(plan.lexical_terms.iter().any(|term| term == "hiking"));
        assert!(plan.lexical_terms.iter().any(|term| term == "amusement"));
        let plan = build_query_plan_with_profile(question, None, Profile::Generic);
        assert!(!plan.lexical_terms.iter().any(|term| term == "amusement"));

        // The generic profile still expands a topic-class trigger, which is
        // not about any particular question.
        let plan = build_query_plan_with_profile(
            "Where does Robin like to go outdoors?",
            None,
            Profile::Generic,
        );
        assert!(plan.lexical_terms.iter().any(|term| term == "camping"));
    }

    #[test]
    fn hyde_preserves_possessive_subjects() {
        let plan = build_query_plan("What do Melanie's kids like?", None);
        let hyde = build_hyde_query("What do Melanie's kids like?", &plan).unwrap();
        assert!(hyde.contains("Melanie kid"));
        assert!(hyde.contains("activities"));
    }

    #[test]
    fn nickname_plan_adds_short_name_aliases() {
        let plan = build_query_plan("What nickname does Nate use for Joanna?", None);
        assert!(plan.lexical_terms.iter().any(|term| term == "nickname"));
        assert!(plan.lexical_terms.iter().any(|term| term == "jo"));
        assert!(plan.lexical_terms.iter().any(|term| term == "joa"));
    }

    #[test]
    fn local_state_plan_uses_home_and_nearby_terms() {
        let plan = build_query_plan("Does Deborah live close to the beach or the mountains?", None);
        assert!(plan.lexical_terms.iter().any(|term| term == "home"));
        assert!(plan.lexical_terms.iter().any(|term| term == "nearby"));
        assert!(plan.lexical_terms.iter().any(|term| term == "beach"));
        assert!(plan.lexical_terms.iter().any(|term| term == "mountain"));
    }

    #[test]
    fn electronics_plan_bridges_device_language() {
        let plan =
            build_query_plan("What electronics issue has been frustrating Sam lately?", None);
        assert!(plan.lexical_terms.iter().any(|term| term == "device"));
        assert!(plan.lexical_terms.iter().any(|term| term == "computer"));
        assert!(plan.lexical_terms.iter().any(|term| term == "issue"));
    }
}
