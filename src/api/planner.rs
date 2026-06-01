pub use crate::api::plan::*;

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
    fn retrieval_rewrite_corrects_common_locomo_typos() {
        assert_eq!(
            rewrite_query_for_retrieval(
                "What fields would Caroline be likely to pursue in her educaton?"
            ),
            "What fields would Caroline be likely to pursue in her education?"
        );
        assert_eq!(
            rewrite_query_for_retrieval(
                "When did Caroline and Melanie go to a pride fesetival together?"
            ),
            "When did Caroline and Melanie go to a pride festival together?"
        );
    }

    #[test]
    fn inference_plan_adds_archetype_expansion_terms() {
        let plan = build_query_plan(
            "What fields would Caroline be likely to pursue in her education?",
            None,
        );
        assert!(plan.lexical_terms.iter().any(|term| term == "career"));
        assert!(plan.lexical_terms.iter().any(|term| term == "education"));
        assert!(plan
            .semantic_queries
            .iter()
            .any(|query| query.contains("career") && query.contains("training")));
    }

    #[test]
    fn park_preference_plan_bridges_to_outdoor_evidence() {
        let plan = build_query_plan(
            "Would Melanie be more interested in going to a national park or a theme park?",
            None,
        );
        assert!(plan.lexical_terms.iter().any(|term| term == "camping"));
        assert!(plan.lexical_terms.iter().any(|term| term == "hiking"));
        assert!(plan.lexical_terms.iter().any(|term| term == "amusement"));
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
