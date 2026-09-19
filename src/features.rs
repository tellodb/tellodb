//! Switches for derived ingest structures, used for ablations.
//!
//! `TELLODB_DISABLE=gist,keywords,...` turns structures off. A disabled
//! structure is neither built at ingest nor consulted at query time, so a run
//! measures both its quality contribution and its cost. Unknown names are a
//! startup error, so a typo cannot silently run the full configuration.

use anyhow::{bail, Result};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    /// Splitting long memories into `::cN` chunks.
    Chunks,
    /// Session gist companion (`::gist`) on a session's first turn.
    Gist,
    /// Keyword companion (`::kw`) on a session's first turn.
    Keywords,
    /// "Canonical fact" companions (`::factN`).
    FactCompanions,
    /// Atomic memory card companions (`::cardN`).
    AtomicCards,
    /// Event companion (`::event`).
    EventCompanions,
    /// Relation companions (`::relN`).
    RelationCompanions,
    /// Memory card rows and the card retrieval lane.
    MemoryCards,
    /// Per-session router records and the session routing lanes.
    SessionRouter,
    /// Preference memories and the preference lane.
    Preferences,
    /// Links from turns that refer back to earlier memories.
    RetrospectiveLinks,
    /// Links between derived records and their source.
    DerivedLinks,
    /// Subject/predicate/object edges and the entity-graph lanes.
    GraphEdges,
    /// Fact version chains (supersession) and fact lookups.
    Facts,
    /// Dropping near-duplicate facts at ingest.
    SemanticDedup,
    /// Background core-profile consolidation.
    Consolidation,
    /// Numeric metric extraction.
    Metrics,
    /// Grouping predicate variants by meaning before fact supersession.
    PredicateCanon,
}

impl Feature {
    pub const ALL: [Feature; 18] = [
        Feature::Chunks,
        Feature::Gist,
        Feature::Keywords,
        Feature::FactCompanions,
        Feature::AtomicCards,
        Feature::EventCompanions,
        Feature::RelationCompanions,
        Feature::MemoryCards,
        Feature::SessionRouter,
        Feature::Preferences,
        Feature::RetrospectiveLinks,
        Feature::DerivedLinks,
        Feature::GraphEdges,
        Feature::Facts,
        Feature::SemanticDedup,
        Feature::Consolidation,
        Feature::Metrics,
        Feature::PredicateCanon,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Feature::Chunks => "chunks",
            Feature::Gist => "gist",
            Feature::Keywords => "keywords",
            Feature::FactCompanions => "fact_companions",
            Feature::AtomicCards => "atomic_cards",
            Feature::EventCompanions => "event_companions",
            Feature::RelationCompanions => "relation_companions",
            Feature::MemoryCards => "memory_cards",
            Feature::SessionRouter => "session_router",
            Feature::Preferences => "preferences",
            Feature::RetrospectiveLinks => "retrospective_links",
            Feature::DerivedLinks => "derived_links",
            Feature::GraphEdges => "graph_edges",
            Feature::Facts => "facts",
            Feature::SemanticDedup => "semantic_dedup",
            Feature::Consolidation => "consolidation",
            Feature::Metrics => "metrics",
            Feature::PredicateCanon => "predicate_canon",
        }
    }

    fn bit(self) -> u32 {
        1 << (self as u32)
    }

    /// The companion structure a derived memory-id tag belongs to.
    pub fn for_companion_tag(tag: &str) -> Option<Feature> {
        let stem = tag.trim_end_matches(|c: char| c.is_ascii_digit());
        match stem {
            "gist" => Some(Feature::Gist),
            "kw" => Some(Feature::Keywords),
            "fact" => Some(Feature::FactCompanions),
            "card" => Some(Feature::AtomicCards),
            "event" => Some(Feature::EventCompanions),
            "rel" => Some(Feature::RelationCompanions),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Features {
    disabled: u32,
}

impl Features {
    pub fn parse(spec: &str) -> Result<Self> {
        let mut disabled = 0;
        for name in spec.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            let name = name.to_ascii_lowercase();
            if name == "none" {
                continue;
            }
            let Some(feature) = Feature::ALL.iter().find(|f| f.name() == name) else {
                let known: Vec<&str> = Feature::ALL.iter().map(|f| f.name()).collect();
                bail!("unknown TELLODB_DISABLE entry '{name}'; known: {}", known.join(","));
            };
            disabled |= feature.bit();
        }
        Ok(Self { disabled })
    }

    pub fn enabled(self, feature: Feature) -> bool {
        self.disabled & feature.bit() == 0
    }

    pub fn disabled_names(self) -> Vec<&'static str> {
        Feature::ALL.iter().filter(|f| !self.enabled(**f)).map(|f| f.name()).collect()
    }
}

static FEATURES: OnceLock<Features> = OnceLock::new();

pub fn init(config: Features) -> Features {
    *FEATURES.get_or_init(|| config)
}

/// Process-wide switches. Everything is enabled until `init_from_env` runs
/// (or if `TELLODB_DISABLE` is unset).
pub fn features() -> Features {
    *FEATURES.get_or_init(Features::default)
}

pub fn enabled(feature: Feature) -> bool {
    features().enabled(feature)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_names_and_rejects_typos() {
        let f = Features::parse("gist, graph_edges,").unwrap();
        assert!(!f.enabled(Feature::Gist) && !f.enabled(Feature::GraphEdges));
        assert!(f.enabled(Feature::Chunks));
        assert_eq!(f.disabled_names(), vec!["gist", "graph_edges"]);
        assert!(Features::parse("gists").is_err());
        assert_eq!(Features::parse("").unwrap(), Features::default());
    }

    #[test]
    fn every_feature_has_a_distinct_bit_and_name() {
        let all =
            Features::parse(&Feature::ALL.iter().map(|f| f.name()).collect::<Vec<_>>().join(","))
                .unwrap();
        assert!(Feature::ALL.iter().all(|f| !all.enabled(*f)));
        assert_eq!(all.disabled_names().len(), Feature::ALL.len());
    }

    #[test]
    fn companion_tags_map_to_features() {
        assert_eq!(Feature::for_companion_tag("card12"), Some(Feature::AtomicCards));
        assert_eq!(Feature::for_companion_tag("kw"), Some(Feature::Keywords));
        assert_eq!(Feature::for_companion_tag("c0"), None);
    }
}
