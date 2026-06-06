/// Canonical set of typed edge relationships used across all tables.
/// String representation is lowercase_snake_case stored in SQLite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeType {
    // Derivation / provenance
    DerivedFrom,
    DerivedVariant,
    // Fact lifecycle
    Supersedes,
    SupersededBy,
    // Card relationships
    Supports,
    Derives,
    Updates,
    // Semantic / ontological
    Prefers,
    WorksAt,
    LivesIn,
    CausedBy,
    LeadsTo,
    // Fallback
    Default,
}

impl EdgeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeType::DerivedFrom    => "derived_from",
            EdgeType::DerivedVariant => "derived_variant",
            EdgeType::Supersedes     => "supersedes",
            EdgeType::SupersededBy   => "superseded_by",
            EdgeType::Supports       => "supports",
            EdgeType::Derives        => "derives",
            EdgeType::Updates        => "updates",
            EdgeType::Prefers        => "prefers",
            EdgeType::WorksAt        => "works_at",
            EdgeType::LivesIn        => "lives_in",
            EdgeType::CausedBy       => "caused_by",
            EdgeType::LeadsTo        => "leads_to",
            EdgeType::Default        => "default",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "derived_from"    => EdgeType::DerivedFrom,
            "derived_variant" => EdgeType::DerivedVariant,
            "supersedes" | "supersede" => EdgeType::Supersedes,
            "superseded_by"   => EdgeType::SupersededBy,
            "supports"        => EdgeType::Supports,
            "derives"         => EdgeType::Derives,
            "updates"         => EdgeType::Updates,
            "prefers"         => EdgeType::Prefers,
            "works_at"        => EdgeType::WorksAt,
            "lives_in"        => EdgeType::LivesIn,
            "caused_by"       => EdgeType::CausedBy,
            "leads_to"        => EdgeType::LeadsTo,
            _                 => EdgeType::Default,
        }
    }

    pub fn default_weight(&self) -> f32 {
        match self {
            EdgeType::CausedBy   => 1.4,
            EdgeType::Supersedes => 0.3,
            EdgeType::Prefers    => 1.2,
            EdgeType::DerivedFrom => 0.7,
            _                    => 1.0,
        }
    }
}

