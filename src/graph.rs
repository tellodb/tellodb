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
    pub const ALL: [Self; 13] = [
        Self::DerivedFrom,
        Self::DerivedVariant,
        Self::Supersedes,
        Self::SupersededBy,
        Self::Supports,
        Self::Derives,
        Self::Updates,
        Self::Prefers,
        Self::WorksAt,
        Self::LivesIn,
        Self::CausedBy,
        Self::LeadsTo,
        Self::Default,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeType::DerivedFrom => "derived_from",
            EdgeType::DerivedVariant => "derived_variant",
            EdgeType::Supersedes => "supersedes",
            EdgeType::SupersededBy => "superseded_by",
            EdgeType::Supports => "supports",
            EdgeType::Derives => "derives",
            EdgeType::Updates => "updates",
            EdgeType::Prefers => "prefers",
            EdgeType::WorksAt => "works_at",
            EdgeType::LivesIn => "lives_in",
            EdgeType::CausedBy => "caused_by",
            EdgeType::LeadsTo => "leads_to",
            EdgeType::Default => "default",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "derived_from" => EdgeType::DerivedFrom,
            "derived_variant" => EdgeType::DerivedVariant,
            "supersedes" | "supersede" => EdgeType::Supersedes,
            "superseded_by" => EdgeType::SupersededBy,
            "supports" => EdgeType::Supports,
            "derives" => EdgeType::Derives,
            "updates" => EdgeType::Updates,
            "prefers" => EdgeType::Prefers,
            "works_at" => EdgeType::WorksAt,
            "lives_in" => EdgeType::LivesIn,
            "caused_by" => EdgeType::CausedBy,
            "leads_to" => EdgeType::LeadsTo,
            _ => EdgeType::Default,
        }
    }

    pub fn default_weight(&self) -> f32 {
        match self {
            EdgeType::CausedBy => 1.4,
            EdgeType::Supersedes => 0.3,
            EdgeType::Prefers => 1.2,
            EdgeType::DerivedFrom => 0.7,
            _ => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    In,
    #[default]
    Out,
    Both,
}

impl Direction {
    pub fn parse(value: Option<&str>) -> Self {
        match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            Some("in" | "inbound") => Self::In,
            Some("both") => Self::Both,
            _ => Self::Out,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::In => "Inbound",
            Self::Out => "Outbound",
            Self::Both => "Both",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Direction, EdgeType};

    #[test]
    fn edge_type_round_trips() {
        for edge_type in EdgeType::ALL {
            assert_eq!(EdgeType::from_str(edge_type.as_str()), edge_type);
        }
    }

    #[test]
    fn direction_parses_and_renders() {
        assert_eq!(Direction::parse(Some("in")), Direction::In);
        assert_eq!(Direction::parse(Some("INBOUND")), Direction::In);
        assert_eq!(Direction::parse(Some("both")), Direction::Both);
        assert_eq!(Direction::parse(Some("out")), Direction::Out);
        assert_eq!(Direction::parse(None).as_str(), "Outbound");
    }
}
