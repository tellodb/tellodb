use std::fmt::{Display, Formatter};

pub type EntityId = String;
pub type SessionId = String;

const SEPARATOR: &str = "::";

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Tag {
    Chunk(u32),
    Companion(String),
    Card(String),
    SyntheticQuery,
    Named(String),
}

impl Tag {
    pub fn from_component(component: &str) -> Self {
        if component == "sq" {
            return Self::SyntheticQuery;
        }
        if let Some(value) = component.strip_prefix('c').and_then(|value| value.parse().ok()) {
            return Self::Chunk(value);
        }
        if component.starts_with("card") {
            return Self::Card(component.to_string());
        }
        if component == "gist"
            || component == "kw"
            || component == "event"
            || component.starts_with("rel")
            || component.starts_with("fact")
        {
            return Self::Companion(component.to_string());
        }
        Self::Named(component.to_string())
    }

    fn component(&self) -> String {
        match self {
            Self::Chunk(value) => format!("c{value}"),
            Self::Companion(value) | Self::Card(value) | Self::Named(value) => value.clone(),
            Self::SyntheticQuery => "sq".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryIdError {
    Empty,
    InvalidEscape,
    InvalidTurn,
}

impl Display for MemoryIdError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::Empty => "memory id is empty",
            Self::InvalidEscape => "memory id contains an invalid escape",
            Self::InvalidTurn => "memory id has an invalid turn index",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for MemoryIdError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryId {
    entity: EntityId,
    session: SessionId,
    turn: u32,
    tags: Vec<Tag>,
    rendered: String,
    structured: bool,
}

impl MemoryId {
    pub fn new(entity: impl Into<EntityId>, session: impl Into<SessionId>, turn: u32) -> Self {
        let entity = entity.into();
        let session = session.into();
        let rendered = render_structured(&entity, &session, turn, &[]);
        Self { entity, session, turn, tags: Vec::new(), rendered, structured: true }
    }

    #[must_use]
    pub fn derived(&self, tag: Tag) -> Self {
        if !self.structured {
            let component = escape_component(&tag.component());
            let rendered = format!("{}{SEPARATOR}{component}", self.rendered);
            return Self { rendered, tags: vec![tag], ..self.clone() };
        }

        let mut tags = self.tags.clone();
        tags.push(tag);
        let component = escape_component(&tags.last().expect("tag was just appended").component());
        let rendered = format!("{}{SEPARATOR}{component}", self.rendered);
        Self { tags, rendered, ..self.clone() }
    }

    pub fn derived_from(parent: &str, tag: &str) -> String {
        let parsed = Self::parse(parent).unwrap_or_else(|_| Self::opaque(parent));
        parsed.derived(Tag::from_component(tag)).as_str().to_string()
    }

    pub fn parse(value: &str) -> Result<Self, MemoryIdError> {
        if value.is_empty() {
            return Err(MemoryIdError::Empty);
        }

        let parts: Vec<&str> = value.split(SEPARATOR).collect();
        if parts.len() < 3 {
            let entity = decode_component(value)?;
            return Ok(Self {
                entity,
                session: String::new(),
                turn: 0,
                tags: Vec::new(),
                rendered: value.to_string(),
                structured: false,
            });
        }

        let entity = decode_component(parts[0])?;
        let session = decode_component(parts[1])?;
        let turn = parts[2].parse::<u32>().map_err(|_parse_error| MemoryIdError::InvalidTurn)?;
        let mut tags = Vec::with_capacity(parts.len().saturating_sub(3));
        for part in &parts[3..] {
            tags.push(Tag::from_component(&decode_component(part)?));
        }
        let rendered = value.to_string();
        Ok(Self { entity, session, turn, tags, rendered, structured: true })
    }

    pub fn as_str(&self) -> &str {
        &self.rendered
    }

    pub fn entity(&self) -> &EntityId {
        &self.entity
    }

    pub fn session(&self) -> &SessionId {
        &self.session
    }

    pub fn turn(&self) -> u32 {
        self.turn
    }

    pub fn tags(&self) -> &[Tag] {
        &self.tags
    }

    pub fn is_structured(&self) -> bool {
        self.structured
    }

    fn opaque(value: &str) -> Self {
        Self {
            entity: value.to_string(),
            session: String::new(),
            turn: 0,
            tags: Vec::new(),
            rendered: value.to_string(),
            structured: false,
        }
    }
}

fn render_structured(entity: &str, session: &str, turn: u32, tags: &[Tag]) -> String {
    let mut rendered = format!(
        "{}{SEPARATOR}{}{SEPARATOR}{turn}",
        escape_component(entity),
        escape_component(session)
    );
    for tag in tags {
        rendered.push_str(SEPARATOR);
        rendered.push_str(&escape_component(&tag.component()));
    }
    rendered
}

fn escape_component(component: &str) -> String {
    let mut escaped = String::with_capacity(component.len());
    for character in component.chars() {
        match character {
            '%' => escaped.push_str("%25"),
            ':' => escaped.push_str("%3A"),
            _ => escaped.push(character),
        }
    }
    escaped
}

fn decode_component(component: &str) -> Result<String, MemoryIdError> {
    let mut decoded = String::with_capacity(component.len());
    let mut chars = component.chars();
    while let Some(character) = chars.next() {
        if character != '%' {
            decoded.push(character);
            continue;
        }
        let high = chars.next().ok_or(MemoryIdError::InvalidEscape)?;
        let low = chars.next().ok_or(MemoryIdError::InvalidEscape)?;
        let high = high.to_digit(16).ok_or(MemoryIdError::InvalidEscape)?;
        let low = low.to_digit(16).ok_or(MemoryIdError::InvalidEscape)?;
        let byte = ((high << 4) | low) as u8;
        match byte {
            b'%' => decoded.push('%'),
            b':' => decoded.push(':'),
            _ => return Err(MemoryIdError::InvalidEscape),
        }
    }
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{MemoryId, Tag};

    fn memory_component() -> impl Strategy<Value = String> {
        prop::collection::vec(
            prop_oneof![
                Just("%".to_string()),
                Just("::".to_string()),
                any::<char>().prop_map(|character| character.to_string()),
            ],
            0..16,
        )
        .prop_map(|parts| {
            parts.into_iter().fold(String::new(), |mut value, part| {
                value.push_str(&part);
                value
            })
        })
    }

    #[test]
    fn structured_ids_round_trip_with_legacy_rendering() {
        let id = MemoryId::new("alice", "chat-1", 3);
        assert_eq!(id.as_str(), "alice::chat-1::3");
        assert_eq!(MemoryId::parse(id.as_str()).unwrap(), id);
    }

    #[test]
    fn components_escape_percent_and_colon() {
        let id = MemoryId::new("team::alpha%beta", "session::one%two", 7);
        assert_eq!(id.as_str(), "team%3A%3Aalpha%25beta::session%3A%3Aone%25two::7");
        assert_eq!(MemoryId::parse(id.as_str()).unwrap(), id);
    }

    #[test]
    fn derived_ids_preserve_parent_and_tags() {
        let id = MemoryId::new("alice", "chat", 2).derived(Tag::Chunk(3));
        assert_eq!(id.as_str(), "alice::chat::2::c3");
        assert_eq!(id.tags(), &[Tag::Chunk(3)]);
        assert_eq!(MemoryId::parse(id.as_str()).unwrap(), id);
    }

    #[test]
    fn synthetic_query_tag_is_typed() {
        let id = MemoryId::new("alice", "chat", 2).derived(Tag::SyntheticQuery);
        assert!(id.tags().contains(&Tag::SyntheticQuery));
    }

    #[test]
    fn legacy_components_with_empty_and_unicode_values_round_trip() {
        for (entity, session, turn) in [("", "", 0), ("用户", "جلسة", 12)] {
            let id = MemoryId::new(entity, session, turn);
            assert_eq!(MemoryId::parse(id.as_str()).unwrap(), id);
        }
    }

    #[test]
    fn opaque_ids_are_preserved_without_identity_inference() {
        let id = MemoryId::parse("opaque-id").unwrap();
        assert!(!id.is_structured());
        assert_eq!(id.as_str(), "opaque-id");
        assert_eq!(id.session(), "");
        assert_eq!(id.turn(), 0);
    }

    #[test]
    fn malformed_percent_escape_is_rejected() {
        assert!(MemoryId::parse("a%2::b::0").is_err());
    }

    #[test]
    fn parsed_structured_ids_preserve_the_callers_spelling() {
        for value in ["a::b::007", "a::b::1::c03", "urn%3Ax::s::0"] {
            assert_eq!(MemoryId::parse(value).unwrap().as_str(), value);
        }
    }

    #[test]
    fn derived_ids_extend_the_original_parent_spelling() {
        assert_eq!(MemoryId::derived_from("a::b::007", "c1"), "a::b::007::c1");
        assert_eq!(MemoryId::derived_from("urn%3Ax::s::0", "fact0"), "urn%3Ax::s::0::fact0");
    }

    proptest! {
        #[test]
        fn generated_components_round_trip(
            entity in memory_component(),
            session in memory_component(),
            turn in any::<u32>(),
        ) {
            let id = MemoryId::new(entity, session, turn);
            let rendered = id.as_str().to_string();
            prop_assert_eq!(MemoryId::parse(&rendered), Ok(id));
        }
    }
}
