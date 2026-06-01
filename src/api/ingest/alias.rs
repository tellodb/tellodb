use crate::api::EngineState;

use axum::http::StatusCode;

pub fn extract_aliases_from_text(text: &str, known_entities: &[String]) -> Vec<(String, String)> {
    let mut aliases = Vec::new();
    let lower = text.to_ascii_lowercase();
    const MAX_ALIASES_PER_ENTITY: usize = 10;
    const MAX_PREFIX_COLLISIONS: usize = 3;

    let mut seen = std::collections::HashSet::new();
    let mut alias_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    let mut add_alias = |alias: String, canonical: String| {
        if alias == canonical.to_ascii_lowercase() {
            return;
        }
        let key = format!("{}|{}", alias, canonical);
        if !seen.insert(key) {
            return;
        }
        let count = alias_counts.entry(canonical.clone()).or_insert(0);
        if *count >= MAX_ALIASES_PER_ENTITY {
            return;
        }
        *count += 1;
        aliases.push((alias, canonical));
    };

    for entity in known_entities {
        let entity_lower = entity.to_ascii_lowercase();

        if entity.len() >= 5 {
            let short = &entity_lower[..3];
            if lower.contains(short) {
                let collisions = known_entities
                    .iter()
                    .filter(|e| e.to_ascii_lowercase().starts_with(short))
                    .count();
                if collisions <= MAX_PREFIX_COLLISIONS {
                    add_alias(short.to_string(), entity.clone());
                }
            }
        }

        if entity.len() >= 6 {
            let short4 = &entity_lower[..4];
            if lower.contains(short4) {
                let collisions = known_entities
                    .iter()
                    .filter(|e| e.to_ascii_lowercase().starts_with(short4))
                    .count();
                if collisions <= MAX_PREFIX_COLLISIONS {
                    add_alias(short4.to_string(), entity.clone());
                }
            }
        }
    }

    let nickname_patterns = [
        "call me ",
        "calls me ",
        "called me ",
        "my nickname is ",
        "nickname is ",
        "they call me ",
        "people call me ",
        "known as ",
        "goes by ",
        "i go by ",
    ];
    for pattern in nickname_patterns {
        if let Some(pos) = lower.find(pattern) {
            let rest = &text[pos + pattern.len()..];
            let nickname: String = rest
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '\'' && c != '-')
                .next()
                .unwrap_or("")
                .to_string();
            if nickname.len() >= 2 {
                for entity in known_entities {
                    if lower.contains(&entity.to_ascii_lowercase()) {
                        add_alias(nickname.to_ascii_lowercase(), entity.clone());
                        break;
                    }
                }
            }
        }
    }

    let relationship_labels = [
        ("hubby", "husband"),
        ("wifey", "wife"),
        ("hubbie", "husband"),
        ("bro", "brother"),
        ("sis", "sister"),
        ("mom", "mother"),
        ("dad", "father"),
        ("kiddo", "child"),
        ("kiddos", "children"),
    ];
    for (alias, _rel_type) in relationship_labels {
        if lower.contains(alias) {
            for entity in known_entities {
                if lower.contains(&entity.to_ascii_lowercase()) {
                    add_alias(alias.to_string(), entity.clone());
                    break;
                }
            }
        }
    }

    aliases
}

pub fn is_semantic_duplicate(
    state: &EngineState,
    entity_id: &str,
    embedding: &[f32],
    threshold: f32,
) -> Result<bool, StatusCode> {
    let candidates = state
        .vector_index
        .search(Some(entity_id), embedding, 5)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    for (_, dist) in candidates {
        let similarity = 1.0 - dist;
        if similarity >= threshold {
            return Ok(true);
        }
    }
    Ok(false)
}


