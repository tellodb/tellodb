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
        let char_count = entity_lower.chars().count();

        // Prefix aliases are counted in characters: slicing by bytes panicked
        // on names like "Zoë" (and a panic aborts the server in release).
        for (min_chars, prefix_chars) in [(5, 3), (6, 4)] {
            if char_count < min_chars {
                continue;
            }
            let prefix: String = entity_lower.chars().take(prefix_chars).collect();
            if lower.contains(&prefix) {
                let collisions = known_entities
                    .iter()
                    .filter(|e| e.to_ascii_lowercase().starts_with(&prefix))
                    .count();
                if collisions <= MAX_PREFIX_COLLISIONS {
                    add_alias(prefix, entity.clone());
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
    vectors: &crate::vector_index::VectorIndex,
    entity_id: &str,
    embedding: &[f32],
    threshold: f32,
) -> Result<bool, StatusCode> {
    let candidates = vectors
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_ascii_entity_names_do_not_panic() {
        let aliases = extract_aliases_from_text(
            "Zoë Smith and José went hiking with 北京朋友",
            &["Zoë Smith".to_string(), "José Álvarez".to_string(), "北京朋友们".to_string()],
        );
        assert!(aliases
            .iter()
            .any(|(alias, canonical)| alias == "zoë" && canonical == "Zoë Smith"));
    }
}
