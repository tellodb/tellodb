use std::collections::HashSet;

use crate::api::types::IngestPayload;
use crate::api::utils::{
    extract_named_phrases, extract_temporal_terms, has_token, normalize_alpha_tokens,
    normalize_fact_text, singularize_token,
};

use super::dialogue::{extract_dialogue_messages, strip_leading_bracketed_prefixes};

pub fn is_numericish(token: &str) -> bool {
    !token.is_empty()
        && token.chars().all(|c| c.is_ascii_digit() || matches!(c, '$' | '.' | ',' | '%'))
}

pub fn sanitize_key_parts(parts: &[&str]) -> Option<String> {
    const DROP: &[&str] = &[
        "my",
        "the",
        "a",
        "an",
        "to",
        "of",
        "for",
        "and",
        "new",
        "current",
        "daily",
        "local",
        "now",
        "currently",
        "really",
        "very",
        "just",
        "that",
        "this",
    ];

    let cleaned = parts
        .iter()
        .filter(|part| !part.is_empty())
        .map(|part| part.trim_matches(|c: char| !c.is_ascii_alphanumeric()))
        .filter(|part| !part.is_empty())
        .filter(|part| !part.chars().all(|c| c.is_ascii_digit()))
        .filter(|part| !DROP.contains(part))
        .collect::<Vec<_>>();

    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned.join("_"))
    }
}

pub fn sanitize_key_parts_owned(parts: &[String]) -> Option<String> {
    let borrowed = parts.iter().map(|part| part.as_str()).collect::<Vec<_>>();
    sanitize_key_parts(&borrowed)
}

pub fn build_contextual_key(
    context: &[String],
    base: &[String],
    suffix: Option<&str>,
) -> Option<String> {
    let mut parts = Vec::new();
    parts.extend(context.iter().cloned());
    for (idx, token) in base.iter().enumerate() {
        if idx + 1 == base.len() {
            parts.push(singularize_token(token));
        } else {
            parts.push(token.clone());
        }
    }
    if let Some(suffix) = suffix {
        parts.push(suffix.to_string());
    }
    sanitize_key_parts_owned(&parts)
}

fn extract_my_attribute_key_from_tokens(tokens: &[String]) -> Option<String> {
    let verbs = [
        "is", "was", "are", "were", "takes", "take", "costs", "cost", "equals", "measures",
        "lasts", "lasted", "called", "named", "uses", "use", "prefers", "prefer",
    ];
    let my_index = tokens.iter().position(|token| token == "my")?;
    let verb_index = tokens
        .iter()
        .enumerate()
        .skip(my_index + 1)
        .find(|(_, token)| verbs.contains(&token.as_str()))?
        .0;
    if verb_index <= my_index + 1 {
        return None;
    }
    let parts = tokens[my_index + 1..verb_index].to_vec();
    build_contextual_key(&[], &parts, None)
}

fn extract_i_have_attribute_key(tokens: &[String]) -> Option<String> {
    let i_index = tokens.iter().position(|token| token == "i")?;
    let have_index = tokens
        .iter()
        .enumerate()
        .skip(i_index + 1)
        .find(|(_, token)| {
            ["have", "has", "had", "own", "owned", "keep", "kept"].contains(&token.as_str())
        })?
        .0;

    let mut idx = have_index + 1;
    while idx < tokens.len()
        && (is_numericish(&tokens[idx])
            || ["a", "an", "the", "my", "now", "currently"].contains(&tokens[idx].as_str()))
    {
        idx += 1;
    }
    if idx >= tokens.len() {
        return None;
    }

    let mut base = Vec::new();
    let mut context = Vec::new();
    let mut reading_context = false;
    let suffix = if tokens[have_index + 1..].iter().any(|token| is_numericish(token)) {
        Some("count")
    } else {
        None
    };

    for token in tokens.iter().skip(idx) {
        if ["called", "named", "titled"].contains(&token.as_str()) {
            return build_contextual_key(&context, &base, Some("name"));
        }
        if ["on", "at", "in", "for", "with"].contains(&token.as_str()) && !base.is_empty() {
            reading_context = true;
            continue;
        }
        if ["is", "was", "are", "were", "that", "which"].contains(&token.as_str()) {
            break;
        }
        if is_numericish(token) {
            continue;
        }
        if reading_context {
            context.push(token.clone());
        } else {
            base.push(token.clone());
        }
    }

    if base.is_empty() {
        None
    } else {
        build_contextual_key(&context, &base, suffix)
    }
}

fn extract_spend_price_key(tokens: &[String]) -> Option<String> {
    let spend_index =
        tokens.iter().position(|token| ["spent", "paid", "cost"].contains(&token.as_str()))?;
    let on_index =
        tokens.iter().enumerate().skip(spend_index + 1).find(|(_, token)| token == &"on")?.0;
    let base = tokens[on_index + 1..]
        .iter()
        .filter(|token| !is_numericish(token))
        .cloned()
        .collect::<Vec<_>>();
    if base.is_empty() {
        None
    } else {
        build_contextual_key(&[], &base, Some("price"))
    }
}

fn extract_identity_or_location_key(tokens: &[String]) -> Option<String> {
    let i_index = tokens.iter().position(|token| token == "i")?;
    let rest = &tokens[i_index + 1..];

    if rest.starts_with(&["live".to_string(), "in".to_string()])
        || rest.starts_with(&["moved".to_string(), "to".to_string()])
    {
        return Some("residence".to_string());
    }
    if rest.starts_with(&["work".to_string(), "at".to_string()]) {
        return Some("employer".to_string());
    }
    if rest.starts_with(&["work".to_string(), "as".to_string()])
        || rest.starts_with(&["am".to_string(), "a".to_string()])
        || rest.starts_with(&["am".to_string(), "an".to_string()])
    {
        return Some("occupation".to_string());
    }
    if rest.starts_with(&["study".to_string(), "at".to_string()])
        || rest.starts_with(&["studied".to_string(), "at".to_string()])
        || rest.starts_with(&["graduated".to_string(), "from".to_string()])
    {
        return Some("school".to_string());
    }
    None
}

pub fn infer_fact_key(text: &str) -> Option<String> {
    let normalized =
        text.trim().strip_prefix("User fact:").unwrap_or(text).trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }

    let tokens = normalize_alpha_tokens(&normalized);

    if ["married", "single", "divorced", "engaged", "dating"]
        .iter()
        .any(|needle| normalized.contains(needle))
    {
        return Some("relationship_status".to_string());
    }
    if (has_token(&tokens, "children")
        || has_token(&tokens, "child")
        || has_token(&tokens, "kids")
        || has_token(&tokens, "kid"))
        && (has_token(&tokens, "have") || has_token(&tokens, "has") || has_token(&tokens, "had"))
    {
        return Some("children_count".to_string());
    }
    if has_token(&tokens, "nickname") || has_token(&tokens, "called") {
        return Some("nickname".to_string());
    }
    if has_token(&tokens, "research") || has_token(&tokens, "researched") {
        return Some("research_topic".to_string());
    }
    if has_token(&tokens, "certificate") || has_token(&tokens, "certified") {
        return Some("certificate".to_string());
    }
    if has_token(&tokens, "team")
        && (has_token(&tokens, "favorite") || has_token(&tokens, "support"))
    {
        return Some("favorite_team".to_string());
    }
    if has_token(&tokens, "hobby") || has_token(&tokens, "hobbies") {
        return Some("hobbies".to_string());
    }
    if has_token(&tokens, "bought")
        || has_token(&tokens, "buy")
        || has_token(&tokens, "purchased")
        || has_token(&tokens, "acquired")
        || (has_token(&tokens, "got") && has_token(&tokens, "new"))
    {
        return Some("purchase".to_string());
    }
    if has_token(&tokens, "recipe") {
        return Some("recipe".to_string());
    }
    if has_token(&tokens, "karate")
        || has_token(&tokens, "judo")
        || has_token(&tokens, "taekwondo")
        || normalized.contains("martial art")
    {
        return Some("martial_arts".to_string());
    }

    if has_token(&tokens, "graduated") && has_token(&tokens, "degree") {
        return Some("degree".to_string());
    }
    if normalized.contains("last name before") || normalized.contains("old name") {
        return Some("previous_last_name".to_string());
    }
    if normalized.contains("previous occupation") {
        return Some("previous_occupation".to_string());
    }
    if has_token(&tokens, "commute") && (has_token(&tokens, "take") || has_token(&tokens, "takes"))
    {
        return Some("commute_duration".to_string());
    }
    if normalized.contains("internet plan")
        && (has_token(&tokens, "mbps")
            || has_token(&tokens, "speed")
            || has_token(&tokens, "upgraded"))
    {
        return Some("internet_plan_speed".to_string());
    }
    if has_token(&tokens, "spotify")
        && (has_token(&tokens, "playlist") || has_token(&tokens, "playlists"))
        && (has_token(&tokens, "created")
            || has_token(&tokens, "called")
            || has_token(&tokens, "named"))
    {
        return Some("spotify_playlist_name".to_string());
    }

    extract_spend_price_key(&tokens)
        .or_else(|| extract_i_have_attribute_key(&tokens))
        .or_else(|| extract_identity_or_location_key(&tokens))
        .or_else(|| extract_my_attribute_key_from_tokens(&tokens))
}

pub fn split_atomic_claims(text: &str) -> Vec<String> {
    text.split(['.', '!', '?', ';'])
        .map(|part| normalize_fact_text(strip_leading_bracketed_prefixes(part)))
        .filter(|part| part.len() >= 12)
        .collect()
}

pub fn is_high_signal_atomic_claim(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let temporal = !extract_temporal_terms(text).is_empty();
    let named = !extract_named_phrases(&[text.to_string()]).is_empty();
    let relation_like = [
        "volunteer",
        "work",
        "worked",
        "works",
        "study",
        "studied",
        "lives",
        "live",
        "likes",
        "loves",
        "prefers",
        "prefer",
        "enjoys",
        "joined",
        "went",
        "visited",
        "watched",
        "bought",
        "started",
        "finished",
        "won",
        "plays",
        "played",
        "learned",
        "teaches",
        "taught",
        "plans",
        "planned",
        "wants",
        "wanted",
        "has",
        "have",
        "had",
        "is",
        "was",
        "are",
        "were",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    let personal_signal =
        [" i ", " my ", " me ", " we ", " our ", " he ", " she ", " his ", " her ", " they "]
            .iter()
            .any(|needle| format!(" {lower} ").contains(needle));

    (temporal || named || personal_signal) && relation_like
}

pub fn build_atomic_memory_card_payloads(
    payload: &IngestPayload,
    entity_id: &str,
    session_id: &str,
    turn_index: usize,
) -> Vec<IngestPayload> {
    let dialogue = extract_dialogue_messages(&payload.textual_content);
    let source_claims = if dialogue.is_empty() {
        payload
            .textual_content
            .lines()
            .filter(|line| !line.trim_start().starts_with('['))
            .map(|line| ("memory".to_string(), line.to_string()))
            .collect::<Vec<_>>()
    } else {
        dialogue
    };

    let mut cards = Vec::new();
    let mut seen = HashSet::new();
    for (speaker, line) in source_claims {
        for claim in split_atomic_claims(&line) {
            if !is_high_signal_atomic_claim(&claim) {
                continue;
            }
            let key = format!("{}|{}", speaker.to_ascii_lowercase(), claim.to_ascii_lowercase());
            if !seen.insert(key) {
                continue;
            }
            let card_idx = cards.len();
            if card_idx >= 4 {
                return cards;
            }

            let card_kind = if preference_signal_strength(&claim, &payload.relations).is_some() {
                "preference"
            } else {
                "fact"
            };
            let fact_key = infer_fact_key(&claim);
            let subject = if speaker.eq_ignore_ascii_case("memory") {
                payload.entity_id.clone()
            } else {
                speaker.clone()
            };
            let text = if speaker.eq_ignore_ascii_case("memory") {
                format!("Atomic memory card: {}", claim)
            } else {
                format!("Atomic memory card: {} said {}", speaker, claim)
            };

            cards.push(IngestPayload {
                entity_id: payload.entity_id.clone(),
                memory_id: format!(
                    "{}::{}::{}",
                    entity_id,
                    session_id,
                    2_200_000 + turn_index * 50 + card_idx
                ),
                timestamp: payload.timestamp,
                textual_content: text,
                relations: payload.relations.clone(),
                kind: Some(card_kind.to_string()),
                fact_key,
                source_memory_id: Some(payload.memory_id.clone()),
                index_semantic: Some(true),
                enable_semantic_dedup: Some(true),
                enable_consolidation: Some(false),
                content_type: payload.content_type.clone(),
                fact_operation: Some("derive".to_string()),
                fact_confidence: Some(0.90),
                fact_subject: Some(subject),
                fact_predicate: None,
                fact_object: Some(claim),
                ..Default::default()
            });
        }
    }

    cards
}

pub fn preference_signal_strength(
    text: &str,
    relations: &[(String, String, String)],
) -> Option<f32> {
    let lower = text.to_ascii_lowercase();
    let mut strength = 0.0f32;

    let weighted_phrases = [
        ("love ", 1.0),
        ("loves ", 1.0),
        ("favorite", 0.95),
        ("prefer ", 0.9),
        ("prefers ", 0.9),
        ("enjoy ", 0.8),
        ("enjoys ", 0.8),
        ("like ", 0.7),
        ("likes ", 0.7),
        ("hate ", 0.85),
        ("hates ", 0.85),
        ("avoid ", 0.8),
        ("avoids ", 0.8),
    ];
    for (phrase, weight) in weighted_phrases {
        if lower.contains(phrase) {
            strength = strength.max(weight);
        }
    }

    for (_, predicate, _) in relations {
        let pred = predicate.trim().to_ascii_lowercase();
        strength = strength.max(match pred.as_str() {
            "love" | "loves" | "favorite" => 1.0,
            "prefer" | "prefers" => 0.9,
            "enjoy" | "enjoys" | "likes" | "like" => 0.8,
            "hate" | "hates" | "avoid" | "avoids" | "dislikes" => 0.85,
            _ => 0.0,
        });
    }

    (strength > 0.0).then_some(strength)
}

pub fn extract_retrospective_reference_query(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let patterns = [
        "remember when ",
        "remember that time ",
        "last time we ",
        "back in ",
        "that time at ",
        "that trip when ",
    ];

    for pattern in patterns {
        if let Some(pos) = lower.find(pattern) {
            let start = pos + pattern.len();
            let rest = text[start..].trim();
            let candidate = rest.split(['.', '!', '?']).next().unwrap_or("").trim();
            if candidate.len() >= 12 {
                return Some(candidate.to_string());
            }
        }
    }

    None
}
