use crate::api::types::IngestPayload;
use crate::api::utils::{
    extract_named_phrases, extract_temporal_terms, has_token, normalize_alpha_tokens,
    normalize_fact_text, singularize_token,
};
use crate::core::memory_id::MemoryId;
use crate::heuristics::{benchmark_tuned_rules, Profile};

use super::dialogue::strip_leading_bracketed_prefixes;

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
    infer_fact_key_with_profile(text, Profile::Generic)
}

pub fn infer_fact_key_with_profile(text: &str, profile: Profile) -> Option<String> {
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
    if benchmark_tuned_rules(profile)
        && (normalized.contains("last name before") || normalized.contains("old name"))
    {
        return Some("previous_last_name".to_string());
    }
    if normalized.contains("previous occupation") {
        return Some("previous_occupation".to_string());
    }
    if benchmark_tuned_rules(profile)
        && has_token(&tokens, "commute")
        && (has_token(&tokens, "take") || has_token(&tokens, "takes"))
    {
        return Some("commute_duration".to_string());
    }
    if benchmark_tuned_rules(profile)
        && normalized.contains("internet plan")
        && (has_token(&tokens, "mbps")
            || has_token(&tokens, "speed")
            || has_token(&tokens, "upgraded"))
    {
        return Some("internet_plan_speed".to_string());
    }
    if benchmark_tuned_rules(profile)
        && has_token(&tokens, "spotify")
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

/// Memory cards for the facts an extractor finds in `payload`.
pub fn build_atomic_memory_card_payloads(payload: &IngestPayload) -> Vec<IngestPayload> {
    let extractor = crate::extract::extractor();
    let ctx = crate::extract::ExtractCtx {
        entity_id: &payload.entity_id,
        timestamp_ms: payload.timestamp,
        relations: &payload.relations,
    };
    build_cards_from_facts(payload, &extractor.extract(&payload.textual_content, &ctx))
}

fn build_cards_from_facts(
    payload: &IngestPayload,
    facts: &[crate::extract::ExtractedFact],
) -> Vec<IngestPayload> {
    facts
        .iter()
        .enumerate()
        .map(|(card_idx, fact)| {
            let text = if fact.speaker.eq_ignore_ascii_case("memory") {
                format!("Atomic memory card: {}", fact.object)
            } else {
                format!("Atomic memory card: {} said {}", fact.speaker, fact.object)
            };
            IngestPayload {
                entity_id: payload.entity_id.clone(),
                memory_id: MemoryId::derived_from(&payload.memory_id, &format!("card{card_idx}")),
                timestamp: payload.timestamp,
                textual_content: text,
                relations: payload.relations.clone(),
                kind: Some(if fact.is_preference { "preference" } else { "fact" }.to_string()),
                fact_key: fact.fact_key.clone(),
                source_memory_id: Some(payload.memory_id.clone()),
                index_semantic: Some(true),
                enable_semantic_dedup: Some(true),
                enable_consolidation: Some(false),
                content_type: payload.content_type.clone(),
                fact_operation: Some("derive".to_string()),
                fact_confidence: Some(fact.confidence),
                fact_subject: Some(fact.subject.clone()),
                fact_predicate: fact.predicate.clone(),
                fact_object: Some(fact.object.clone()),
                ..Default::default()
            }
        })
        .collect()
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
#[cfg(test)]
mod moved_tests {
    use super::*;

    #[test]
    fn test_infer_fact_key_marriage() {
        assert_eq!(infer_fact_key("I am married"), Some("relationship_status".to_string()));
    }

    #[test]
    fn test_infer_fact_key_children() {
        assert_eq!(infer_fact_key("I have two children"), Some("children_count".to_string()));
    }

    #[test]
    fn test_infer_fact_key_nickname() {
        assert_eq!(infer_fact_key("my nickname is Bob"), Some("nickname".to_string()));
    }

    #[test]
    fn test_infer_fact_key_hobbies() {
        assert_eq!(infer_fact_key("my hobbies include reading"), Some("hobbies".to_string()));
    }

    #[test]
    fn test_infer_fact_key_purchase() {
        assert_eq!(infer_fact_key("I bought a car"), Some("purchase".to_string()));
    }

    #[test]
    fn test_infer_fact_key_recipe() {
        assert_eq!(infer_fact_key("I found a great recipe"), Some("recipe".to_string()));
    }

    #[test]
    fn test_infer_fact_key_research() {
        assert_eq!(infer_fact_key("I research AI"), Some("research_topic".to_string()));
    }

    #[test]
    fn test_infer_fact_key_certificate() {
        assert_eq!(infer_fact_key("I got a certificate"), Some("certificate".to_string()));
    }

    #[test]
    fn test_infer_fact_key_favorite_team() {
        assert_eq!(
            infer_fact_key("my favorite team is the Lakers"),
            Some("favorite_team".to_string())
        );
    }

    #[test]
    fn test_infer_fact_key_degree() {
        assert_eq!(infer_fact_key("graduated with a degree"), Some("degree".to_string()));
    }

    #[test]
    fn test_infer_fact_key_previous_last_name() {
        assert_eq!(
            infer_fact_key_with_profile("my old name is Smith", Profile::LegacyTuned),
            Some("previous_last_name".to_string())
        );
        assert_ne!(
            infer_fact_key_with_profile("my old name is Smith", Profile::Generic),
            Some("previous_last_name".to_string())
        );
    }

    #[test]
    fn test_infer_fact_key_previous_occupation() {
        assert_eq!(
            infer_fact_key("previous occupation was teacher"),
            Some("previous_occupation".to_string())
        );
    }

    #[test]
    fn test_infer_fact_key_commute_duration() {
        assert_eq!(
            infer_fact_key_with_profile("my commute takes 30 minutes", Profile::LegacyTuned),
            Some("commute_duration".to_string())
        );
        assert_ne!(
            infer_fact_key_with_profile("my commute takes 30 minutes", Profile::Generic),
            Some("commute_duration".to_string())
        );
    }

    #[test]
    fn test_infer_fact_key_internet_plan() {
        assert_eq!(
            infer_fact_key_with_profile(
                "I upgraded my internet plan to 500 mbps",
                Profile::LegacyTuned,
            ),
            Some("internet_plan_speed".to_string())
        );
        assert_ne!(
            infer_fact_key_with_profile(
                "I upgraded my internet plan to 500 mbps",
                Profile::Generic,
            ),
            Some("internet_plan_speed".to_string())
        );
    }

    #[test]
    fn test_infer_fact_key_spotify_playlist() {
        assert_eq!(
            infer_fact_key_with_profile(
                "I created a spotify playlist named vibes",
                Profile::LegacyTuned,
            ),
            Some("spotify_playlist_name".to_string())
        );
        assert_ne!(
            infer_fact_key_with_profile(
                "I created a spotify playlist named vibes",
                Profile::Generic,
            ),
            Some("spotify_playlist_name".to_string())
        );
    }

    #[test]
    fn test_infer_fact_key_called_triggers_nickname() {
        assert_eq!(infer_fact_key("people called me Bob"), Some("nickname".to_string()));
    }

    #[test]
    fn test_infer_fact_key_martial_arts() {
        assert_eq!(infer_fact_key("I practice karate"), Some("martial_arts".to_string()));
    }

    #[test]
    fn test_infer_fact_key_martial_arts_phrase() {
        assert_eq!(infer_fact_key("I study martial arts"), Some("martial_arts".to_string()));
    }

    #[test]
    fn test_infer_fact_key_spend_price() {
        let result = infer_fact_key("I spent $50 on books").unwrap();
        assert!(result.ends_with("_price"));
    }

    #[test]
    fn test_infer_fact_key_identity_residence() {
        assert_eq!(infer_fact_key("I live in New York"), Some("residence".to_string()));
    }

    #[test]
    fn test_infer_fact_key_identity_employer() {
        assert_eq!(infer_fact_key("I work at Google"), Some("employer".to_string()));
    }

    #[test]
    fn test_infer_fact_key_identity_occupation() {
        assert_eq!(infer_fact_key("I am a doctor"), Some("occupation".to_string()));
    }

    #[test]
    fn test_infer_fact_key_identity_school() {
        assert_eq!(infer_fact_key("I study at MIT"), Some("school".to_string()));
    }

    #[test]
    fn test_infer_fact_key_prefixed_user_fact_stripped() {
        assert_eq!(
            infer_fact_key("User fact: I am married"),
            Some("relationship_status".to_string())
        );
    }

    #[test]
    fn test_infer_fact_key_irrelevant_returns_none() {
        assert_eq!(infer_fact_key("The sky is blue"), None);
    }

    #[test]
    fn test_infer_fact_key_empty_none() {
        assert_eq!(infer_fact_key(""), None);
    }

    #[test]
    fn test_infer_fact_key_has_new_and_got_returns_purchase() {
        assert_eq!(infer_fact_key("I got a new phone"), Some("purchase".to_string()));
    }

    #[test]
    fn test_infer_fact_key_child_count_no_have_verb() {
        assert_eq!(infer_fact_key("the children are playing"), None);
    }

    #[test]
    fn test_is_high_signal_atomic_claim_temporal_and_relation() {
        assert!(is_high_signal_atomic_claim("I visited New York last year"));
    }

    #[test]
    fn test_is_high_signal_atomic_claim_named_and_relation() {
        assert!(is_high_signal_atomic_claim("Alice works at Google"));
    }

    #[test]
    fn test_is_high_signal_atomic_claim_personal_signal() {
        assert!(is_high_signal_atomic_claim("I work at Google"));
    }

    #[test]
    fn test_is_high_signal_atomic_claim_no_relation_word() {
        assert!(!is_high_signal_atomic_claim("I am fine"));
    }

    #[test]
    fn test_is_high_signal_atomic_claim_empty() {
        assert!(!is_high_signal_atomic_claim(""));
    }

    #[test]
    fn test_is_high_signal_atomic_claim_he_she_personal() {
        assert!(is_high_signal_atomic_claim("He works at Microsoft"));
    }

    #[test]
    fn test_is_high_signal_atomic_claim_missing_relation() {
        assert!(!is_high_signal_atomic_claim("I the ball"));
    }

    #[test]
    fn test_sanitize_key_parts_basic() {
        assert_eq!(
            sanitize_key_parts(&["my", "favorite", "color"]),
            Some("favorite_color".to_string())
        );
    }

    #[test]
    fn test_sanitize_key_parts_filters_stop_words() {
        assert_eq!(sanitize_key_parts(&["the", "a", "an", "of"]), None);
    }

    #[test]
    fn test_sanitize_key_parts_filters_numbers() {
        assert_eq!(sanitize_key_parts(&["hello", "123", "world"]), Some("hello_world".to_string()));
    }

    #[test]
    fn test_sanitize_key_parts_filters_punctuation() {
        assert_eq!(sanitize_key_parts(&["hello!", "world?"]), Some("hello_world".to_string()));
    }

    #[test]
    fn test_sanitize_key_parts_all_filtered_returns_none() {
        assert_eq!(sanitize_key_parts(&["the", "a", "123", "!@#"]), None);
    }

    #[test]
    fn test_sanitize_key_parts_empty_input() {
        assert_eq!(sanitize_key_parts(&[]), None);
    }

    #[test]
    fn test_sanitize_key_parts_owned() {
        let parts = vec!["my".to_string(), "test".to_string()];
        assert_eq!(sanitize_key_parts_owned(&parts), Some("test".to_string()));
    }

    #[test]
    fn test_is_numericish_digits() {
        assert!(is_numericish("123"));
    }

    #[test]
    fn test_is_numericish_with_dollar() {
        assert!(is_numericish("$50"));
    }

    #[test]
    fn test_is_numericish_with_decimal() {
        assert!(is_numericish("3.14"));
    }

    #[test]
    fn test_is_numericish_with_comma() {
        assert!(is_numericish("1,000"));
    }

    #[test]
    fn test_is_numericish_with_percent() {
        assert!(is_numericish("99%"));
    }

    #[test]
    fn test_is_numericish_text_false() {
        assert!(!is_numericish("hello"));
    }

    #[test]
    fn test_is_numericish_empty_false() {
        assert!(!is_numericish(""));
    }

    #[test]
    fn test_split_atomic_claims_basic() {
        let claims = split_atomic_claims("I like pizza. I have a dog.");
        assert_eq!(claims.len(), 2);
    }

    #[test]
    fn test_split_atomic_claims_filters_short() {
        let claims = split_atomic_claims("Hi. I like pizza. Ok.");
        assert_eq!(claims.len(), 1);
        assert!(claims[0].contains("I like pizza"));
    }

    #[test]
    fn test_split_atomic_claims_strips_bracketed() {
        let claims = split_atomic_claims("[meta] I like pizza.");
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0], "I like pizza");
    }

    #[test]
    fn test_split_atomic_claims_semicolons() {
        let claims = split_atomic_claims("I like pizza; I have a dog");
        assert_eq!(claims.len(), 2);
    }

    #[test]
    fn test_build_contextual_key_basic() {
        let base = vec!["coffee".to_string()];
        let key = build_contextual_key(&[], &base, None);
        assert_eq!(key, Some("coffee".to_string()));
    }

    #[test]
    fn test_build_contextual_key_with_suffix() {
        let base = vec!["coffee".to_string()];
        let key = build_contextual_key(&[], &base, Some("price"));
        assert_eq!(key, Some("coffee_price".to_string()));
    }

    #[test]
    fn test_build_contextual_key_with_context() {
        let ctx = vec!["morning".to_string()];
        let base = vec!["coffee".to_string()];
        let key = build_contextual_key(&ctx, &base, None);
        assert_eq!(key, Some("morning_coffee".to_string()));
    }

    #[test]
    fn test_build_contextual_key_removes_stop_words() {
        let base = vec!["the".to_string(), "coffee".to_string()];
        let key = build_contextual_key(&[], &base, None);
        assert_eq!(key, Some("coffee".to_string()));
    }
}
