use std::collections::HashSet;

use super::types::CoverageFacet;
use super::types::QueryIntent;
use super::types::QueryRequirement;
use crate::api::utils::{
    dedupe_preserve_order, extract_salient_terms, extract_temporal_terms, has_token,
    is_low_signal_keyword, normalize_alpha_tokens, normalize_fact_text, singularize_token,
};
use crate::fts::tokenize_for_similarity;

pub fn build_keyword_query(query: &str) -> Option<String> {
    let generic_drop = [
        "what",
        "when",
        "where",
        "who",
        "why",
        "how",
        "would",
        "could",
        "should",
        "did",
        "does",
        "do",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "likely",
        "might",
        "will",
        "can",
        "have",
        "has",
        "had",
        "kind",
        "kinds",
        "type",
        "types",
        "sort",
        "sorts",
        "thing",
        "things",
        "person",
        "people",
        "event",
        "events",
        "activity",
        "activities",
        "cause",
        "causes",
        "item",
        "items",
        "area",
        "areas",
        "status",
        "focus",
    ];

    let mut tokens = tokenize_for_similarity(query)
        .into_iter()
        .map(|token| singularize_token(token.as_str()))
        .collect::<Vec<_>>();

    if tokens.len() > 5 {
        tokens.retain(|token| !generic_drop.contains(&token.as_str()));
    }

    let keyword_query = dedupe_preserve_order(tokens).join(" ");
    (!keyword_query.is_empty()).then_some(keyword_query)
}

fn push_expansion_terms(out: &mut Vec<String>, terms: &[&str]) {
    for term in terms {
        push_expansion_term(out, term, 3);
    }
}

fn push_expansion_term(out: &mut Vec<String>, term: &str, min_len: usize) {
    let term = singularize_token(term.trim().to_ascii_lowercase().as_str());
    if term.len() >= min_len && !out.iter().any(|existing| existing == &term) {
        out.push(term);
    }
}

struct ExpansionRule {
    trigger_tokens: &'static [&'static str],
    expansions: &'static [&'static str],
}

const EXPANSION_RULES: &[ExpansionRule] = &[
    ExpansionRule {
        trigger_tokens: &["kid", "kids", "child", "children", "daughter", "son", "family"],
        expansions: &[
            "kid", "child", "children", "daughter", "son", "family", "museum", "dinosaur", "park",
            "camping", "nature", "love", "fun",
        ],
    },
    ExpansionRule {
        trigger_tokens: &["destress", "de-stress", "relax", "unwind", "stress"],
        expansions: &[
            "destress", "relax", "unwind", "stress", "relief", "escape", "dance", "music", "yoga",
            "nature", "art",
        ],
    },
    ExpansionRule {
        trigger_tokens: &["volunteer", "volunteering", "charity", "donate", "donation"],
        expansions: &[
            "volunteer",
            "volunteering",
            "charity",
            "community",
            "church",
            "shelter",
            "donate",
            "homeless",
            "fundraiser",
        ],
    },
    ExpansionRule {
        trigger_tokens: &["patriotic", "political", "politics", "office", "vote", "campaign"],
        expansions: &[
            "country",
            "community",
            "office",
            "politics",
            "campaign",
            "vote",
            "public",
            "service",
            "serve",
            "serving",
            "proud",
            "volunteer",
            "rights",
            "activist",
        ],
    },
    ExpansionRule {
        trigger_tokens: &["pet", "dog", "animal", "turtle"],
        expansions: &[
            "pet",
            "dog",
            "animal",
            "companion",
            "family",
            "training",
            "shelter",
            "veterinarian",
        ],
    },
    ExpansionRule {
        trigger_tokens: &["workshop", "course", "class", "training", "mentor", "mentorship"],
        expansions: &["workshop", "course", "class", "training", "mentor", "program"],
    },
    ExpansionRule {
        trigger_tokens: &["paint", "painting", "pottery", "art", "artist", "draw", "drawing"],
        expansions: &["art", "paint", "painting", "draw", "drawing", "creative", "pottery"],
    },
    ExpansionRule {
        trigger_tokens: &["promotion", "promoted", "promote"],
        expansions: &["promoted", "promotion", "role", "position", "manager", "lead"],
    },
    ExpansionRule {
        trigger_tokens: &["electronic", "electronics", "device", "smartwatch", "watch"],
        expansions: &[
            "device",
            "phone",
            "laptop",
            "computer",
            "watch",
            "smartwatch",
            "fitness",
            "tracker",
            "broken",
            "issue",
        ],
    },
    ExpansionRule {
        trigger_tokens: &["digestive", "stomach", "indigestion", "nausea"],
        expansions: &["digestive", "stomach", "indigestion", "nausea", "sick", "ache"],
    },
    ExpansionRule {
        trigger_tokens: &["popular", "fanbase", "famous", "audience", "followers"],
        expansions: &["popular", "fanbase", "audience", "follower", "brand", "global", "music"],
    },
    ExpansionRule {
        trigger_tokens: &["health", "fitness", "lifestyle", "stress", "challenge", "cope"],
        expansions: &[
            "health",
            "fitness",
            "exercise",
            "diet",
            "stress",
            "challenge",
            "cope",
            "wellbeing",
            "mental",
            "nature",
            "creative",
        ],
    },
];

pub fn build_query_expansion_terms(
    query: &str,
    slot_key: Option<&str>,
    intent: QueryIntent,
    subject_entities: &[String],
) -> Vec<String> {
    let lower = query.to_ascii_lowercase();
    let tokens = normalize_alpha_tokens(query)
        .into_iter()
        .map(|token| singularize_token(&token))
        .collect::<Vec<_>>();
    let has = |needle: &str| tokens.iter().any(|token| token == needle);
    let has_any_singular = |needles: &[&str]| needles.iter().any(|needle| has(needle));
    let mut terms = Vec::new();

    match slot_key {
        Some("relationship_status") => {
            push_expansion_terms(&mut terms, &["married", "single", "dating", "partner"]);
        }
        Some("children_count") => {
            push_expansion_terms(&mut terms, &["child", "children", "kid", "daughter", "son"]);
        }
        Some("martial_arts") => {
            push_expansion_terms(&mut terms, &["karate", "judo", "boxing", "taekwondo"]);
        }
        Some("certificate") => {
            push_expansion_terms(&mut terms, &["certificate", "certified", "course", "training"]);
        }
        Some("financial_status") => {
            push_expansion_terms(
                &mut terms,
                &["money", "cash", "income", "afford", "expensive", "earning"],
            );
        }
        Some("health_issue") => {
            push_expansion_terms(
                &mut terms,
                &["health", "symptom", "doctor", "pain", "injury", "diagnosis"],
            );
        }
        Some("favorite_team") => {
            push_expansion_terms(&mut terms, &["favorite", "team", "support", "fan"]);
        }
        Some("nickname") => {
            push_expansion_terms(
                &mut terms,
                &["nickname", "called", "call", "name", "named", "short"],
            );
            for entity in subject_entities {
                let alpha = entity
                    .to_ascii_lowercase()
                    .chars()
                    .filter(|ch| ch.is_ascii_alphabetic())
                    .collect::<String>();
                if alpha.len() >= 4 {
                    push_expansion_term(&mut terms, &alpha[..2], 2);
                    push_expansion_term(&mut terms, &alpha[..3], 3);
                }
            }
        }
        Some("hobbies") => {
            push_expansion_terms(
                &mut terms,
                &["hobby", "activity", "enjoy", "like", "interest", "practice"],
            );
        }
        Some("purchase") => {
            push_expansion_terms(
                &mut terms,
                &["buy", "bought", "purchase", "new", "recently", "acquired"],
            );
        }
        Some("travel_location") => {
            push_expansion_terms(
                &mut terms,
                &["travel", "trip", "visit", "visited", "vacation", "city", "state"],
            );
        }
        _ => {}
    }

    if slot_key.is_none() {
        for entity in subject_entities {
            let alpha = entity
                .to_ascii_lowercase()
                .chars()
                .filter(|ch| ch.is_ascii_alphabetic())
                .collect::<String>();
            if alpha.len() >= 4 {
                push_expansion_term(&mut terms, &alpha[..2], 2);
                push_expansion_term(&mut terms, &alpha[..3], 3);
            }
        }
    }

    for rule in EXPANSION_RULES {
        if has_any_singular(rule.trigger_tokens) {
            push_expansion_terms(&mut terms, rule.expansions);
        }
    }

    if has_any_singular(&["education", "edu", "educaton", "field", "career", "job", "profession"])
        || lower.contains("career path")
        || lower.contains("career option")
        || lower.contains("future job")
    {
        push_expansion_terms(
            &mut terms,
            &["education", "study", "school", "career", "job", "profession", "training"],
        );
    }

    if has_any_singular(&["book", "books", "bookshelf", "library", "read", "reading"])
        || lower.contains("dr. seuss")
        || lower.contains("dr seuss")
    {
        push_expansion_terms(
            &mut terms,
            &[
                "book",
                "library",
                "story",
                "stories",
                "read",
                "reading",
                "classic",
                "educational",
                "children",
                "kid",
            ],
        );
    }

    if lower.contains("national park") || has_any_singular(&["outdoor", "outdoors", "nature"]) {
        push_expansion_terms(
            &mut terms,
            &[
                "nature", "outdoor", "outdoors", "camping", "hiking", "trail", "mountain", "beach",
                "campfire", "wildlife",
            ],
        );
    }
    if lower.contains("theme park") || lower.contains("amusement park") {
        push_expansion_terms(
            &mut terms,
            &["theme", "amusement", "ride", "roller", "coaster", "fun"],
        );
    }

    let location_trigger = has_any_singular(&[
        "state", "city", "country", "location", "place", "travel", "trip", "visit",
    ]);
    let living_trigger =
        has_any_singular(&["live", "lives", "living", "located", "home", "near", "close"]);
    if living_trigger
        && has_any_singular(&["state", "city", "country", "location", "place", "beach", "mountain"])
    {
        push_expansion_terms(
            &mut terms,
            &[
                "live",
                "lives",
                "living",
                "home",
                "local",
                "nearby",
                "neighborhood",
                "beach",
                "mountain",
                "state",
                "city",
            ],
        );
    } else if location_trigger {
        push_expansion_terms(
            &mut terms,
            &[
                "travel", "trip", "visit", "visited", "went", "vacation", "city", "state",
                "country", "place",
            ],
        );
    }

    if lower.contains("another country")
        || lower.contains("move to")
        || lower.contains("moving")
        || lower.contains("relocate")
    {
        push_expansion_terms(
            &mut terms,
            &[
                "move", "moving", "relocate", "abroad", "country", "overseas", "travel",
                "language", "culture", "open",
            ],
        );
    }

    if lower.contains("feel about")
        || lower.contains("felt about")
        || lower.contains("think about")
        || lower.contains("say about")
        || has_any_singular(&["supporting", "support", "supported"])
    {
        push_expansion_terms(
            &mut terms,
            &[
                "feel",
                "felt",
                "think",
                "support",
                "proud",
                "grateful",
                "thankful",
                "strength",
                "motivation",
                "rock",
                "family",
            ],
        );
    }

    if matches!(intent, QueryIntent::Inference | QueryIntent::Recommendation) {
        push_expansion_terms(
            &mut terms,
            &["prefer", "enjoy", "like", "value", "believe", "support", "interest", "passion"],
        );
    }

    terms.truncate(24);
    terms
}

pub fn build_expansion_query(entities: &[String], expansion_terms: &[String]) -> Option<String> {
    if expansion_terms.is_empty() {
        return None;
    }
    let mut parts = entities.iter().take(2).cloned().collect::<Vec<_>>();
    parts.extend(expansion_terms.iter().take(10).cloned());
    (!parts.is_empty()).then_some(parts.join(" "))
}

pub fn build_hypothetical_semantic_queries(
    query: &str,
    entities: &[String],
    cross_entity: bool,
    intent: QueryIntent,
) -> Vec<String> {
    let mut expanded = Vec::new();
    let entity_span = if entities.is_empty() {
        "the people in this dialogue".to_string()
    } else {
        entities.join(" and ")
    };

    expanded.push(format!("A memory snippet about {entity_span} that answers: {query}"));

    if cross_entity {
        expanded.push(format!(
            "A shared event involving {} with concrete details, dates, and context",
            entities.join(" and ")
        ));
    }

    match intent {
        QueryIntent::Inference => expanded.push(format!(
            "Evidence describing preferences, values, attitudes, and likely opinions for {entity_span}"
        )),
        QueryIntent::PeripheralMention => expanded.push(format!(
            "An offhand mention about nicknames, names, childhood details, or side facts for {entity_span}"
        )),
        QueryIntent::TemporalAggregation => expanded.push(format!(
            "A timeline statement with explicit dates and time references for {entity_span}"
        )),
        QueryIntent::NumericAggregation => expanded.push(format!(
            "A countable statement with quantities, frequency, or totals for {entity_span}"
        )),
        QueryIntent::Recommendation | QueryIntent::General => {}
    }

    dedupe_preserve_order(expanded)
}

pub fn build_inference_semantic_hints(query: &str, entities: &[String]) -> Vec<String> {
    let mut hints = Vec::new();
    hints.push(format!("{query} based on preferences values opinions personality behaviors"));
    for entity in entities {
        hints.push(format!("{entity} preferences beliefs values personality"));
    }
    hints
}

pub fn build_peripheral_fts_query(query: &str, entities: &[String]) -> Option<String> {
    let cue_tokens =
        ["nickname", "called", "named", "name", "child", "childhood", "as", "kid", "pet", "middle"];
    let tokens = normalize_alpha_tokens(query);
    let mut selected = tokens
        .into_iter()
        .filter(|t| cue_tokens.contains(&t.as_str()) || t.len() >= 5)
        .take(4)
        .collect::<Vec<_>>();

    for entity in entities.iter().take(2) {
        selected.push(entity.to_ascii_lowercase());
    }

    let selected = dedupe_preserve_order(selected);
    if selected.is_empty() {
        return None;
    }
    Some(selected.into_iter().map(|t| format!("\"{t}\"")).collect::<Vec<_>>().join(" "))
}

pub fn build_fact_slot_queries(
    query: &str,
    entities: &[String],
    slot_key: &str,
) -> (Vec<String>, Vec<String>) {
    let slot = humanize_slot_key(slot_key);
    let mut semantic =
        vec![format!("canonical fact {slot} {query}"), format!("fact about {slot} {query}")];
    let mut lexical = vec![format!("\"{slot}\""), format!("\"canonical fact\" \"{slot}\"")];

    for entity in entities.iter().take(2) {
        semantic.push(format!("{entity} canonical fact {slot}"));
        lexical.push(format!("\"{entity}\" \"{slot}\""));
    }

    (dedupe_preserve_order(semantic), dedupe_preserve_order(lexical))
}

pub fn build_cross_entity_subqueries(
    query: &str,
    entities: &[String],
) -> (Vec<String>, Vec<String>) {
    if entities.len() < 2 {
        return (Vec::new(), Vec::new());
    }

    let mut semantic = Vec::new();
    let mut lexical = Vec::new();
    for entity in entities {
        semantic.push(format!("{entity} {query}"));
        lexical.push(format!("\"{entity}\" {query}"));
    }

    let pair = entities.iter().take(2).cloned().collect::<Vec<_>>();
    if pair.len() == 2 {
        semantic.push(format!("{} {} shared common", pair[0], pair[1]));
        lexical.push(format!("\"{}\" \"{}\"", pair[0], pair[1]));
    }

    (semantic, lexical)
}

pub fn build_purchase_queries(query: &str, entities: &[String]) -> (Vec<String>, Vec<String>) {
    let mut semantic = vec![
        format!("{query} bought purchased acquired new item"),
        format!("purchase bought recently new item {query}"),
    ];
    let mut lexical = vec![
        format!("{query} bought purchased acquired"),
        "\"bought\" \"new\"".to_string(),
        "\"bought\" \"recently\"".to_string(),
        "\"purchased\" \"recently\"".to_string(),
    ];

    for entity in entities.iter().take(2) {
        semantic.push(format!("{entity} bought purchased acquired new item"));
        lexical.push(format!("\"{entity}\" \"bought\""));
        lexical.push(format!("\"{entity}\" \"purchased\""));
    }

    (dedupe_preserve_order(semantic), dedupe_preserve_order(lexical))
}

pub fn humanize_slot_key(slot_key: &str) -> String {
    slot_key.replace('_', " ")
}

pub fn infer_query_fact_key(query: &str) -> Option<String> {
    let lower = query.to_ascii_lowercase();
    let tokens = normalize_alpha_tokens(query);

    if lower.contains("relationship status")
        || ["married", "single", "divorced", "engaged", "dating"]
            .iter()
            .any(|needle| lower.contains(needle))
    {
        return Some("relationship_status".to_string());
    }
    if lower.contains("nickname")
        || (lower.contains("what is the name of") && lower.contains("pet"))
        || lower.contains("called")
    {
        return Some("nickname".to_string());
    }
    if (has_token(&tokens, "children") || has_token(&tokens, "kids") || has_token(&tokens, "child"))
        && lower.contains("how many")
    {
        return Some("children_count".to_string());
    }
    if lower.contains("martial art")
        || ["karate", "judo", "taekwondo", "jiujitsu", "jitsu", "boxing"]
            .iter()
            .any(|needle| lower.contains(needle))
    {
        return Some("martial_arts".to_string());
    }
    if has_token(&tokens, "research") {
        return Some("research_topic".to_string());
    }
    if has_token(&tokens, "certificate") || has_token(&tokens, "certified") {
        return Some("certificate".to_string());
    }
    if has_token(&tokens, "financial") && has_token(&tokens, "status") {
        return Some("financial_status".to_string());
    }
    if has_token(&tokens, "health") && has_token(&tokens, "problem") {
        return Some("health_issue".to_string());
    }
    if has_token(&tokens, "team")
        && (has_token(&tokens, "support") || has_token(&tokens, "favorite"))
    {
        return Some("favorite_team".to_string());
    }
    if has_token(&tokens, "recipe") || lower.contains("ice cream") {
        return Some("recipe".to_string());
    }
    if has_token(&tokens, "hobby")
        || lower.contains("activities does")
        || lower.contains("activities has")
    {
        return Some("hobbies".to_string());
    }
    if is_purchase_query(query) {
        return Some("purchase".to_string());
    }
    if has_token(&tokens, "state")
        && (has_token(&tokens, "visit")
            || has_token(&tokens, "travel")
            || has_token(&tokens, "vacation"))
    {
        return Some("travel_location".to_string());
    }

    None
}

pub fn is_purchase_query(query: &str) -> bool {
    let tokens = normalize_alpha_tokens(query);
    let lower = query.to_ascii_lowercase();
    has_token(&tokens, "buy")
        || has_token(&tokens, "bought")
        || has_token(&tokens, "purchase")
        || has_token(&tokens, "purchased")
        || has_token(&tokens, "acquired")
        || ((has_token(&tokens, "get") || has_token(&tokens, "got"))
            && (has_token(&tokens, "new")
                || has_token(&tokens, "recently")
                || lower.contains("what did")))
}

pub fn make_coverage_facet(text: String, entities: Vec<String>) -> Option<CoverageFacet> {
    let normalized = normalize_fact_text(&text);
    if normalized.is_empty() {
        return None;
    }
    Some(CoverageFacet {
        text: normalized.clone(),
        lexical_terms: dedupe_preserve_order(
            tokenize_for_similarity(&normalized)
                .into_iter()
                .map(|token| singularize_token(&token))
                .filter(|token| !is_low_signal_keyword(token))
                .collect(),
        ),
        temporal_terms: extract_temporal_terms(&normalized),
        entities: dedupe_preserve_order(entities),
    })
}

pub fn make_query_requirement(
    text: String,
    entities: Vec<String>,
    require_all_entities: bool,
) -> Option<QueryRequirement> {
    let normalized = normalize_fact_text(&text);
    if normalized.is_empty() {
        return None;
    }
    Some(QueryRequirement {
        text: normalized.clone(),
        lexical_terms: dedupe_preserve_order(
            tokenize_for_similarity(&normalized)
                .into_iter()
                .map(|token| singularize_token(&token))
                .filter(|token| !is_low_signal_keyword(token))
                .collect(),
        ),
        temporal_terms: extract_temporal_terms(&normalized),
        entities: dedupe_preserve_order(entities),
        require_all_entities,
    })
}

pub fn build_coverage_facets(
    query: &str,
    keyword_query: Option<&str>,
    subject_entities: &[String],
    cross_entity: bool,
    ordinal_rank: Option<usize>,
    slot_key: Option<&str>,
    expansion_terms: &[String],
) -> Vec<CoverageFacet> {
    let mut facets = Vec::new();
    let mut seen = HashSet::new();
    let mut push_facet = |facet: CoverageFacet| {
        let key = format!(
            "{}|{}",
            facet.text.to_ascii_lowercase(),
            facet.entities.join("|").to_ascii_lowercase()
        );
        if seen.insert(key) {
            facets.push(facet);
        }
    };
    if let Some(facet) = make_coverage_facet(query.to_string(), subject_entities.to_vec()) {
        push_facet(facet);
    }
    if let Some(keyword_query) = keyword_query {
        if !keyword_query.eq_ignore_ascii_case(query) {
            if let Some(facet) =
                make_coverage_facet(keyword_query.to_string(), subject_entities.to_vec())
            {
                push_facet(facet);
            }
        }
    }
    if let Some(slot_key) = slot_key {
        let slot = humanize_slot_key(slot_key);
        if let Some(facet) =
            make_coverage_facet(format!("canonical fact {slot}"), subject_entities.to_vec())
        {
            push_facet(facet);
        }
    }
    if let Some(expansion_query) = build_expansion_query(subject_entities, expansion_terms) {
        if let Some(facet) = make_coverage_facet(expansion_query, subject_entities.to_vec()) {
            push_facet(facet);
        }
    }
    if cross_entity {
        let base = keyword_query.unwrap_or(query);
        for entity in subject_entities.iter().take(3) {
            if let Some(facet) =
                make_coverage_facet(format!("{entity} {base}"), vec![entity.clone()])
            {
                push_facet(facet);
            }
        }
    }
    if ordinal_rank.is_some() {
        let stripped = crate::api::plan::intent::strip_ordinal_tokens(query);
        if !stripped.is_empty() && !stripped.eq_ignore_ascii_case(query) {
            if let Some(facet) = make_coverage_facet(stripped, subject_entities.to_vec()) {
                push_facet(facet);
            }
        }
    }

    facets.into_iter().take(8).collect()
}

pub fn build_query_requirements(
    query: &str,
    keyword_query: Option<&str>,
    subject_entities: &[String],
    cross_entity: bool,
    ordinal_rank: Option<usize>,
    slot_key: Option<&str>,
    expansion_terms: &[String],
) -> Vec<QueryRequirement> {
    let mut requirements = Vec::new();
    let mut seen = HashSet::new();
    let requirement_focus_terms = extract_salient_terms(keyword_query.unwrap_or(query), 3);
    let mut push_requirement = |requirement: QueryRequirement| {
        let key = format!(
            "{}|{}|{}",
            requirement.text.to_ascii_lowercase(),
            requirement.entities.join("|").to_ascii_lowercase(),
            requirement.require_all_entities
        );
        if seen.insert(key) {
            requirements.push(requirement);
        }
    };

    let require_joint_entities = cross_entity && !subject_entities.is_empty();

    if let Some(requirement) =
        make_query_requirement(query.to_string(), subject_entities.to_vec(), require_joint_entities)
    {
        push_requirement(requirement);
    }
    if let Some(keyword_query) = keyword_query {
        if !keyword_query.eq_ignore_ascii_case(query) {
            if let Some(requirement) = make_query_requirement(
                keyword_query.to_string(),
                subject_entities.to_vec(),
                require_joint_entities,
            ) {
                push_requirement(requirement);
            }
        }
    }
    if let Some(slot_key) = slot_key {
        let slot = humanize_slot_key(slot_key);
        if let Some(requirement) = make_query_requirement(
            format!("canonical fact {slot}"),
            subject_entities.to_vec(),
            false,
        ) {
            push_requirement(requirement);
        }
    }
    if let Some(expansion_query) = build_expansion_query(subject_entities, expansion_terms) {
        if let Some(requirement) =
            make_query_requirement(expansion_query, subject_entities.to_vec(), false)
        {
            push_requirement(requirement);
        }
    }
    if cross_entity && !subject_entities.is_empty() {
        if let Some(requirement) =
            make_query_requirement(query.to_string(), subject_entities.to_vec(), true)
        {
            push_requirement(requirement);
        }
        for entity in subject_entities.iter().take(3) {
            let focus = requirement_focus_terms
                .iter()
                .filter(|term| !subject_entities.iter().any(|name| name.eq_ignore_ascii_case(term)))
                .take(2)
                .cloned()
                .collect::<Vec<_>>();
            let entity_requirement_text = if focus.is_empty() {
                entity.clone()
            } else {
                format!("{entity} {}", focus.join(" "))
            };
            if let Some(requirement) =
                make_query_requirement(entity_requirement_text, vec![entity.clone()], false)
            {
                push_requirement(requirement);
            }
        }
    }
    if ordinal_rank.is_some() {
        let stripped = crate::api::plan::intent::strip_ordinal_tokens(query);
        if !stripped.is_empty() && !stripped.eq_ignore_ascii_case(query) {
            if let Some(requirement) =
                make_query_requirement(stripped, subject_entities.to_vec(), false)
            {
                push_requirement(requirement);
            }
        }
    }

    requirements.into_iter().take(8).collect()
}
