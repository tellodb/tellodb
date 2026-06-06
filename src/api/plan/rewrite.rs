use super::types::QueryPlan;
use crate::api::utils::{normalize_alpha_tokens, singularize_token};

pub fn rewrite_query_for_retrieval(query: &str) -> String {
    let q = query.trim();
    let ql = q.to_lowercase();

    let advice_prefixes: &[&str] = &[
        "any tips for",
        "any tips on",
        "any suggestions for",
        "any suggestions on",
        "any advice for",
        "any advice on",
        "any ideas for",
        "any ideas on",
        "any recommendations for",
        "any recommendations on",
        "can you recommend",
        "can you suggest",
        "could you recommend",
        "could you suggest",
        "do you have any tips",
        "do you have any suggestions",
        "do you have any advice",
        "do you have any ideas",
        "do you think it would be",
        "do you think i should",
        "what should i",
        "what would you suggest",
        "how can i find",
        "how can i get",
        "how can i improve",
        "i've been having trouble with",
        "i've been struggling with",
        "i've been thinking about",
        "i was thinking about",
        "i'm thinking about",
        "i'm planning",
        "i am planning",
        "i'm trying to decide",
    ];

    for prefix in advice_prefixes {
        if ql.starts_with(prefix) {
            let stripped = &q[prefix.len()..].trim_start_matches([',', ' ']);
            let cleaned = stripped.trim_end_matches(['?', '.']);
            if !cleaned.is_empty() {
                return cleaned.to_string();
            }
        }
    }

    let advice_suffixes: &[&str] = &[
        "any tips?",
        "any tips.",
        "any advice?",
        "any advice.",
        "any suggestions?",
        "any suggestions.",
        "any ideas?",
        "any ideas.",
        "any recommendations?",
        "any recommendations.",
        "do you have any tips?",
        "what do you think?",
        "any thoughts?",
        "what should i do?",
    ];
    for suffix in advice_suffixes {
        if ql.ends_with(suffix) {
            let trimmed =
                q[..q.len() - suffix.len()].trim().trim_end_matches(['.', ',']);
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }

    if ql.starts_with("how many ") {
        let rest = &q[9..];
        let rest_clean = rest
            .replace(" did I ", " ")
            .replace(" have I ", " ")
            .replace(" do I ", " ")
            .replace(" am I ", " ")
            .replace(" did i ", " ")
            .replace(" have i ", " ")
            .replace(" do i ", " ")
            .trim_end_matches('?')
            .trim()
            .to_string();
        if !rest_clean.is_empty() {
            return rest_clean;
        }
    }

    if ql.starts_with("how long")
        || (ql.starts_with("how many days") || ql.starts_with("how many weeks"))
    {
        let nouns: String = q
            .split_whitespace()
            .filter(|w| {
                let wl = w.to_lowercase();
                ![
                    "how", "many", "long", "days", "weeks", "did", "it", "take", "for", "after",
                    "i", "me", "my", "was", "were", "have", "spend", "spent", "in", "a", "an",
                    "the",
                ]
                .contains(&wl.as_str())
            })
            .collect::<Vec<_>>()
            .join(" ");
        let nouns_clean = nouns.trim_end_matches('?').trim().to_string();
        if nouns_clean.len() > 4 {
            return nouns_clean;
        }
    }

    q.to_string()
}


pub fn build_hyde_query(query: &str, plan: &QueryPlan) -> Option<String> {
    let lower = query.to_ascii_lowercase();
    let q = query.trim().trim_end_matches('?');
    let possessive_subject = extract_possessive_subject_phrase(q, &plan.subject_entities);

    if lower.starts_with("what ") {
        if lower.contains(" like") || lower.contains(" enjoy") || lower.contains(" love") {
            let entities = plan.subject_entities.iter().take(2).cloned().collect::<Vec<_>>();
            let subject = if let Some(subject) = possessive_subject.clone() {
                subject
            } else if entities.is_empty() {
                extract_subject_from_what_query(q)
            } else {
                entities.join(" and ")
            };
            if !subject.is_empty() {
                return Some(format!("{subject} likes enjoys activities hobbies interests sports"));
            }
        }

        if lower.contains("favorite") || lower.contains("prefer") {
            let subject = if let Some(subject) = possessive_subject.clone() {
                subject
            } else if !plan.subject_entities.is_empty() {
                plan.subject_entities[0].clone()
            } else {
                extract_subject_from_what_query(q)
            };
            if !subject.is_empty() {
                return Some(format!("{subject} favorite preferred {}", extract_object_noun(q)));
            }
        }

        if lower.contains(" does ") || lower.contains(" did ") {
            let entities = plan.subject_entities.iter().take(2).cloned().collect::<Vec<_>>();
            let subject = possessive_subject
                .clone()
                .or_else(|| (!entities.is_empty()).then(|| entities.join(" ")));
            if let Some(subject) = subject {
                let keywords =
                    plan.lexical_terms.iter().take(4).cloned().collect::<Vec<_>>().join(" ");
                return Some(format!("{subject} {keywords}"));
            }
        }

        if lower.contains("attribute") || lower.contains("describe") || lower.contains("type of") {
            let entities = plan.subject_entities.iter().take(2).cloned().collect::<Vec<_>>();
            if !entities.is_empty() {
                let kw = plan.lexical_terms.iter().take(3).cloned().collect::<Vec<_>>().join(" ");
                return Some(format!("{} is known for {kw}", entities.join(" and ")));
            }
        }
    }

    if lower.starts_with("how many ") {
        let entities = plan.subject_entities.iter().take(2).cloned().collect::<Vec<_>>();
        let subject = possessive_subject
            .clone()
            .or_else(|| (!entities.is_empty()).then(|| entities.join(" ")));
        if let Some(subject) = subject {
            let kw = plan.lexical_terms.iter().take(3).cloned().collect::<Vec<_>>().join(" ");
            return Some(format!("{subject} has {kw}"));
        }
    }

    if lower.starts_with("how long") {
        let entities = plan.subject_entities.iter().take(2).cloned().collect::<Vec<_>>();
        let subject = possessive_subject
            .clone()
            .or_else(|| (!entities.is_empty()).then(|| entities.join(" ")));
        if let Some(subject) = subject {
            let kw = plan.lexical_terms.iter().take(3).cloned().collect::<Vec<_>>().join(" ");
            return Some(format!("{subject} for years since {kw}"));
        }
    }

    if lower.starts_with("when ") {
        let entities = plan.subject_entities.iter().take(2).cloned().collect::<Vec<_>>();
        let subject = possessive_subject
            .clone()
            .or_else(|| (!entities.is_empty()).then(|| entities.join(" and ")));
        if let Some(subject) = subject {
            let kw = plan.lexical_terms.iter().take(4).cloned().collect::<Vec<_>>().join(" ");
            return Some(format!("{subject} {kw} date time"));
        }
    }

    if lower.starts_with("would ") || lower.starts_with("could ") || lower.contains("likely") {
        let entities = plan.subject_entities.iter().take(2).cloned().collect::<Vec<_>>();
        let subject = possessive_subject
            .clone()
            .or_else(|| (!entities.is_empty()).then(|| entities.join(" ")));
        if let Some(subject) = subject {
            let kw = plan.lexical_terms.iter().take(4).cloned().collect::<Vec<_>>().join(" ");
            return Some(format!("{subject} enjoys prefers interested in {kw}"));
        }
    }

    None
}

fn extract_possessive_subject_phrase(q: &str, entities: &[String]) -> Option<String> {
    let lower = q.to_ascii_lowercase();
    let stop = [
        "like", "likes", "enjoy", "enjoys", "love", "loves", "have", "has", "had", "did", "does",
        "do", "is", "are", "was", "were", "would", "could", "should", "to", "for", "about", "as",
        "in", "on", "with", "from", "during", "when", "where", "why", "how",
    ];

    for entity in entities {
        let entity_lower = entity.to_ascii_lowercase();
        let Some(pos) = lower.find(&entity_lower) else {
            continue;
        };
        let after = &q[pos + entity.len()..];
        let after = after.trim_start();
        let possessive_tail = after.strip_prefix("'s").or_else(|| after.strip_prefix("s'"))?;
        let words = normalize_alpha_tokens(possessive_tail)
            .into_iter()
            .map(|word| singularize_token(&word))
            .take_while(|word| !stop.contains(&word.as_str()))
            .take(3)
            .collect::<Vec<_>>();
        if !words.is_empty() {
            return Some(format!("{} {}", entity, words.join(" ")));
        }
    }

    None
}

fn extract_subject_from_what_query(q: &str) -> String {
    let lower = q.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("what do ") {
        let words: Vec<&str> = rest.split_whitespace().collect();
        let end = words
            .iter()
            .position(|&w| matches!(w, "like" | "enjoy" | "love" | "have" | "own"))
            .unwrap_or(2);
        return words[..end.min(words.len())].join(" ");
    }
    if let Some(rest) = lower.strip_prefix("what is ") {
        let words: Vec<&str> = rest.split_whitespace().collect();
        let end = words
            .iter()
            .position(|&w| w.ends_with("'s") || w.ends_with("s'"))
            .map(|i| i + 1)
            .unwrap_or(2);
        return words[..end.min(words.len())].join(" ");
    }
    String::new()
}

fn extract_object_noun(q: &str) -> String {
    let lower = q.to_ascii_lowercase();
    if let Some(pos) = lower.find("favorite") {
        let rest = &q[pos + 8..].trim();
        return rest.split_whitespace().take(3).collect::<Vec<_>>().join(" ");
    }
    String::new()
}
