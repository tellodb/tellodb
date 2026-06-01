use super::types::QueryIntent;
use crate::ml::QueryIntentClassifier;

pub fn classify_query_intent(
    query: &str,
    classifier: Option<&QueryIntentClassifier>,
) -> QueryIntent {
    if let Some(ml) = classifier {
        if let Some(intent) = ml.predict(query) {
            return intent;
        }
    }

    let lower = query.to_ascii_lowercase();
    let months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    let has_specific_month = months.iter().any(|m| lower.contains(&format!(" in {m}")));
    let is_numeric = lower.contains("how many")
        || lower.contains("number of")
        || lower.contains("in total")
        || lower.contains("total amount")
        || lower.contains("total money")
        || lower.contains("combined")
        || lower.contains("average")
        || lower.contains("altogether");
    let is_temporal = lower.starts_with("when ")
        || has_specific_month
        || lower.contains(" last month")
        || lower.contains(" this year")
        || lower.contains(" past ")
        || lower.contains(" over the past")
        || lower.contains(" before ")
        || lower.contains(" after ");
    let is_recommendation = lower.contains("recommend")
        || lower.contains("suggest")
        || lower.contains("advice")
        || lower.contains("tips")
        || lower.contains("ideas");
    let is_inference = lower.contains("would")
        || lower.contains("likely")
        || lower.contains("might")
        || lower.contains("considered")
        || lower.contains("be open to")
        || lower.contains("political leaning")
        || lower.contains("more interested");
    let is_peripheral = lower.contains("nickname")
        || lower.contains("called")
        || lower.contains("named")
        || lower.contains("as a child")
        || lower.contains("as a kid")
        || lower.contains("childhood")
        || lower.contains("middle name");

    if is_numeric {
        if has_specific_month || lower.contains(" before ") || lower.contains(" after ") {
            QueryIntent::TemporalAggregation
        } else {
            QueryIntent::NumericAggregation
        }
    } else if is_peripheral {
        QueryIntent::PeripheralMention
    } else if is_inference {
        QueryIntent::Inference
    } else if is_temporal {
        QueryIntent::TemporalAggregation
    } else if is_recommendation {
        QueryIntent::Recommendation
    } else {
        QueryIntent::General
    }
}

pub fn extract_ordinal_rank(query: &str) -> Option<usize> {
    let lower = query.to_ascii_lowercase();
    for rank in 1..=10 {
        if let Some(word) = ordinal_word(rank) {
            if lower.contains(word) {
                return Some(rank);
            }
        }
    }
    for rank in 1..=20 {
        let suffix = if rank % 10 == 1 && rank % 100 != 11 {
            "st"
        } else if rank % 10 == 2 && rank % 100 != 12 {
            "nd"
        } else if rank % 10 == 3 && rank % 100 != 13 {
            "rd"
        } else {
            "th"
        };
        let token = format!("{rank}{suffix}");
        if lower.contains(&token) {
            return Some(rank);
        }
    }
    None
}

pub fn ordinal_word(rank: usize) -> Option<&'static str> {
    match rank {
        1 => Some("first"),
        2 => Some("second"),
        3 => Some("third"),
        4 => Some("fourth"),
        5 => Some("fifth"),
        6 => Some("sixth"),
        7 => Some("seventh"),
        8 => Some("eighth"),
        9 => Some("ninth"),
        10 => Some("tenth"),
        _ => None,
    }
}

pub fn strip_ordinal_tokens(query: &str) -> String {
    use crate::api::utils::normalize_fact_text;
    let drop = [
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
        "tenth", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
    ];

    let cleaned = query
        .split_whitespace()
        .filter(|raw| {
            let token = raw.trim_matches(|c: char| !c.is_ascii_alphanumeric()).to_ascii_lowercase();
            !drop.contains(&token.as_str())
        })
        .collect::<Vec<_>>()
        .join(" ");

    normalize_fact_text(cleaned.as_str())
}
