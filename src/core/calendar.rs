use std::collections::HashSet;

pub const MONTHS: [&str; 12] = [
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

pub const MILLIS_PER_DAY: u64 = 86_400_000;
pub const MIN_YEAR: i32 = 1990;
pub const MAX_YEAR: i32 = 2100;
pub const MAX_DAY_OF_MONTH: u32 = 31;
pub const FIRST_WEEK_END_DAY: u32 = 7;
pub const YEAR_DIGITS: usize = 4;
pub const LAST_WEEK_DAY_OFFSET: u32 = 6;

pub fn month_index(value: &str) -> Option<u32> {
    let month = value.trim_matches(|c: char| !c.is_ascii_alphabetic()).to_ascii_lowercase();
    match month.as_str() {
        "jan" | "january" => Some(1),
        "feb" | "february" => Some(2),
        "mar" | "march" => Some(3),
        "apr" | "april" => Some(4),
        "may" => Some(5),
        "jun" | "june" => Some(6),
        "jul" | "july" => Some(7),
        "aug" | "august" => Some(8),
        "sep" | "sept" | "september" => Some(9),
        "oct" | "october" => Some(10),
        "nov" | "november" => Some(11),
        "dec" | "december" => Some(12),
        _ => None,
    }
}

pub fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

pub fn days_since_epoch(year: i32, month: u32, day: u32) -> i64 {
    let a = (14 - month as i64) / 12;
    let y = year as i64 + 4800 - a;
    let m = month as i64 + 12 * a - 3;
    let jdn = day as i64 + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
    const UNIX_EPOCH_JDN: i64 = 2_440_588;
    jdn - UNIX_EPOCH_JDN
}

pub fn month_to_ms(year: i32, month: u32, day: u32) -> u64 {
    days_since_epoch(year, month, day) as u64 * MILLIS_PER_DAY
}

pub fn extract_temporal_terms(query: &str) -> Vec<String> {
    let lower = query.to_ascii_lowercase();
    let mut terms = Vec::with_capacity(4);

    for token in lower.split(|c: char| !c.is_ascii_alphanumeric()) {
        if token.len() == YEAR_DIGITS && token.chars().all(|c| c.is_ascii_digit()) {
            terms.push(token.to_string());
        }
    }

    let temporal_terms = [
        "spring",
        "summer",
        "fall",
        "autumn",
        "winter",
        "weekend",
        "week",
        "month",
        "year",
        "yesterday",
        "today",
        "tomorrow",
        "recently",
        "latest",
        "last",
        "recent",
    ];
    for month in MONTHS {
        if lower.contains(month) {
            terms.push(month.to_string());
        }
    }
    for term in temporal_terms {
        if lower.contains(term) {
            terms.push(term.to_string());
        }
    }

    let mut seen = HashSet::new();
    terms.into_iter().filter(|term| seen.insert(term.clone())).collect()
}

pub fn parse_temporal_window(query: &str, reference_time_ms: Option<u64>) -> Option<(u64, u64)> {
    let lower = query.to_ascii_lowercase();
    let tokens: Vec<&str> = lower.split_whitespace().collect();

    for token in &tokens {
        let clean =
            token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '/' && c != '-');
        let parts: Vec<&str> = clean.split(['/', '-']).collect();
        if parts.len() == 3 {
            if let (Ok(y), Ok(m), Ok(d)) =
                (parts[0].parse::<i32>(), parts[1].parse::<u32>(), parts[2].parse::<u32>())
            {
                if (MIN_YEAR..=MAX_YEAR).contains(&y)
                    && (1..=12).contains(&m)
                    && (1..=MAX_DAY_OF_MONTH).contains(&d)
                {
                    let d = d.min(days_in_month(y, m));
                    let start_ms = month_to_ms(y, m, d);
                    return Some((start_ms, start_ms + MILLIS_PER_DAY));
                }
            }
        }
    }

    let year: Option<i32> = tokens.iter().find_map(|token| {
        let digits: String = token.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() == 4 {
            digits.parse().ok().filter(|&y: &i32| (MIN_YEAR..=MAX_YEAR).contains(&y))
        } else {
            None
        }
    });

    if let Some(year) = year {
        for (season, start_month, end_month) in [
            ("spring", 3, 5),
            ("summer", 6, 8),
            ("fall", 9, 11),
            ("autumn", 9, 11),
            ("winter", 12, 2),
        ] {
            if lower.contains(season) {
                let start_ms = month_to_ms(year, start_month, 1);
                let end_ms = if end_month < start_month {
                    month_to_ms(year + 1, end_month, days_in_month(year + 1, end_month))
                } else {
                    month_to_ms(year, end_month, days_in_month(year, end_month))
                };
                return Some((start_ms, end_ms + MILLIS_PER_DAY));
            }
        }

        let found_month = tokens.iter().find_map(|token| month_index(token)).or_else(|| {
            MONTHS
                .iter()
                .enumerate()
                .find_map(|(index, month)| lower.contains(month).then_some(index as u32 + 1))
        });

        if let Some(month) = found_month {
            let is_last_week = lower.contains("last week");
            let is_first_week = lower.contains("first week");
            let day: Option<u32> = tokens.iter().find_map(|token| {
                let digits: String = token.chars().filter(|c| c.is_ascii_digit()).collect();
                if digits.len() <= 2 {
                    digits.parse::<u32>().ok().filter(|&d| (1..=MAX_DAY_OF_MONTH).contains(&d))
                } else {
                    None
                }
            });

            let dom = days_in_month(year, month);
            let (start_ms, end_ms) = if is_last_week {
                let last_day = dom;
                let first_day = last_day.saturating_sub(LAST_WEEK_DAY_OFFSET).max(1);
                (
                    month_to_ms(year, month, first_day),
                    month_to_ms(year, month, last_day) + MILLIS_PER_DAY,
                )
            } else if is_first_week {
                (
                    month_to_ms(year, month, 1),
                    month_to_ms(year, month, FIRST_WEEK_END_DAY) + MILLIS_PER_DAY,
                )
            } else if let Some(day) = day {
                let d = day.min(dom);
                (month_to_ms(year, month, d), month_to_ms(year, month, d) + MILLIS_PER_DAY)
            } else {
                (month_to_ms(year, month, 1), month_to_ms(year, month, dom) + MILLIS_PER_DAY)
            };
            return Some((start_ms, end_ms));
        }
    }

    let ref_ms = reference_time_ms.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    });

    if lower.contains("yesterday") {
        let ref_day_start = (ref_ms / MILLIS_PER_DAY) * MILLIS_PER_DAY;
        return Some((ref_day_start.saturating_sub(MILLIS_PER_DAY), ref_day_start));
    }
    if lower.contains("today") {
        let ref_day_start = (ref_ms / MILLIS_PER_DAY) * MILLIS_PER_DAY;
        return Some((ref_day_start, ref_day_start + MILLIS_PER_DAY));
    }
    if lower.contains("last week") || lower.contains("past week") {
        return Some((ref_ms.saturating_sub(7 * MILLIS_PER_DAY), ref_ms));
    }
    if lower.contains("this week") {
        let ref_day_start = (ref_ms / MILLIS_PER_DAY) * MILLIS_PER_DAY;
        return Some((
            ref_day_start.saturating_sub(6 * MILLIS_PER_DAY),
            ref_day_start + MILLIS_PER_DAY,
        ));
    }
    if lower.contains("last month") || lower.contains("past month") {
        return Some((ref_ms.saturating_sub(30 * MILLIS_PER_DAY), ref_ms));
    }
    if lower.contains("this month") {
        return Some((ref_ms.saturating_sub(30 * MILLIS_PER_DAY), ref_ms + MILLIS_PER_DAY));
    }
    if lower.contains("last year") || lower.contains("past year") {
        return Some((ref_ms.saturating_sub(365 * MILLIS_PER_DAY), ref_ms));
    }

    for i in 0..tokens.len().saturating_sub(2) {
        if tokens[i] == "last" || tokens[i] == "past" {
            if let Ok(n) = tokens[i + 1].parse::<u64>() {
                let unit = tokens[i + 2].trim_matches(|c: char| !c.is_alphabetic());
                if unit.starts_with("day") {
                    return Some((ref_ms.saturating_sub(n * MILLIS_PER_DAY), ref_ms));
                }
                if unit.starts_with("week") {
                    return Some((ref_ms.saturating_sub(n * 7 * MILLIS_PER_DAY), ref_ms));
                }
                if unit.starts_with("month") {
                    return Some((ref_ms.saturating_sub(n * 30 * MILLIS_PER_DAY), ref_ms));
                }
            }
        }
    }

    None
}
