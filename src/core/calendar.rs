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
    const UNIX_EPOCH_JDN: i64 = 2_440_588;
    let a = (14 - i64::from(month)) / 12;
    let y = i64::from(year) + 4800 - a;
    let m = i64::from(month) + 12 * a - 3;
    let jdn = i64::from(day) + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
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

#[allow(clippy::too_many_lines)]
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
        let digits: String = token.chars().filter(char::is_ascii_digit).collect();
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
                let digits: String = token.chars().filter(char::is_ascii_digit).collect();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_temporal_terms_finds_years() {
        let result = extract_temporal_terms("events in 2023 and 1999");
        assert!(result.contains(&"2023".to_string()));
        assert!(result.contains(&"1999".to_string()));
    }

    #[test]
    fn extract_temporal_terms_finds_months() {
        let result = extract_temporal_terms("meeting in January and March");
        assert!(result.contains(&"january".to_string()));
        assert!(result.contains(&"march".to_string()));
    }

    #[test]
    fn extract_temporal_terms_finds_seasons() {
        let result = extract_temporal_terms("summer vacation 2024");
        assert_eq!(&*result, &["2024".to_string(), "summer".to_string()]);
    }

    #[test]
    fn extract_temporal_terms_special_terms() {
        let result = extract_temporal_terms("what happened yesterday and today");
        assert!(result.contains(&"yesterday".to_string()));
        assert!(result.contains(&"today".to_string()));
    }

    #[test]
    fn extract_temporal_terms_deduplicates() {
        let result = extract_temporal_terms("2024 in January and 2024 also january");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn extract_temporal_terms_no_temporal_terms_returns_empty() {
        assert!(extract_temporal_terms("hello world").is_empty());
    }

    #[test]
    fn parse_temporal_window_month_year() {
        let result = parse_temporal_window("October 2023", None).unwrap();
        let expected = (month_to_ms(2023, 10, 1), month_to_ms(2023, 10, 31) + MILLIS_PER_DAY);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_season_year() {
        let result = parse_temporal_window("summer 2022", None).unwrap();
        let expected = (month_to_ms(2022, 6, 1), month_to_ms(2022, 8, 31) + MILLIS_PER_DAY);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_winter_wraps_year() {
        let result = parse_temporal_window("winter 2024", None).unwrap();
        let expected = (
            month_to_ms(2024, 12, 1),
            month_to_ms(2025, 2, days_in_month(2025, 2)) + MILLIS_PER_DAY,
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_last_week_of_month() {
        let result = parse_temporal_window("last week of October 2023", None).unwrap();
        let expected = (month_to_ms(2023, 10, 25), month_to_ms(2023, 10, 31) + MILLIS_PER_DAY);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_specific_day() {
        let result = parse_temporal_window("May 1 2022", None).unwrap();
        let expected = (month_to_ms(2022, 5, 1), month_to_ms(2022, 5, 1) + MILLIS_PER_DAY);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_iso_formatted_date() {
        let result = parse_temporal_window("Where was I living as of 2024/05/12?", None).unwrap();
        let expected_start = month_to_ms(2024, 5, 12);
        assert_eq!(result, (expected_start, expected_start + MILLIS_PER_DAY));

        let result2 = parse_temporal_window("events on 2023-11-05", None).unwrap();
        let expected_start2 = month_to_ms(2023, 11, 5);
        assert_eq!(result2, (expected_start2, expected_start2 + MILLIS_PER_DAY));
    }

    #[test]
    fn parse_temporal_window_no_temporal_info_returns_none() {
        assert_eq!(parse_temporal_window("no time mentioned here", None), None);
    }

    #[test]
    fn parse_temporal_window_first_week_of_month() {
        let result = parse_temporal_window("first week of March 2023", None).unwrap();
        let expected = (month_to_ms(2023, 3, 1), month_to_ms(2023, 3, 7) + MILLIS_PER_DAY);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_temporal_window_relative_yesterday() {
        let ref_ms = 1_700_000_000_000_u64;
        let ref_day = (ref_ms / MILLIS_PER_DAY) * MILLIS_PER_DAY;
        let result = parse_temporal_window("what did I do yesterday?", Some(ref_ms)).unwrap();
        assert_eq!(result, (ref_day - MILLIS_PER_DAY, ref_day));
    }

    #[test]
    fn parse_temporal_window_relative_last_week() {
        let ref_ms = 1_700_000_000_000_u64;
        let result = parse_temporal_window("what happened last week?", Some(ref_ms)).unwrap();
        assert_eq!(result, (ref_ms - 7 * MILLIS_PER_DAY, ref_ms));
    }

    #[test]
    fn parse_temporal_window_relative_diff_ref_times() {
        let t1 = 1_650_000_000_000_u64;
        let t2 = 1_720_000_000_000_u64;
        let query = "show activities from last week";
        let win1 = parse_temporal_window(query, Some(t1)).unwrap();
        let win2 = parse_temporal_window(query, Some(t2)).unwrap();
        assert_ne!(win1, win2);
        assert_eq!(win1.1, t1);
        assert_eq!(win2.1, t2);
    }

    #[test]
    fn parse_temporal_window_past_n_days() {
        let ref_ms = 1_700_000_000_000_u64;
        let result = parse_temporal_window("updates in the past 5 days", Some(ref_ms)).unwrap();
        assert_eq!(result, (ref_ms - 5 * MILLIS_PER_DAY, ref_ms));
    }

    #[test]
    fn days_in_month_all_months() {
        assert_eq!(days_in_month(2023, 1), 31);
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(2023, 3), 31);
        assert_eq!(days_in_month(2023, 4), 30);
        assert_eq!(days_in_month(2023, 5), 31);
        assert_eq!(days_in_month(2023, 6), 30);
        assert_eq!(days_in_month(2023, 7), 31);
        assert_eq!(days_in_month(2023, 8), 31);
        assert_eq!(days_in_month(2023, 9), 30);
        assert_eq!(days_in_month(2023, 10), 31);
        assert_eq!(days_in_month(2023, 11), 30);
        assert_eq!(days_in_month(2023, 12), 31);
    }

    #[test]
    fn days_in_month_leap_year_feb() {
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2000, 2), 29);
    }

    #[test]
    fn days_in_month_non_leap_feb() {
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(1900, 2), 28);
    }

    #[test]
    fn month_index_accepts_full_and_short_names() {
        assert_eq!(month_index("January"), Some(1));
        assert_eq!(month_index("sept"), Some(9));
        assert_eq!(month_index("unknown"), None);
    }
}
