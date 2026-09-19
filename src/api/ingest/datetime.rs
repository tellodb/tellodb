use super::dialogue::extract_bracketed_header_value;

pub fn extract_document_time_ms(text: &str, fallback_ms: u64) -> u64 {
    extract_bracketed_header_value(text, "Session Date")
        .and_then(|date| parse_date_to_epoch_ms(&date, fallback_ms))
        .unwrap_or(fallback_ms)
}

pub fn extract_event_time_ms(text: &str, document_time_ms: u64) -> Option<u64> {
    if let Some(date_ms) = parse_date_to_epoch_ms(text, document_time_ms) {
        return Some(date_ms);
    }

    let lower = text.to_ascii_lowercase();
    let day_ms = 86_400_000u64;
    if lower.contains("tomorrow") {
        Some(document_time_ms.saturating_add(day_ms))
    } else if lower.contains("yesterday") {
        Some(document_time_ms.saturating_sub(day_ms))
    } else if lower.contains("next week") {
        Some(document_time_ms.saturating_add(day_ms * 7))
    } else if lower.contains("last week") || lower.contains("past week") {
        Some(document_time_ms.saturating_sub(day_ms * 7))
    } else if lower.contains("next month") {
        Some(document_time_ms.saturating_add(day_ms * 30))
    } else if lower.contains("last month") || lower.contains("past month") {
        Some(document_time_ms.saturating_sub(day_ms * 30))
    } else if lower.contains("today") {
        Some(document_time_ms)
    } else {
        None
    }
}

pub fn parse_date_to_epoch_ms(text: &str, reference_ms: u64) -> Option<u64> {
    for raw in text.split_whitespace() {
        let token = raw.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-');
        if let Some((year, month, day)) = parse_iso_date(token) {
            return epoch_ms_from_ymd(year, month, day);
        }
    }

    let tokens = text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return None;
    }

    let (reference_year, _, _) = civil_from_epoch_ms(reference_ms);
    for (idx, token) in tokens.iter().enumerate() {
        let Some(month) = month_number(token) else {
            continue;
        };

        let mut day = 1u32;
        let mut year = reference_year;
        let mut explicit_numeric_part = false;

        if let Some(next) = tokens.get(idx + 1) {
            if let Ok(parsed_day) = next.parse::<u32>() {
                if (1..=31).contains(&parsed_day) {
                    day = parsed_day;
                    explicit_numeric_part = true;
                    if let Some(after) =
                        tokens.get(idx + 2).and_then(|value| value.parse::<i32>().ok())
                    {
                        if (1900..=2200).contains(&after) {
                            year = after;
                            explicit_numeric_part = true;
                        }
                    }
                } else if (1900..=2200).contains(&(parsed_day as i32)) {
                    year = parsed_day as i32;
                    explicit_numeric_part = true;
                }
            }
        }

        if idx > 0 {
            if let Some(prev_day) = tokens.get(idx - 1).and_then(|value| value.parse::<u32>().ok())
            {
                if (1..=31).contains(&prev_day) {
                    day = prev_day;
                    explicit_numeric_part = true;
                }
            }
        }

        if token == "may" && !explicit_numeric_part {
            continue;
        }

        return epoch_ms_from_ymd(year, month, day);
    }

    for token in &tokens {
        if token.len() == 4 {
            if let Ok(year) = token.parse::<i32>() {
                if (1900..=2200).contains(&year) {
                    return epoch_ms_from_ymd(year, 1, 1);
                }
            }
        }
    }

    None
}

pub fn parse_iso_date(token: &str) -> Option<(i32, u32, u32)> {
    let parts = token.split('-').collect::<Vec<_>>();
    if parts.len() != 3 {
        return None;
    }
    let year = parts[0].parse::<i32>().ok()?;
    let month = parts[1].parse::<u32>().ok()?;
    let day = parts[2].parse::<u32>().ok()?;
    if !(1900..=2200).contains(&year) || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some((year, month, day))
}

pub fn month_number(token: &str) -> Option<u32> {
    match token {
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

pub fn epoch_ms_from_ymd(year: i32, month: u32, day: u32) -> Option<u64> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let days = days_from_civil(year, month, day);
    if days < 0 {
        return None;
    }
    Some(days as u64 * 86_400_000)
}

pub fn civil_from_epoch_ms(ms: u64) -> (i32, u32, u32) {
    civil_from_days((ms / 86_400_000) as i64)
}

pub fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = year - if month <= 2 { 1 } else { 0 };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let mp = month as i32 + if month > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + day as i32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146097 + doe - 719468) as i64
}

pub fn civil_from_days(days_since_epoch: i64) -> (i32, u32, u32) {
    let z = days_since_epoch + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    ((y + if m <= 2 { 1 } else { 0 }) as i32, m as u32, d as u32)
}
#[cfg(test)]
mod moved_tests {
    use super::*;

    #[test]
    fn test_parse_date_to_epoch_ms_iso_date() {
        let ms = parse_date_to_epoch_ms("2024-01-15", 0).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!(y, 2024);
        assert_eq!(m, 1);
        assert_eq!(d, 15);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_month_day_year() {
        let ms = parse_date_to_epoch_ms("January 15 2024", 0).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!(y, 2024);
        assert_eq!(m, 1);
        assert_eq!(d, 15);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_year_only() {
        let ref_ms = epoch_ms_from_ymd(2024, 6, 15).unwrap();
        let ms = parse_date_to_epoch_ms("2025", ref_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!(y, 2025);
        assert_eq!(m, 1);
        assert_eq!(d, 1);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_may_disambiguation() {
        let ref_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = parse_date_to_epoch_ms("May 2024", ref_ms);
        assert!(ms.is_some(), "May with year should succeed");
        let (y, m, d) = civil_from_epoch_ms(ms.unwrap());
        assert_eq!((y, m, d), (2024, 5, 1));
    }

    #[test]
    fn test_parse_date_to_epoch_ms_may_with_day() {
        let ref_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = parse_date_to_epoch_ms("May 15 2024", ref_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!(y, 2024);
        assert_eq!(m, 5);
        assert_eq!(d, 15);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_day_before_month() {
        let ref_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = parse_date_to_epoch_ms("15 January 2024", ref_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!(y, 2024);
        assert_eq!(m, 1);
        assert_eq!(d, 15);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_invalid_text() {
        assert_eq!(parse_date_to_epoch_ms("not a date", 0), None);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_iso_with_punctuation() {
        let ms = parse_date_to_epoch_ms("date: 2024-03-15!", 0).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!(y, 2024);
        assert_eq!(m, 3);
        assert_eq!(d, 15);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_short_month() {
        let ref_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = parse_date_to_epoch_ms("Jan 15 2024", ref_ms).unwrap();
        let (_y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!(m, 1);
        assert_eq!(d, 15);
    }

    #[test]
    fn test_parse_date_to_epoch_ms_year_out_of_range() {
        assert_eq!(parse_date_to_epoch_ms("1899-01-01", 0), None);
        assert_eq!(parse_date_to_epoch_ms("2201-01-01", 0), None);
    }

    #[test]
    fn test_extract_event_time_ms_tomorrow() {
        let doc_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = extract_event_time_ms("see you tomorrow", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 1, 2));
    }

    #[test]
    fn test_extract_event_time_ms_yesterday() {
        let doc_ms = epoch_ms_from_ymd(2024, 1, 15).unwrap();
        let ms = extract_event_time_ms("it happened yesterday", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 1, 14));
    }

    #[test]
    fn test_extract_event_time_ms_next_week() {
        let doc_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = extract_event_time_ms("next week", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 1, 8));
    }

    #[test]
    fn test_extract_event_time_ms_last_week() {
        let doc_ms = epoch_ms_from_ymd(2024, 1, 15).unwrap();
        let ms = extract_event_time_ms("last week", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 1, 8));
    }

    #[test]
    fn test_extract_event_time_ms_today() {
        let doc_ms = epoch_ms_from_ymd(2024, 6, 15).unwrap();
        let ms = extract_event_time_ms("today", doc_ms).unwrap();
        assert_eq!(ms, doc_ms);
    }

    #[test]
    fn test_extract_event_time_ms_next_month() {
        let doc_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = extract_event_time_ms("next month", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 1, 31));
    }

    #[test]
    fn test_extract_event_time_ms_absolute_date_overrides_relative() {
        let doc_ms = epoch_ms_from_ymd(2024, 1, 1).unwrap();
        let ms = extract_event_time_ms("meeting on 2024-06-15", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 6, 15));
    }

    #[test]
    fn test_extract_event_time_ms_no_match() {
        assert_eq!(extract_event_time_ms("no temporal info", 0), None);
    }

    #[test]
    fn test_extract_event_time_ms_past_week() {
        let doc_ms = epoch_ms_from_ymd(2024, 3, 15).unwrap();
        let ms = extract_event_time_ms("past week", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 3, 8));
    }

    #[test]
    fn test_extract_event_time_ms_past_month() {
        let doc_ms = epoch_ms_from_ymd(2024, 3, 15).unwrap();
        let ms = extract_event_time_ms("past month", doc_ms).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 2, 14));
    }

    #[test]
    fn test_extract_document_time_ms_from_header() {
        let text = "[Session Date: 2024-06-15]\ncontent";
        let ms = extract_document_time_ms(text, 0);
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 6, 15));
    }

    #[test]
    fn test_extract_document_time_ms_fallback() {
        let text = "no header here";
        let ms = extract_document_time_ms(text, 5000000);
        assert_eq!(ms, 5000000);
    }

    #[test]
    fn test_epoch_ms_roundtrip() {
        let ms = epoch_ms_from_ymd(2024, 6, 15).unwrap();
        let (y, m, d) = civil_from_epoch_ms(ms);
        assert_eq!((y, m, d), (2024, 6, 15));
    }

    #[test]
    fn test_epoch_ms_from_ymd_invalid_month() {
        assert_eq!(epoch_ms_from_ymd(2024, 0, 1), None);
        assert_eq!(epoch_ms_from_ymd(2024, 13, 1), None);
    }

    #[test]
    fn test_epoch_ms_from_ymd_invalid_day() {
        assert_eq!(epoch_ms_from_ymd(2024, 1, 0), None);
        assert_eq!(epoch_ms_from_ymd(2024, 1, 32), None);
    }

    #[test]
    fn test_parse_iso_date_valid() {
        assert_eq!(parse_iso_date("2024-01-15"), Some((2024, 1, 15)));
    }

    #[test]
    fn test_parse_iso_date_invalid_format() {
        assert_eq!(parse_iso_date("2024/01/15"), None);
    }

    #[test]
    fn test_parse_iso_date_out_of_range() {
        assert_eq!(parse_iso_date("1800-01-01"), None);
    }

    #[test]
    fn test_parse_iso_date_not_enough_parts() {
        assert_eq!(parse_iso_date("2024-01"), None);
    }

    #[test]
    fn test_month_number_full_names() {
        assert_eq!(month_number("january"), Some(1));
        assert_eq!(month_number("february"), Some(2));
        assert_eq!(month_number("december"), Some(12));
    }

    #[test]
    fn test_month_number_abbreviations() {
        assert_eq!(month_number("jan"), Some(1));
        assert_eq!(month_number("feb"), Some(2));
        assert_eq!(month_number("dec"), Some(12));
    }

    #[test]
    fn test_month_number_sept_variant() {
        assert_eq!(month_number("sept"), Some(9));
        assert_eq!(month_number("sep"), Some(9));
    }

    #[test]
    fn test_month_number_invalid() {
        assert_eq!(month_number("xyz"), None);
    }

    #[test]
    fn test_civil_date_roundtrip() {
        let days = days_from_civil(2024, 6, 15);
        let (y, m, d) = civil_from_days(days);
        assert_eq!((y, m, d), (2024, 6, 15));
    }

    #[test]
    fn test_civil_date_epoch() {
        let days = days_from_civil(1970, 1, 1);
        assert_eq!(days, 0);
    }
}
