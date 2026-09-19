use axum::http::{HeaderMap, HeaderName, HeaderValue};
use std::time::Instant;

pub fn elapsed_ms_and_us(start: Instant) -> (u64, u64) {
    let elapsed = start.elapsed();
    (elapsed.as_millis() as u64, elapsed.as_micros() as u64)
}

pub fn insert_u64_header(headers: &mut HeaderMap, name: &str, value: u64) {
    if let (Ok(header_name), Ok(header_value)) =
        (HeaderName::from_bytes(name.as_bytes()), HeaderValue::from_str(&value.to_string()))
    {
        headers.insert(header_name, header_value);
    }
}

pub fn insert_stage_timing_headers(headers: &mut HeaderMap, base: &str, millis: u64, micros: u64) {
    insert_u64_header(headers, &format!("{base}-ms"), millis);
    insert_u64_header(headers, &format!("{base}-us"), micros);
}

pub fn insert_f32_header(headers: &mut HeaderMap, name: &str, value: f32) {
    if let (Ok(header_name), Ok(header_value)) =
        (HeaderName::from_bytes(name.as_bytes()), HeaderValue::from_str(&format!("{value:.4}")))
    {
        headers.insert(header_name, header_value);
    }
}

pub fn clip_profile_to_budget(profile_json: &str, max_fields: usize) -> String {
    let Ok(val) = serde_json::from_str::<serde_json::Value>(profile_json) else {
        return profile_json.to_string();
    };
    let Some(obj) = val.as_object() else {
        return profile_json.to_string();
    };
    let clipped: serde_json::Map<String, serde_json::Value> =
        obj.iter().take(max_fields).map(|(k, v)| (k.clone(), v.clone())).collect();
    serde_json::to_string(&clipped).unwrap_or_else(|_| profile_json.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_profile_to_budget_valid_json_clipped() {
        let input = r#"{"a":1,"b":2,"c":3}"#;
        let result = clip_profile_to_budget(input, 2);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.len(), 2);
    }

    #[test]
    fn clip_profile_to_budget_invalid_json_unchanged() {
        let input = "not valid json";
        assert_eq!(clip_profile_to_budget(input, 5), input);
    }

    #[test]
    fn clip_profile_to_budget_non_object_json_unchanged() {
        assert_eq!(clip_profile_to_budget("[1,2,3]", 2), "[1,2,3]");
    }

    #[test]
    fn elapsed_ms_and_us_returns_positive_values() {
        let (ms, us) = elapsed_ms_and_us(Instant::now());
        assert!(us >= ms * 1000);
    }
}
