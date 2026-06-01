#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = tellodb::api::ingest_utils::extract_temporal_terms(text);
        let _ = tellodb::api::ingest_utils::normalize_fact_text(text);
        let _ = tellodb::api::ingest_utils::extract_named_phrases(&[text.to_string()]);
    }
});
