#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        if text.len() < 1024 {
            let _ = tellodb::api::planner::build_query_plan(text, None);
        }
    }
});
