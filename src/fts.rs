pub fn tokenize_for_similarity(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() > 1)
        .map(str::to_ascii_lowercase)
        .collect()
}
