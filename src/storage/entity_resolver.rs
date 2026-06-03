use std::cmp::Ordering;
use thiserror::Error;

// ── Error Type ──

#[derive(Debug, Error)]
pub enum ResolutionError {
    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    EmbeddingDimensionMismatch { expected: usize, actual: usize },
}

// ── Tier Enum ──

#[derive(Debug, Clone, PartialEq)]
pub enum ResolverTier {
    Exact,
    Fuzzy(f32),
    Phonetic,
    Embedding(f32),
}

impl ResolverTier {
    pub fn confidence(&self) -> f32 {
        match self {
            Self::Exact => 1.0,
            Self::Fuzzy(s) => *s,
            Self::Phonetic => 0.85,
            Self::Embedding(s) => *s,
        }
    }

    pub fn priority(&self) -> u8 {
        match self {
            Self::Exact => 4,
            Self::Fuzzy(_) => 3,
            Self::Embedding(_) => 2,
            Self::Phonetic => 1,
        }
    }
}

// ── Resolution Result ──

#[derive(Debug, Clone)]
pub struct EntityResolution {
    pub matched_name: Option<String>,
    pub tier: ResolverTier,
}

// ── Config ──

#[derive(Debug, Clone)]
pub struct ResolutionConfig {
    pub fuzzy_threshold: f32,
    pub embedding_threshold: f32,
    pub max_candidates: usize,
}

impl Default for ResolutionConfig {
    fn default() -> Self {
        Self {
            fuzzy_threshold: 0.92,
            embedding_threshold: 0.88,
            max_candidates: 20,
        }
    }
}

// ── Candidate Data ──

#[derive(Debug, Clone)]
pub struct EntityCandidate {
    pub name: String,
    pub aliases: Vec<String>,
    pub soundex_key: String,
    pub embedding: Option<Vec<f32>>,
}

// ── Merge Proposal Status ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStatus {
    Pending,
    Accepted,
    Rejected,
}

impl MergeStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "accepted" => Self::Accepted,
            "rejected" => Self::Rejected,
            _ => Self::Pending,
        }
    }
}

// ── Pure Functions ──

fn soundex_code(c: char) -> char {
    match c {
        'b' | 'f' | 'p' | 'v' => '1',
        'c' | 'g' | 'j' | 'k' | 'q' | 's' | 'x' | 'z' => '2',
        'd' | 't' => '3',
        'l' => '4',
        'm' | 'n' => '5',
        'r' => '6',
        _ => '0',
    }
}

/// Compute a Soundex key (basic variant).
///
/// Drops non-ASCII-alpha characters, keeps the first letter,
/// then encodes subsequent letters into digit codes, collapsing
/// adjacent identical codes and padding/truncating to 4 chars.
pub fn soundex_key(name: &str) -> String {
    let lower: String = name
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_lowercase())
        .collect();

    if lower.is_empty() {
        return String::new();
    }

    let first = lower.chars().next().unwrap();
    let mut key = String::with_capacity(4);
    key.push(first);

    let mut prev_code = soundex_code(first);
    for c in lower.chars().skip(1) {
        let code = soundex_code(c);
        if code != '0' && code != prev_code {
            key.push(code);
            if key.len() >= 4 {
                break;
            }
        }
        prev_code = code;
    }

    while key.len() < 4 {
        key.push('0');
    }

    key
}

/// Compute a phonetic key with digraph rewrites for challenging name pairs.
///
/// Rewrites: PH→F, CK→K, KN→N, WR→R before computing the Soundex key.
/// This makes "Phillip" and "Filip" collide, which pure Soundex misses.
pub fn phonetic_key(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    let mut transformed = String::with_capacity(lower.len());

    let chars: Vec<char> = lower.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let (replacement, advance) = match chars[i] {
            'p' if i + 1 < chars.len() && chars[i + 1] == 'h' => ('f', 2),
            'c' if i + 1 < chars.len() && chars[i + 1] == 'k' => ('k', 2),
            'k' if i + 1 < chars.len() && chars[i + 1] == 'n' => ('n', 2),
            'w' if i + 1 < chars.len() && chars[i + 1] == 'r' => ('r', 2),
            _ => (chars[i], 1),
        };
        transformed.push(replacement);
        i += advance;
    }

    soundex_key(&transformed)
}

/// Cosine similarity between two f32 slices.
///
/// Returns 0.0 if either vector is empty or dimensions differ.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Run the 4-tier resolution pipeline against a set of known candidates.
///
/// Tiers execute in priority order, stopping at the first confident match:
///   1. **Exact** — case-insensitive equality against name or aliases.
///   2. **Fuzzy** — Jaro-Winkler similarity ≥ `config.fuzzy_threshold`.
///   3. **Phonetic** — Soundex-with-digraph-rewrite key collision.
///   4. **Embedding** — cosine similarity ≥ `config.embedding_threshold`.
///
/// If multiple tiers produce a match against different candidates,
/// the higher-priority tier wins. Returns an `EntityResolution` with
/// the best matched name (or `None` if no tier matched).
pub fn resolve_name(
    name: &str,
    candidates: &[EntityCandidate],
    name_embedding: Option<&[f32]>,
    config: &ResolutionConfig,
) -> EntityResolution {
    let trimmed = name.trim();
    if trimmed.len() < 2 {
        return EntityResolution {
            matched_name: None,
            tier: ResolverTier::Exact,
        };
    }

    let name_lower = trimmed.to_ascii_lowercase();

    // Tier 1: Exact match against canonical name or any alias.
    for c in candidates {
        if c.name.to_ascii_lowercase() == name_lower
            || c.aliases
                .iter()
                .any(|a| a.to_ascii_lowercase() == name_lower)
        {
            return EntityResolution {
                matched_name: Some(c.name.clone()),
                tier: ResolverTier::Exact,
            };
        }
    }

    // Tier 2: Fuzzy match via Jaro-Winkler.
    let fuzzy_threshold = config.fuzzy_threshold as f64;
    let mut fuzzy_matches: Vec<(&EntityCandidate, f64)> = candidates
        .iter()
        .filter_map(|c| {
            let score = strsim::jaro_winkler(name_lower.as_str(), &c.name.to_ascii_lowercase());
            if score >= fuzzy_threshold {
                Some((c, score))
            } else {
                // Also check aliases.
                c.aliases
                    .iter()
                    .map(|a| strsim::jaro_winkler(name_lower.as_str(), &a.to_ascii_lowercase()))
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .filter(|s| *s >= fuzzy_threshold)
                    .map(|score| (c, score))
            }
        })
        .collect();
    fuzzy_matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    if let Some(&(best, score)) = fuzzy_matches.first() {
        return EntityResolution {
            matched_name: Some(best.name.clone()),
            tier: ResolverTier::Fuzzy(score as f32),
        };
    }

    // Tier 3: Phonetic key collision.
    let name_key = phonetic_key(trimmed);
    if !name_key.is_empty() {
        for c in candidates {
            if c.soundex_key == name_key {
                return EntityResolution {
                    matched_name: Some(c.name.clone()),
                    tier: ResolverTier::Phonetic,
                };
            }
        }
    }

    // Tier 4: Embedding similarity.
    if let Some(embedding) = name_embedding {
        let mut emb_matches: Vec<(&EntityCandidate, f32)> = candidates
            .iter()
            .filter_map(|c| {
                c.embedding.as_ref().and_then(|c_emb| {
                    let sim = cosine_similarity(embedding, c_emb);
                    if sim >= config.embedding_threshold {
                        Some((c, sim))
                    } else {
                        None
                    }
                })
            })
            .collect();
        emb_matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        if let Some(&(best, sim)) = emb_matches.first() {
            return EntityResolution {
                matched_name: Some(best.name.clone()),
                tier: ResolverTier::Embedding(sim),
            };
        }
    }

    EntityResolution {
        matched_name: None,
        tier: ResolverTier::Exact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(name: &str) -> EntityCandidate {
        EntityCandidate {
            name: name.to_string(),
            aliases: Vec::new(),
            soundex_key: phonetic_key(name),
            embedding: None,
        }
    }

    fn candidate_with_aliases(name: &str, aliases: &[&str]) -> EntityCandidate {
        EntityCandidate {
            name: name.to_string(),
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
            soundex_key: phonetic_key(name),
            embedding: None,
        }
    }

    #[test]
    fn soundex_robert_rupert() {
        assert_eq!(soundex_key("Robert"), soundex_key("Rupert"));
    }

    #[test]
    fn soundex_smith_smythe() {
        assert_eq!(soundex_key("Smith"), soundex_key("Smythe"));
    }

    #[test]
    fn soundex_ashcraft_ashcroft() {
        assert_eq!(soundex_key("Ashcraft"), soundex_key("Ashcroft"));
    }

    #[test]
    fn soundex_empty_returns_empty() {
        assert!(soundex_key("").is_empty());
    }

    #[test]
    fn phonetic_phillip_filip() {
        assert_eq!(phonetic_key("Phillip"), phonetic_key("Filip"));
    }

    #[test]
    fn phonetic_knight_night() {
        assert_eq!(phonetic_key("Knight"), phonetic_key("Night"));
    }

    #[test]
    fn phonetic_wright_right() {
        assert_eq!(phonetic_key("Wright"), phonetic_key("Right"));
    }

    #[test]
    fn resolve_exact_match() {
        let candidates = vec![candidate("Sarah")];
        let result = resolve_name("Sarah", &candidates, None, &ResolutionConfig::default());
        assert_eq!(result.matched_name, Some("Sarah".to_string()));
        assert_eq!(result.tier, ResolverTier::Exact);
    }

    #[test]
    fn resolve_exact_via_alias() {
        let candidates = vec![candidate_with_aliases("Sarah", &["the engineering lead"])];
        let result = resolve_name("the engineering lead", &candidates, None, &ResolutionConfig::default());
        assert_eq!(result.matched_name, Some("Sarah".to_string()));
        assert_eq!(result.tier, ResolverTier::Exact);
    }

    #[test]
    fn resolve_fuzzy_jaro_winkler() {
        let candidates = vec![candidate("Sarah")];
        let result = resolve_name("Sara", &candidates, None, &ResolutionConfig::default());
        assert_eq!(result.matched_name, Some("Sarah".to_string()));
        assert!(matches!(result.tier, ResolverTier::Fuzzy(_)));
    }

    #[test]
    fn resolve_fuzzy_below_threshold() {
        let candidates = vec![candidate("Sarah")];
        let result = resolve_name("Jonathan", &candidates, None, &ResolutionConfig::default());
        assert_eq!(result.matched_name, None);
    }

    #[test]
    fn resolve_phonetic() {
        let candidates = vec![candidate("Phillip")];
        let result = resolve_name("Filip", &candidates, None, &ResolutionConfig::default());
        assert!(result.matched_name.is_some());
        assert_eq!(result.tier, ResolverTier::Phonetic);
    }

    #[test]
    fn resolve_empty_name_returns_none() {
        let result = resolve_name("", &[], None, &ResolutionConfig::default());
        assert_eq!(result.matched_name, None);
    }

    #[test]
    fn resolve_short_name_returns_none() {
        let result = resolve_name("X", &[], None, &ResolutionConfig::default());
        assert_eq!(result.matched_name, None);
    }

    #[test]
    fn resolve_no_candidates_returns_none() {
        let result = resolve_name("Sarah", &[], None, &ResolutionConfig::default());
        assert_eq!(result.matched_name, None);
    }

    #[test]
    fn cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_empty_returns_zero() {
        assert!((cosine_similarity(&[], &[])).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_dim_mismatch_returns_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn merge_status_roundtrip() {
        assert_eq!(MergeStatus::from_str("pending"), MergeStatus::Pending);
        assert_eq!(MergeStatus::from_str("accepted"), MergeStatus::Accepted);
        assert_eq!(MergeStatus::from_str("rejected"), MergeStatus::Rejected);
        assert_eq!(MergeStatus::from_str("unknown"), MergeStatus::Pending);
        assert_eq!(MergeStatus::Pending.as_str(), "pending");
        assert_eq!(MergeStatus::Accepted.as_str(), "accepted");
        assert_eq!(MergeStatus::Rejected.as_str(), "rejected");
    }

    #[test]
    fn tier_priority_order() {
        assert!(ResolverTier::Exact.priority() > ResolverTier::Fuzzy(0.95).priority());
        assert!(ResolverTier::Fuzzy(0.95).priority() > ResolverTier::Embedding(0.9).priority());
        assert!(ResolverTier::Embedding(0.9).priority() > ResolverTier::Phonetic.priority());
    }
}
