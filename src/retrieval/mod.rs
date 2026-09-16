pub mod scoring;
pub use scoring::ScoringWeights;

use std::collections::HashMap;

/// Reciprocal Rank Fusion over multiple ranked result lanes.
///
/// Each lane is a slice of (item_id, score) where higher score = better. We rank
/// each lane (ties broken by score, then by appearance order), then aggregate:
///
///   S(item) = sum_lane  1 / (c + rank_lane(item))
///
/// c=60 is the standard RRF constant (Cormack et al., 2009).
pub fn rrf_fuse(lanes: &[Vec<(String, f32)>], c: f32) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for lane in lanes {
        let mut sorted: Vec<&(String, f32)> = lane.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, (item, _score)) in sorted.iter().enumerate() {
            let r = (rank + 1) as f32;
            *scores.entry((*item).clone()).or_insert(0.0) += 1.0 / (c + r);
        }
    }
    let mut out: Vec<(String, f32)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}
