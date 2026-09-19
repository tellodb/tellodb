pub mod lanes;
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
        // Lanes are often built from hash maps; ties break by id so ranks do
        // not depend on hash order.
        sorted.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))
        });
        for (rank, (item, _score)) in sorted.iter().enumerate() {
            let r = (rank + 1) as f32;
            *scores.entry((*item).clone()).or_insert(0.0) += 1.0 / (c + r);
        }
    }
    let mut out: Vec<(String, f32)> = scores.into_iter().collect();
    // Ties (common: equal ranks in different lanes) break by id, not by hash order.
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))
    });
    out
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn rrf_lane() -> impl Strategy<Value = Vec<(String, f32)>> {
        prop::collection::vec(
            (
                prop::collection::vec(any::<char>(), 0..8)
                    .prop_map(|characters| characters.into_iter().collect::<String>()),
                any::<f32>().prop_filter("finite score", |score| score.is_finite()),
            ),
            0..8,
        )
    }

    #[test]
    fn rrf_ties_do_not_depend_on_input_order() {
        let lane_a =
            vec![("m2".to_string(), 1.0), ("m1".to_string(), 1.0), ("m3".to_string(), 0.5)];
        let lane_b =
            vec![("m1".to_string(), 1.0), ("m2".to_string(), 1.0), ("m3".to_string(), 0.5)];
        assert_eq!(
            rrf_fuse(std::slice::from_ref(&lane_a), 60.0),
            rrf_fuse(std::slice::from_ref(&lane_b), 60.0)
        );
        let fused = rrf_fuse(&[lane_a, lane_b], 60.0);
        assert_eq!(fused.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), ["m1", "m2", "m3"]);
    }

    proptest! {
        #[test]
        fn adding_an_empty_lane_changes_nothing(
            lanes in prop::collection::vec(rrf_lane(), 0..5),
            c in 1.0f32..1000.0,
        ) {
            let mut with_empty = lanes.clone();
            with_empty.push(Vec::new());
            prop_assert_eq!(rrf_fuse(&lanes, c), rrf_fuse(&with_empty, c));
        }

        #[test]
        fn rrf_score_is_monotone_in_rank(count in 2usize..64) {
            let lane = (0..count)
                .map(|rank| (format!("item-{rank:03}"), 1.0))
                .collect::<Vec<_>>();
            let fused = rrf_fuse(&[lane], 60.0);

            for pair in fused.windows(2) {
                prop_assert!(pair[0].1 >= pair[1].1);
            }
            for (rank, (item, score)) in fused.iter().enumerate() {
                prop_assert_eq!(item, &format!("item-{rank:03}"));
                let expected = 1.0 / (60.0 + (rank + 1) as f32);
                prop_assert!((*score - expected).abs() <= f32::EPSILON);
            }
        }
    }
}
