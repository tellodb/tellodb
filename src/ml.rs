use crate::api::planner::QueryIntent;
use crate::semantic::SemanticInference;
use anyhow::Result;
use std::sync::Arc;

const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.38;

pub struct QueryIntentClassifier {
    prototypes: Vec<(QueryIntent, Vec<f32>)>,
    semantic: Arc<SemanticInference>,
    threshold: f32,
}

impl QueryIntentClassifier {
    pub fn new(semantic: Arc<SemanticInference>) -> Result<Self> {
        let prototype_texts: &[(QueryIntent, &[&str])] = &[
            (
                QueryIntent::NumericAggregation,
                &[
                    "how many items are there in total",
                    "count the number of things",
                    "what is the total amount spent",
                    "how much money was spent altogether",
                    "what is the average across everything",
                    "give me an overall tally and sum",
                ],
            ),
            (
                QueryIntent::TemporalAggregation,
                &[
                    "when did that event happen",
                    "what occurred in the month of march",
                    "what happened before the move",
                    "what happened after the meeting last week",
                    "tell me the timeline of past events",
                    "when was the last time this occurred",
                ],
            ),
            (
                QueryIntent::Recommendation,
                &[
                    "what would you recommend I do",
                    "do you have any suggestions for me",
                    "give me some advice on this topic",
                    "what tips do you have based on what you know",
                    "what ideas can you offer me",
                ],
            ),
            (
                QueryIntent::Inference,
                &[
                    "would I enjoy that activity",
                    "what is my likely opinion on this",
                    "might I be interested in trying that",
                    "what would I probably think about this",
                    "am I open to considering this option",
                ],
            ),
            (
                QueryIntent::PeripheralMention,
                &[
                    "what is my nickname or what am I called",
                    "what was I called as a child growing up",
                    "what is my middle name",
                    "what was my childhood like",
                ],
            ),
            (
                QueryIntent::General,
                &[
                    "tell me what you know about this",
                    "describe this to me in detail",
                    "what is this thing about",
                    "who is this person you mentioned",
                    "explain this concept to me",
                ],
            ),
        ];

        let mut prototypes = Vec::with_capacity(prototype_texts.len());
        for (intent, texts) in prototype_texts {
            let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
            let mut embeddings: Vec<Vec<f32>> = Vec::new();
            for t in &texts {
                if let Ok(emb) = semantic.generate_query_embedding(t) {
                    embeddings.push(emb);
                }
            }
            if embeddings.is_empty() {
                continue;
            }
            let dim = embeddings[0].len();
            let mut centroid = vec![0.0f32; dim];
            for emb in &embeddings {
                for (i, val) in emb.iter().enumerate() {
                    centroid[i] += val;
                }
            }
            let n = embeddings.len() as f32;
            for val in centroid.iter_mut() {
                *val /= n;
            }
            prototypes.push((*intent, centroid));
        }

        Ok(Self { prototypes, semantic, threshold: DEFAULT_CONFIDENCE_THRESHOLD })
    }

    #[cfg(test)]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn prototype_count(&self) -> usize {
        self.prototypes.len()
    }

    pub fn predict(&self, query: &str) -> Option<QueryIntent> {
        let embedding = self.semantic.generate_query_embedding(query).ok()?;

        if embedding.iter().all(|&v| v == 0.0) {
            return None;
        }

        let mut best_intent = QueryIntent::General;
        let mut best_score = -1.0f32;

        for (intent, prototype) in &self.prototypes {
            let score = cosine_similarity(&embedding, prototype);
            if score > best_score {
                best_score = score;
                best_intent = *intent;
            }
        }
        let normalized = (best_score + 1.0) * 0.5;

        if normalized >= self.threshold {
            Some(best_intent)
        } else {
            None
        }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_both_zero() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let result = cosine_similarity(&a, &b);
        assert!(!result.is_nan());
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_b() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_small_values() {
        let a = vec![1e-8, 2e-8, 3e-8];
        let b = vec![1e-8, 2e-8, 3e-8];
        let result = cosine_similarity(&a, &b);
        assert!(!result.is_nan());
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_near_identical() {
        let a = vec![100.0, 200.0, 300.0];
        let b = vec![100.1, 200.1, 300.1];
        let result = cosine_similarity(&a, &b);
        assert!(!result.is_nan());
        assert!(result > 0.99);
    }

    #[test]
    fn test_cosine_similarity_different_dimensions() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_single_element() {
        let a = vec![5.0];
        let b = vec![5.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        let c = vec![-5.0];
        assert!((cosine_similarity(&a, &c) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_large_values() {
        let a = vec![1e10, 2e10, 3e10];
        let b = vec![1e10, 2e10, 3e10];
        let result = cosine_similarity(&a, &b);
        assert!(!result.is_nan());
        assert!((result - 1.0).abs() < 0.001);
    }
}
