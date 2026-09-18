//! Numeric memory: quantities stated in memories ("ran 5 miles", "$1,200
//! rent") extracted deterministically at ingest and aggregated with SQL, so
//! "how much / how many" answers are computed rather than guessed by an LLM.
//!
//! Values are normalised to one canonical unit per dimension (meters,
//! seconds, kg, celsius, bytes, USD), so a label such as `distance` can be
//! summed across memories that used different units.

use crate::storage::{TenantDatabaseManager, TenantStore};
use anyhow::{Context, Result};
use regex::Regex;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedMetric {
    /// Canonical label: `money`, `distance`, `duration`, `weight`,
    /// `temperature`, `data_size`, `percentage` or `count_<noun>`.
    pub label: String,
    /// Value in the canonical unit.
    pub value: f64,
    pub unit: String,
    /// Matched text, for provenance.
    pub source_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateResult {
    pub sum: f64,
    pub count: usize,
    pub avg: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketedAggregate {
    pub bucket_start_ms: u64,
    pub bucket_end_ms: u64,
    pub result: AggregateResult,
}

#[derive(Debug, Clone, Copy)]
pub enum TemporalBucket {
    Hour,
    Day,
    Week,
    Month,
    Year,
}

impl TemporalBucket {
    fn duration_ms(self) -> u64 {
        match self {
            TemporalBucket::Hour => 3_600_000,
            TemporalBucket::Day => 86_400_000,
            TemporalBucket::Week => 604_800_000,
            TemporalBucket::Month => 2_592_000_000,
            TemporalBucket::Year => 31_536_000_000,
        }
    }
}

/// At most this many buckets are returned for one aggregation.
const MAX_BUCKETS: i64 = 10_000;

/// One unit family: a pattern whose captures are `lo` (optional range start),
/// `value` and `unit`, and a converter to the canonical unit.
struct Dimension {
    label: &'static str,
    unit: &'static str,
    pattern: Regex,
    /// Returns the factor/offset conversion for a matched unit word, or `None`
    /// if the word does not belong to this dimension.
    convert: fn(&str, f64) -> Option<f64>,
}

const NUMBER: &str = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?";

fn unit_pattern(units: &str) -> Regex {
    // Optional range ("5-10 miles", "5 to 10 miles") reported as its midpoint.
    Regex::new(&format!(
        r"(?i)\b(?:(?P<lo>{NUMBER})\s*(?:-|–|to)\s*)?(?P<value>{NUMBER})\s*(?P<unit>{units})\b"
    ))
    .expect("static metric pattern")
}

fn parse_number(text: &str) -> Option<f64> {
    text.replace(',', "").parse::<f64>().ok()
}

fn singular(word: &str) -> String {
    let lower = word.to_ascii_lowercase();
    if let Some(stem) = lower.strip_suffix("ies") {
        format!("{stem}y")
    } else if lower.ends_with("ss") {
        lower
    } else {
        lower.strip_suffix('s').map(str::to_string).unwrap_or(lower)
    }
}

pub struct MetricExtractor {
    money_prefixed: Regex,
    dimensions: Vec<Dimension>,
    count: Regex,
}

impl MetricExtractor {
    pub fn new() -> Self {
        let dimensions = vec![
            Dimension {
                label: "money",
                unit: "USD",
                pattern: unit_pattern("dollars?|usd"),
                convert: |_, v| Some(v),
            },
            Dimension {
                label: "distance",
                unit: "meters",
                // Bare "m" and "in" are excluded: "5 in the morning" is not a length.
                pattern: unit_pattern(
                    "miles?|mi|kilometers?|kilometres?|km|meters?|metres?|feet|foot|ft|yards?|yd|centimeters?|cm|millimeters?|mm|inches|inch",
                ),
                convert: |u, v| {
                    let f = match u {
                        "mile" | "miles" | "mi" => 1609.344,
                        "kilometer" | "kilometers" | "kilometre" | "kilometres" | "km" => 1000.0,
                        "meter" | "meters" | "metre" | "metres" => 1.0,
                        "feet" | "foot" | "ft" => 0.3048,
                        "yard" | "yards" | "yd" => 0.9144,
                        "centimeter" | "centimeters" | "cm" => 0.01,
                        "millimeter" | "millimeters" | "mm" => 0.001,
                        "inch" | "inches" => 0.0254,
                        _ => return None,
                    };
                    Some(v * f)
                },
            },
            Dimension {
                label: "duration",
                unit: "seconds",
                pattern: unit_pattern(
                    "hours?|hrs?|minutes?|mins?|seconds?|secs?|milliseconds?|ms|days?|weeks?",
                ),
                convert: |u, v| {
                    let f = match u {
                        "hour" | "hours" | "hr" | "hrs" => 3600.0,
                        "minute" | "minutes" | "min" | "mins" => 60.0,
                        "second" | "seconds" | "sec" | "secs" => 1.0,
                        "millisecond" | "milliseconds" | "ms" => 0.001,
                        "day" | "days" => 86_400.0,
                        "week" | "weeks" => 604_800.0,
                        _ => return None,
                    };
                    Some(v * f)
                },
            },
            Dimension {
                label: "weight",
                unit: "kg",
                pattern: unit_pattern(
                    "pounds?|lbs?|ounces?|oz|kilograms?|kg|grams?|g|milligrams?|mg|tonnes?|tons?",
                ),
                convert: |u, v| {
                    let f = match u {
                        "pound" | "pounds" | "lb" | "lbs" => 0.453_592,
                        "ounce" | "ounces" | "oz" => 0.028_349_5,
                        "kilogram" | "kilograms" | "kg" => 1.0,
                        "gram" | "grams" | "g" => 0.001,
                        "milligram" | "milligrams" | "mg" => 0.000_001,
                        "ton" | "tons" | "tonne" | "tonnes" => 1000.0,
                        _ => return None,
                    };
                    Some(v * f)
                },
            },
            Dimension {
                label: "temperature",
                unit: "celsius",
                // Requires a degree sign or a spelled-out scale: "5 k" or "3 c"
                // are far more often not temperatures.
                pattern: Regex::new(&format!(
                    r"(?i)(?P<value>-?(?:{NUMBER}))\s*(?:°\s*(?P<unit>[fck])\b|degrees?\s+(?P<unit2>fahrenheit|celsius|kelvin)|(?P<unit3>fahrenheit|celsius|kelvin)\b)"
                ))
                .expect("static metric pattern"),
                convert: |u, v| match u {
                    "f" | "fahrenheit" => Some((v - 32.0) * 5.0 / 9.0),
                    "c" | "celsius" => Some(v),
                    "k" | "kelvin" => Some(v - 273.15),
                    _ => None,
                },
            },
            Dimension {
                label: "data_size",
                unit: "bytes",
                pattern: unit_pattern(
                    "terabytes?|gigabytes?|megabytes?|kilobytes?|bytes?|tb|gb|mb|kb",
                ),
                convert: |u, v| {
                    let f = match u {
                        "kb" | "kilobyte" | "kilobytes" => 1024.0,
                        "mb" | "megabyte" | "megabytes" => 1_048_576.0,
                        "gb" | "gigabyte" | "gigabytes" => 1_073_741_824.0,
                        "tb" | "terabyte" | "terabytes" => 1_099_511_627_776.0,
                        "byte" | "bytes" => 1.0,
                        _ => return None,
                    };
                    Some(v * f)
                },
            },
            Dimension {
                label: "percentage",
                unit: "%",
                pattern: Regex::new(&format!(
                    r"(?i)(?P<value>{NUMBER})\s*(?P<unit>%|percent\b|pct\b)"
                ))
                .expect("static metric pattern"),
                convert: |_, v| Some(v),
            },
        ];
        Self {
            money_prefixed: Regex::new(&format!(
                r"(?i)(?:\$|\busd\s*)(?P<lo>{NUMBER})(?:\s*(?:-|–|to)\s*\$?(?P<value>{NUMBER}))?"
            ))
            .expect("static metric pattern"),
            dimensions,
            count: Regex::new(&format!(
                r"(?i)\b(?P<value>{NUMBER})\s+(?P<unit>times|people|persons|items|units|cars|houses|books|files|projects|tasks|events|meetings|emails|messages|calls|visits|orders|products|customers|users|accounts|transactions|countries|cities|games|movies|songs|miles run|pages|classes|lessons|trips|photos|pets|kids|children)\b"
            ))
            .expect("static metric pattern"),
        }
    }

    /// Extracts quantities in text order. Overlapping matches are resolved in
    /// favour of the earlier-checked, more specific pattern, so a quantity is
    /// never counted twice (e.g. "about 5 miles" once, not as distance and as
    /// an approximate value).
    pub fn extract(&self, text: &str) -> Vec<ExtractedMetric> {
        let mut accepted: Vec<(usize, usize, ExtractedMetric)> = Vec::new();
        let overlaps = |accepted: &[(usize, usize, ExtractedMetric)], s: usize, e: usize| {
            accepted.iter().any(|(as_, ae, _)| s < *ae && *as_ < e)
        };

        for cap in self.money_prefixed.captures_iter(text) {
            let whole = cap.get(0).expect("match");
            let lo = cap.name("lo").and_then(|m| parse_number(m.as_str()));
            let hi = cap.name("value").and_then(|m| parse_number(m.as_str()));
            let value = match (lo, hi) {
                (Some(lo), Some(hi)) => (lo + hi) / 2.0,
                (Some(v), None) => v,
                _ => continue,
            };
            if !overlaps(&accepted, whole.start(), whole.end()) {
                accepted.push((
                    whole.start(),
                    whole.end(),
                    ExtractedMetric {
                        label: "money".into(),
                        value,
                        unit: "USD".into(),
                        source_text: whole.as_str().to_string(),
                    },
                ));
            }
        }

        for dim in &self.dimensions {
            for cap in dim.pattern.captures_iter(text) {
                let whole = cap.get(0).expect("match");
                if overlaps(&accepted, whole.start(), whole.end()) {
                    continue;
                }
                let Some(unit_word) = ["unit", "unit2", "unit3"]
                    .iter()
                    .find_map(|n| cap.name(n))
                    .map(|m| m.as_str().to_ascii_lowercase())
                else {
                    continue;
                };
                let Some(raw) = cap.name("value").and_then(|m| parse_number(m.as_str())) else {
                    continue;
                };
                let raw = match cap.name("lo").and_then(|m| parse_number(m.as_str())) {
                    Some(lo) => (lo + raw) / 2.0,
                    None => raw,
                };
                let Some(value) = (dim.convert)(unit_word.as_str(), raw) else {
                    continue;
                };
                accepted.push((
                    whole.start(),
                    whole.end(),
                    ExtractedMetric {
                        label: dim.label.into(),
                        value,
                        unit: dim.unit.into(),
                        source_text: whole.as_str().to_string(),
                    },
                ));
            }
        }

        for cap in self.count.captures_iter(text) {
            let whole = cap.get(0).expect("match");
            if overlaps(&accepted, whole.start(), whole.end()) {
                continue;
            }
            let (Some(value), Some(noun)) =
                (cap.name("value").and_then(|m| parse_number(m.as_str())), cap.name("unit"))
            else {
                continue;
            };
            let noun = singular(noun.as_str().split_whitespace().last().unwrap_or_default());
            accepted.push((
                whole.start(),
                whole.end(),
                ExtractedMetric {
                    label: format!("count_{noun}"),
                    value,
                    unit: noun,
                    source_text: whole.as_str().to_string(),
                },
            ));
        }

        accepted.sort_by_key(|(start, _, _)| *start);
        accepted.into_iter().map(|(_, _, metric)| metric).collect()
    }
}

impl Default for MetricExtractor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MetricVault {
    tenant_manager: Arc<TenantDatabaseManager>,
    extractor: MetricExtractor,
}

impl MetricVault {
    pub fn new(tenant_manager: Arc<TenantDatabaseManager>) -> Self {
        Self { tenant_manager, extractor: MetricExtractor::new() }
    }

    pub fn extractor(&self) -> &MetricExtractor {
        &self.extractor
    }

    /// Replaces the metrics recorded for `memory_id` with those extracted from
    /// `text`. Idempotent, so re-ingesting a memory never double-counts.
    pub fn record_memory(
        &self,
        tenant: &TenantStore,
        entity_id: &str,
        memory_id: &str,
        timestamp_ms: u64,
        text: &str,
    ) -> Result<usize> {
        let metrics = self.extractor.extract(text);
        let mut conn = tenant.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        tx.execute("DELETE FROM metrics WHERE memory_id = ?1", params![memory_id])?;
        {
            let mut insert = tx.prepare_cached(
                "INSERT INTO metrics (memory_id, ordinal, entity_id, timestamp_ms, label, value, unit, source_text)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?;
            for (ordinal, m) in metrics.iter().enumerate() {
                insert.execute(params![
                    memory_id,
                    ordinal as i64,
                    entity_id,
                    timestamp_ms as i64,
                    m.label,
                    m.value,
                    m.unit,
                    m.source_text
                ])?;
            }
        }
        tx.commit()?;
        Ok(metrics.len())
    }

    fn tenant(&self, tenant_id: &str) -> Result<Arc<TenantStore>> {
        self.tenant_manager.get_tenant(tenant_id).context("failed to get tenant for aggregation")
    }

    pub fn aggregate_range(
        &self,
        tenant_id: &str,
        entity_id: &str,
        label: &str,
        start_ms: u64,
        end_ms: u64,
    ) -> Result<AggregateResult> {
        let tenant = self.tenant(tenant_id)?;
        let conn = tenant.get_conn()?;
        let row = conn.query_row(
            "SELECT COALESCE(SUM(value), 0), COUNT(*), COALESCE(MIN(value), 0),
                    COALESCE(MAX(value), 0), COALESCE(SUM(value * value), 0)
             FROM metrics
             WHERE entity_id = ?1 AND label = ?2 AND timestamp_ms >= ?3 AND timestamp_ms <= ?4",
            params![entity_id, label, clamp_ms(start_ms), clamp_ms(end_ms)],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
        )?;
        Ok(aggregate_from(row))
    }

    /// Aggregates per time bucket in a single query; only non-empty buckets
    /// are returned. (The previous implementation looped over every bucket
    /// between `start` and `end`, which never finished for an open-ended
    /// range.)
    pub fn aggregate_bucketed(
        &self,
        tenant_id: &str,
        entity_id: &str,
        label: &str,
        start_ms: u64,
        end_ms: u64,
        bucket: TemporalBucket,
    ) -> Result<Vec<BucketedAggregate>> {
        let bucket_ms = bucket.duration_ms() as i64;
        let aligned_start = (clamp_ms(start_ms) / bucket_ms) * bucket_ms;
        let tenant = self.tenant(tenant_id)?;
        let conn = tenant.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT (timestamp_ms - ?3) / ?5 AS bucket,
                    SUM(value), COUNT(*), MIN(value), MAX(value), SUM(value * value)
             FROM metrics
             WHERE entity_id = ?1 AND label = ?2 AND timestamp_ms >= ?3 AND timestamp_ms <= ?4
             GROUP BY bucket ORDER BY bucket LIMIT ?6",
        )?;
        let rows = stmt.query_map(
            params![entity_id, label, aligned_start, clamp_ms(end_ms), bucket_ms, MAX_BUCKETS],
            |row| {
                let bucket: i64 = row.get(0)?;
                Ok((bucket, (row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?)))
            },
        )?;
        let mut out = Vec::new();
        for row in rows {
            let (bucket, stats) = row?;
            let start = (aligned_start + bucket * bucket_ms) as u64;
            out.push(BucketedAggregate {
                bucket_start_ms: start,
                bucket_end_ms: start + bucket_ms as u64,
                result: aggregate_from(stats),
            });
        }
        Ok(out)
    }
}

fn clamp_ms(ms: u64) -> i64 {
    ms.min(i64::MAX as u64) as i64
}

fn aggregate_from((sum, count, min, max, sum_sq): (f64, i64, f64, f64, f64)) -> AggregateResult {
    let n = count.max(0) as usize;
    let avg = if n > 0 { sum / n as f64 } else { 0.0 };
    let variance = if n > 1 { (sum_sq / n as f64 - avg * avg).max(0.0) } else { 0.0 };
    AggregateResult { sum, count: n, avg, min, max, stddev: variance.sqrt() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(text: &str) -> Vec<(String, f64)> {
        MetricExtractor::new().extract(text).into_iter().map(|m| (m.label, m.value)).collect()
    }

    #[test]
    fn extracts_normalized_quantities_once() {
        let got = labels("I ran about 5 miles, paid $1,200 rent and slept 8 hours.");
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].0, "distance");
        assert!((got[0].1 - 8046.72).abs() < 1e-6);
        assert_eq!(got[1], ("money".to_string(), 1200.0));
        assert_eq!(got[2], ("duration".to_string(), 28_800.0));
    }

    #[test]
    fn ranges_use_midpoint_and_counts_are_singular() {
        assert_eq!(labels("It costs 10-20 dollars"), vec![("money".to_string(), 15.0)]);
        assert_eq!(
            labels("I have visited 12 countries"),
            vec![("count_country".to_string(), 12.0)]
        );
    }

    #[test]
    fn ambiguous_units_are_not_metrics() {
        assert!(labels("I woke up at 5 in the morning and read 3 c chapters").is_empty());
        assert_eq!(labels("It was 72°F")[0].0, "temperature");
    }

    #[test]
    fn record_is_idempotent_and_aggregates_by_bucket() {
        let temp = tempfile::tempdir().unwrap();
        let paths = crate::runtime_paths::RuntimePaths::from_root(temp.path().to_path_buf());
        let manager =
            Arc::new(TenantDatabaseManager::new(paths, crate::vector_index::VectorConfig::new(3)));
        let vault = MetricVault::new(manager.clone());
        let tenant = manager.get_tenant("default").unwrap();
        let day = 86_400_000;
        vault.record_memory(&tenant, "u", "m1", day, "I spent $10 and $20").unwrap();
        vault.record_memory(&tenant, "u", "m1", day, "I spent $10 and $20").unwrap();
        vault.record_memory(&tenant, "u", "m2", 3 * day, "Then $5").unwrap();

        let total = vault.aggregate_range("default", "u", "money", 0, u64::MAX).unwrap();
        assert_eq!((total.sum, total.count), (35.0, 3));

        let buckets = vault
            .aggregate_bucketed("default", "u", "money", 0, u64::MAX, TemporalBucket::Day)
            .unwrap();
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0].result.sum, 30.0);
        assert_eq!(buckets[1].bucket_start_ms, 3 * day);
    }
}
