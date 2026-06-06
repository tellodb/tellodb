#![allow(dead_code)]
use crate::storage::TenantDatabaseManager;
use anyhow::{Context, Result};
use regex::Regex;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;

// ── Named Constants ──────────────────────────────────────────────────────────

// Confidence values for extraction sources
const CONFIDENCE_DETERMINISTIC: f64 = 1.0;
const CONFIDENCE_NEURAL: f64 = 0.6;
const CONFIDENCE_MERGED: f64 = 0.7;

// Distance conversion factors (to meters)
const MILES_TO_METERS: f64 = 1609.344;
const KM_TO_METERS: f64 = 1000.0;
const FEET_TO_METERS: f64 = 0.3048;
const YARDS_TO_METERS: f64 = 0.9144;
const CM_TO_METERS: f64 = 0.01;
const MM_TO_METERS: f64 = 0.001;
const INCHES_TO_METERS: f64 = 0.0254;

// Weight conversion factors (to kg)
const POUNDS_TO_KG: f64 = 0.453592;
const OUNCES_TO_KG: f64 = 0.0283495;
const GRAMS_TO_KG: f64 = 0.001;
const MILLIGRAMS_TO_KG: f64 = 0.000001;
const TONS_TO_KG: f64 = 1000.0;

// Temperature conversion constants
const FAHRENHEIT_OFFSET: f64 = 32.0;
const FAHRENHEIT_NUMERATOR: f64 = 5.0;
const FAHRENHEIT_DENOMINATOR: f64 = 9.0;
const KELVIN_TO_CELSIUS_OFFSET: f64 = 273.15;

// Duration conversion factors (to seconds)
const HOURS_TO_SECONDS: f64 = 3600.0;
const MINUTES_TO_SECONDS: f64 = 60.0;
const MS_TO_SECONDS: f64 = 0.001;
const DAYS_TO_SECONDS: f64 = 86400.0;

// Data size conversion factors (to bytes)
const KB_TO_BYTES: f64 = 1024.0;
const MB_TO_BYTES: f64 = 1_048_576.0;
const GB_TO_BYTES: f64 = 1_073_741_824.0;
const TB_TO_BYTES: f64 = 1_099_511_627_776.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRecord {
    pub timestamp_ms: u64,
    pub entity_id: String,
    pub label: String,
    pub value: f64,
    pub unit: Option<String>,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub source: ExtractionSource,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[derive(Default)]
pub enum ExtractionSource {
    #[default]
    Deterministic,
    Neural,
    Merged,
}

impl ExtractionSource {
    fn as_str(&self) -> &'static str {
        match self {
            ExtractionSource::Deterministic => "deterministic",
            ExtractionSource::Neural => "neural",
            ExtractionSource::Merged => "merged",
        }
    }
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy)]
enum MetricCategory {
    Distance,
    Money,
    Percentage,
    Duration,
    Weight,
    Count,
}

impl MetricCategory {
    fn from_unit(unit: Option<&str>) -> Self {
        match unit {
            Some(u) if u.contains("mile") || u.contains("km") => MetricCategory::Distance,
            Some(u) if u.contains("dollar") || u == "$" => MetricCategory::Money,
            Some("%") => MetricCategory::Percentage,
            Some(u) if u.contains("hour") || u.contains("minute") || u.contains("day") => {
                MetricCategory::Duration
            }
            Some(u) if u.contains("lb") || u.contains("kg") => MetricCategory::Weight,
            _ => MetricCategory::Count,
        }
    }

    fn label(&self, suffix: &str) -> String {
        match self {
            MetricCategory::Distance => format!("distance_{}", suffix),
            MetricCategory::Money => format!("money_{}", suffix),
            MetricCategory::Percentage => format!("percentage_{}", suffix),
            MetricCategory::Duration => format!("duration_{}", suffix),
            MetricCategory::Weight => format!("weight_{}", suffix),
            MetricCategory::Count => format!("count_{}", suffix),
        }
    }
}

pub struct MetricVault {
    tenant_manager: Arc<TenantDatabaseManager>,
    extractor: MetricExtractor,
    semantic: Arc<crate::semantic::SemanticInference>,
}

impl MetricVault {
    pub fn new(
        tenant_manager: Arc<TenantDatabaseManager>,
        semantic: Arc<crate::semantic::SemanticInference>,
    ) -> Result<Self> {
        Ok(Self {
            tenant_manager,
            extractor: MetricExtractor::new().context("failed to create MetricExtractor")?,
            semantic,
        })
    }

    pub fn process_text(
        &self,
        user_id: &str,
        entity_id: &str,
        timestamp_ms: u64,
        text: &str,
    ) -> Result<()> {
        let tenant =
            self.tenant_manager.get_tenant(user_id).context("failed to get tenant for user")?;
        let lower_text = text.to_lowercase();

        let is_preference = lower_text.contains("love")
            || lower_text.contains("hate")
            || lower_text.contains("favorite")
            || lower_text.contains("always")
            || lower_text.contains("never")
            || lower_text.contains("prefer");

        let deterministic_metrics = self.extractor.extract(text);

        let mut neural_metrics: Vec<(String, f64, Option<String>)> = Vec::new();
        let mut entities: Vec<(String, String)> = Vec::new();

        if let Ok(extracted) = self.semantic.extract_entities(text) {
            entities = extracted;
            for (etype, ename) in &entities {
                neural_metrics.push((
                    format!("entity_{}", etype.to_lowercase()),
                    1.0,
                    Some(ename.clone()),
                ));
            }
        }

        let merged = Self::merge_extractions(&deterministic_metrics, &neural_metrics);

        for (label, value, unit, source, confidence) in merged {
            let (norm_value, norm_unit) = normalize_unit(&label, value, unit.as_deref());
            let record = MetricRecord {
                timestamp_ms,
                entity_id: entity_id.to_string(),
                label: if norm_unit != unit { format!("{}_normalized", label) } else { label },
                value: norm_value,
                unit: norm_unit,
                confidence,
                source,
            };
            self.insert_metric(user_id, &record).context("failed to insert merged metric")?;
        }

        if is_preference {
            let mut triples = Vec::new();
            for (_, name) in &entities {
                triples.push((
                    entity_id.to_string(),
                    "has_preference".to_string(),
                    name.clone(),
                    timestamp_ms,
                ));
            }
            if !triples.is_empty() {
                let conn =
                    tenant.get_conn().context("failed to get connection for preference edges")?;
                let mut stmt = conn.prepare_cached("
                    INSERT OR REPLACE INTO edges (edge_id, source, target, edge_type, label, weight, timestamp_ms)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                ").context("failed to prepare edge insert statement")?;
                for (src, rel, dst, ts) in triples {
                    let edge_id = format!("{}:{}:{}", src, rel, dst);
                    stmt.execute(params![edge_id, src, dst, rel, rel, 1.0, ts])
                        .context("failed to execute edge insert")?;
                }
            }
        }

        if entities.len() > 1 {
            let mut triples = Vec::new();
            for i in 0..entities.len() {
                for j in i + 1..entities.len() {
                    let (label_a, name_a) = &entities[i];
                    let (label_b, name_b) = &entities[j];
                    let predicate = match (label_a.as_str(), label_b.as_str()) {
                        ("PER", "ORG") => Some("associated_with"),
                        ("PER", "LOC") => Some("located_in"),
                        ("ORG", "LOC") => Some("headquartered_in"),
                        ("PER", "PER") => Some("knows"),
                        _ => None,
                    };
                    if let Some(pred) = predicate {
                        triples.push((
                            name_a.clone(),
                            pred.to_string(),
                            name_b.clone(),
                            timestamp_ms,
                        ));
                    }
                }
            }
            if !triples.is_empty() {
                let conn =
                    tenant.get_conn().context("failed to get connection for entity edges")?;
                let mut stmt = conn.prepare_cached("
                    INSERT OR REPLACE INTO edges (edge_id, source, target, edge_type, label, weight, timestamp_ms)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                ").context("failed to prepare entity edge insert statement")?;
                for (src, rel, dst, ts) in triples {
                    let edge_id = format!("{}:{}:{}", src, rel, dst);
                    stmt.execute(params![edge_id, src, dst, rel, rel, 1.0, ts])
                        .context("failed to execute entity edge insert")?;
                }
            }
        }

        Ok(())
    }

    fn merge_extractions(
        deterministic: &[(String, f64, Option<String>)],
        neural: &[(String, f64, Option<String>)],
    ) -> Vec<(String, f64, Option<String>, ExtractionSource, f64)> {
        let mut det_map: HashMap<String, Vec<(f64, Option<String>)>> = HashMap::new();
        for (label, value, unit) in deterministic {
            det_map.entry(label.clone()).or_default().push((*value, unit.clone()));
        }

        let mut neu_map: HashMap<String, Vec<(f64, Option<String>)>> = HashMap::new();
        for (label, value, unit) in neural {
            neu_map.entry(label.clone()).or_default().push((*value, unit.clone()));
        }

        let mut merged = Vec::new();

        for (label, det_vals) in &det_map {
            for (value, unit) in det_vals {
                merged.push((
                    label.clone(),
                    *value,
                    unit.clone(),
                    ExtractionSource::Deterministic,
                    CONFIDENCE_DETERMINISTIC,
                ));
            }
        }

        for (label, neu_vals) in &neu_map {
            if det_map.contains_key(label) {
                if let Some(det_vals) = det_map.get(label) {
                    for (n_val, n_unit) in neu_vals {
                        let dominated =
                            det_vals.iter().any(|(d_val, _)| (d_val - n_val).abs() < f64::EPSILON);
                        if !dominated {
                            merged.push((
                                label.clone(),
                                *n_val,
                                n_unit.clone(),
                                ExtractionSource::Merged,
                                CONFIDENCE_MERGED,
                            ));
                        }
                    }
                }
            } else {
                for (value, unit) in neu_vals {
                    merged.push((
                        label.clone(),
                        *value,
                        unit.clone(),
                        ExtractionSource::Neural,
                        CONFIDENCE_NEURAL,
                    ));
                }
            }
        }

        merged
    }

    pub fn insert_metric(&self, user_id: &str, record: &MetricRecord) -> Result<()> {
        let tenant = self
            .tenant_manager
            .get_tenant(user_id)
            .context("failed to get tenant for metric insert")?;
        let conn = tenant.get_conn().context("failed to get connection for metric insert")?;

        let content_hash = compute_content_hash(
            record.timestamp_ms,
            &record.entity_id,
            &record.label,
            record.value,
        );

        let existing: Option<String> = conn.query_row(
            "SELECT content_hash FROM metrics WHERE timestamp_ms = ?1 AND entity_id = ?2 AND label = ?3",
            params![record.timestamp_ms, record.entity_id, record.label],
            |row| row.get(0),
        ).ok();

        if let Some(existing_hash) = existing {
            if existing_hash == content_hash {
                return Ok(());
            }
        }

        conn.execute(
            "INSERT OR REPLACE INTO metrics (timestamp_ms, entity_id, label, value, unit, content_hash, confidence, source)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                record.timestamp_ms,
                record.entity_id,
                record.label,
                record.value,
                record.unit,
                content_hash,
                record.confidence,
                record.source.as_str(),
            ],
        ).context("failed to upsert metric record")?;
        Ok(())
    }

    pub fn aggregate_range(
        &self,
        user_id: &str,
        entity_id: &str,
        label: &str,
        start_ms: u64,
        end_ms: u64,
    ) -> Result<AggregateResult> {
        let tenant = self
            .tenant_manager
            .get_tenant(user_id)
            .context("failed to get tenant for aggregation")?;
        let conn = tenant.get_conn().context("failed to get connection for aggregation")?;
        let mut stmt = conn
            .prepare(
                "
            SELECT
                COALESCE(SUM(value), 0.0),
                COUNT(*),
                COALESCE(AVG(value), 0.0),
                COALESCE(MIN(value), 0.0),
                COALESCE(MAX(value), 0.0)
            FROM metrics
            WHERE entity_id = ?1 AND label = ?2 AND timestamp_ms >= ?3 AND timestamp_ms <= ?4
        ",
            )
            .context("failed to prepare aggregate query")?;
        let (sum, count, avg, min, max): (f64, usize, f64, f64, f64) = stmt
            .query_row(params![entity_id, label, start_ms, end_ms], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?))
            })
            .context("failed to execute aggregate query")?;

        let stddev = if count > 1 {
            let mut var_stmt = conn
                .prepare(
                    "
                SELECT value FROM metrics
                WHERE entity_id = ?1 AND label = ?2 AND timestamp_ms >= ?3 AND timestamp_ms <= ?4
            ",
                )
                .context("failed to prepare variance query")?;
            let values: Vec<f64> = var_stmt
                .query_map(params![entity_id, label, start_ms, end_ms], |row| row.get(0))
                .context("failed to execute variance query")?
                .filter_map(|r| r.ok())
                .collect();
            let mean = avg;
            let variance: f64 =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count as f64);
            variance.sqrt()
        } else {
            0.0
        };

        Ok(AggregateResult { sum, count, avg, min, max, stddev })
    }

    pub fn aggregate_bucketed(
        &self,
        user_id: &str,
        entity_id: &str,
        label: &str,
        start_ms: u64,
        end_ms: u64,
        bucket: TemporalBucket,
    ) -> Result<Vec<BucketedAggregate>> {
        let bucket_ms = bucket.duration_ms();
        let aligned_start = (start_ms / bucket_ms) * bucket_ms;
        let mut results = Vec::new();
        let mut cursor = aligned_start;

        while cursor < end_ms {
            let bucket_end = cursor + bucket_ms;
            let result = self
                .aggregate_range(user_id, entity_id, label, cursor, bucket_end.min(end_ms))
                .context("failed to compute bucket aggregate")?;
            if result.count > 0 {
                results.push(BucketedAggregate {
                    bucket_start_ms: cursor,
                    bucket_end_ms: bucket_end,
                    result,
                });
            }
            cursor = bucket_end;
        }

        Ok(results)
    }
}

fn compute_content_hash(timestamp_ms: u64, entity_id: &str, label: &str, value: f64) -> String {
    let mut hasher = Sha256::new();
    hasher.update(timestamp_ms.to_le_bytes());
    hasher.update(entity_id.as_bytes());
    hasher.update(label.as_bytes());
    hasher.update(value.to_le_bytes());
    let result = hasher.finalize();
    result.iter().map(|b| format!("{:02x}", b)).collect()
}

fn normalize_unit(label: &str, value: f64, unit: Option<&str>) -> (f64, Option<String>) {
    let unit_str = match unit {
        Some(u) => u.to_lowercase(),
        None => return (value, unit.map(|s| s.to_string())),
    };

    if label.starts_with("distance_") {
        match unit_str.as_str() {
            "mile" | "miles" | "mi" => (value * MILES_TO_METERS, Some("meters".to_string())),
            "km" | "kilometer" | "kilometers" => (value * KM_TO_METERS, Some("meters".to_string())),
            "m" | "meter" | "meters" => (value, Some("meters".to_string())),
            "ft" | "foot" | "feet" => (value * FEET_TO_METERS, Some("meters".to_string())),
            "yd" | "yard" | "yards" => (value * YARDS_TO_METERS, Some("meters".to_string())),
            "cm" | "centimeter" | "centimeters" => {
                (value * CM_TO_METERS, Some("meters".to_string()))
            }
            "mm" | "millimeter" | "millimeters" => {
                (value * MM_TO_METERS, Some("meters".to_string()))
            }
            "in" | "inch" | "inches" => (value * INCHES_TO_METERS, Some("meters".to_string())),
            _ => (value, unit.map(|s| s.to_string())),
        }
    } else if label.starts_with("weight_") {
        match unit_str.as_str() {
            "lb" | "lbs" | "pound" | "pounds" => (value * POUNDS_TO_KG, Some("kg".to_string())),
            "oz" | "ounce" | "ounces" => (value * OUNCES_TO_KG, Some("kg".to_string())),
            "g" | "gram" | "grams" => (value * GRAMS_TO_KG, Some("kg".to_string())),
            "mg" | "milligram" | "milligrams" => (value * MILLIGRAMS_TO_KG, Some("kg".to_string())),
            "ton" | "tons" | "tonne" | "tonnes" => (value * TONS_TO_KG, Some("kg".to_string())),
            "kg" | "kilogram" | "kilograms" => (value, Some("kg".to_string())),
            _ => (value, unit.map(|s| s.to_string())),
        }
    } else if label.starts_with("temperature_") {
        match unit_str.as_str() {
            "f" | "fahrenheit" => (
                (value - FAHRENHEIT_OFFSET) * FAHRENHEIT_NUMERATOR / FAHRENHEIT_DENOMINATOR,
                Some("celsius".to_string()),
            ),
            "k" | "kelvin" => (value - KELVIN_TO_CELSIUS_OFFSET, Some("celsius".to_string())),
            "c" | "celsius" => (value, Some("celsius".to_string())),
            _ => (value, unit.map(|s| s.to_string())),
        }
    } else if label.starts_with("duration_") {
        match unit_str.as_str() {
            "hour" | "hours" | "hr" | "hrs" | "h" => {
                (value * HOURS_TO_SECONDS, Some("seconds".to_string()))
            }
            "minute" | "minutes" | "min" | "mins" => {
                (value * MINUTES_TO_SECONDS, Some("seconds".to_string()))
            }
            "second" | "seconds" | "sec" | "secs" | "s" => (value, Some("seconds".to_string())),
            "ms" | "millisecond" | "milliseconds" => {
                (value * MS_TO_SECONDS, Some("seconds".to_string()))
            }
            "day" | "days" => (value * DAYS_TO_SECONDS, Some("seconds".to_string())),
            _ => (value, unit.map(|s| s.to_string())),
        }
    } else if label.starts_with("data_") {
        match unit_str.as_str() {
            "kb" | "kilobyte" | "kilobytes" => (value * KB_TO_BYTES, Some("bytes".to_string())),
            "mb" | "megabyte" | "megabytes" => (value * MB_TO_BYTES, Some("bytes".to_string())),
            "gb" | "gigabyte" | "gigabytes" => (value * GB_TO_BYTES, Some("bytes".to_string())),
            "tb" | "terabyte" | "terabytes" => (value * TB_TO_BYTES, Some("bytes".to_string())),
            "byte" | "bytes" | "b" => (value, Some("bytes".to_string())),
            _ => (value, unit.map(|s| s.to_string())),
        }
    } else {
        (value, unit.map(|s| s.to_string()))
    }
}

pub struct MetricExtractor {
    money_re: Regex,
    distance_re: Regex,
    count_re: Regex,
    percentage_re: Regex,
    duration_re: Regex,
    temperature_re: Regex,
    weight_re: Regex,
    data_size_re: Regex,
    range_re: Regex,
    approx_re: Regex,
}

impl MetricExtractor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            money_re: Regex::new(
                r"(?i)(?:\$|USD)\s*(\d+(?:[,\d]{1,3})*(?:\.\d+)?)|(\d+(?:[,\d]{1,3})*(?:\.\d+)?)\s*(?:\$|USD|dollars)",
            )?,
            distance_re: Regex::new(
                r"(?i)(\d+(?:\.\d+)?)\s*(miles?|mi|km|kilometers?|m|meters?|ft|feet|foot|yd|yards?|cm|centimeters?|mm|millimeters?|in|inches?|inch)\b",
            )?,
            count_re: Regex::new(
                r"(?i)(\d+(?:\.\d+)?)\s*(times?|people|persons?|items?|units?|cars?|houses?|books?|files?|projects?|tasks?|events?|meetings?|emails?|messages?|calls?|visits?|orders?|products?|customers?|users?|accounts?|transactions?)\b",
            )?,
            percentage_re: Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(%|percent|pct)\b")?,
            duration_re: Regex::new(
                r"(?i)(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?|ms|milliseconds?|days?)\b",
            )?,
            temperature_re: Regex::new(
                r"(?i)(\d+(?:\.\d+)?)\s*(?:°?\s*)(f|fahrenheit|c|celsius|k|kelvin)\b",
            )?,
            weight_re: Regex::new(
                r"(?i)(\d+(?:\.\d+)?)\s*(lbs?|pounds?|oz|ounces?|kg|kilograms?|g|grams?|mg|milligrams?|tons?|tonnes?)\b",
            )?,
            data_size_re: Regex::new(
                r"(?i)(\d+(?:\.\d+)?)\s*(kb|mb|gb|tb|bytes?|kilobytes?|megabytes?|gigabytes?|terabytes?)\b",
            )?,
            range_re: Regex::new(
                r"(?i)(?:between\s+)?(\d+(?:\.\d+)?)\s*(?:[-–]\s*|to\s+|and\s+)(\d+(?:\.\d+)?)\s*(miles?|km|dollars?|\$|%|hours?|minutes?|days?|lbs?|kg|items?|people|times?)?\b",
            )?,
            approx_re: Regex::new(
                r"(?i)(?:about|approximately|around|roughly|nearly|~)\s*(\d+(?:\.\d+)?)\s*(miles?|km|dollars?|\$|%|hours?|minutes?|days?|lbs?|kg|items?|people|times?)?\b",
            )?,
        })
    }

    pub fn extract(&self, text: &str) -> Vec<(String, f64, Option<String>)> {
        let mut results = Vec::new();

        for cap in self.money_re.captures_iter(text) {
            let val_str = cap.get(1).or_else(|| cap.get(2)).map(|m| m.as_str());
            if let Some(s) = val_str {
                let cleaned = s.replace(',', "");
                if let Ok(v) = cleaned.parse::<f64>() {
                    results.push(("money".to_string(), v, Some("USD".to_string())));
                }
            }
        }

        for cap in self.distance_re.captures_iter(text) {
            if let (Some(v_match), Some(u_match)) = (cap.get(1), cap.get(2)) {
                if let Ok(v) = v_match.as_str().parse::<f64>() {
                    results.push((
                        format!("distance_{}", u_match.as_str().to_lowercase()),
                        v,
                        Some(u_match.as_str().to_string()),
                    ));
                }
            }
        }

        for cap in self.count_re.captures_iter(text) {
            if let (Some(v_match), Some(u_match)) = (cap.get(1), cap.get(2)) {
                if let Ok(v) = v_match.as_str().parse::<f64>() {
                    results.push((
                        format!("count_{}", u_match.as_str().to_lowercase()),
                        v,
                        Some(u_match.as_str().to_string()),
                    ));
                }
            }
        }

        for cap in self.percentage_re.captures_iter(text) {
            if let Some(v_match) = cap.get(1) {
                if let Ok(v) = v_match.as_str().parse::<f64>() {
                    results.push(("percentage".to_string(), v, Some("%".to_string())));
                }
            }
        }

        for cap in self.duration_re.captures_iter(text) {
            if let (Some(v_match), Some(u_match)) = (cap.get(1), cap.get(2)) {
                if let Ok(v) = v_match.as_str().parse::<f64>() {
                    results.push((
                        format!("duration_{}", u_match.as_str().to_lowercase()),
                        v,
                        Some(u_match.as_str().to_string()),
                    ));
                }
            }
        }

        for cap in self.temperature_re.captures_iter(text) {
            if let (Some(v_match), Some(u_match)) = (cap.get(1), cap.get(2)) {
                if let Ok(v) = v_match.as_str().parse::<f64>() {
                    results.push((
                        format!("temperature_{}", u_match.as_str().to_lowercase()),
                        v,
                        Some(u_match.as_str().to_string()),
                    ));
                }
            }
        }

        for cap in self.weight_re.captures_iter(text) {
            if let (Some(v_match), Some(u_match)) = (cap.get(1), cap.get(2)) {
                if let Ok(v) = v_match.as_str().parse::<f64>() {
                    results.push((
                        format!("weight_{}", u_match.as_str().to_lowercase()),
                        v,
                        Some(u_match.as_str().to_string()),
                    ));
                }
            }
        }

        for cap in self.data_size_re.captures_iter(text) {
            if let (Some(v_match), Some(u_match)) = (cap.get(1), cap.get(2)) {
                if let Ok(v) = v_match.as_str().parse::<f64>() {
                    results.push((
                        format!("data_{}", u_match.as_str().to_lowercase()),
                        v,
                        Some(u_match.as_str().to_string()),
                    ));
                }
            }
        }

        for cap in self.range_re.captures_iter(text) {
            if let (Some(lo_match), Some(hi_match)) = (cap.get(1), cap.get(2)) {
                if let (Ok(lo), Ok(hi)) =
                    (lo_match.as_str().parse::<f64>(), hi_match.as_str().parse::<f64>())
                {
                    let midpoint = (lo + hi) / 2.0;
                    let unit_str = cap.get(3).map(|m| m.as_str().to_string());
                    let category = MetricCategory::from_unit(unit_str.as_deref());
                    results.push((category.label("range"), midpoint, unit_str));
                }
            }
        }

        for cap in self.approx_re.captures_iter(text) {
            if let Some(v_match) = cap.get(1) {
                if let Ok(v) = v_match.as_str().parse::<f64>() {
                    let unit_str = cap.get(2).map(|m| m.as_str().to_string());
                    let category = MetricCategory::from_unit(unit_str.as_deref());
                    results.push((category.label("approx"), v, unit_str));
                }
            }
        }

        results
    }
}
