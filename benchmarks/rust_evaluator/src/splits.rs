//! Frozen dev/test splits.
//!
//! Tuning happens on `dev`; paper numbers come from `test`. Split files are
//! generated once with `make-splits`, committed, and only read afterwards so
//! the partition can never drift between runs.

use anyhow::{Context, Result};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::{DatasetKind, Instance};

#[derive(Copy, Clone, Debug, ValueEnum, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Dev,
    Test,
    All,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Dev => "dev",
            Split::Test => "test",
            Split::All => "all",
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SplitManifest {
    pub dataset: String,
    pub seed: u64,
    pub dev_fraction: f64,
    /// What a split unit is: `question` (LongMemEval, each instance has its
    /// own haystack) or `conversation` (LoCoMo, questions share a haystack).
    pub unit: String,
    pub dev: Vec<String>,
    pub test: Vec<String>,
}

/// 64-bit FNV-1a. Stable across Rust versions, unlike `DefaultHasher`.
fn fnv1a(seed: u64, text: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64 ^ seed;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn question_id(instance: &Instance, index: usize) -> String {
    instance.question_id.clone().unwrap_or_else(|| format!("idx-{index}"))
}

/// Split unit for an instance. LoCoMo questions about one conversation must
/// land on the same side, otherwise dev tuning leaks into test. LongMemEval
/// abstention variants (`<id>_abs`) reuse their base question's haystack, so
/// they are grouped with it.
fn unit_id(kind: DatasetKind, instance: &Instance, index: usize) -> String {
    match kind {
        DatasetKind::Longmemeval => {
            let id = question_id(instance, index);
            id.strip_suffix("_abs").map(str::to_string).unwrap_or(id)
        }
        DatasetKind::Locomo => {
            instance.entity_id.clone().unwrap_or_else(|| question_id(instance, index))
        }
    }
}

pub fn default_manifest_path(dataset_path: &str) -> PathBuf {
    let stem = Path::new(dataset_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "dataset".to_string());
    let file = format!("{stem}.json");
    for dir in ["benchmarks/splits", "../splits"] {
        if Path::new(dir).is_dir() {
            return Path::new(dir).join(&file);
        }
    }
    Path::new("benchmarks/splits").join(file)
}

pub fn build_manifest(
    kind: DatasetKind,
    dataset_path: &str,
    dataset: &[Instance],
    dev_fraction: f64,
    seed: u64,
) -> SplitManifest {
    // Strata: question type for LongMemEval, a single stratum for LoCoMo
    // (conversations mix all categories).
    let mut strata: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut seen = HashSet::new();
    for (index, instance) in dataset.iter().enumerate() {
        let unit = unit_id(kind, instance, index);
        if !seen.insert(unit.clone()) {
            continue;
        }
        let stratum = match kind {
            DatasetKind::Longmemeval => {
                instance.question_type.clone().unwrap_or_else(|| "unknown".to_string())
            }
            DatasetKind::Locomo => "all".to_string(),
        };
        strata.entry(stratum).or_default().push(unit);
    }

    let mut dev = Vec::new();
    let mut test = Vec::new();
    for units in strata.values_mut() {
        units.sort_by_key(|unit| (fnv1a(seed, unit), unit.clone()));
        let dev_count = ((units.len() as f64) * dev_fraction).round() as usize;
        let dev_count = dev_count.clamp(usize::from(units.len() > 1), units.len());
        dev.extend(units[..dev_count].iter().cloned());
        test.extend(units[dev_count..].iter().cloned());
    }
    dev.sort();
    test.sort();

    SplitManifest {
        dataset: Path::new(dataset_path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        seed,
        dev_fraction,
        unit: match kind {
            DatasetKind::Longmemeval => "question",
            DatasetKind::Locomo => "conversation",
        }
        .to_string(),
        dev,
        test,
    }
}

pub fn write_manifest(path: &Path, manifest: &SplitManifest, force: bool) -> Result<()> {
    if path.exists() && !force {
        anyhow::bail!(
            "Split file {} already exists. Splits are frozen; pass --force only if you are \
             deliberately re-partitioning (this invalidates earlier dev/test results).",
            path.display()
        );
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    let body = serde_json::to_string_pretty(manifest)?;
    fs::write(path, body + "\n").with_context(|| format!("Failed to write {}", path.display()))
}

pub fn apply_split(
    kind: DatasetKind,
    dataset: Vec<Instance>,
    split: Split,
    manifest_path: &Path,
) -> Result<Vec<Instance>> {
    if split == Split::All {
        return Ok(dataset);
    }
    let data = fs::read_to_string(manifest_path).with_context(|| {
        format!(
            "Failed to read split file {}. Generate it once with `make-splits`, or pass --split all.",
            manifest_path.display()
        )
    })?;
    let manifest: SplitManifest = serde_json::from_str(&data)
        .with_context(|| format!("Failed to parse split file {}", manifest_path.display()))?;

    let known: HashSet<&str> =
        manifest.dev.iter().chain(manifest.test.iter()).map(String::as_str).collect();
    let wanted: HashSet<&str> = match split {
        Split::Dev => manifest.dev.iter().map(String::as_str).collect(),
        Split::Test => manifest.test.iter().map(String::as_str).collect(),
        Split::All => unreachable!(),
    };

    let total = dataset.len();
    let mut unknown = 0usize;
    let filtered: Vec<Instance> = dataset
        .into_iter()
        .enumerate()
        .filter_map(|(index, instance)| {
            let unit = unit_id(kind, &instance, index);
            if !known.contains(unit.as_str()) {
                unknown += 1;
            }
            wanted.contains(unit.as_str()).then_some(instance)
        })
        .collect();

    if unknown > 0 {
        anyhow::bail!(
            "{unknown}/{total} dataset questions are not in split file {}. The dataset changed \
             since the split was frozen; results would not be comparable.",
            manifest_path.display()
        );
    }
    if filtered.is_empty() {
        anyhow::bail!("Split '{}' selected zero questions.", split.as_str());
    }
    println!("Split: {} ({} of {} questions)", split.as_str(), filtered.len(), total);
    Ok(filtered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instance(id: &str, entity: &str, qtype: &str) -> Instance {
        Instance {
            question_id: Some(id.to_string()),
            entity_id: Some(entity.to_string()),
            question_type: Some(qtype.to_string()),
            question_date: None,
            question: String::new(),
            haystack_dates: vec![],
            haystack_sessions: vec![],
            haystack_session_ids: vec![],
            answer_session_ids: vec![],
            answer: None,
        }
    }

    #[test]
    fn fnv_is_stable() {
        assert_eq!(fnv1a(0, ""), 0xcbf29ce484222325);
        assert_eq!(fnv1a(0, "a"), 0xaf63dc4c8601ec8c);
    }

    #[test]
    fn stratified_split_is_deterministic_and_disjoint() {
        let data: Vec<_> = (0..40)
            .map(|i| instance(&format!("q{i}"), "e", if i % 2 == 0 { "a" } else { "b" }))
            .collect();
        let m1 = build_manifest(DatasetKind::Longmemeval, "x.json", &data, 0.3, 7);
        let m2 = build_manifest(DatasetKind::Longmemeval, "x.json", &data, 0.3, 7);
        assert_eq!(m1.dev, m2.dev);
        assert_eq!(m1.dev.len(), 12);
        assert_eq!(m1.dev.len() + m1.test.len(), 40);
        let dev: HashSet<_> = m1.dev.iter().collect();
        assert!(m1.test.iter().all(|id| !dev.contains(id)));
    }

    #[test]
    fn locomo_keeps_conversations_together() {
        let data: Vec<_> = (0..30)
            .map(|i| instance(&format!("c{}/q{i}", i % 10), &format!("c{}", i % 10), "x"))
            .collect();
        let m = build_manifest(DatasetKind::Locomo, "locomo10.json", &data, 0.3, 7);
        assert_eq!(m.dev.len(), 3);
        assert_eq!(m.test.len(), 7);
    }
}
