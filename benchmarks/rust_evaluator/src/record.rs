//! Reproducible run records.
//!
//! Every recall/LLM run writes one JSON file with provenance (commit, config,
//! engine, host, dataset hash, split) and metrics computed from per-question
//! rows. Paper tables are generated only from these files (`report`).

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::QueryTimings;

const BOOTSTRAP_RESAMPLES: usize = 1000;
const BOOTSTRAP_SEED: u64 = 0x5eed_7e11_0db0;

#[derive(Serialize, Clone, Debug)]
pub struct QuestionRecord {
    pub question_id: String,
    pub question_type: String,
    /// At least one gold session in the top-k sessions (LongMemEval recall_any).
    pub hit_any: bool,
    /// Every gold session in the top-k sessions (LongMemEval recall_all).
    pub hit_all: bool,
    pub ndcg: f64,
    /// `None` in recall-only runs.
    pub answer_correct: Option<bool>,
    /// The question failed with an error; counted as incorrect, never dropped.
    pub errored: bool,
    pub query_ms: u64,
    pub context_tokens: u64,
    pub timings: QueryTimings,
}

#[derive(Default, Serialize, Clone, Debug)]
pub struct IngestStats {
    pub entities: u64,
    pub memories: u64,
    pub wall_ms: u64,
    pub timestamp_parse_failures: u64,
}

pub struct RunContext<'a> {
    pub mode: &'a str,
    pub dataset_kind: &'a str,
    pub dataset_path: &'a str,
    pub split: &'a str,
    pub top_k: usize,
    pub config: Value,
    pub engine_url: &'a str,
    pub engine_api_key: Option<&'a str>,
    pub started_ms: u64,
}

/// Session-level retrieval quality for one question.
pub fn session_metrics(retrieved: &[String], gold: &[String], top_k: usize) -> (bool, bool, f64) {
    if gold.is_empty() {
        return (false, false, 0.0);
    }
    let hit_any = gold.iter().any(|g| retrieved.contains(g));
    let hit_all = gold.iter().all(|g| retrieved.contains(g));
    let dcg: f64 = retrieved
        .iter()
        .enumerate()
        .filter(|(_, s)| gold.contains(s))
        .map(|(rank, _)| 1.0 / ((rank + 2) as f64).log2())
        .sum();
    let ideal: f64 =
        (0..gold.len().min(top_k.max(1))).map(|rank| 1.0 / ((rank + 2) as f64).log2()).sum();
    (hit_any, hit_all, if ideal > 0.0 { dcg / ideal } else { 0.0 })
}

pub fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn percentile_u64(samples: &[u64], pct: f64) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut ordered = samples.to_vec();
    ordered.sort_unstable();
    let rank = ((ordered.len() as f64) * pct / 100.0).ceil() as usize;
    ordered[rank.saturating_sub(1).min(ordered.len() - 1)]
}

fn latency_summary(samples: &[u64]) -> Value {
    let mean = if samples.is_empty() {
        0.0
    } else {
        samples.iter().sum::<u64>() as f64 / samples.len() as f64
    };
    json!({
        "mean": mean,
        "p50": percentile_u64(samples, 50.0),
        "p95": percentile_u64(samples, 95.0),
        "p99": percentile_u64(samples, 99.0),
        "max": samples.iter().copied().max().unwrap_or(0),
    })
}

/// splitmix64; deterministic so CIs are reproducible from the record alone.
struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

/// Mean with a 95% percentile-bootstrap confidence interval.
pub fn mean_with_ci(values: &[f64]) -> Value {
    if values.is_empty() {
        return json!({ "mean": null, "ci95": null, "n": 0 });
    }
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let mut rng = SplitMix(BOOTSTRAP_SEED);
    let mut means: Vec<f64> = (0..BOOTSTRAP_RESAMPLES)
        .map(|_| (0..n).map(|_| values[(rng.next() % n as u64) as usize]).sum::<f64>() / n as f64)
        .collect();
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = means[(BOOTSTRAP_RESAMPLES as f64 * 0.025) as usize];
    let hi = means[((BOOTSTRAP_RESAMPLES as f64 * 0.975) as usize).min(BOOTSTRAP_RESAMPLES - 1)];
    json!({ "mean": mean, "ci95": [lo, hi], "n": n })
}

fn bools(rows: &[&QuestionRecord], f: impl Fn(&QuestionRecord) -> Option<bool>) -> Vec<f64> {
    rows.iter().filter_map(|r| f(r)).map(|b| if b { 1.0 } else { 0.0 }).collect()
}

fn quality_block(rows: &[&QuestionRecord]) -> Value {
    json!({
        "n": rows.len(),
        "errored": rows.iter().filter(|r| r.errored).count(),
        // Errored questions count as failures everywhere: dropping them would
        // silently inflate every metric.
        "recall_any": mean_with_ci(&bools(rows, |r| Some(r.hit_any && !r.errored))),
        "recall_all": mean_with_ci(&bools(rows, |r| Some(r.hit_all && !r.errored))),
        "ndcg": mean_with_ci(&rows.iter().map(|r| if r.errored { 0.0 } else { r.ndcg }).collect::<Vec<_>>()),
        "accuracy": mean_with_ci(&bools(rows, |r| {
            if r.errored { Some(false) } else { r.answer_correct }
        })),
    })
}

fn stage_latencies(rows: &[&QuestionRecord]) -> Value {
    type Getter = fn(&QueryTimings) -> u64;
    let stages: [(&str, Getter); 15] = [
        ("planning", |t| t.planning_ms),
        ("route", |t| t.route_ms),
        ("embed", |t| t.embed_ms),
        ("ann", |t| t.ann_ms),
        ("fts", |t| t.fts_ms),
        ("cards", |t| t.card_ms),
        ("rerank", |t| t.rerank_ms),
        ("preference", |t| t.preference_ms),
        ("graph", |t| t.graph_ms),
        ("session", |t| t.session_ms),
        ("fuse", |t| t.fuse_ms),
        ("hydrate", |t| t.hydrate_ms),
        ("hydrate_obs", |t| t.hydrate_obs_ms),
        ("trace", |t| t.trace_ms),
        ("engine_total", |t| t.total_ms),
    ];
    let ok: Vec<&&QuestionRecord> = rows.iter().filter(|r| !r.errored).collect();
    let mut out = serde_json::Map::new();
    for (name, get) in stages {
        let samples: Vec<u64> = ok.iter().map(|r| get(&r.timings)).collect();
        out.insert(name.to_string(), latency_summary(&samples));
    }
    Value::Object(out)
}

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn command_line(cmd: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(cmd).args(args).output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (output.status.success() && !text.is_empty()).then_some(text)
}

fn host_info() -> Value {
    let cpu = command_line("sysctl", &["-n", "machdep.cpu.brand_string"]).or_else(|| {
        fs::read_to_string("/proc/cpuinfo").ok().and_then(|info| {
            info.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
    });
    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        "cpu": cpu,
        "mem_bytes": command_line("sysctl", &["-n", "hw.memsize"]).and_then(|s| s.parse::<u64>().ok()),
        "gpu": command_line("nvidia-smi", &["--query-gpu=name,memory.total", "--format=csv,noheader"]),
        "hostname": command_line("hostname", &[]),
    })
}

fn file_fnv(path: &str) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Some(format!("{hash:016x}"))
}

async fn engine_info(client: &reqwest::Client, url: &str, api_key: Option<&str>) -> Value {
    let mut request = client.get(format!("{url}/version"));
    if let Some(key) = api_key {
        request = request.header("x-api-key", key);
    }
    match request.send().await {
        Ok(response) if response.status().is_success() => {
            response.json::<Value>().await.unwrap_or(Value::Null)
        }
        _ => Value::Null,
    }
}

pub fn runs_dir() -> PathBuf {
    for dir in ["benchmarks/runs", "../runs"] {
        if Path::new(dir).is_dir() {
            return PathBuf::from(dir);
        }
    }
    PathBuf::from("benchmarks/runs")
}

pub async fn write_run_record(
    client: &reqwest::Client,
    ctx: &RunContext<'_>,
    rows: &[QuestionRecord],
    ingest: &IngestStats,
    out_dir: &Path,
) -> Result<PathBuf> {
    let all: Vec<&QuestionRecord> = rows.iter().collect();
    let mut by_type: BTreeMap<&str, Vec<&QuestionRecord>> = BTreeMap::new();
    for row in rows {
        by_type.entry(row.question_type.as_str()).or_default().push(row);
    }
    let per_type: BTreeMap<&str, Value> =
        by_type.iter().map(|(k, v)| (*k, quality_block(v))).collect();

    let ok: Vec<&QuestionRecord> = rows.iter().filter(|r| !r.errored).collect();
    let query_ms: Vec<u64> = ok.iter().map(|r| r.query_ms).collect();
    let context_tokens: Vec<u64> = ok.iter().map(|r| r.context_tokens).collect();
    let finished_ms = now_ms();

    let record = json!({
        "schema": 1,
        "mode": ctx.mode,
        "started_ms": ctx.started_ms,
        "finished_ms": finished_ms,
        "git": {
            "commit": git(&["rev-parse", "HEAD"]),
            "dirty": git(&["status", "--porcelain", "--untracked-files=no"]).map(|s| !s.is_empty()),
        },
        "argv": std::env::args().collect::<Vec<_>>(),
        "dataset": {
            "kind": ctx.dataset_kind,
            "path": ctx.dataset_path,
            "fnv64": file_fnv(ctx.dataset_path),
            "split": ctx.split,
        },
        "top_k": ctx.top_k,
        "config": ctx.config,
        "engine": engine_info(client, ctx.engine_url, ctx.engine_api_key).await,
        "host": host_info(),
        "metrics": {
            "overall": quality_block(&all),
            "per_type": per_type,
            "latency_ms": {
                "query_client": latency_summary(&query_ms),
                "stages": stage_latencies(&all),
            },
            "context_tokens": latency_summary(&context_tokens),
            "ingest": {
                "entities": ingest.entities,
                "memories": ingest.memories,
                "wall_ms": ingest.wall_ms,
                "timestamp_parse_failures": ingest.timestamp_parse_failures,
                "memories_per_sec": if ingest.wall_ms > 0 {
                    ingest.memories as f64 * 1000.0 / ingest.wall_ms as f64
                } else { 0.0 },
            },
        },
        "questions": rows,
    });

    fs::create_dir_all(out_dir)
        .with_context(|| format!("Failed to create runs dir {}", out_dir.display()))?;
    let stem = Path::new(ctx.dataset_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "dataset".into());
    let path = out_dir.join(format!("{}_{}_{}_{}.json", finished_ms, stem, ctx.split, ctx.mode));
    fs::write(&path, serde_json::to_string_pretty(&record)? + "\n")
        .with_context(|| format!("Failed to write {}", path.display()))?;
    Ok(path)
}

/// Render a markdown comparison table from run record files.
pub fn report(paths: &[String]) -> Result<String> {
    let mut out = String::from(
        "| run | commit | dataset | split | n | err | recall_any (95% CI) | recall_all | nDCG | accuracy (95% CI) | query p50/p95/p99 ms | ingest mem/s |\n\
         |---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    let fmt_ci = |v: &Value| -> String {
        match (v["mean"].as_f64(), v["ci95"].as_array()) {
            (Some(m), Some(ci)) => format!(
                "{:.1} ({:.1}–{:.1})",
                m * 100.0,
                ci[0].as_f64().unwrap_or(0.0) * 100.0,
                ci[1].as_f64().unwrap_or(0.0) * 100.0
            ),
            _ => "–".to_string(),
        }
    };
    let fmt_mean =
        |v: &Value| v["mean"].as_f64().map_or("–".into(), |m| format!("{:.1}", m * 100.0));
    for path in paths {
        let data = fs::read_to_string(path).with_context(|| format!("Failed to read {path}"))?;
        let r: Value = serde_json::from_str(&data).with_context(|| format!("Bad record {path}"))?;
        let m = &r["metrics"];
        let q = &m["latency_ms"]["query_client"];
        let commit = r["git"]["commit"].as_str().map(|c| &c[..c.len().min(8)]).unwrap_or("?");
        let dirty = if r["git"]["dirty"].as_bool().unwrap_or(false) { "*" } else { "" };
        out.push_str(&format!(
            "| {} | {}{} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{}/{} | {:.1} |\n",
            Path::new(path).file_stem().map(|s| s.to_string_lossy()).unwrap_or_default(),
            commit,
            dirty,
            r["dataset"]["kind"].as_str().unwrap_or("?"),
            r["dataset"]["split"].as_str().unwrap_or("?"),
            m["overall"]["n"],
            m["overall"]["errored"],
            fmt_ci(&m["overall"]["recall_any"]),
            fmt_mean(&m["overall"]["recall_all"]),
            fmt_mean(&m["overall"]["ndcg"]),
            fmt_ci(&m["overall"]["accuracy"]),
            q["p50"],
            q["p95"],
            q["p99"],
            m["ingest"]["memories_per_sec"].as_f64().unwrap_or(0.0),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn session_metrics_basic() {
        let (any, all, ndcg) = session_metrics(&s(&["a", "b", "c"]), &s(&["a"]), 3);
        assert!(any && all);
        assert!((ndcg - 1.0).abs() < 1e-9);

        let (any, all, ndcg) = session_metrics(&s(&["x", "a"]), &s(&["a", "b"]), 2);
        assert!(any && !all);
        let expected = (1.0 / 3f64.log2()) / (1.0 + 1.0 / 3f64.log2());
        assert!((ndcg - expected).abs() < 1e-9);

        let (any, _, ndcg) = session_metrics(&s(&["x"]), &s(&["a"]), 1);
        assert!(!any && ndcg == 0.0);
    }

    #[test]
    fn bootstrap_ci_brackets_mean() {
        let values: Vec<f64> = (0..200).map(|i| if i % 4 == 0 { 1.0 } else { 0.0 }).collect();
        let v = mean_with_ci(&values);
        let mean = v["mean"].as_f64().unwrap();
        let ci = v["ci95"].as_array().unwrap();
        assert!((mean - 0.25).abs() < 1e-9);
        assert!(ci[0].as_f64().unwrap() < mean && mean < ci[1].as_f64().unwrap());
        assert_eq!(mean_with_ci(&values), v, "bootstrap must be deterministic");
    }

    #[test]
    fn percentiles() {
        let samples: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile_u64(&samples, 50.0), 50);
        assert_eq!(percentile_u64(&samples, 99.0), 99);
        assert_eq!(percentile_u64(&[], 99.0), 0);
    }
}
