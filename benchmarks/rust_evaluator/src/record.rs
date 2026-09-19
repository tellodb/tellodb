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
    /// Has at least one gold evidence session. Abstention questions have
    /// none, so retrieval recall is undefined for them and they are excluded
    /// from recall/nDCG (they still count for answer accuracy).
    pub answerable: bool,
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
    /// Summed `x-tm-ingest-counts` (expanded, embedded, per derived structure).
    pub counts: BTreeMap<String, u64>,
    /// Database bytes per memory sent, one sample per ingested entity.
    pub db_bytes_per_memory: Vec<f64>,
}

impl IngestStats {
    pub fn add_counts(&mut self, counts: &BTreeMap<String, u64>) {
        for (k, v) in counts {
            *self.counts.entry(k.clone()).or_default() += v;
        }
    }
}

/// Parses `k=v,k=v` from the engine's `x-tm-ingest-counts` header.
pub fn parse_counts_header(value: &str) -> BTreeMap<String, u64> {
    value
        .split(',')
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            Some((k.trim().to_string(), v.trim().parse().ok()?))
        })
        .collect()
}

pub struct RunContext<'a> {
    pub mode: &'a str,
    pub dataset_kind: &'a str,
    pub dataset_path: &'a str,
    pub split: &'a str,
    pub tier: &'a str,
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
        "unanswerable": rows.iter().filter(|r| !r.answerable).count(),
        "recall_any": mean_with_ci(&bools(rows, |r| r.answerable.then_some(r.hit_any && !r.errored))),
        "recall_all": mean_with_ci(&bools(rows, |r| r.answerable.then_some(r.hit_all && !r.errored))),
        "ndcg": mean_with_ci(
            &rows
                .iter()
                .filter(|r| r.answerable)
                .map(|r| if r.errored { 0.0 } else { r.ndcg })
                .collect::<Vec<_>>(),
        ),
        "accuracy": mean_with_ci(&bools(rows, |r| {
            if r.errored { Some(false) } else { r.answer_correct }
        })),
    })
}

fn stage_latencies(rows: &[&QuestionRecord]) -> Value {
    type Getter = fn(&QueryTimings) -> u64;
    let stages: [(&str, Getter); 24] = [
        ("planning", |t| t.planning_ms),
        ("route", |t| t.route_ms),
        ("embed", |t| t.embed_ms),
        ("ann", |t| t.ann_ms),
        ("fts", |t| t.fts_ms),
        ("cards", |t| t.card_ms),
        ("rerank", |t| t.rerank_ms),
        ("preference", |t| t.preference_ms),
        ("graph", |t| t.graph_ms),
        ("score_loop", |t| t.score_loop_us / 1000),
        ("factver", |t| t.factver_us / 1000),
        ("build_cards", |t| t.build_cards_us / 1000),
        ("proof", |t| t.proof_us / 1000),
        ("confidence", |t| t.confidence_us / 1000),
        ("graph_seeds_wall", |t| t.graph_seeds_wall_us / 1000),
        ("graph_links", |t| t.graph_links_us / 1000),
        ("graph_edges", |t| t.graph_edges_us / 1000),
        ("graph_entities", |t| t.graph_entities_us / 1000),
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

fn host_info(tier: &str) -> Value {
    let cpu = command_line("sysctl", &["-n", "machdep.cpu.brand_string"]).or_else(|| {
        fs::read_to_string("/proc/cpuinfo").ok().and_then(|info| {
            info.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
    });
    json!({
        "tier": tier,
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
        "tier": ctx.tier,
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
        "host": host_info(ctx.tier),
        "metrics": {
            "overall": quality_block(&all),
            "per_type": per_type,
            "latency_ms": {
                "query_client": latency_summary(&query_ms),
                "stages": stage_latencies(&all),
            },
            "context_tokens": latency_summary(&context_tokens),
            "rerank_reasons": rerank_reasons(&ok),
            "rerank_applied_rate": if ok.is_empty() { 0.0 } else {
                ok.iter().filter(|r| r.timings.rerank_applied).count() as f64 / ok.len() as f64
            },
            "ingest": {
                "entities": ingest.entities,
                "memories": ingest.memories,
                "wall_ms": ingest.wall_ms,
                "timestamp_parse_failures": ingest.timestamp_parse_failures,
                "counts": ingest.counts,
                "embedded_per_memory": ingest.counts.get("embedded").map(|e| {
                    if ingest.memories > 0 { *e as f64 / ingest.memories as f64 } else { 0.0 }
                }),
                "db_bytes_per_memory": mean(&ingest.db_bytes_per_memory),
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
        "| run | tier | commit | dataset | split | n | err | recall_any (95% CI) | recall_all | nDCG | accuracy (95% CI) | query p50/p95/p99 ms | ingest mem/s |\n\
         |---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
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
        let tier = r["tier"].as_str().or_else(|| r["host"]["tier"].as_str()).unwrap_or("–");
        out.push_str(&format!(
            "| {} | {} | {}{} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{}/{} | {:.1} |\n",
            Path::new(path).file_stem().map(|s| s.to_string_lossy()).unwrap_or_default(),
            tier,
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

    out.push_str("\n### Query Latency Breakdown (Mean ms)\n\n");
    out.push_str("| run | tier | plan | route | embed | ann | fts | cards | rerank | pref | graph | session | fuse | hydr | total |\n");
    out.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for path in paths {
        let data = fs::read_to_string(path).with_context(|| format!("Failed to read {path}"))?;
        let r: Value = serde_json::from_str(&data).with_context(|| format!("Bad record {path}"))?;
        let tier = r["tier"].as_str().or_else(|| r["host"]["tier"].as_str()).unwrap_or("–");
        let st = &r["metrics"]["latency_ms"]["stages"];
        let stage_mean = |name: &str| -> String {
            st[name]["mean"].as_f64().map_or("–".into(), |m| format!("{:.1}", m))
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            Path::new(path).file_stem().map(|s| s.to_string_lossy()).unwrap_or_default(),
            tier,
            stage_mean("planning"),
            stage_mean("route"),
            stage_mean("embed"),
            stage_mean("ann"),
            stage_mean("fts"),
            stage_mean("cards"),
            stage_mean("rerank"),
            stage_mean("preference"),
            stage_mean("graph"),
            stage_mean("session"),
            stage_mean("fuse"),
            stage_mean("hydrate"),
            stage_mean("engine_total"),
        ));
    }

    Ok(out)
}

/// Paired bootstrap 95% CI of `mean(b - a)` over aligned samples.
pub fn paired_delta_ci(a: &[f64], b: &[f64]) -> Option<(f64, f64, f64)> {
    if a.is_empty() || a.len() != b.len() {
        return None;
    }
    let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| y - x).collect();
    let v = mean_with_ci(&diffs);
    let ci = v["ci95"].as_array()?;
    Some((v["mean"].as_f64()?, ci[0].as_f64()?, ci[1].as_f64()?))
}

/// Per-question `(recall_any, ndcg)` of answerable questions, by question id.
fn question_scores(record: &Value) -> BTreeMap<String, (f64, f64)> {
    record["questions"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .filter(|q| q["answerable"].as_bool().unwrap_or(true))
                .filter_map(|q| {
                    let errored = q["errored"].as_bool().unwrap_or(false);
                    let hit = q["hit_any"].as_bool().unwrap_or(false) && !errored;
                    let ndcg = if errored { 0.0 } else { q["ndcg"].as_f64().unwrap_or(0.0) };
                    Some((q["question_id"].as_str()?.to_string(), (f64::from(u8::from(hit)), ndcg)))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// A run's label: the configuration directory it was written to (the sweep
/// scripts name directories after the configuration), else its disabled
/// structures.
fn ablation_label(path: &str, record: &Value) -> String {
    if let Some(dir) = Path::new(path).parent().and_then(|p| p.file_name()) {
        let dir = dir.to_string_lossy();
        if !dir.is_empty() && dir != "runs" && !dir.chars().all(|c| c.is_ascii_digit() || c == '_') {
            return dir.into_owned();
        }
    }
    let disabled: Vec<&str> = record["engine"]["disabled_structures"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if disabled.is_empty() {
        "(none)".to_string()
    } else {
        format!("-{}", disabled.join(",-"))
    }
}

/// Markdown table of each run's change against `baseline`: paired deltas on
/// the questions both runs answered, plus ingest and latency cost changes.
/// "keep?" is `drop` when both quality deltas' CIs contain zero.
pub fn ablation_report(baseline: &str, runs: &[String]) -> Result<String> {
    let load = |path: &str| -> Result<Value> {
        let text = fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {path}"))
    };
    let base = load(baseline)?;
    let base_scores = question_scores(&base);
    let ingest = |r: &Value, key: &str| r["metrics"]["ingest"][key].as_f64();
    let p95 = |r: &Value| r["metrics"]["latency_ms"]["query_client"]["p95"].as_f64();
    let pct = |new: Option<f64>, old: Option<f64>| match (new, old) {
        (Some(n), Some(o)) if o.abs() > f64::EPSILON => format!("{:+.1}%", (n - o) / o * 100.0),
        _ => "–".to_string(),
    };
    let fmt_delta = |d: Option<(f64, f64, f64)>| match d {
        Some((m, lo, hi)) => format!("{:+.1} ({:+.1}…{:+.1})", m * 100.0, lo * 100.0, hi * 100.0),
        None => "–".to_string(),
    };

    let mut out = format!(
        "Baseline: `{}` {} — recall_any {:.1}, nDCG {:.1}, ingest {:.1} mem/s, {:.0} B/mem, {:.2} embedded/mem, query p95 {} ms\n\n",
        baseline,
        ablation_label(baseline, &base),
        base["metrics"]["overall"]["recall_any"]["mean"].as_f64().unwrap_or(0.0) * 100.0,
        base["metrics"]["overall"]["ndcg"]["mean"].as_f64().unwrap_or(0.0) * 100.0,
        ingest(&base, "memories_per_sec").unwrap_or(0.0),
        ingest(&base, "db_bytes_per_memory").unwrap_or(0.0),
        ingest(&base, "embedded_per_memory").unwrap_or(0.0),
        p95(&base).map(|v| v.to_string()).unwrap_or_else(|| "–".into()),
    );
    out.push_str(
        "| config | n paired | Δ recall_any (95% CI) | Δ nDCG (95% CI) | Δ ingest mem/s | Δ bytes/mem | Δ embedded/mem | Δ query p95 | rerank rate | keep? |\n\
         |---|---|---|---|---|---|---|---|---|---|\n",
    );
    for path in runs {
        let run = load(path)?;
        let scores = question_scores(&run);
        let (mut a_hit, mut b_hit, mut a_ndcg, mut b_ndcg) = (vec![], vec![], vec![], vec![]);
        for (qid, (hit, ndcg)) in &base_scores {
            if let Some((run_hit, run_ndcg)) = scores.get(qid) {
                a_hit.push(*hit);
                b_hit.push(*run_hit);
                a_ndcg.push(*ndcg);
                b_ndcg.push(*run_ndcg);
            }
        }
        let d_hit = paired_delta_ci(&a_hit, &b_hit);
        let d_ndcg = paired_delta_ci(&a_ndcg, &b_ndcg);
        let spans_zero =
            |d: Option<(f64, f64, f64)>| d.is_some_and(|(_, lo, hi)| lo <= 0.0 && hi >= 0.0);
        let verdict = if spans_zero(d_hit) && spans_zero(d_ndcg) { "drop" } else { "keep" };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {:.0}% | {} |\n",
            ablation_label(path, &run),
            a_hit.len(),
            fmt_delta(d_hit),
            fmt_delta(d_ndcg),
            pct(ingest(&run, "memories_per_sec"), ingest(&base, "memories_per_sec")),
            pct(ingest(&run, "db_bytes_per_memory"), ingest(&base, "db_bytes_per_memory")),
            pct(ingest(&run, "embedded_per_memory"), ingest(&base, "embedded_per_memory")),
            pct(p95(&run), p95(&base)),
            run["metrics"]["rerank_applied_rate"].as_f64().unwrap_or(0.0) * 100.0,
            verdict,
        ));
    }
    Ok(out)
}

/// Share of questions per engine rerank decision.
fn rerank_reasons(rows: &[&QuestionRecord]) -> Value {
    const NAMES: [&str; 8] = [
        "disabled",
        "too_few_candidates",
        "heuristic_applied",
        "heuristic_skipped",
        "always",
        "gate_uncertain",
        "gate_confident",
        "requested",
    ];
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for row in rows {
        let name = NAMES.get(row.timings.rerank_reason as usize).copied().unwrap_or("unknown");
        *counts.entry(name).or_default() += 1;
    }
    let total = rows.len().max(1) as f64;
    Value::Object(counts.into_iter().map(|(k, v)| (k.to_string(), json!(v as f64 / total))).collect())
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
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
    fn paired_delta_detects_consistent_change() {
        let a: Vec<f64> = (0..100).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        let same = paired_delta_ci(&a, &a).unwrap();
        assert!(same.1 <= 0.0 && same.2 >= 0.0);
        let better: Vec<f64> = a.iter().map(|_| 1.0).collect();
        let (mean, lo, _) = paired_delta_ci(&a, &better).unwrap();
        assert!((mean - 0.5).abs() < 1e-9 && lo > 0.0);
        assert!(paired_delta_ci(&a, &a[..10]).is_none());
    }

    #[test]
    fn percentiles() {
        let samples: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile_u64(&samples, 50.0), 50);
        assert_eq!(percentile_u64(&samples, 99.0), 99);
        assert_eq!(percentile_u64(&[], 99.0), 0);
    }
}
