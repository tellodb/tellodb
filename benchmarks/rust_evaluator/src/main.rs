use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_ENGINE_URL: &str = "http://127.0.0.1:3000";
const DEFAULT_DATASET_PATH: &str = "benchmarks/LongMemEval/data/longmemeval_s_cleaned.json";
const DEFAULT_LOCOMO_DATASET_PATH: &str = "benchmarks/LoCoMo/data/locomo10.json";
const DEFAULT_OPENROUTER_MODEL: &str = "openai/gpt-4.1-mini";
const DEFAULT_OPENROUTER_JUDGE_MODEL: &str = "openai/gpt-4.1-mini";
const DEFAULT_GROQ_MODEL: &str = "openai/gpt-oss-120b";
const DEFAULT_GROQ_JUDGE_MODEL: &str = "openai/gpt-oss-120b";
const DEFAULT_INGEST_BATCH_SIZE: usize = 8;
const RESET_CONFIRM_PHRASE: &str = "delete-all-data";
const OPENROUTER_MAX_ATTEMPTS: usize = 4;

const DIRECT_ANSWER_PROMPT: &str = "\
You are a question-answering system. Based on the retrieved conversation history below, answer the question.\n\n\
Question: {question}\n\
Question Date: {question_date}\n\n\
History Chats:\n{context}\n\n\
Instructions:\n\
{answer_rules}\n\
{focus_rules}\n\
- Extract relevant information from the conversation history to answer the question\n\
- Consider any temporal/date information present in the data\n\
- If the context contains enough information, provide a clear, concise answer\n\
- If the context does not contain enough information, respond with \"I don't know\"\n\
- Base your answer ONLY on the provided context\n\
- Answer in at most 15 words.\n\
- Output ONLY the answer. No introductory text, no explanation, no commentary.\n\
- Do not start with \"We need to\", \"Let me\", \"Based on\", \"I think\", \"Thus\", \"So\", or similar.\n\n\
Examples:\n\
  Q: Where did Caroline move from 4 years ago?\n\
  A: Sweden\n\n\
  Q: What activities does Melanie partake in?\n\
  A: pottery, camping, painting, swimming\n\n\
Answer:";

const CON_ANSWER_PROMPT: &str = "\
You are a question-answering system. Based on the retrieved conversation history below, answer the question.\n\n\
Question: {question}\n\
Question Date: {question_date}\n\n\
History Chats:\n{context}\n\n\
Instructions:\n\
{answer_rules}\n\
{focus_rules}\n\
- Extract relevant information from the conversation history to answer the question\n\
- Consider any temporal/date information present in the data\n\
- If the context contains enough information, provide a clear, concise answer\n\
- If the context does not contain enough information, respond with \"I don't know\"\n\
- Base your answer ONLY on the provided context\n\
- Answer in at most 15 words.\n\
- Output ONLY the answer. No introductory text, no explanation, no commentary.\n\
- Do not start with \"We need to\", \"Let me\", \"Based on\", \"I think\", \"Thus\", \"So\", or similar.\n\n\
Examples:\n\
  Q: Where did Caroline move from 4 years ago?\n\
  A: Sweden\n\n\
  Q: What activities does Melanie partake in?\n\
  A: pottery, camping, painting, swimming\n\n\
Answer:";

const CON_NOTES_PROMPT: &str = "\
I will give you a chat history between you and a user, as well as a question from the user. Write compact reading notes that extract all relevant facts needed to answer the question.\n\
Keep dates, names, quantities, and list items exact. Resolve relative dates against the session date when possible, but preserve uncertainty if the exact calendar answer is not supported. Include all distinct relevant facts needed for the final answer and avoid unrelated facts. If no relevant information is found, output exactly \"empty\".\n\
Output ONLY the extracted notes. No introductory text, no explanation, no commentary.\n\n\
Chat History:\n\
Session Date: {session_date}\n\
Session Content:\n{session_content}\n\n\
Question Date: {question_date}\n\
Question: {question}\n\
Extracted note:";

const CON_SEPARATE_ANSWER_PROMPT: &str = "\
You are a question-answering system. Based on the reading notes extracted from history chats, answer the question.\n\n\
Question: {question}\n\
Current Date: {question_date}\n\n\
Reading Notes:\n{notes}\n\n\
Instructions:\n\
{answer_rules}\n\
{focus_rules}\n\
- Extract relevant information from the notes to answer the question\n\
- Consider any temporal/date information present in the notes\n\
- If the notes contain enough information, provide a clear, concise final answer\n\
- Do not quote evidence, do not mention sessions, and do not explain your reasoning\n\
- If the notes do not contain enough information, respond with \"I don't know\"\n\
- Base your answer ONLY on the provided notes\n\
- Answer in at most 15 words.\n\
- Output ONLY the answer. No introductory text, no explanation, no commentary.\n\
- Do not start with \"We need to\", \"Let me\", \"Based on\", \"I think\", \"Thus\", \"So\", or similar.\n\n\
Examples:\n\
  Q: Where did Caroline move from 4 years ago?\n\
  A: Sweden\n\n\
  Q: What activities does Melanie partake in?\n\
  A: pottery, camping, painting, swimming\n\n\
Answer:";

const CON_SUMMARIZE_PROMPT: &str = "\
I will give you several history chats between you and a user, and a question. Extract all relevant facts needed to answer the question into a dense bulleted list.\n\
Keep dates, names, quantities, and list items exact. Ignore conversational noise. If no relevant information is found, output exactly \"empty\".\n\
Output ONLY the bulleted list. No introductory text, no explanation, no commentary.\n\n\
Chat History:\n\
{context}\n\n\
Question Date: {question_date}\n\
Question: {question}\n\
Extracted Notes:";

const NUMERIC_EXTRACTION_PROMPT: &str = "\
You are a strict numeric extraction engine for memory QA.\n\
From the provided evidence, extract the numbers needed to answer the question.\n\
Return JSON only (no markdown) with this exact schema:\n\
{\"operation\":\"count|sum|average|difference|none\",\"values\":[number],\"unit\":\"string\",\"no_answer\":true|false}\n\
Rules:\n\
- Use only evidence-supported values.\n\
- values must be plain numbers (no currency symbols, commas, or words).\n\
- For count questions, include one value per counted item/event (usually 1s, or explicit counts).\n\
- For sum questions, include each additive amount once.\n\
- For average questions, include all component values explicitly referenced by the question.\n\
- Set operation to \"none\" and no_answer=true if evidence is insufficient.\n\n\
Question: {question}\n\
Evidence:\n{evidence}\n\
JSON:";

const ANSWER_VERIFY_PROMPT: &str = "\
You are validating a candidate answer against evidence.\n\
If the candidate is correct and fully supported, reply exactly PASS.\n\
If the candidate is incorrect but evidence supports a short corrected answer, reply exactly CORRECT: <answer>.\n\
If evidence is insufficient, reply exactly IDK.\n\n\
Question: {question}\n\
Candidate: {candidate}\n\
Evidence:\n{evidence}\n\
Verifier Output:";

#[derive(Copy, Clone, Debug, ValueEnum, Eq, PartialEq)]
enum ReaderMode {
    Direct,
    Con,
    ConSeparate,
    Summary,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum AnswerIntent {
    NumericAggregation,
    TemporalAggregation,
    Recommendation,
    General,
}

#[derive(Copy, Clone, Debug, ValueEnum, Eq, PartialEq)]
enum DatasetKind {
    Longmemeval,
    Locomo,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct IngestTimings {
    pub embed: u64,
    pub storage: u64,
    pub fts: u64,
    pub vector: u64,
    pub graph: u64,
    pub fact: u64,
    pub derived_embed: u64,
    pub derived_ner: u64,
    pub total: u64,
}

#[derive(Parser, Debug)]
#[command(name = "rust_evaluator")]
#[command(
    about = "Memory benchmark evaluator for LongMemEval and LoCoMo with optional OpenRouter grading"
)]
struct Cli {
    #[command(subcommand)]
    mode: EvalMode,

    #[arg(
        long,
        global = true,
        env = "TELLODB_ENGINE_URL",
        default_value = DEFAULT_ENGINE_URL
    )]
    engine_url: String,

    #[arg(long, global = true, env = "TELLODB_API_KEY")]
    engine_api_key: Option<String>,

    #[arg(long, global = true)]
    dataset: Option<String>,

    #[arg(long, global = true, value_enum, default_value_t = DatasetKind::Longmemeval)]
    dataset_kind: DatasetKind,

    #[arg(long, global = true, default_value_t = 500)]
    limit: usize,

    #[arg(long, global = true, default_value_t = 0)]
    start_index: usize,

    #[arg(long, global = true, default_value_t = 6)]
    top_k: usize,

    #[arg(long, global = true, default_value_t = 1)]
    ingest_concurrency: usize,

    #[arg(long, global = true)]
    dev_fast: bool,

    #[arg(long, global = true)]
    enable_neural_rerank: bool,

    #[arg(long, global = true)]
    enable_semantic_dedup: bool,

    #[arg(long, global = true)]
    enable_consolidation: bool,

    #[arg(long, global = true, value_enum, default_value_t = ReaderMode::Con)]
    reader_mode: ReaderMode,

    #[arg(long, global = true, default_value_t = 3)]
    max_chunks_per_session: usize,

    #[arg(long, global = true)]
    reset_first: bool,

    #[arg(long, global = true)]
    dump_candidates_jsonl: Option<String>,

    #[arg(long, global = true)]
    dump_gold_ranks_jsonl: Option<String>,
}

#[derive(Subcommand, Debug)]
enum EvalMode {
    Recall,
    Llm {
        #[arg(long, default_value = DEFAULT_OPENROUTER_MODEL)]
        openrouter_model: String,

        #[arg(long, default_value = DEFAULT_OPENROUTER_JUDGE_MODEL)]
        openrouter_judge_model: String,

        #[arg(long, default_value_t = false)]
        use_groq: bool,

        #[arg(long, default_value = DEFAULT_GROQ_MODEL)]
        groq_model: String,

        #[arg(long, default_value_t = false)]
        use_groq_judge: bool,

        #[arg(long, default_value = DEFAULT_GROQ_JUDGE_MODEL)]
        groq_judge_model: String,

        #[arg(long)]
        output_jsonl: Option<String>,
    },
    Smoke {
        #[arg(long, default_value = DEFAULT_OPENROUTER_MODEL)]
        openrouter_model: String,

        #[arg(long, default_value = DEFAULT_OPENROUTER_JUDGE_MODEL)]
        openrouter_judge_model: String,

        #[arg(long, default_value_t = false)]
        use_groq: bool,

        #[arg(long, default_value = DEFAULT_GROQ_MODEL)]
        groq_model: String,

        #[arg(long, default_value_t = false)]
        use_groq_judge: bool,

        #[arg(long, default_value = DEFAULT_GROQ_JUDGE_MODEL)]
        groq_judge_model: String,
    },
    AnalyzeGoldRanks {
        #[arg(long)]
        input: String,

        #[arg(long, default_value_t = 20)]
        examples: usize,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Turn {
    role: String,
    content: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
struct Instance {
    question_id: Option<String>,
    #[serde(default)]
    entity_id: Option<String>,
    question_type: Option<String>,
    question_date: Option<String>,
    question: String,
    haystack_dates: Vec<String>,
    haystack_sessions: Vec<Vec<Turn>>,
    haystack_session_ids: Vec<String>,
    answer_session_ids: Vec<String>,
    answer: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct LocomoSample {
    #[serde(default)]
    sample_id: Option<String>,
    conversation: LocomoConversation,
    #[serde(default)]
    qa: Vec<LocomoQuestion>,
}

#[derive(Deserialize, Debug)]
struct LocomoConversation {
    #[serde(default, rename = "speaker_a")]
    _speaker_a: Option<String>,
    #[serde(default, rename = "speaker_b")]
    _speaker_b: Option<String>,
    #[serde(flatten)]
    fields: HashMap<String, serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct LocomoQuestion {
    question: String,
    #[serde(default)]
    answer: Option<serde_json::Value>,
    #[serde(default, rename = "adversarial_answer")]
    _adversarial_answer: Option<serde_json::Value>,
    #[serde(default)]
    evidence: Vec<String>,
    #[serde(default)]
    category: Option<u64>,
}

#[derive(Deserialize, Debug)]
struct LocomoTurn {
    speaker: String,
    dia_id: String,
    text: String,
}

#[derive(Serialize, Clone)]
struct IngestPayload<'a> {
    entity_id: &'a str,
    memory_id: String,
    timestamp: u64,
    textual_content: String,
    relations: Vec<(&'a str, &'a str, &'a str)>,
    enable_semantic_dedup: bool,
    enable_consolidation: bool,
}

#[derive(Serialize, Clone)]
struct BatchIngestPayload<'a> {
    items: Vec<IngestPayload<'a>>,
}

#[derive(Serialize)]
struct QueryPayload<'a> {
    textual_query: &'a str,
    limit: usize,
    entity_id: Option<&'a str>,
    enable_neural_rerank: bool,
}

#[derive(Deserialize, Debug, Clone)]
struct QueryResult {
    memory_id: String,
    textual_content: String,
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    turn_index: usize,
    #[serde(default)]
    similarity: f32,
    #[serde(default)]
    created_at_ms: u64,
}

#[derive(Deserialize, Debug)]
struct NumericExtraction {
    operation: String,
    #[serde(default)]
    values: Vec<f64>,
    #[serde(default)]
    unit: String,
    #[serde(default)]
    no_answer: bool,
}

#[derive(Clone)]
struct EvalConfig {
    dataset_kind: DatasetKind,
    engine_url: String,
    engine_api_key: Option<String>,
    top_k: usize,
    ingest_concurrency: usize,
    dev_fast: bool,
    enable_neural_rerank: bool,
    enable_semantic_dedup: bool,
    enable_consolidation: bool,
    reader_mode: ReaderMode,
    max_chunks_per_session: usize,
    reset_first: bool,
    dump_candidates_jsonl: Option<String>,
    dump_gold_ranks_jsonl: Option<String>,
}

#[derive(Default)]
struct EvalTotals {
    retrieval_correct: usize,
    answer_correct: usize,
    skipped: usize,
    evaluated: usize,
    category_stats: BTreeMap<String, CategoryStats>,
    timings: AggregateTimings,
    quality: QualityStats,
    latency_samples: LatencySamples,
}

#[derive(Default)]
struct CategoryStats {
    evaluated: usize,
    retrieval_correct: usize,
    answer_correct: usize,
}

#[derive(Default)]
struct QualityStats {
    retrieval_hit_answer_pass: usize,
    retrieval_hit_answer_fail: usize,
    retrieval_miss_answer_pass: usize,
    retrieval_miss_answer_fail: usize,
    idk_answers: usize,
    context_tokens_total: u128,
    context_token_samples: Vec<u128>,
}

#[derive(Default)]
struct LatencySamples {
    query_ms: Vec<u128>,
    answer_ms: Vec<u128>,
    judge_ms: Vec<u128>,
    total_ms: Vec<u128>,
}

#[derive(Default)]
struct AggregateTimings {
    ingest_ms: u128,
    query_ms: u128,
    pack_ms: u128,
    answer_ms: u128,
    judge_ms: u128,
    total_ms: u128,
    inner_embed_ms: u128,
    inner_ann_ms: u128,
    inner_rerank_ms: u128,
    inner_fts_ms: u128,
    inner_card_ms: u128,
    inner_fuse_ms: u128,
    inner_hydrate_ms: u128,
    inner_session_ms: u128,
    inner_query_total_ms: u128,
    inner_route_ms: u128,
    inner_planning_ms: u128,
    inner_hydrate_obs_ms: u128,
    inner_trace_ms: u128,
    inner_preference_ms: u128,
    inner_graph_ms: u128,
    routed_sessions: u128,
    memory_card_hits: u128,
    temporal_event_hits: u128,
    shadow_question_hits: u128,
    facet_posting_hits: u128,
    mem_scene_hits: u128,
    scoped_ann_attempts: u128,
    scoped_primary_hits: u128,
}

#[derive(Default, Clone, Copy)]
struct QueryTimings {
    route_ms: u64,
    embed_ms: u64,
    ann_ms: u64,
    rerank_ms: u64,
    fts_ms: u64,
    card_ms: u64,
    fuse_ms: u64,
    hydrate_ms: u64,
    session_ms: u64,
    preference_ms: u64,
    graph_ms: u64,
    planning_ms: u64,
    hydrate_obs_ms: u64,
    trace_ms: u64,
    total_ms: u64,
    routed_sessions: u64,
    memory_card_hits: u64,
    temporal_event_hits: u64,
    shadow_question_hits: u64,
    facet_posting_hits: u64,
    mem_scene_hits: u64,
    scoped_ann_attempts: u64,
    scoped_primary_hits: u64,
}

struct QueryResponse {
    results: Vec<QueryResult>,
    timings: QueryTimings,
}

struct PackedSession {
    session_id: String,
    session_date: String,
    session_focus: String,
    chunks: Vec<QueryResult>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(90))
        .build()
        .context("Failed to build HTTP client")?;

    let config = EvalConfig {
        dataset_kind: cli.dataset_kind,
        engine_url: cli.engine_url,
        engine_api_key: cli.engine_api_key,
        top_k: cli.top_k,
        ingest_concurrency: cli.ingest_concurrency,
        dev_fast: cli.dev_fast,
        enable_neural_rerank: cli.enable_neural_rerank,
        enable_semantic_dedup: cli.enable_semantic_dedup,
        enable_consolidation: cli.enable_consolidation,
        reader_mode: cli.reader_mode,
        max_chunks_per_session: cli.max_chunks_per_session,
        reset_first: cli.reset_first,
        dump_candidates_jsonl: cli.dump_candidates_jsonl.clone(),
        dump_gold_ranks_jsonl: cli.dump_gold_ranks_jsonl.clone(),
    };

    match cli.mode {
        EvalMode::Recall => {
            let dataset_path = resolve_dataset_path(cli.dataset_kind, cli.dataset.as_deref())?;
            let dataset = load_dataset(cli.dataset_kind, &dataset_path)?;
            run_recall(&client, &dataset, cli.start_index, cli.limit, &config).await?
        }
        EvalMode::Llm {
            openrouter_model,
            openrouter_judge_model,
            use_groq,
            groq_model,
            use_groq_judge,
            groq_judge_model,
            output_jsonl,
        } => {
            let dataset_path = resolve_dataset_path(cli.dataset_kind, cli.dataset.as_deref())?;
            let dataset = load_dataset(cli.dataset_kind, &dataset_path)?;
            run_llm(
                &client,
                &dataset,
                cli.start_index,
                cli.limit,
                &config,
                &openrouter_model,
                &openrouter_judge_model,
                use_groq,
                &groq_model,
                use_groq_judge,
                &groq_judge_model,
                output_jsonl.as_deref(),
            )
            .await?
        }
        EvalMode::Smoke {
            openrouter_model,
            openrouter_judge_model,
            use_groq,
            groq_model,
            use_groq_judge,
            groq_judge_model,
        } => {
            let dataset = vec![smoke_instance()];
            let mut smoke_config = config.clone();
            smoke_config.dataset_kind = DatasetKind::Locomo;
            run_llm(
                &client,
                &dataset,
                0,
                1,
                &smoke_config,
                &openrouter_model,
                &openrouter_judge_model,
                use_groq,
                &groq_model,
                use_groq_judge,
                &groq_judge_model,
                None,
            )
            .await?
        }
        EvalMode::AnalyzeGoldRanks { input, examples } => {
            analyze_gold_rank_dump(&input, examples)?;
        }
    }

    Ok(())
}

fn resolve_dataset_path(dataset_kind: DatasetKind, user_path: Option<&str>) -> Result<String> {
    if let Some(path) = user_path {
        return Ok(path.to_string());
    }

    let candidates: &[&str] = match dataset_kind {
        DatasetKind::Longmemeval => &[
            "../LongMemEval/data/longmemeval_s_cleaned.json",
            DEFAULT_DATASET_PATH,
        ],
        DatasetKind::Locomo => &[
            "../LoCoMo/data/locomo10.json",
            "../locomo/data/locomo10.json",
            DEFAULT_LOCOMO_DATASET_PATH,
            "benchmarks/locomo/data/locomo10.json",
            "benchmarks/locomo10.json",
        ],
    };

    for candidate in candidates {
        if Path::new(candidate).exists() {
            return Ok(candidate.to_string());
        }
    }

    let benchmark_name = match dataset_kind {
        DatasetKind::Longmemeval => "LongMemEval",
        DatasetKind::Locomo => "LoCoMo",
    };
    anyhow::bail!("Could not find {benchmark_name} dataset. Pass --dataset explicitly.");
}

fn load_dataset(dataset_kind: DatasetKind, path: &str) -> Result<Vec<Instance>> {
    let data =
        fs::read_to_string(path).with_context(|| format!("Failed to read dataset at {path}"))?;
    match dataset_kind {
        DatasetKind::Longmemeval => {
            let dataset = serde_json::from_str(&data)
                .with_context(|| format!("Failed to parse dataset JSON at {path}"))?;
            Ok(dataset)
        }
        DatasetKind::Locomo => {
            let samples: Vec<LocomoSample> = serde_json::from_str(&data)
                .with_context(|| format!("Failed to parse LoCoMo dataset JSON at {path}"))?;
            let dataset = normalize_locomo_samples(samples);
            if dataset.is_empty() {
                anyhow::bail!(
                    "LoCoMo normalization produced zero evaluable questions. Check the dataset file."
                );
            }
            Ok(dataset)
        }
    }
}

fn smoke_instance() -> Instance {
    let entity_id = "smoke-sample".to_string();
    let haystack_session_ids = vec!["session_1".to_string(), "session_2".to_string()];
    let haystack_dates = vec!["5 March 2024".to_string(), "12 March 2024".to_string()];
    let haystack_sessions = vec![
        vec![
            Turn {
                role: "user".to_string(),
                content: serde_json::Value::String(
                    "I redesigned my bookstore last weekend. I picked warm wooden shelves, comfy reading chairs, and lots of plants."
                        .to_string(),
                ),
            },
            Turn {
                role: "assistant".to_string(),
                content: serde_json::Value::String(
                    "That sounds cozy. The shelves, chairs, and plants should make the store feel welcoming."
                        .to_string(),
                ),
            },
        ],
        vec![
            Turn {
                role: "user".to_string(),
                content: serde_json::Value::String(
                    "Customers keep saying the reading chairs make the shop comfortable, and the plants make it feel calm."
                        .to_string(),
                ),
            },
            Turn {
                role: "assistant".to_string(),
                content: serde_json::Value::String(
                    "The redesign seems to be working well."
                        .to_string(),
                ),
            },
        ],
    ];

    Instance {
        question_id: Some("smoke/q1".to_string()),
        entity_id: Some(entity_id),
        question_type: Some("locomo-factoid".to_string()),
        question_date: Some("12 March 2024".to_string()),
        question: "What three things did Alex add to the bookstore redesign?".to_string(),
        haystack_dates,
        haystack_sessions,
        haystack_session_ids,
        answer_session_ids: vec!["session_1".to_string()],
        answer: Some(serde_json::Value::String(
            "wooden shelves, reading chairs, and plants".to_string(),
        )),
    }
}

fn normalize_locomo_samples(samples: Vec<LocomoSample>) -> Vec<Instance> {
    let mut normalized = Vec::new();

    for (sample_idx, sample) in samples.into_iter().enumerate() {
        let sample_id = sample
            .sample_id
            .unwrap_or_else(|| format!("locomo-sample-{}", sample_idx + 1));
        let ordered_sessions = ordered_locomo_sessions(&sample.conversation);
        if ordered_sessions.is_empty() {
            continue;
        }

        let haystack_session_ids = ordered_sessions
            .iter()
            .map(|(session_id, _, _)| session_id.clone())
            .collect::<Vec<_>>();
        let haystack_dates = ordered_sessions
            .iter()
            .map(|(_, session_date, _)| session_date.clone())
            .collect::<Vec<_>>();
        let haystack_sessions = ordered_sessions
            .iter()
            .map(|(_, _, turns)| {
                turns
                    .iter()
                    .map(|turn| Turn {
                        role: turn.speaker.clone(),
                        content: serde_json::Value::String(if turn.dia_id.is_empty() {
                            turn.text.clone()
                        } else {
                            format!("[{}] {}", turn.dia_id, turn.text)
                        }),
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let question_date = haystack_dates.last().cloned();

        for (qa_idx, qa) in sample.qa.into_iter().enumerate() {
            let Some(answer) = qa.answer else {
                continue;
            };
            if qa.evidence.is_empty() {
                continue;
            }

            let answer_session_ids = qa
                .evidence
                .iter()
                .filter_map(|dialog_id| locomo_dialog_id_to_session_id(dialog_id))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();

            if answer_session_ids.is_empty() {
                continue;
            }

            normalized.push(Instance {
                question_id: Some(format!("{sample_id}/q{}", qa_idx + 1)),
                entity_id: Some(sample_id.clone()),
                question_type: Some(locomo_category_name(qa.category)),
                question_date: question_date.clone(),
                question: qa.question,
                haystack_dates: haystack_dates.clone(),
                haystack_sessions: haystack_sessions.clone(),
                haystack_session_ids: haystack_session_ids.clone(),
                answer_session_ids,
                answer: Some(answer),
            });
        }
    }

    normalized
}

fn ordered_locomo_sessions(
    conversation: &LocomoConversation,
) -> Vec<(String, String, Vec<LocomoTurn>)> {
    let mut indices = BTreeSet::new();
    for key in conversation.fields.keys() {
        if let Some(index) = parse_locomo_session_index(key) {
            indices.insert(index);
        }
    }

    let mut sessions = Vec::new();
    for index in indices {
        let session_key = format!("session_{index}");
        let date_key = format!("session_{index}_date_time");
        let Some(session_value) = conversation.fields.get(&session_key) else {
            continue;
        };
        let Ok(turns) = serde_json::from_value::<Vec<LocomoTurn>>(session_value.clone()) else {
            continue;
        };
        let session_date = conversation
            .fields
            .get(&date_key)
            .and_then(|value| value.as_str())
            .unwrap_or("unknown")
            .to_string();
        sessions.push((session_key, session_date, turns));
    }

    sessions
}

fn parse_locomo_session_index(key: &str) -> Option<usize> {
    let rest = key.strip_prefix("session_")?;
    let digits = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        None
    } else {
        digits.parse::<usize>().ok()
    }
}

fn locomo_dialog_id_to_session_id(dialog_id: &str) -> Option<String> {
    let rest = dialog_id.strip_prefix('D')?;
    let session_digits = rest.split(':').next()?;
    let session_number = session_digits.parse::<usize>().ok()?;
    Some(format!("session_{session_number}"))
}

fn locomo_category_name(category: Option<u64>) -> String {
    match category {
        Some(1) => "locomo-factoid".to_string(),
        Some(2) => "temporal-reasoning".to_string(),
        Some(3) => "multi-session".to_string(),
        Some(4) => "locomo-open-domain".to_string(),
        Some(5) => "locomo-adversarial".to_string(),
        Some(other) => format!("locomo-category-{other}"),
        None => "locomo-unknown".to_string(),
    }
}

fn breakdown_label(dataset_kind: DatasetKind, question_type: &str) -> Option<&'static str> {
    match dataset_kind {
        DatasetKind::Locomo => match question_type {
            "locomo-factoid" => Some("Single-Hop"),
            "multi-session" => Some("Multi-Hop"),
            "locomo-open-domain" => Some("Open Domain"),
            "temporal-reasoning" => Some("Temporal"),
            _ => None,
        },
        DatasetKind::Longmemeval => match question_type {
            "single-session-user" => Some("Single-User"),
            "single-session-assistant" => Some("Assistant"),
            "single-session-preference" => Some("Preference"),
            "multi-session" => Some("Multi-Session"),
            "temporal-reasoning" => Some("Temporal"),
            "knowledge-update" => Some("Knowledge"),
            _ => None,
        },
    }
}

fn update_category_stats(
    totals: &mut EvalTotals,
    dataset_kind: DatasetKind,
    question_type: &str,
    retrieval_correct: bool,
    answer_correct: bool,
) {
    let Some(label) = breakdown_label(dataset_kind, question_type) else {
        return;
    };
    let entry = totals.category_stats.entry(label.to_string()).or_default();
    entry.evaluated += 1;
    if retrieval_correct {
        entry.retrieval_correct += 1;
    }
    if answer_correct {
        entry.answer_correct += 1;
    }
}

async fn run_recall(
    client: &Client,
    dataset: &[Instance],
    start_index: usize,
    limit: usize,
    config: &EvalConfig,
) -> Result<()> {
    println!("Running memory recall benchmark");
    println!("Engine URL: {}", config.engine_url);
    println!(
        "Engine auth: {}",
        if config.engine_api_key.is_some() {
            "x-api-key"
        } else {
            "disabled"
        }
    );
    let available = dataset.len().saturating_sub(start_index);
    println!("Questions: {}", limit.min(available));
    println!("Start index: {}", start_index);
    println!("Top-K sessions: {}", config.top_k);
    println!("Neural rerank: {}", config.enable_neural_rerank);
    println!("Semantic dedup: {}", config.enable_semantic_dedup);
    println!("Consolidation: {}", config.enable_consolidation);
    println!("Dev fast: {}", config.dev_fast);
    println!("Max chunks per session: {}", config.max_chunks_per_session);
    println!("Reset first: {}", config.reset_first);

    if config.reset_first {
        reset_engine(client, config).await?;
    }

    let mut candidate_dump_writer = if let Some(path) = config.dump_candidates_jsonl.as_deref() {
        Some(BufWriter::new(
            File::create(path).with_context(|| format!("Failed to create {path}"))?,
        ))
    } else {
        None
    };
    let mut gold_rank_writer = if let Some(path) = config.dump_gold_ranks_jsonl.as_deref() {
        Some(BufWriter::new(
            File::create(path).with_context(|| format!("Failed to create {path}"))?,
        ))
    } else {
        None
    };
    let mut totals = EvalTotals::default();
    let max_questions = limit.min(available);
    let mut active_entity_id: Option<String> = None;

    for (i, instance) in dataset
        .iter()
        .enumerate()
        .skip(start_index)
        .take(max_questions)
    {
        let display_index = i - start_index;
        let entity_id = benchmark_entity_id(instance, i);
        let eval_question_id = evaluation_question_id(instance, i);

        if instance.haystack_sessions.is_empty() {
            println!("Skipping empty haystack");
            totals.skipped += 1;
            continue;
        }

        let total_start = Instant::now();

        let ingest_ms = if active_entity_id.as_deref() != Some(entity_id.as_str()) {
            if config.dataset_kind == DatasetKind::Locomo && active_entity_id.is_some() {
                reset_engine(client, config).await?;
            }
            let ingest_start = Instant::now();
            ingest_instance(client, config, &entity_id, instance).await?;
            active_entity_id = Some(entity_id.clone());
            ingest_start.elapsed().as_millis()
        } else {
            0
        };

        let query_start = Instant::now();
        let query = query_engine(client, config, &entity_id, &instance.question).await?;
        let query_ms = query_start.elapsed().as_millis();

        let pack_start = Instant::now();
        let retrieved_sessions = extract_top_sessions(&entity_id, &query.results, config.top_k);
        let hit = instance
            .answer_session_ids
            .iter()
            .any(|ans| retrieved_sessions.contains(ans));
        let question_type = instance
            .question_type
            .as_deref()
            .unwrap_or("single-session-user");
        if let Some(writer) = candidate_dump_writer.as_mut() {
            write_candidate_dump(
                writer,
                &eval_question_id,
                instance,
                question_type,
                &retrieved_sessions,
                &query.results,
                config.top_k * 8,
            )?;
        }
        if let Some(writer) = gold_rank_writer.as_mut() {
            write_gold_rank_dump(
                writer,
                &eval_question_id,
                instance,
                question_type,
                &retrieved_sessions,
                &query.results,
            )?;
        }
        let pack_ms = pack_start.elapsed().as_millis();

        let total_ms = total_start.elapsed().as_millis();
        totals.evaluated += 1;
        totals.timings.ingest_ms += ingest_ms;
        totals.timings.query_ms += query_ms;
        totals.timings.pack_ms += pack_ms;
        totals.timings.total_ms += total_ms;
        totals.timings.inner_embed_ms += query.timings.embed_ms as u128;
        totals.timings.inner_route_ms += query.timings.route_ms as u128;
        totals.timings.inner_ann_ms += query.timings.ann_ms as u128;
        totals.timings.inner_rerank_ms += query.timings.rerank_ms as u128;
        totals.timings.inner_fts_ms += query.timings.fts_ms as u128;
        totals.timings.inner_card_ms += query.timings.card_ms as u128;
        totals.timings.inner_fuse_ms += query.timings.fuse_ms as u128;
        totals.timings.inner_hydrate_ms += query.timings.hydrate_ms as u128;
        totals.timings.inner_preference_ms += query.timings.preference_ms as u128;
        totals.timings.inner_graph_ms += query.timings.graph_ms as u128;
        totals.timings.inner_session_ms += query.timings.session_ms as u128;
        totals.timings.inner_query_total_ms += query.timings.total_ms as u128;
        add_query_diagnostics(&mut totals.timings, query.timings);

        if hit {
            totals.retrieval_correct += 1;
        } else {
            println!(
                "\n\n--- Question {}/{} [{}] ---",
                display_index + 1,
                max_questions,
                eval_question_id
            );
            println!("FAIL: evidence session not found in Top-{}", config.top_k);
            println!("  Question: {}", instance.question);
            println!("  Question Type: {}", question_type);
            println!(
                "  Expected Answer Sessions: {:?}",
                instance.answer_session_ids
            );
            println!("  Retrieved Sessions: {:?}", retrieved_sessions);
            println!(
                "  timings: ingest={}ms | query={}ms (route={} embed={} ann={} rerank={} fts={} card={} pref={} graph={} session={} fuse={} hydrate={} visible={} other={} total={}) | pack={}ms",
                ingest_ms,
                query_ms,
                query.timings.route_ms,
                query.timings.embed_ms,
                query.timings.ann_ms,
                query.timings.rerank_ms,
                query.timings.fts_ms,
                query.timings.card_ms,
                query.timings.preference_ms,
                query.timings.graph_ms,
                query.timings.session_ms,
                query.timings.fuse_ms,
                query.timings.hydrate_ms,
                query.timings.visible_stage_ms(),
                query.timings.other_ms(),
                query.timings.total_ms,
                pack_ms
            );
            println!(
                "  routing: routed_sessions={} cards={} events={} shadows={} facets={} scenes={} scoped_ann_attempts={} scoped_primary_hits={}",
                query.timings.routed_sessions,
                query.timings.memory_card_hits,
                query.timings.temporal_event_hits,
                query.timings.shadow_question_hits,
                query.timings.facet_posting_hits,
                query.timings.mem_scene_hits,
                query.timings.scoped_ann_attempts,
                query.timings.scoped_primary_hits,
            );
            println!("--------------------------------------------------");
        }
        update_category_stats(&mut totals, config.dataset_kind, question_type, hit, false);

        let _recall = totals.retrieval_correct as f64 / totals.evaluated as f64 * 100.0;
        // print!(
        //     "\rSTATUS: [ {}/{} ] | Recall@{}: {:.1}%        ",
        //     i + 1,
        //     max_questions,
        //     config.top_k,
        //     recall
        // );
        let _ = std::io::stdout().flush();
    }

    print_recall_summary(&totals, config.top_k, config.dataset_kind);
    Ok(())
}

async fn run_llm(
    client: &Client,
    dataset: &[Instance],
    start_index: usize,
    limit: usize,
    config: &EvalConfig,
    openrouter_model: &str,
    openrouter_judge_model: &str,
    use_groq: bool,
    groq_model: &str,
    use_groq_judge: bool,
    groq_judge_model: &str,
    output_jsonl: Option<&str>,
) -> Result<()> {
    let answer_model = if use_groq {
        groq_model
    } else {
        openrouter_model
    };
    let judge_model = if use_groq_judge {
        groq_judge_model
    } else {
        openrouter_judge_model
    };

    println!("Running memory end-to-end LLM benchmark");
    println!("Engine URL: {}", config.engine_url);
    println!(
        "Engine auth: {}",
        if config.engine_api_key.is_some() {
            "x-api-key"
        } else {
            "disabled"
        }
    );
    let available = dataset.len().saturating_sub(start_index);
    println!("Questions: {}", limit.min(available));
    println!("Start index: {}", start_index);
    println!("Answer model: {}", answer_model);
    println!(
        "Answer provider: {}",
        if use_groq { "Groq" } else { "OpenRouter" }
    );
    println!("Judge model: {}", judge_model);
    println!(
        "Judge provider: {}",
        if use_groq_judge { "Groq" } else { "OpenRouter" }
    );
    println!("Reader mode: {:?}", config.reader_mode);
    println!("Top-K sessions: {}", config.top_k);
    println!("Max chunks per session: {}", config.max_chunks_per_session);
    println!("Neural rerank: {}", config.enable_neural_rerank);
    println!("Semantic dedup: {}", config.enable_semantic_dedup);
    println!("Consolidation: {}", config.enable_consolidation);
    println!("Dev fast: {}", config.dev_fast);
    println!("Reset first: {}", config.reset_first);

    if config.reset_first {
        reset_engine(client, config).await?;
    }

    let mut output_writer = if let Some(path) = output_jsonl {
        Some(BufWriter::new(
            File::create(path).with_context(|| format!("Failed to create {path}"))?,
        ))
    } else {
        None
    };
    let mut candidate_dump_writer = if let Some(path) = config.dump_candidates_jsonl.as_deref() {
        Some(BufWriter::new(
            File::create(path).with_context(|| format!("Failed to create {path}"))?,
        ))
    } else {
        None
    };
    let mut gold_rank_writer = if let Some(path) = config.dump_gold_ranks_jsonl.as_deref() {
        Some(BufWriter::new(
            File::create(path).with_context(|| format!("Failed to create {path}"))?,
        ))
    } else {
        None
    };

    let mut totals = EvalTotals::default();
    let max_questions = limit.min(available);
    let mut active_entity_id: Option<String> = None;

    for (i, instance) in dataset
        .iter()
        .enumerate()
        .skip(start_index)
        .take(max_questions)
    {
        let display_index = i - start_index;
        let entity_id = benchmark_entity_id(instance, i);
        let eval_question_id = evaluation_question_id(instance, i);
        println!(
            "\n--- Question {}/{} [{}] ---",
            display_index + 1,
            max_questions,
            eval_question_id
        );
        println!("Q: {}", instance.question);

        if instance.haystack_sessions.is_empty() {
            println!("Skipping empty haystack");
            totals.skipped += 1;
            continue;
        }

        let total_start = Instant::now();

        let ingest_ms = if active_entity_id.as_deref() != Some(entity_id.as_str()) {
            if config.dataset_kind == DatasetKind::Locomo && active_entity_id.is_some() {
                reset_engine(client, config).await?;
            }
            let ingest_start = Instant::now();
            ingest_instance(client, config, &entity_id, instance).await?;
            active_entity_id = Some(entity_id.clone());
            ingest_start.elapsed().as_millis()
        } else {
            println!("Reusing indexed conversation for entity {}", entity_id);
            0
        };

        let query_start = Instant::now();
        let query = query_engine(client, config, &entity_id, &instance.question).await?;
        let query_ms = query_start.elapsed().as_millis();
        let question_type = instance
            .question_type
            .as_deref()
            .unwrap_or("single-session-user");

        let pack_start = Instant::now();
        let retrieved_sessions = extract_top_sessions(&entity_id, &query.results, config.top_k);
        let retrieval_hit = instance
            .answer_session_ids
            .iter()
            .any(|ans| retrieved_sessions.contains(ans));
        if let Some(writer) = candidate_dump_writer.as_mut() {
            write_candidate_dump(
                writer,
                &eval_question_id,
                instance,
                question_type,
                &retrieved_sessions,
                &query.results,
                config.top_k * 8,
            )?;
        }
        if let Some(writer) = gold_rank_writer.as_mut() {
            write_gold_rank_dump(
                writer,
                &eval_question_id,
                instance,
                question_type,
                &retrieved_sessions,
                &query.results,
            )?;
        }
        let packed_sessions = pack_top_sessions(
            instance,
            &query.results,
            config.top_k,
            config.max_chunks_per_session,
        );
        let pack_ms = pack_start.elapsed().as_millis();

        totals.evaluated += 1;
        if retrieval_hit {
            totals.retrieval_correct += 1;
        }

        if packed_sessions.is_empty() {
            println!("No packed session context retrieved");
            continue;
        }

        let answer_start = Instant::now();
        let prediction = generate_answer(
            client,
            config.reader_mode,
            answer_model,
            instance,
            &packed_sessions,
            use_groq,
        )
        .await
        .with_context(|| format!("Failed to answer question {}", eval_question_id))?;
        let answer_ms = answer_start.elapsed().as_millis();

        if let Some(writer) = output_writer.as_mut() {
            writeln!(
                writer,
                "{}",
                json!({
                    "question_id": eval_question_id,
                    "hypothesis": prediction.clone(),
                })
            )?;
            writer.flush()?;
        }

        let ground_truth = ground_truth(instance);
        // Strip reasoning preamble for judging so the judge sees a clean answer.
        // The raw prediction (with reasoning) is already saved to outputs.jsonl above.
        let prediction_for_judge = strip_model_reasoning(&prediction);
        let judge_start = Instant::now();
        let judge_prompt = get_anscheck_prompt(
            question_type,
            &instance.question,
            &ground_truth,
            &prediction_for_judge,
            eval_question_id.ends_with("_abs"),
        );
        let verdict = call_answer_model(
            client,
            &judge_prompt,
            judge_model,
            4096,
            "none",
            use_groq_judge,
        )
        .await
        .with_context(|| format!("Failed to judge question {}", eval_question_id))?;
        let judge_ms = judge_start.elapsed().as_millis();

        let answer_correct = parse_judge_verdict(&verdict);
        if answer_correct {
            totals.answer_correct += 1;
        }
        update_category_stats(
            &mut totals,
            config.dataset_kind,
            question_type,
            retrieval_hit,
            answer_correct,
        );

        let total_ms = total_start.elapsed().as_millis();
        totals.timings.ingest_ms += ingest_ms;
        totals.timings.query_ms += query_ms;
        totals.timings.pack_ms += pack_ms;
        totals.timings.answer_ms += answer_ms;
        totals.timings.judge_ms += judge_ms;
        totals.timings.total_ms += total_ms;
        totals.timings.inner_embed_ms += query.timings.embed_ms as u128;
        totals.timings.inner_route_ms += query.timings.route_ms as u128;
        totals.timings.inner_ann_ms += query.timings.ann_ms as u128;
        totals.timings.inner_rerank_ms += query.timings.rerank_ms as u128;
        totals.timings.inner_fts_ms += query.timings.fts_ms as u128;
        totals.timings.inner_card_ms += query.timings.card_ms as u128;
        totals.timings.inner_fuse_ms += query.timings.fuse_ms as u128;
        totals.timings.inner_hydrate_ms += query.timings.hydrate_ms as u128;
        totals.timings.inner_preference_ms += query.timings.preference_ms as u128;
        totals.timings.inner_graph_ms += query.timings.graph_ms as u128;
        totals.timings.inner_session_ms += query.timings.session_ms as u128;
        totals.timings.inner_planning_ms += query.timings.planning_ms as u128;
        totals.timings.inner_hydrate_obs_ms += query.timings.hydrate_obs_ms as u128;
        totals.timings.inner_trace_ms += query.timings.trace_ms as u128;
        totals.timings.inner_query_total_ms += query.timings.total_ms as u128;
        add_query_diagnostics(&mut totals.timings, query.timings);

        let retrieval_symbol = if retrieval_hit {
            "retrieval=pass"
        } else {
            "retrieval=fail"
        };
        let answer_symbol = if answer_correct {
            "answer=pass"
        } else {
            "answer=fail"
        };
        let context_tokens = packed_sessions
            .iter()
            .flat_map(|s| s.chunks.iter())
            .map(|c| c.textual_content.split_whitespace().count())
            .sum::<usize>();
        record_llm_diagnostics(
            &mut totals,
            retrieval_hit,
            answer_correct,
            &prediction,
            context_tokens,
            query_ms,
            answer_ms,
            judge_ms,
            total_ms,
        );

        println!("{retrieval_symbol} | {answer_symbol}");
        println!("Predicted: {}", truncate_chars(&prediction_for_judge, 160));
        println!("Ground truth: {}", truncate_chars(&ground_truth, 160));
        println!(
            "timings: ingest={}ms | query={}ms (plan={} route={} embed={} ann={} rerank={} fts={} card={} pref={} graph={} session={} fuse={} hydrate={} hyd_obs={} trace={} visible={} other={} total={}) | pack={}ms | answer={}ms | judge={}ms",
            ingest_ms,
            query_ms,
            query.timings.planning_ms,
            query.timings.route_ms,
            query.timings.embed_ms,
            query.timings.ann_ms,
            query.timings.rerank_ms,
            query.timings.fts_ms,
            query.timings.card_ms,
            query.timings.preference_ms,
            query.timings.graph_ms,
            query.timings.session_ms,
            query.timings.fuse_ms,
            query.timings.hydrate_ms,
            query.timings.hydrate_obs_ms,
            query.timings.trace_ms,
            query.timings.visible_stage_ms(),
            query.timings.other_ms(),
            query.timings.total_ms,
            pack_ms,
            answer_ms,
            judge_ms,
        );
        println!(
            "routing: routed_sessions={} cards={} events={} shadows={} facets={} scenes={} scoped_ann_attempts={} scoped_primary_hits={}",
            query.timings.routed_sessions,
            query.timings.memory_card_hits,
            query.timings.temporal_event_hits,
            query.timings.shadow_question_hits,
            query.timings.facet_posting_hits,
            query.timings.mem_scene_hits,
            query.timings.scoped_ann_attempts,
            query.timings.scoped_primary_hits,
        );

        let recall_pct = totals.retrieval_correct as f64 / totals.evaluated as f64 * 100.0;
        let accuracy_pct = totals.answer_correct as f64 / totals.evaluated as f64 * 100.0;
        println!(
            "📊 STATUS: [ {}/{} ] | Recall: {:.1}% | Accuracy: {:.1}%",
            i + 1,
            max_questions,
            recall_pct,
            accuracy_pct
        );
        println!(
            "📈 MemScore: {:.1}% / {:.1}% / {}ms / {}tok",
            recall_pct, accuracy_pct, query_ms, context_tokens,
        );
    }

    print_llm_summary(&totals, config.top_k, config.dataset_kind);
    Ok(())
}

fn with_engine_auth(
    builder: reqwest::RequestBuilder,
    config: &EvalConfig,
) -> reqwest::RequestBuilder {
    if let Some(api_key) = config.engine_api_key.as_deref() {
        builder.header("x-api-key", api_key)
    } else {
        builder
    }
}

async fn reset_engine(client: &Client, config: &EvalConfig) -> Result<()> {
    println!("Resetting engine state...");
    let payload = json!({
        "confirm": RESET_CONFIRM_PHRASE,
        "clear_embedding_cache": true,
    });
    let reset_paths = ["/admin/reset", "/v1/admin/reset"];
    let mut last_error = None;

    for reset_path in reset_paths {
        let url = format!("{}{}", config.engine_url, reset_path);
        let response = with_engine_auth(client.post(&url), config)
            .json(&payload)
            .send()
            .await;

        match response {
            Ok(response) if response.status().is_success() => {
                let total_ms = header_u64(response.headers(), "x-tm-total-ms");
                println!(
                    "Engine reset complete via {} (engine_total={}ms)",
                    reset_path, total_ms
                );
                return Ok(());
            }
            Ok(response) => {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                last_error = Some(anyhow::anyhow!(
                    "Reset failed on {reset_path} with HTTP {status}: {body}"
                ));
            }
            Err(error) => {
                last_error = Some(error.into());
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Reset failed for unknown reason")))
}

async fn ingest_instance(
    client: &Client,
    config: &EvalConfig,
    q_id: &str,
    instance: &Instance,
) -> Result<()> {
    println!("Ingesting {} sessions...", instance.haystack_sessions.len());
    let mut payloads = Vec::new();
    let session_cap = if config.dev_fast { 12 } else { usize::MAX };
    let turn_cap = if config.dev_fast { 4 } else { usize::MAX };

    for (s_idx, session) in instance
        .haystack_sessions
        .iter()
        .enumerate()
        .take(session_cap)
    {
        let session_id = &instance.haystack_session_ids[s_idx];
        let session_date = instance
            .haystack_dates
            .get(s_idx)
            .map(String::as_str)
            .unwrap_or("unknown");
        let session_focus = build_session_focus(session);

        for t_idx in 0..session.len().min(turn_cap) {
            let window_text =
                build_enriched_window(session_id, session_date, &session_focus, session, t_idx);
            if window_text.is_empty() {
                continue;
            }

            let memory_id = format!("{q_id}::{session_id}::{t_idx}");
            let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64
                + (s_idx as u64 * 1000)
                + t_idx as u64;

            let payload = IngestPayload {
                entity_id: q_id,
                memory_id,
                timestamp,
                textual_content: window_text,
                relations: vec![("", "BELONGS_TO", q_id)],
                enable_semantic_dedup: config.enable_semantic_dedup,
                enable_consolidation: config.enable_consolidation,
            };
            payloads.push(payload);
        }
    }

    let ingest_tasks = payloads
        .chunks(DEFAULT_INGEST_BATCH_SIZE)
        .map(|chunk| BatchIngestPayload {
            items: chunk.to_vec(),
        })
        .map(|payload| {
            let url = format!("{}/ingest/batch", config.engine_url);
            let request_client = client.clone();
            let engine_api_key = config.engine_api_key.clone();
            async move {
                let mut request = request_client.post(&url);
                if let Some(api_key) = engine_api_key.as_deref() {
                    request = request.header("x-api-key", api_key);
                }
                let response = request.json(&payload).send().await?;
                let headers = response.headers();
                let timings = IngestTimings {
                    embed: header_u64(headers, "x-tm-embed-ms"),
                    derived_embed: header_u64(headers, "x-tm-derived-embed-ms"),
                    storage: header_u64(headers, "x-tm-storage-ms"),
                    fts: header_u64(headers, "x-tm-fts-ms"),
                    vector: header_u64(headers, "x-tm-vector-ms"),
                    graph: header_u64(headers, "x-tm-graph-ms"),
                    fact: header_u64(headers, "x-tm-fact-bridge-ms"),
                    derived_ner: header_u64(headers, "x-tm-ner-ms"),
                    total: header_u64(headers, "x-tm-total-ms"),
                };
                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    anyhow::bail!("Batch ingest failed with HTTP {status}: {body}");
                }
                Ok(timings)
            }
        })
        .collect::<Vec<_>>();

    let total_batches = ingest_tasks.len();
    let mut ingest_count = 0u64;
    let mut total_diag = IngestTimings::default();
    let mut completed_batches = 0usize;
    let mut ingest_stream =
        futures::stream::iter(ingest_tasks).buffer_unordered(config.ingest_concurrency);

    while let Some(result) = ingest_stream.next().await {
        let t = result?;
        total_diag.embed += t.embed;
        total_diag.derived_embed += t.derived_embed;
        total_diag.derived_ner += t.derived_ner;
        total_diag.storage += t.storage;
        total_diag.fts += t.fts;
        total_diag.vector += t.vector;
        total_diag.graph += t.graph;
        total_diag.fact += t.fact;
        total_diag.total += t.total;
        ingest_count += 1;
        completed_batches += 1;

        // let should_log_progress = total_batches > 1
        //     && (completed_batches == 1
        //         || completed_batches == total_batches
        //         || completed_batches % 5 == 0);
        let should_log_progress = false;
        if should_log_progress {
            println!(
                "Ingest progress: {}/{} batches complete (avg_total={}ms, avg_embed={}ms)",
                completed_batches,
                total_batches,
                total_diag.total / ingest_count,
                total_diag.embed / ingest_count
            );
        }
    }
    if ingest_count > 0 {
        println!(
            "Ingestion complete. Avg Batch Ingest: {}ms (embed={}ms, derived={}ms, ner={}ms, storage={}ms, fts={}ms, vector={}ms, graph={}ms, fact={}ms)",
            total_diag.total / ingest_count,
            total_diag.embed / ingest_count,
            total_diag.derived_embed / ingest_count,
            total_diag.derived_ner / ingest_count,
            total_diag.storage / ingest_count,
            total_diag.fts / ingest_count,
            total_diag.vector / ingest_count,
            total_diag.graph / ingest_count,
            total_diag.fact / ingest_count
        );
    }

    Ok(())
}

async fn query_engine(
    client: &Client,
    config: &EvalConfig,
    q_id: &str,
    question: &str,
) -> Result<QueryResponse> {
    // println!("Querying engine...");
    let query_payload = QueryPayload {
        textual_query: question,
        limit: config.top_k * 6,
        entity_id: Some(q_id),
        enable_neural_rerank: config.enable_neural_rerank,
    };

    let response = with_engine_auth(
        client.post(format!("{}/query/semantic", config.engine_url)),
        config,
    )
    .json(&query_payload)
    .send()
    .await
    .context("Failed to query semantic endpoint")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("Query failed with HTTP {status}: {body}");
    }

    let timings = parse_query_timings(response.headers());
    let mut results = response
        .json::<Vec<QueryResult>>()
        .await
        .context("Failed to decode query results")?;
    for result in &mut results {
        if result.session_id.is_empty() {
            result.session_id = session_id_from_memory_id(&result.memory_id);
        }
        if result.turn_index == 0 {
            result.turn_index = turn_index_from_memory_id(&result.memory_id);
        }
    }

    Ok(QueryResponse { results, timings })
}

fn header_u64(headers: &reqwest::header::HeaderMap, name: &str) -> u64 {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
}

fn parse_query_timings(headers: &reqwest::header::HeaderMap) -> QueryTimings {
    QueryTimings {
        route_ms: header_u64(headers, "x-tm-route-ms"),
        embed_ms: header_u64(headers, "x-tm-embed-ms"),
        ann_ms: header_u64(headers, "x-tm-ann-ms"),
        rerank_ms: header_u64(headers, "x-tm-rerank-ms"),
        fts_ms: header_u64(headers, "x-tm-fts-ms"),
        card_ms: header_u64(headers, "x-tm-card-ms"),
        fuse_ms: header_u64(headers, "x-tm-fuse-ms"),
        hydrate_ms: header_u64(headers, "x-tm-hydrate-ms"),
        preference_ms: header_u64(headers, "x-tm-preference-ms"),
        graph_ms: header_u64(headers, "x-tm-graph-bridge-ms"),
        session_ms: header_u64(headers, "x-tm-session-ms"),
        planning_ms: header_u64(headers, "x-tm-planning-ms"),
        hydrate_obs_ms: header_u64(headers, "x-tm-hydrate-obs-ms"),
        trace_ms: header_u64(headers, "x-tm-trace-ms"),
        total_ms: header_u64(headers, "x-tm-total-ms"),
        routed_sessions: header_u64(headers, "x-tm-routed-sessions"),
        memory_card_hits: header_u64(headers, "x-tm-memory-card-hits"),
        temporal_event_hits: header_u64(headers, "x-tm-temporal-event-hits"),
        shadow_question_hits: header_u64(headers, "x-tm-shadow-question-hits"),
        facet_posting_hits: header_u64(headers, "x-tm-facet-posting-hits"),
        mem_scene_hits: header_u64(headers, "x-tm-mem-scene-hits"),
        scoped_ann_attempts: header_u64(headers, "x-tm-scoped-ann-attempts"),
        scoped_primary_hits: header_u64(headers, "x-tm-scoped-primary-hits"),
    }
}

impl QueryTimings {
    fn visible_stage_ms(self) -> u64 {
        self.route_ms
            + self.embed_ms
            + self.ann_ms
            + self.rerank_ms
            + self.fts_ms
            + self.card_ms
            + self.preference_ms
            + self.graph_ms
            + self.session_ms
            + self.fuse_ms
            + self.hydrate_ms
            + self.planning_ms
            + self.hydrate_obs_ms
            + self.trace_ms
    }

    fn other_ms(self) -> i64 {
        self.total_ms as i64 - self.visible_stage_ms() as i64
    }
}

fn add_query_diagnostics(timings: &mut AggregateTimings, query: QueryTimings) {
    timings.routed_sessions += query.routed_sessions as u128;
    timings.memory_card_hits += query.memory_card_hits as u128;
    timings.temporal_event_hits += query.temporal_event_hits as u128;
    timings.shadow_question_hits += query.shadow_question_hits as u128;
    timings.facet_posting_hits += query.facet_posting_hits as u128;
    timings.mem_scene_hits += query.mem_scene_hits as u128;
    timings.scoped_ann_attempts += query.scoped_ann_attempts as u128;
    timings.scoped_primary_hits += query.scoped_primary_hits as u128;
}

fn write_candidate_dump<W: Write>(
    writer: &mut W,
    question_id: &str,
    instance: &Instance,
    question_type: &str,
    retrieved_sessions: &[String],
    results: &[QueryResult],
    limit: usize,
) -> Result<()> {
    let gold: HashSet<&str> = instance
        .answer_session_ids
        .iter()
        .map(String::as_str)
        .collect();
    let candidates = results
        .iter()
        .take(limit)
        .enumerate()
        .map(|(rank, result)| {
            let session_id = session_id_for_result(result);
            json!({
                "rank": rank + 1,
                "memory_id": &result.memory_id,
                "source_session_id": session_id,
                "source_turn_index": result.turn_index,
                "final_score": result.similarity,
                "created_at_ms": result.created_at_ms,
                "lexical_hits": lexical_overlap_count(&instance.question, &result.textual_content),
                "entity_hits": entity_overlap_count(&instance.question, &result.textual_content),
                "temporal_hits": temporal_overlap_count(&instance.question, &result.textual_content),
                "is_gold_session": gold.contains(session_id.as_str()),
                "text_preview": truncate_chars(&normalize_text(&result.textual_content), 240),
            })
        })
        .collect::<Vec<_>>();
    let gold_rank_positions = instance
        .answer_session_ids
        .iter()
        .map(|gold_session| {
            let rank = results
                .iter()
                .position(|result| session_id_for_result(result) == *gold_session)
                .map(|idx| idx + 1);
            json!({
                "session_id": gold_session,
                "rank": rank,
                "bucket": rank.map(rank_bucket).unwrap_or("missing"),
            })
        })
        .collect::<Vec<_>>();

    writeln!(
        writer,
        "{}",
        json!({
            "question_id": question_id,
            "question": &instance.question,
            "question_type": question_type,
            "expected_answer_sessions": &instance.answer_session_ids,
            "retrieved_sessions": retrieved_sessions,
            "gold_rank_positions": gold_rank_positions,
            "top_candidates": candidates,
        })
    )?;
    Ok(())
}

fn write_gold_rank_dump<W: Write>(
    writer: &mut W,
    question_id: &str,
    instance: &Instance,
    question_type: &str,
    retrieved_sessions: &[String],
    results: &[QueryResult],
) -> Result<()> {
    let mut first_rank_by_session: HashMap<String, usize> = HashMap::new();
    for (rank, result) in results.iter().enumerate() {
        let session_id = session_id_for_result(result);
        first_rank_by_session.entry(session_id).or_insert(rank + 1);
    }

    let mut gold_rank_positions = Vec::new();
    let mut missing_gold_sessions = Vec::new();
    for gold_session in &instance.answer_session_ids {
        if let Some(rank) = first_rank_by_session.get(gold_session) {
            gold_rank_positions.push(json!({
                "session_id": gold_session,
                "rank": rank,
                "bucket": rank_bucket(*rank),
            }));
        } else {
            missing_gold_sessions.push(gold_session.clone());
        }
    }

    let top_wrong_sessions = retrieved_sessions
        .iter()
        .filter(|sid| !instance.answer_session_ids.contains(*sid))
        .take(16)
        .cloned()
        .collect::<Vec<_>>();

    writeln!(
        writer,
        "{}",
        json!({
            "question_id": question_id,
            "question": &instance.question,
            "question_type": question_type,
            "expected_answer_sessions": &instance.answer_session_ids,
            "retrieved_sessions": retrieved_sessions,
            "gold_rank_positions": gold_rank_positions,
            "missing_gold_sessions": missing_gold_sessions,
            "top_wrong_sessions": top_wrong_sessions,
        })
    )?;
    Ok(())
}

fn rank_bucket(rank: usize) -> &'static str {
    match rank {
        1..=8 => "1-8",
        9..=16 => "9-16",
        17..=32 => "17-32",
        33..=64 => "33-64",
        _ => "65+",
    }
}

#[derive(Default)]
struct GoldRankBucketStats {
    questions: usize,
    question_buckets: BTreeMap<String, usize>,
    gold_session_buckets: BTreeMap<String, usize>,
    missing_gold_sessions: usize,
    expected_gold_sessions: usize,
}

impl GoldRankBucketStats {
    fn add_question(&mut self, bucket: &str, expected: usize, missing: usize) {
        self.questions += 1;
        *self.question_buckets.entry(bucket.to_string()).or_default() += 1;
        self.expected_gold_sessions += expected;
        self.missing_gold_sessions += missing;
    }

    fn add_gold_session_bucket(&mut self, bucket: &str, count: usize) {
        *self.gold_session_buckets.entry(bucket.to_string()).or_default() += count;
    }
}

fn analyze_gold_rank_dump(input: &str, example_limit: usize) -> Result<()> {
    let file = File::open(input).with_context(|| format!("Failed to open gold-rank dump: {input}"))?;
    let reader = BufReader::new(file);
    let mut overall = GoldRankBucketStats::default();
    let mut by_type: BTreeMap<String, GoldRankBucketStats> = BTreeMap::new();
    let mut examples = Vec::new();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("Failed to read line {}", line_idx + 1))?;
        if line.trim().is_empty() {
            continue;
        }

        let value: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("Invalid JSON on line {}", line_idx + 1))?;
        let question_type = value
            .get("question_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let expected = value
            .get("expected_answer_sessions")
            .and_then(|v| v.as_array())
            .map(|v| v.len())
            .unwrap_or(0);
        let missing = value
            .get("missing_gold_sessions")
            .and_then(|v| v.as_array())
            .map(|v| v.len())
            .unwrap_or(0);

        let mut best_bucket = "missing".to_string();
        let mut seen_gold_positions = 0usize;
        if let Some(positions) = value.get("gold_rank_positions").and_then(|v| v.as_array()) {
            for pos in positions {
                if let Some(bucket) = pos.get("bucket").and_then(|v| v.as_str()) {
                    seen_gold_positions += 1;
                    overall.add_gold_session_bucket(bucket, 1);
                    by_type
                        .entry(question_type.clone())
                        .or_default()
                        .add_gold_session_bucket(bucket, 1);
                    if bucket_order(bucket) < bucket_order(&best_bucket) {
                        best_bucket = bucket.to_string();
                    }
                }
            }
        }

        if missing > 0 {
            overall.add_gold_session_bucket("missing", missing);
            by_type
                .entry(question_type.clone())
                .or_default()
                .add_gold_session_bucket("missing", missing);
        }
        if seen_gold_positions == 0 && missing == 0 {
            best_bucket = "missing".to_string();
        }

        overall.add_question(&best_bucket, expected, missing);
        by_type
            .entry(question_type.clone())
            .or_default()
            .add_question(&best_bucket, expected, missing);

        if best_bucket != "1-8" && examples.len() < example_limit {
            examples.push(format_gold_rank_example(&value, &best_bucket));
        }
    }

    println!("Gold-rank dump: {input}");
    print_gold_rank_stats("Overall", &overall);
    println!();
    println!("By question type:");
    for (question_type, stats) in &by_type {
        print_gold_rank_stats(question_type, stats);
    }

    if !examples.is_empty() {
        println!();
        println!("Examples outside top-8:");
        for example in examples {
            println!("{example}");
        }
    }

    Ok(())
}

fn print_gold_rank_stats(label: &str, stats: &GoldRankBucketStats) {
    let top8 = stats.question_buckets.get("1-8").copied().unwrap_or(0);
    let top16 = top8 + stats.question_buckets.get("9-16").copied().unwrap_or(0);
    let top32 = top16 + stats.question_buckets.get("17-32").copied().unwrap_or(0);
    let pct = |count: usize| -> f64 {
        if stats.questions == 0 {
            0.0
        } else {
            count as f64 / stats.questions as f64 * 100.0
        }
    };

    println!(
        "{label}: questions={} top8={:.1}% top16={:.1}% top32={:.1}% missing_gold_sessions={}/{}",
        stats.questions,
        pct(top8),
        pct(top16),
        pct(top32),
        stats.missing_gold_sessions,
        stats.expected_gold_sessions
    );
    println!(
        "  question buckets: {}",
        bucket_counts_line(&stats.question_buckets)
    );
    println!(
        "  gold-session buckets: {}",
        bucket_counts_line(&stats.gold_session_buckets)
    );
}

fn bucket_counts_line(counts: &BTreeMap<String, usize>) -> String {
    ["1-8", "9-16", "17-32", "33-64", "65+", "missing"]
        .iter()
        .map(|bucket| format!("{bucket}={}", counts.get(*bucket).copied().unwrap_or(0)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn bucket_order(bucket: &str) -> usize {
    match bucket {
        "1-8" => 0,
        "9-16" => 1,
        "17-32" => 2,
        "33-64" => 3,
        "65+" => 4,
        _ => 5,
    }
}

fn format_gold_rank_example(value: &serde_json::Value, bucket: &str) -> String {
    let question_id = value.get("question_id").and_then(|v| v.as_str()).unwrap_or("unknown");
    let question_type = value.get("question_type").and_then(|v| v.as_str()).unwrap_or("unknown");
    let question = value.get("question").and_then(|v| v.as_str()).unwrap_or("");
    let expected = value
        .get("expected_answer_sessions")
        .map(|v| v.to_string())
        .unwrap_or_else(|| "[]".to_string());
    let retrieved = value
        .get("retrieved_sessions")
        .and_then(|v| v.as_array())
        .map(|sessions| {
            sessions
                .iter()
                .take(8)
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();

    format!("  {question_id} [{question_type}] best={bucket} gold={expected} retrieved_top8=[{retrieved}] q={question}")
}

fn lexical_overlap_count(query: &str, text: &str) -> usize {
    let query_terms = query
        .to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|term| term.len() >= 4)
        .map(|term| term.to_string())
        .collect::<HashSet<_>>();
    let lower = text.to_ascii_lowercase();
    query_terms
        .into_iter()
        .filter(|term| lower.contains(term.as_str()))
        .count()
}

fn entity_overlap_count(query: &str, text: &str) -> usize {
    let lower = text.to_ascii_lowercase();
    query
        .split_whitespace()
        .filter(|word| {
            word.chars()
                .next()
                .map(|ch| ch.is_ascii_uppercase())
                .unwrap_or(false)
        })
        .map(|word| {
            word.trim_matches(|c: char| !c.is_ascii_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|word| word.len() > 2 && lower.contains(word.as_str()))
        .collect::<HashSet<_>>()
        .len()
}

fn temporal_overlap_count(query: &str, text: &str) -> usize {
    let lower = text.to_ascii_lowercase();
    let query_lower = query.to_ascii_lowercase();
    let temporal_terms = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "yesterday",
        "today",
        "tomorrow",
        "week",
        "month",
        "year",
        "before",
        "after",
        "recent",
        "latest",
    ];
    temporal_terms
        .iter()
        .filter(|term| query_lower.contains(**term) && lower.contains(**term))
        .count()
}

fn extract_top_sessions(q_id: &str, results: &[QueryResult], top_k: usize) -> Vec<String> {
    let mut retrieved_sessions = Vec::new();
    let prefix = format!("{q_id}::");

    for result in results {
        if !result.memory_id.starts_with(&prefix) {
            continue;
        }
        let session_id = session_id_for_result(result);
        if session_id.is_empty() {
            continue;
        }
        if !retrieved_sessions.contains(&session_id) {
            retrieved_sessions.push(session_id);
        }
        if retrieved_sessions.len() == top_k {
            break;
        }
    }

    retrieved_sessions
}

fn pack_top_sessions(
    instance: &Instance,
    results: &[QueryResult],
    top_k: usize,
    max_chunks_per_session: usize,
) -> Vec<PackedSession> {
    let mut session_order = Vec::new();
    for result in results {
        let session_id = session_id_for_result(result);
        if session_id.is_empty() {
            continue;
        }
        if !session_order.contains(&session_id) {
            session_order.push(session_id);
        }
        if session_order.len() == top_k {
            break;
        }
    }
    if session_order.is_empty() {
        return Vec::new();
    }

    let allowed: HashSet<&str> = session_order.iter().map(String::as_str).collect();
    let mut grouped: HashMap<String, Vec<QueryResult>> = HashMap::new();

    for result in results {
        let session_id = session_id_for_result(result);
        if session_id.is_empty() || !allowed.contains(session_id.as_str()) {
            continue;
        }
        let entry = grouped.entry(session_id).or_default();
        if entry.len() < max_chunks_per_session {
            entry.push(result.clone());
        }
    }

    let session_meta = build_session_meta(instance);
    let mut packed = Vec::new();

    for session_id in session_order {
        let mut chunks = grouped.remove(&session_id).unwrap_or_default();
        chunks.sort_by_key(|chunk| (chunk.turn_index, chunk.created_at_ms));
        chunks
            .dedup_by(|a, b| a.memory_id == b.memory_id || a.textual_content == b.textual_content);
        if chunks.is_empty() {
            continue;
        }
        let (session_date, session_focus) = session_meta
            .get(&session_id)
            .cloned()
            .unwrap_or_else(|| ("unknown".to_string(), String::new()));
        packed.push(PackedSession {
            session_id,
            session_date,
            session_focus,
            chunks,
        });
    }

    packed
}

fn build_session_meta(instance: &Instance) -> HashMap<String, (String, String)> {
    let mut meta = HashMap::new();
    for ((session_id, session), session_date) in instance
        .haystack_session_ids
        .iter()
        .zip(instance.haystack_sessions.iter())
        .zip(instance.haystack_dates.iter())
    {
        meta.insert(
            session_id.clone(),
            (session_date.clone(), build_session_focus(session)),
        );
    }
    meta
}

fn turn_index_from_memory_id(memory_id: &str) -> usize {
    memory_id
        .rsplit("::")
        .next()
        .and_then(|part| part.parse::<usize>().ok())
        .unwrap_or(0)
}

fn build_enriched_window(
    session_id: &str,
    session_date: &str,
    session_focus: &str,
    session: &[Turn],
    t_idx: usize,
) -> String {
    let start_i = t_idx.saturating_sub(1);
    let end_i = (t_idx + 2).min(session.len());

    let mut lines = Vec::new();
    for turn in &session[start_i..end_i] {
        let content = json_value_to_text(&turn.content);
        let content = normalize_text(&content);
        if !content.is_empty() {
            lines.push(format!("{}: {}", turn.role, content));
        }
    }

    if lines.is_empty() {
        return String::new();
    }

    let mut enriched = String::new();
    enriched.push_str(&format!("[Session ID: {}]\n", session_id));
    enriched.push_str(&format!("[Session Date: {}]\n", session_date));
    if !session_focus.is_empty() {
        enriched.push_str(&format!("[Session Focus: {}]\n", session_focus));
    }
    enriched.push_str(&format!("[Window Turns: {}-{}]\n", start_i + 1, end_i));
    enriched.push_str(&lines.join("\n"));
    enriched
}

fn build_session_focus(session: &[Turn]) -> String {
    let mut user_lines = Vec::new();
    let mut all_lines = Vec::new();

    for turn in session {
        let content = normalize_text(&json_value_to_text(&turn.content));
        if content.is_empty() {
            continue;
        }
        all_lines.push(content.clone());
        if turn.role.eq_ignore_ascii_case("user") {
            user_lines.push(content);
        }
    }

    let source = if user_lines.is_empty() {
        all_lines
    } else {
        user_lines
    };
    truncate_chars(&source.join(" | "), 320)
}

async fn generate_answer(
    client: &Client,
    reader_mode: ReaderMode,
    model: &str,
    instance: &Instance,
    packed_sessions: &[PackedSession],
    use_groq: bool,
) -> Result<String> {
    let question_date = instance.question_date.as_deref().unwrap_or("unknown");
    let answer_rules = answer_rules_for_instance(instance);
    let intent = classify_answer_intent(instance);
    let focus_rules = focus_rules_for_intent(intent);

    let max_answer_tokens = 1024usize;
    let (evidence_text, mut answer) = match reader_mode {
        ReaderMode::Direct => {
            let context = render_session_context(packed_sessions);
            let prompt = DIRECT_ANSWER_PROMPT
                .replace("{answer_rules}", &answer_rules)
                .replace("{focus_rules}", focus_rules)
                .replace("{context}", &context)
                .replace("{question_date}", question_date)
                .replace("{question}", &instance.question);
            let answer =
                call_answer_model(client, &prompt, model, max_answer_tokens, "high", use_groq).await?;
            (context, answer)
        }
        ReaderMode::Con => {
            let context = render_session_context(packed_sessions);
            let prompt = CON_ANSWER_PROMPT
                .replace("{answer_rules}", &answer_rules)
                .replace("{focus_rules}", focus_rules)
                .replace("{context}", &context)
                .replace("{question_date}", question_date)
                .replace("{question}", &instance.question);
            let answer =
                call_answer_model(client, &prompt, model, max_answer_tokens, "high", use_groq).await?;
            (context, answer)
        }
        ReaderMode::ConSeparate => {
            let mut notes = Vec::new();
            for session in packed_sessions {
                let session_content = render_single_session(session);
                let prompt = CON_NOTES_PROMPT
                    .replace("{session_date}", &session.session_date)
                    .replace("{session_content}", &session_content)
                    .replace("{question_date}", question_date)
                    .replace("{question}", &instance.question);
                let note =
                    call_answer_model(client, &prompt, model, 300, "high", use_groq).await?;
                if !note.trim().is_empty() && note.trim().to_ascii_lowercase() != "empty" {
                    notes.push(format!(
                        "Session ID: {}\nSession Date: {}\nNotes: {}",
                        session.session_id,
                        session.session_date,
                        note.trim()
                    ));
                }
            }

            if notes.is_empty() {
                let context = render_session_context(packed_sessions);
                let prompt = DIRECT_ANSWER_PROMPT
                    .replace("{answer_rules}", &answer_rules)
                    .replace("{focus_rules}", focus_rules)
                    .replace("{context}", &context)
                    .replace("{question_date}", question_date)
                    .replace("{question}", &instance.question);
                let answer =
                    call_answer_model(client, &prompt, model, max_answer_tokens, "high", use_groq).await?;
                (context, answer)
            } else {
                let notes_text = notes.join("\n\n---\n\n");
                let prompt = CON_SEPARATE_ANSWER_PROMPT
                    .replace("{answer_rules}", &answer_rules)
                    .replace("{focus_rules}", focus_rules)
                    .replace("{notes}", &notes_text)
                    .replace("{question_date}", question_date)
                    .replace("{question}", &instance.question);
                let answer =
                    call_answer_model(client, &prompt, model, max_answer_tokens, "high", use_groq).await?;
                (notes_text, answer)
            }
        }
        ReaderMode::Summary => {
            let context = render_session_context(packed_sessions);
            let prompt = CON_SUMMARIZE_PROMPT
                .replace("{context}", &context)
                .replace("{question_date}", question_date)
                .replace("{question}", &instance.question);
            let notes = call_answer_model(client, &prompt, model, 512, "high", use_groq).await?;
            if notes.trim().is_empty() || notes.trim().to_ascii_lowercase() == "empty" {
                (context, "I don't know".to_string())
            } else {
                let answer_prompt = CON_SEPARATE_ANSWER_PROMPT
                    .replace("{answer_rules}", &answer_rules)
                    .replace("{focus_rules}", focus_rules)
                    .replace("{notes}", &notes)
                    .replace("{question_date}", question_date)
                    .replace("{question}", &instance.question);
                let answer =
                    call_answer_model(client, &answer_prompt, model, max_answer_tokens, "high", use_groq)
                        .await?;
                (notes, answer)
            }
        }
    };

    if intent == AnswerIntent::NumericAggregation {
        if let Some(deterministic_answer) = derive_deterministic_numeric_answer(
            client,
            model,
            &instance.question,
            &evidence_text,
            use_groq,
        )
        .await?
        {
            let deterministic_check = verify_answer_candidate(
                client,
                model,
                &instance.question,
                &evidence_text,
                &deterministic_answer,
                use_groq,
            )
            .await?;
            if deterministic_check == "PASS" {
                answer = deterministic_answer;
            } else if is_idk_answer(&answer) {
                answer = deterministic_answer;
            }
        }
    }

    if matches!(
        intent,
        AnswerIntent::NumericAggregation | AnswerIntent::TemporalAggregation
    ) {
        let verdict = verify_answer_candidate(
            client,
            model,
            &instance.question,
            &evidence_text,
            &answer,
            use_groq,
        )
        .await?;
        if let Some(corrected) = verdict.strip_prefix("CORRECT:") {
            answer = corrected.trim().to_string();
        } else if is_idk_answer(&verdict) {
            answer = "I don't know".to_string();
        }
    }

    Ok(strip_model_reasoning(&answer))
}

fn classify_answer_intent(instance: &Instance) -> AnswerIntent {
    let question = instance.question.to_ascii_lowercase();
    let question_type = instance.question_type.as_deref().unwrap_or_default();
    let is_numeric_aggregation_question = question.contains("how many")
        || question.contains("number of")
        || question.contains("in total")
        || question.contains("total amount")
        || question.contains("total money")
        || question.contains("combined")
        || question.contains("average")
        || question.contains("altogether");
    let is_temporal_question = question_type == "temporal-reasoning"
        || question.starts_with("when ")
        || question.contains(" in march")
        || question.contains(" in april")
        || question.contains(" in may")
        || question.contains(" in june")
        || question.contains(" in july")
        || question.contains(" in august")
        || question.contains(" in september")
        || question.contains(" in october")
        || question.contains(" in november")
        || question.contains(" in december")
        || question.contains(" last month")
        || question.contains(" this year")
        || question.contains(" past ");
    let is_recommendation_question = question.contains("recommend")
        || question.contains("suggest")
        || question.contains("advice")
        || question.contains("any tips")
        || question.contains("any ideas")
        || question.contains("activities");

    if is_numeric_aggregation_question {
        AnswerIntent::NumericAggregation
    } else if is_temporal_question {
        AnswerIntent::TemporalAggregation
    } else if is_recommendation_question {
        AnswerIntent::Recommendation
    } else {
        AnswerIntent::General
    }
}

fn focus_rules_for_intent(intent: AnswerIntent) -> &'static str {
    match intent {
        AnswerIntent::NumericAggregation => {
            "Focus: numeric aggregation. Identify all qualifying items first, then compute once. Prefer a normalized final numeric result."
        }
        AnswerIntent::TemporalAggregation => {
            "Focus: temporal constraints. Resolve time windows and only count facts that satisfy the asked period."
        }
        AnswerIntent::Recommendation => {
            "Focus: personalized recommendations. Use user-specific preferences from evidence and avoid generic suggestions."
        }
        AnswerIntent::General => "Focus: direct factual answer with minimal extra detail.",
    }
}

async fn derive_deterministic_numeric_answer(
    client: &Client,
    model: &str,
    question: &str,
    evidence: &str,
    use_groq: bool,
) -> Result<Option<String>> {
    let prompt = NUMERIC_EXTRACTION_PROMPT
        .replace("{question}", question)
        .replace("{evidence}", evidence);
    let extraction_text = call_answer_model(client, &prompt, model, 2048, "none", use_groq).await?;
    let mut cleaned_extraction = extraction_text.to_string();
    while let Some(start_idx) = cleaned_extraction.find("<think>") {
        if let Some(end_idx) = cleaned_extraction.find("</think>") {
            cleaned_extraction = format!(
                "{}{}",
                &cleaned_extraction[..start_idx],
                &cleaned_extraction[end_idx + "</think>".len()..]
            );
        } else {
            cleaned_extraction = cleaned_extraction[..start_idx].to_string();
            break;
        }
    }
    let extraction_json = match extract_first_json_object(&cleaned_extraction) {
        Some(value) => value,
        None => return Ok(None),
    };
    let extraction: NumericExtraction = match serde_json::from_str(&extraction_json) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    if extraction.no_answer {
        return Ok(None);
    }

    let op = extraction.operation.trim().to_ascii_lowercase();
    let values = extraction.values;
    if values.is_empty() {
        return Ok(None);
    }

    let computed = match op.as_str() {
        "count" | "sum" => values.iter().sum::<f64>(),
        "average" => values.iter().sum::<f64>() / values.len() as f64,
        "difference" => {
            if values.len() < 2 {
                return Ok(None);
            }
            let mut diff = values[0] - values[1];
            if question.to_ascii_lowercase().contains("more") && diff < 0.0 {
                diff = -diff;
            }
            diff
        }
        _ => return Ok(None),
    };

    Ok(Some(format_numeric_answer(
        computed,
        &extraction.unit,
        question,
    )))
}

async fn verify_answer_candidate(
    client: &Client,
    model: &str,
    question: &str,
    evidence: &str,
    candidate: &str,
    use_groq: bool,
) -> Result<String> {
    let prompt = ANSWER_VERIFY_PROMPT
        .replace("{question}", question)
        .replace("{candidate}", candidate)
        .replace("{evidence}", evidence);
    let verdict = call_answer_model(client, &prompt, model, 2048, "none", use_groq).await?;
    let mut cleaned_verdict = verdict.to_string();
    while let Some(start_idx) = cleaned_verdict.find("<think>") {
        if let Some(end_idx) = cleaned_verdict.find("</think>") {
            cleaned_verdict = format!(
                "{}{}",
                &cleaned_verdict[..start_idx],
                &cleaned_verdict[end_idx + "</think>".len()..]
            );
        } else {
            cleaned_verdict = cleaned_verdict[..start_idx].to_string();
            break;
        }
    }
    let trimmed = cleaned_verdict.trim().trim_end_matches(|c: char| c == '.' || c == '!' || c == '?').trim();
    Ok(trimmed.to_string())
}

fn extract_first_json_object(text: &str) -> Option<String> {
    let mut start = None;
    let mut depth = 0usize;
    for (idx, ch) in text.char_indices() {
        if ch == '{' {
            if start.is_none() {
                start = Some(idx);
            }
            depth += 1;
        } else if ch == '}' {
            if depth == 0 {
                continue;
            }
            depth -= 1;
            if depth == 0 {
                let s = start?;
                return Some(text[s..=idx].to_string());
            }
        }
    }
    None
}

fn format_numeric_answer(value: f64, unit: &str, question: &str) -> String {
    let rounded = if (value.fract()).abs() < 1e-9 {
        format!("{}", value as i64)
    } else {
        let mut text = format!("{value:.2}");
        while text.contains('.') && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        text
    };

    let lower_question = question.to_ascii_lowercase();
    if unit.trim().starts_with('$')
        || lower_question.contains("money")
        || lower_question.contains("spent")
        || lower_question.contains("cost")
        || lower_question.contains("raise")
        || lower_question.contains("amount")
    {
        format!("${rounded}")
    } else if unit.trim().is_empty() {
        rounded
    } else {
        format!("{rounded} {}", unit.trim())
    }
}

fn answer_rules_for_instance(instance: &Instance) -> String {
    let question = instance.question.to_ascii_lowercase();
    let question_type = instance.question_type.as_deref().unwrap_or_default();
    let is_recommendation_question = question.contains("recommend")
        || question.contains("suggest")
        || question.contains("advice")
        || question.contains("any tips")
        || question.contains("any ideas");
    let is_numeric_aggregation_question = question.contains("how many")
        || question.contains("number of")
        || question.contains("in total")
        || question.contains("total amount")
        || question.contains("total money")
        || question.contains("combined")
        || question.contains("average")
        || question.contains("altogether");
    let is_list_question = question.contains("what events")
        || question.contains("what activities")
        || question.contains("what types")
        || question.contains("what are ")
        || question.contains("which ")
        || question.contains("who ")
        || question.contains("names");
    let mut rules = vec![
        "Use only facts supported by the provided history.".to_string(),
        "Do not say \"I don't know\" if the answer can be inferred by combining retrieved facts, resolving dates, counting, or normalizing relative references.".to_string(),
        "Reply exactly \"I don't know\" only when the history truly does not support an answer.".to_string(),
        "If multiple facts satisfy the question, include every supported item and avoid unsupported extras.".to_string(),
        "Keep the answer compact, but not cryptic: answer the question directly in its normalized final form.".to_string(),
    ];

    if question_type == "temporal-reasoning"
        || question.contains("when ")
        || question.starts_with("when")
        || question.contains("how long")
    {
        rules.push(
            "For time questions, resolve relative references carefully using the session dates and the question date."
                .to_string(),
        );
        rules.push(
            "Do not answer with a session timestamp, chat timestamp, or raw phrases like yesterday/last week/this month if they can be normalized to a calendar date, month, year, or explicitly anchored relative period."
                .to_string(),
        );
        rules.push(
            "Prefer the normalized final answer, such as \"5 July 2023\", \"July 2023\", \"2022\", or \"the week before 9 June 2023\"."
                .to_string(),
        );
        rules.push(
            "If the question asks about one event, return one best-matching time answer only and do not list additional dates from similar events."
                .to_string(),
        );
    }

    if is_list_question {
        rules.push(
            "For list questions, give a complete comma-separated list of the items that directly answer the question, with no commentary and no loosely related extras."
                .to_string(),
        );
        rules.push(
            "Prefer the main answer items, not examples, sub-activities, evidence snippets, or explanatory detail."
                .to_string(),
        );
    }

    if question_type == "locomo-open-domain"
        || question.contains("would ")
        || question.contains("likely")
        || question.contains("political leaning")
        || question.contains("relationship status")
    {
        rules.push(
            "For likely, preference, or open-domain questions, start with the single best supported conclusion, then add only the shortest needed qualifier."
                .to_string(),
        );
        rules.push(
            "Do not answer \"I don't know\" if one conclusion is clearly better supported than the alternatives."
                .to_string(),
        );
    }

    if is_numeric_aggregation_question || question.starts_with("how many") {
        rules.push(
            "For counting questions, return the supported final count only, unless the question explicitly asks for more detail."
                .to_string(),
        );
        rules.push(
            "For aggregate totals, collect all relevant events first, then compute the final number; do not answer from a single event mention."
                .to_string(),
        );
        rules.push(
            "For average questions, include every entity explicitly named in the question before dividing."
                .to_string(),
        );
        rules.push(
            "Double-check arithmetic once before answering; avoid off-by-one mistakes.".to_string(),
        );
    }

    if question_type == "single-session-preference"
        || question_type == "locomo-open-domain"
        || is_recommendation_question
    {
        rules.push(
            "For recommendations, advice, or suggestions, personalize the answer using user-specific preferences, habits, goals, and constraints from the retrieved history."
                .to_string(),
        );
        rules.push(
            "Avoid generic tips when personal signals are available; prioritize suggestions that match the user's stated likes/dislikes."
                .to_string(),
        );
        rules.push(
            "If the question asks for recommendations, provide concise actionable suggestions, not \"I don't know\", unless the history truly has no relevant personal preference signal."
                .to_string(),
        );
    }

    let is_singular_answer_question = !is_list_question
        && (question.starts_with("what is ")
            || question.starts_with("what was ")
            || question.starts_with("what did ")
            || question.starts_with("what does ")
            || question.starts_with("what book ")
            || question.starts_with("what kind of ")
            || question.starts_with("where ")
            || question.starts_with("why ")
            || question.starts_with("who "))
        && !question.contains("what are ")
        && !question.contains("what books")
        && !question.contains("what plans")
        && !question.contains("what changes")
        && !question.contains("what ways");

    if is_singular_answer_question {
        rules.push(
            "If the question asks for one item, reason, place, title, status, or event, return a single best answer only and do not append secondary candidates, sibling examples, or extra detail."
                .to_string(),
        );
    }

    if question.contains("relationship status")
        || question.contains("political leaning")
        || question.contains("identity")
        || question.contains("what subject")
        || question.contains("what field")
        || question.contains("what kind of")
    {
        rules.push(
            "For label, identity, status, leaning, field, or category questions, answer with the shortest canonical label or phrase that directly answers the question."
                .to_string(),
        );
    }

    if question.contains("relationship status") {
        rules.push(
            "For relationship status, return only the status label itself, such as Single, Married, or In a relationship."
                .to_string(),
        );
    }

    if question.contains("what book")
        || question.contains("favorite book")
        || question.contains("what was ")
        || question.contains("what does ")
    {
        rules.push(
            "When a short noun phrase directly answers the question, prefer that phrase over a full explanatory sentence."
                .to_string(),
        );
    }

    if (question.starts_with("would ")
        || question.contains(" likely ")
        || question.starts_with("is "))
        && question.contains(" or ")
    {
        rules.push(
            "If the question presents explicit alternatives, choose the single best-supported option from those alternatives."
                .to_string(),
        );
    }

    if question.contains("what do ") && question.contains(" like") {
        rules.push(
            "For questions about what someone likes, return only the liked things themselves, not nearby activities, examples, or situations where those preferences were mentioned."
                .to_string(),
        );
    }

    rules.join(" ")
}

fn render_session_context(packed_sessions: &[PackedSession]) -> String {
    packed_sessions
        .iter()
        .enumerate()
        .map(|(idx, session)| {
            format!(
                "### Session {}\n{}",
                idx + 1,
                render_single_session(session)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn render_single_session(session: &PackedSession) -> String {
    let mut text = String::new();
    text.push_str(&format!("Session ID: {}\n", session.session_id));
    text.push_str(&format!("Session Date: {}\n", session.session_date));
    if !session.session_focus.is_empty() {
        text.push_str(&format!("Session Focus: {}\n", session.session_focus));
    }
    text.push_str("Relevant Chunks:\n");
    for chunk in &session.chunks {
        text.push_str(&format!(
            "- Turn {} (score {:.4})\n{}\n",
            chunk.turn_index,
            chunk.similarity,
            chunk.textual_content.trim()
        ));
    }
    text.trim().to_string()
}

fn benchmark_entity_id(instance: &Instance, idx: usize) -> String {
    if let Some(entity_id) = instance.entity_id.clone() {
        return entity_id;
    }
    instance
        .question_id
        .clone()
        .unwrap_or_else(|| format!("q{idx}"))
}

fn evaluation_question_id(instance: &Instance, idx: usize) -> String {
    instance
        .question_id
        .clone()
        .unwrap_or_else(|| format!("q{idx}"))
}

fn session_id_for_result(result: &QueryResult) -> String {
    if !result.session_id.is_empty() {
        return result.session_id.clone();
    }
    session_id_from_memory_id(&result.memory_id)
}

fn session_id_from_memory_id(memory_id: &str) -> String {
    let parts: Vec<&str> = memory_id.split("::").collect();
    if parts.len() > 1 {
        parts[1].to_string()
    } else {
        String::new()
    }
}

fn ground_truth(instance: &Instance) -> String {
    if let Some(answer) = instance.answer.as_ref() {
        let answer = json_value_to_text(answer);
        if !answer.is_empty() {
            return answer;
        }
    }

    let answer_sessions = instance
        .haystack_session_ids
        .iter()
        .zip(instance.haystack_sessions.iter())
        .filter(|(session_id, _)| instance.answer_session_ids.contains(session_id))
        .map(|(_, session)| session)
        .collect::<Vec<_>>();

    if let Some(session) = answer_sessions.first() {
        session
            .iter()
            .rev()
            .take(2)
            .map(|turn| json_value_to_text(&turn.content))
            .filter(|content| !content.is_empty())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        "unknown".to_string()
    }
}

fn get_anscheck_prompt(
    task: &str,
    question: &str,
    answer: &str,
    response: &str,
    abstention: bool,
) -> String {
    if abstention {
        return format!(
            "You are evaluating an abstention question. The correct answer is that the information was NOT in the conversation, so the system should abstain or say it doesn't know.\n\nThe hypothesis is CORRECT if the system correctly abstains, says \"I don't know\", indicates uncertainty, or explicitly states the information is not available. It is INCORRECT if the system makes up an answer or hallucinates.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.",
            question, answer, response,
        );
    }

    match task {
        "temporal-reasoning" => format!(
            "I will give you a time-related question, a correct answer, and a response from a model. Answer yes if the response refers to the same date, month, year, weekday, or relative time period as the correct answer, even if the format differs. Extra non-conflicting detail should not cause a no. Answer no only if the response gives a different time, misses the main time reference, or gives an incompatible answer. Do not penalize off-by-one errors for the number of days.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
            question, answer, response,
        ),
        "knowledge-update" => format!(
            "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct updated answer. Extra historical or contextual detail should not cause a no as long as the final factual answer is correct. Answer no only if the response contradicts the correct updated answer or omits it entirely.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
            question, answer, response,
        ),
        "single-session-preference" => format!(
            "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and uses the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
            question, answer, response,
        ),
        "locomo-open-domain" => format!(
            "I will give you a question, a correct answer, and a response from a model for an open-domain conversational memory task. Answer yes if the response reaches the same main conclusion as the correct answer or a clearly consistent paraphrase. For likely or preference questions, accept concise conclusions such as likely/probably/would if they align with the gold answer. Extra supporting rationale is optional and wording does not need to match exactly. For list questions, all key gold items must be present. Answer no only if the main conclusion is different, unsupported, or missing.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
            question, answer, response,
        ),
        _ => format!(
            "I will give you a question, a correct answer, and a response from a model. Answer yes if the response states the same fact as the correct answer or a clearly correct paraphrase. Extra non-conflicting detail should not cause a no. For list or set questions, answer yes only if the response includes all key gold items; missing a core item is no. For time questions, accept equivalent absolute or relative dates that refer to the same time period. Answer no only if the response contradicts the correct answer, omits the main fact entirely, or gives an incompatible answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
            question, answer, response,
        ),
    }
}

/// Parse a judge verdict that may be either:
///  - Supermemory JSON format: `{"score": 1, "label": "correct", "explanation": "..."}`
///  - Legacy plain-text: `yes` / `no`
///
/// Returns true if the judge considers the answer correct.
fn parse_judge_verdict(verdict: &str) -> bool {
    let mut cleaned = verdict.to_string();
    while let Some(start_idx) = cleaned.find("<think>") {
        if let Some(end_idx) = cleaned.find("</think>") {
            cleaned = format!(
                "{}{}",
                &cleaned[..start_idx],
                &cleaned[end_idx + "</think>".len()..]
            );
        } else {
            cleaned = cleaned[..start_idx].to_string();
            break;
        }
    }
    let trimmed = cleaned.trim();

    // Try to parse JSON score first.
    if let Some(json_start) = trimmed.find('{') {
        let json_candidate = &trimmed[json_start..];
        // Find matching closing brace.
        let mut depth = 0usize;
        let mut end = None;
        for (i, ch) in json_candidate.char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        if let Some(end_idx) = end {
            let json_str = &json_candidate[..=end_idx];
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Some(score) = value.get("score").and_then(|v| v.as_u64()) {
                    return score == 1;
                }
                // Some models return label instead of score.
                if let Some(label) = value.get("label").and_then(|v| v.as_str()) {
                    let lower = label.to_ascii_lowercase();
                    return lower == "correct" || lower == "yes";
                }
            }
        }
    }

    // Fall back to plain-text yes/no/correct/incorrect.
    let lower = trimmed.to_ascii_lowercase();
    let cleaned_lower = lower.trim_end_matches(|c: char| c.is_ascii_punctuation()).trim();

    if cleaned_lower == "yes" || cleaned_lower == "correct" || cleaned_lower == "true" {
        return true;
    }
    if cleaned_lower == "no" || cleaned_lower == "incorrect" || cleaned_lower == "false" {
        return false;
    }

    // Check for sentences containing yes/correct/true
    if cleaned_lower.contains("yes") || cleaned_lower.contains("correct") || cleaned_lower.contains("true") {
        if cleaned_lower.contains("not correct") || cleaned_lower.contains("incorrect") || cleaned_lower.contains(" not ") {
            return false;
        }
        if cleaned_lower.starts_with("no") && (cleaned_lower.len() == 2 || cleaned_lower.chars().nth(2).unwrap().is_ascii_whitespace() || cleaned_lower.chars().nth(2).unwrap() == ',') {
            return false;
        }
        return true;
    }

    false
}

async fn call_openrouter(
    client: &Client,
    prompt: &str,
    model: &str,
    max_tokens: usize,
    reasoning_effort: &str,
) -> Result<String> {
    let api_key = std::env::var("OPENROUTER_API_KEY")
        .context("OPENROUTER_API_KEY must be set for llm mode")?;
    let requested_max_tokens = effective_openrouter_max_tokens(model, max_tokens);

    let body = openrouter_request(
        client,
        &api_key,
        prompt,
        model,
        requested_max_tokens,
        reasoning_effort,
    )
    .await?;

    if let Some(text) = extract_openrouter_text(&body) {
        return Ok(text);
    }

    if should_retry_openrouter_without_reasoning(&body) {
        let retry_tokens = retry_openrouter_max_tokens(model, requested_max_tokens);
        let retry_body =
            openrouter_request(client, &api_key, prompt, model, retry_tokens, "none").await?;
        if let Some(text) = extract_openrouter_text(&retry_body) {
            return Ok(text);
        }

        anyhow::bail!(
            "OpenRouter response did not contain a message after retry. Raw body: {}",
            truncate_chars(&retry_body.to_string(), 1200)
        );
    }

    anyhow::bail!(
        "OpenRouter response did not contain a message. Raw body: {}",
        truncate_chars(&body.to_string(), 1200)
    )
}

async fn call_answer_model(
    client: &Client,
    prompt: &str,
    model: &str,
    max_tokens: usize,
    reasoning_effort: &str,
    use_groq: bool,
) -> Result<String> {
    if use_groq {
        call_groq(client, prompt, model, max_tokens, reasoning_effort).await
    } else {
        call_openrouter(client, prompt, model, max_tokens, reasoning_effort).await
    }
}

async fn call_groq(
    client: &Client,
    prompt: &str,
    model: &str,
    max_tokens: usize,
    reasoning_effort: &str,
) -> Result<String> {
    let api_key =
        std::env::var("GROQ_API_KEY").context("GROQ_API_KEY must be set for --use-groq")?;
    let body = groq_request(
        client,
        &api_key,
        prompt,
        model,
        max_tokens,
        reasoning_effort,
    )
    .await?;

    if let Some(text) = extract_openrouter_text(&body) {
        return Ok(text);
    }

    anyhow::bail!(
        "Groq response did not contain a message. Raw body: {}",
        truncate_chars(&body.to_string(), 1200)
    )
}

async fn groq_request(
    client: &Client,
    api_key: &str,
    prompt: &str,
    model: &str,
    max_tokens: usize,
    reasoning_effort: &str,
) -> Result<serde_json::Value> {
    let mut request_body = serde_json::Map::new();
    request_body.insert("model".to_string(), json!(model));
    request_body.insert(
        "messages".to_string(),
        json!([{ "role": "user", "content": prompt }]),
    );
    request_body.insert("temperature".to_string(), json!(0.0));
    request_body.insert("top_p".to_string(), json!(1.0));
    request_body.insert("stream".to_string(), json!(false));
    request_body.insert("max_completion_tokens".to_string(), json!(max_tokens));
    if let Some(mapped_effort) = groq_reasoning_effort(reasoning_effort) {
        request_body.insert("reasoning_effort".to_string(), json!(mapped_effort));
    }
    let request_value = serde_json::Value::Object(request_body);
    let url = "https://api.groq.com/openai/v1/chat/completions".to_string();

    for attempt in 1..=OPENROUTER_MAX_ATTEMPTS {
        let response = client
            .post(&url)
            .bearer_auth(api_key)
            .json(&request_value)
            .send()
            .await;

        match response {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    return response
                        .json()
                        .await
                        .context("Failed to decode Groq response");
                }

                let body = response.text().await.unwrap_or_default();
                if attempt < OPENROUTER_MAX_ATTEMPTS && is_retryable_openrouter_status(status) {
                    eprintln!(
                        "Groq returned HTTP {} on attempt {}/{}; retrying...",
                        status, attempt, OPENROUTER_MAX_ATTEMPTS
                    );
                    tokio::time::sleep(openrouter_backoff(attempt)).await;
                    continue;
                }

                anyhow::bail!("Groq returned HTTP {status}: {body}");
            }
            Err(err) => {
                let retryable = err.is_timeout() || err.is_connect() || err.is_request();
                if attempt < OPENROUTER_MAX_ATTEMPTS && retryable {
                    eprintln!(
                        "Groq request failed on attempt {}/{}: {}; retrying...",
                        attempt, OPENROUTER_MAX_ATTEMPTS, err
                    );
                    tokio::time::sleep(openrouter_backoff(attempt)).await;
                    continue;
                }

                return Err(err).context("Groq request failed");
            }
        }
    }

    anyhow::bail!(
        "Groq request failed after {} attempts",
        OPENROUTER_MAX_ATTEMPTS
    )
}

async fn openrouter_request(
    client: &Client,
    api_key: &str,
    prompt: &str,
    model: &str,
    max_tokens: usize,
    reasoning_effort: &str,
) -> Result<serde_json::Value> {
    let reasoning_payload = reasoning_payload_for_model(model, reasoning_effort);
    let mut request_body = serde_json::Map::new();
    request_body.insert("model".to_string(), json!(model));
    request_body.insert(
        "messages".to_string(),
        json!([{ "role": "user", "content": prompt }]),
    );
    request_body.insert("temperature".to_string(), json!(0.0));
    request_body.insert("max_tokens".to_string(), json!(max_tokens));
    if !reasoning_payload.is_null() {
        request_body.insert("reasoning".to_string(), reasoning_payload);
    }
    let request_value = serde_json::Value::Object(request_body);
    let url = std::env::var("LLM_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1/chat/completions".to_string());

    for attempt in 1..=OPENROUTER_MAX_ATTEMPTS {
        let response = client
            .post(&url)
            .bearer_auth(api_key)
            .json(&request_value)
            .send()
            .await;

        match response {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    return response
                        .json()
                        .await
                        .context("Failed to decode OpenRouter response");
                }

                let body = response.text().await.unwrap_or_default();
                if attempt < OPENROUTER_MAX_ATTEMPTS && is_retryable_openrouter_status(status) {
                    eprintln!(
                        "OpenRouter returned HTTP {} on attempt {}/{}; retrying...",
                        status, attempt, OPENROUTER_MAX_ATTEMPTS
                    );
                    tokio::time::sleep(openrouter_backoff(attempt)).await;
                    continue;
                }

                anyhow::bail!("OpenRouter returned HTTP {status}: {body}");
            }
            Err(err) => {
                let retryable = err.is_timeout() || err.is_connect() || err.is_request();
                if attempt < OPENROUTER_MAX_ATTEMPTS && retryable {
                    eprintln!(
                        "OpenRouter request failed on attempt {}/{}: {}; retrying...",
                        attempt, OPENROUTER_MAX_ATTEMPTS, err
                    );
                    tokio::time::sleep(openrouter_backoff(attempt)).await;
                    continue;
                }

                return Err(err).context("OpenRouter request failed");
            }
        }
    }

    anyhow::bail!(
        "OpenRouter request failed after {} attempts",
        OPENROUTER_MAX_ATTEMPTS
    )
}

fn is_retryable_openrouter_status(status: reqwest::StatusCode) -> bool {
    status.as_u16() == 408
        || status.as_u16() == 409
        || status.as_u16() == 429
        || status.is_server_error()
}

fn openrouter_backoff(attempt: usize) -> Duration {
    Duration::from_secs((attempt as u64).saturating_mul(2))
}

fn is_minimax_m2_model(model: &str) -> bool {
    model.contains("minimax/minimax-m2")
}

fn effective_openrouter_max_tokens(model: &str, requested_max_tokens: usize) -> usize {
    if is_minimax_m2_model(model) {
        requested_max_tokens.max(1024)
    } else {
        requested_max_tokens
    }
}

fn retry_openrouter_max_tokens(model: &str, requested_max_tokens: usize) -> usize {
    if is_minimax_m2_model(model) {
        requested_max_tokens.saturating_mul(2).min(2048)
    } else {
        requested_max_tokens.saturating_mul(2).min(512)
    }
}

fn groq_reasoning_effort(reasoning_effort: &str) -> Option<&'static str> {
    match reasoning_effort {
        "none" => None,
        "low" => Some("low"),
        "high" | "xhigh" => Some("high"),
        "minimal" | "medium" => Some("medium"),
        _ => Some("medium"),
    }
}

fn reasoning_payload_for_model(model: &str, reasoning_effort: &str) -> serde_json::Value {
    if is_minimax_m2_model(model) {
        let max_tokens = match reasoning_effort {
            "none" => 32,
            "minimal" => 96,
            "low" => 160,
            "medium" => 256,
            "high" => 384,
            "xhigh" => 512,
            _ => 96,
        };
        return json!({
            "enabled": true,
            "exclude": true,
            "max_tokens": max_tokens
        });
    }

    let normalized_effort = if model.contains("gpt-5") {
        if reasoning_effort == "none" {
            "minimal"
        } else {
            reasoning_effort
        }
    } else {
        reasoning_effort
    };

    if model.contains("gpt-5") {
        json!({ "effort": normalized_effort })
    } else if normalized_effort == "none" {
        serde_json::Value::Null
    } else {
        json!({ "effort": normalized_effort })
    }
}

fn should_retry_openrouter_without_reasoning(body: &serde_json::Value) -> bool {
    let choice = body.get("choices").and_then(|choices| choices.get(0));
    let finish_reason = choice
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let has_reasoning = choice
        .and_then(|choice| choice.get("message"))
        .map(|message| {
            let has_reasoning_details = message
                .get("reasoning_details")
                .and_then(|details| details.as_array())
                .map(|details| !details.is_empty())
                .unwrap_or(false);
            let has_reasoning_text = message
                .get("reasoning")
                .and_then(|value| value.as_str())
                .map(|text| !text.trim().is_empty())
                .unwrap_or(false);
            has_reasoning_details || has_reasoning_text
        })
        .unwrap_or(false);
    let content_is_null = choice
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .map(|content| content.is_null())
        .unwrap_or(false);
    let has_hidden_reasoning_tokens = body
        .get("usage")
        .and_then(|usage| usage.get("completion_tokens_details"))
        .and_then(|details| details.get("reasoning_tokens"))
        .and_then(|value| value.as_u64())
        .map(|tokens| tokens > 0)
        .unwrap_or(false);
    let completion_tokens = body
        .get("usage")
        .and_then(|usage| usage.get("completion_tokens"))
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let provider_returned_empty_completion = content_is_null && completion_tokens == 0;

    (finish_reason == "length" && content_is_null && (has_reasoning || has_hidden_reasoning_tokens))
        || provider_returned_empty_completion
}

fn extract_openrouter_text(body: &serde_json::Value) -> Option<String> {
    let message = body
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"))?;

    if let Some(text) = value_to_openrouter_text(message.get("content")) {
        return Some(text);
    }
    // Reasoning models (DeepSeek V4, etc.) may return reasoning_content instead.
    if let Some(text) = value_to_openrouter_text(message.get("reasoning_content")) {
        return Some(text);
    }

    if let Some(text) = body
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("text"))
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
    {
        return Some(text);
    }

    body.get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("refusal"))
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

/// Strip model reasoning from a response string.
///
/// DeepSeek V4-flash (and other reasoning-class models) emit their chain-of-thought
/// directly in the `content` field in two forms:
///
/// 1. Wrapped in `<think>…</think>` tags — strip the entire block.
/// 2. As a plain reasoning preamble (e.g. "We are asked:", "Thinking.", "1. **Analyze") —
///    extract only the last meaningful paragraph / final answer sentence.
///
/// For non-reasoning models the function is a no-op (returns the text unchanged).
fn strip_model_reasoning(text: &str) -> String {
    // 1. Strip <think>...</think> blocks (used by DeepSeek, Kimi, etc.).
    let mut result = text.to_string();
    loop {
        match (result.find("<think>"), result.find("</think>")) {
            (Some(start), Some(end)) if end >= start => {
                result = format!(
                    "{}{}",
                    &result[..start],
                    &result[end + "</think>".len()..]
                );
            }
            (Some(start), None) => {
                result = result[..start].to_string();
                break;
            }
            _ => break,
        }
    }

    let trimmed = result.trim();

    // 2. Try to extract text after "Answer:" marker (used in our prompts).
    if let Some(ans_pos) = trimmed.rfind("Answer:") {
        let after_answer = trimmed[ans_pos + "Answer:".len()..].trim();
        if !after_answer.is_empty() && !is_reasoning_text(after_answer) {
            return after_answer.to_string();
        }
    }

    // 3. Split into lines and strip leading reasoning lines.
    let lines: Vec<&str> = trimmed.lines().collect();
    let mut start_idx = 0;
    for (i, line) in lines.iter().enumerate() {
        let lower = line.trim().to_ascii_lowercase();
        let is_continuation = i > 0
            && (lower.starts_with("based on")
                || lower.starts_with("the context")
                || lower.starts_with("the history")
                || lower.starts_with("from the"));
        if !is_reasoning_line(lower.as_str()) && !is_continuation {
            // If it's just a short line after reasoning, skip it too
            if line.trim().len() <= 3 && i < lines.len() - 1 {
                continue;
            }
            start_idx = i;
            break;
        }
    }

    // 4. Collect remaining lines, stopping if we hit a new reasoning block.
    let mut clean_lines: Vec<&str> = Vec::new();
    for line in &lines[start_idx..] {
        let lower = line.trim().to_ascii_lowercase();
        if is_reasoning_line(lower.as_str()) && !clean_lines.is_empty() {
            // Check if this is truly a new reasoning block or just a coordinating phrase
            let has_answer_content = clean_lines.iter().any(|l| {
                let t = l.trim();
                t.len() > 20 || t.contains(':') || t.contains(',')
            });
            if has_answer_content {
                break;
            }
        }
        clean_lines.push(line);
    }

    let candidate = clean_lines.join("\n").trim().to_string();
    if !candidate.is_empty() && !is_reasoning_text(&candidate) {
        // If still long, try extracting the last sentence.
        if candidate.len() > 80 {
            if let Some(last_sentence) = extract_last_sentence(&candidate) {
                if !is_reasoning_text(&last_sentence) {
                    return last_sentence;
                }
            }
        }
        return candidate;
    }

    // 5. Fallback: extract last sentence from original trimmed text.
    if let Some(last) = extract_last_sentence(trimmed) {
        if !is_reasoning_text(&last) {
            return last;
        }
    }

    trimmed.to_string()
}

fn is_reasoning_line(lower: &str) -> bool {
    lower.starts_with("thinking")
        || lower.starts_with("we need")
        || lower.starts_with("we are")
        || lower.starts_with("we can")
        || lower.starts_with("let me")
        || lower.starts_with("let's ")
        || lower.starts_with("step ")
        || lower.starts_with("first,")
        || lower.starts_with("first ")
        || lower.starts_with("second,")
        || lower.starts_with("second ")
        || lower.starts_with("1. ")
        || lower.starts_with("1)")
        || lower.starts_with("2. ")
        || lower.starts_with("2)")
        || lower.starts_with("3. ")
        || lower.starts_with("3)")
        || lower.starts_with("- ")
        || lower.starts_with("* ")
        || lower.starts_with("thus,")
        || lower.starts_with("thus ")
        || lower.starts_with("therefore")
        || lower.starts_with("so,")
        || lower.starts_with("so ")
        || lower.starts_with("this means")
        || lower.starts_with("in summary")
        || lower.starts_with("to answer")
        || lower.starts_with("the question")
        || lower.starts_with("the answer")
        || lower.starts_with("based on the")
        || lower.starts_with("looking at")
        || lower.starts_with("analyzing")
        || lower.starts_with("checking")
        || lower.starts_with("i think")
        || lower.starts_with("i need")
        || lower.starts_with("i should")
        || lower.starts_with("i can")
        || lower.starts_with("i will")
        || lower.starts_with("from the")
        || lower == "answer:"
        || lower.starts_with("answer: ")
}

fn is_reasoning_text(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    // If it starts with a reasoning marker, it's reasoning.
    if is_reasoning_line(lower.as_str()) {
        return true;
    }
    // If it has a reasoning preamble pattern like "We are asked: ..."
    let first_sentence_end = lower.find(|c: char| c == '.' || c == '\n');
    if let Some(end) = first_sentence_end {
        let first_sentence = &lower[..=end];
        if first_sentence.starts_with("we need")
            || first_sentence.starts_with("we are")
            || first_sentence.starts_with("let me")
            || first_sentence.starts_with("thinking")
            || first_sentence.starts_with("based on")
            || first_sentence.starts_with("to answer")
            || first_sentence.starts_with("the question")
            || first_sentence.starts_with("the answer")
        {
            return true;
        }
    }
    false
}

fn extract_last_sentence(text: &str) -> Option<String> {
    let trimmed = text.trim();
    // Try sentence-ending punctuation.
    for sep in &["\n\n", ". ", "! ", "? "] {
        if let Some(pos) = trimmed.rfind(sep) {
            let candidate = trimmed[pos + sep.len()..].trim().to_string();
            if !candidate.is_empty() && candidate.len() < 200 && !candidate.contains("  ") {
                return Some(candidate);
            }
        }
    }
    // Try last comma-separated phrase.
    if let Some(pos) = trimmed.rfind(", ") {
        let candidate = trimmed[pos + 2..].trim().to_string();
        if !candidate.is_empty() && candidate.len() < 100 {
            return Some(candidate);
        }
    }
    None
}



fn value_to_openrouter_text(value: Option<&serde_json::Value>) -> Option<String> {
    match value? {
        serde_json::Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        serde_json::Value::Array(parts) => {
            let joined = parts
                .iter()
                .filter_map(openrouter_content_part_text)
                .collect::<Vec<_>>()
                .join("");
            let trimmed = joined.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        _ => None,
    }
}

fn openrouter_content_part_text(part: &serde_json::Value) -> Option<&str> {
    part.get("text")
        .and_then(|text| text.as_str())
        .or_else(|| part.get("content").and_then(|text| text.as_str()))
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn normalize_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn json_value_to_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Number(number) => number.to_string(),
        serde_json::Value::Bool(boolean) => boolean.to_string(),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => value.to_string(),
    }
}

fn print_recall_summary(totals: &EvalTotals, top_k: usize, dataset_kind: DatasetKind) {
    println!("\n============================================================");
    if totals.evaluated == 0 {
        println!("No evaluable questions");
    } else {
        let recall = totals.retrieval_correct as f64 / totals.evaluated as f64 * 100.0;
        println!("Recall@{top_k}: {recall:.1}%");
        print_timing_summary(&totals.timings, totals.evaluated);
        println!("Evaluated: {}", totals.evaluated);
        println!("Skipped: {}", totals.skipped);
        print_dataset_breakdown(totals, top_k, false, dataset_kind);
    }
    println!("============================================================");
}

fn print_llm_summary(totals: &EvalTotals, top_k: usize, dataset_kind: DatasetKind) {
    println!("\n============================================================");
    if totals.evaluated == 0 {
        println!("No evaluable questions");
    } else {
        let recall = totals.retrieval_correct as f64 / totals.evaluated as f64 * 100.0;
        let accuracy = totals.answer_correct as f64 / totals.evaluated as f64 * 100.0;
        println!("Retrieval Recall@{top_k}: {recall:.1}%");
        println!("Answer Accuracy: {accuracy:.1}%");
        print_timing_summary(&totals.timings, totals.evaluated);
        println!("Evaluated: {}", totals.evaluated);
        println!("Skipped: {}", totals.skipped);
        print_dataset_breakdown(totals, top_k, true, dataset_kind);
        print_speed_quality_summary(totals);
    }
    println!("============================================================");
}

fn record_llm_diagnostics(
    totals: &mut EvalTotals,
    retrieval_hit: bool,
    answer_correct: bool,
    prediction: &str,
    context_tokens: usize,
    query_ms: u128,
    answer_ms: u128,
    judge_ms: u128,
    total_ms: u128,
) {
    match (retrieval_hit, answer_correct) {
        (true, true) => totals.quality.retrieval_hit_answer_pass += 1,
        (true, false) => totals.quality.retrieval_hit_answer_fail += 1,
        (false, true) => totals.quality.retrieval_miss_answer_pass += 1,
        (false, false) => totals.quality.retrieval_miss_answer_fail += 1,
    }

    if is_idk_answer(prediction) {
        totals.quality.idk_answers += 1;
    }

    let context_tokens = context_tokens as u128;
    totals.quality.context_tokens_total += context_tokens;
    totals.quality.context_token_samples.push(context_tokens);

    totals.latency_samples.query_ms.push(query_ms);
    totals.latency_samples.answer_ms.push(answer_ms);
    totals.latency_samples.judge_ms.push(judge_ms);
    totals.latency_samples.total_ms.push(total_ms);
}

fn is_idk_answer(prediction: &str) -> bool {
    let normalized = prediction.trim().to_ascii_lowercase();
    let cleaned = normalized.trim_end_matches(|c: char| c.is_ascii_punctuation()).trim().to_string();
    cleaned == "i don't know"
        || cleaned == "i do not know"
        || cleaned == "idk"
        || cleaned.starts_with("i don't know ")
        || cleaned.starts_with("i do not know ")
        || cleaned.starts_with("idk ")
}

fn print_speed_quality_summary(totals: &EvalTotals) {
    if totals.evaluated == 0 {
        return;
    }

    let retrieval_hit_total =
        totals.quality.retrieval_hit_answer_pass + totals.quality.retrieval_hit_answer_fail;
    let retrieval_miss_total =
        totals.quality.retrieval_miss_answer_pass + totals.quality.retrieval_miss_answer_fail;
    let conditioned_accuracy = if retrieval_hit_total == 0 {
        0.0
    } else {
        totals.quality.retrieval_hit_answer_pass as f64 / retrieval_hit_total as f64 * 100.0
    };
    let idk_rate = totals.quality.idk_answers as f64 / totals.evaluated as f64 * 100.0;
    let answer_failures =
        totals.quality.retrieval_hit_answer_fail + totals.quality.retrieval_miss_answer_fail;
    let query_p50 = percentile(&totals.latency_samples.query_ms, 50);
    let query_p95 = percentile(&totals.latency_samples.query_ms, 95);
    let answer_p50 = percentile(&totals.latency_samples.answer_ms, 50);
    let answer_p95 = percentile(&totals.latency_samples.answer_ms, 95);
    let judge_p50 = percentile(&totals.latency_samples.judge_ms, 50);
    let judge_p95 = percentile(&totals.latency_samples.judge_ms, 95);
    let total_p50 = percentile(&totals.latency_samples.total_ms, 50);
    let total_p95 = percentile(&totals.latency_samples.total_ms, 95);
    let context_avg = if totals.evaluated == 0 {
        0
    } else {
        totals.quality.context_tokens_total / totals.evaluated as u128
    };
    let context_p50 = percentile(&totals.quality.context_token_samples, 50);
    let context_p95 = percentile(&totals.quality.context_token_samples, 95);
    let total_avg_ms = totals.timings.total_ms as f64 / totals.evaluated as f64;
    let throughput_qph = if total_avg_ms <= 0.0 {
        0.0
    } else {
        3_600_000.0 / total_avg_ms
    };
    let total_time = totals.timings.total_ms.max(1) as f64;

    println!();
    println!("Quality profile:");
    println!("Answer Accuracy | retrieval hit: {conditioned_accuracy:.1}%");
    println!(
        "Fail breakdown: retrieval hit + answer fail={} | retrieval miss + answer fail={} | retrieval miss + answer pass={}",
        totals.quality.retrieval_hit_answer_fail,
        totals.quality.retrieval_miss_answer_fail,
        totals.quality.retrieval_miss_answer_pass
    );
    if answer_failures > 0 {
        println!(
            "Reader miss share among all answer fails: {:.1}%",
            totals.quality.retrieval_hit_answer_fail as f64 / answer_failures as f64 * 100.0
        );
    }
    println!(
        "\"I don't know\" rate: {:.1}% ({}/{})",
        idk_rate, totals.quality.idk_answers, totals.evaluated
    );
    println!(
        "Retrieval miss rate: {:.1}% ({}/{})",
        retrieval_miss_total as f64 / totals.evaluated as f64 * 100.0,
        retrieval_miss_total,
        totals.evaluated
    );

    println!();
    println!("Speed profile:");
    println!(
        "Time split: ingest={:.1}% | query={:.1}% | answer={:.1}% | judge={:.1}%",
        totals.timings.ingest_ms as f64 / total_time * 100.0,
        totals.timings.query_ms as f64 / total_time * 100.0,
        totals.timings.answer_ms as f64 / total_time * 100.0,
        totals.timings.judge_ms as f64 / total_time * 100.0,
    );
    println!(
        "P50/P95 (ms): query={}/{} | answer={}/{} | judge={}/{} | total={}/{}",
        query_p50, query_p95, answer_p50, answer_p95, judge_p50, judge_p95, total_p50, total_p95
    );
    println!(
        "Estimated throughput: {:.1} questions/hour end-to-end",
        throughput_qph
    );
    println!(
        "Packed context (est tok): avg={} | p50={} | p95={}",
        context_avg, context_p50, context_p95
    );
}

fn percentile(samples: &[u128], percentile: usize) -> u128 {
    if samples.is_empty() {
        return 0;
    }
    let mut ordered = samples.to_vec();
    ordered.sort_unstable();
    let percentile = percentile.clamp(0, 100);
    let rank = (((ordered.len() as f64) * (percentile as f64 / 100.0)).ceil() as usize)
        .saturating_sub(1)
        .min(ordered.len() - 1);
    ordered[rank]
}

fn print_dataset_breakdown(
    totals: &EvalTotals,
    top_k: usize,
    include_accuracy: bool,
    dataset_kind: DatasetKind,
) {
    if totals.category_stats.is_empty() {
        return;
    }

    let (title, categories): (&str, Vec<&str>) = match dataset_kind {
        DatasetKind::Locomo => (
            "Backboard-style LoCoMo breakdown:",
            vec!["Single-Hop", "Multi-Hop", "Open Domain", "Temporal"],
        ),
        DatasetKind::Longmemeval => (
            "Backboard-style LongMemEval breakdown:",
            vec![
                "Single-User",
                "Assistant",
                "Preference",
                "Multi-Session",
                "Temporal",
                "Knowledge",
            ],
        ),
    };
    println!();
    println!("{title}");

    let mut header = format!("{:<20}", "Method");
    for category in &categories {
        header.push_str(&format!(" {:>13}", category));
    }
    header.push_str(&format!(" {:>11}", "Overall"));
    println!("{header}");

    let mut recall_row = format!("{:<20}", format!("Recall@{}", top_k));
    for category in &categories {
        recall_row.push_str(&format!(
            " {:>12.1}%",
            category_metric(totals, category, false)
        ));
    }
    recall_row.push_str(&format!(
        " {:>10.1}%",
        totals.retrieval_correct as f64 / totals.evaluated as f64 * 100.0
    ));
    println!("{recall_row}");

    if include_accuracy {
        let mut accuracy_row = format!("{:<20}", "Answer Accuracy");
        for category in &categories {
            accuracy_row.push_str(&format!(
                " {:>12.1}%",
                category_metric(totals, category, true)
            ));
        }
        accuracy_row.push_str(&format!(
            " {:>10.1}%",
            totals.answer_correct as f64 / totals.evaluated as f64 * 100.0
        ));
        println!("{accuracy_row}");
    }

    let mut counts_row = format!("{:<20}", "Counts");
    for category in &categories {
        counts_row.push_str(&format!(" {:>13}", category_count(totals, category)));
    }
    counts_row.push_str(&format!(" {:>11}", totals.evaluated));
    println!("{counts_row}");
}

fn category_metric(totals: &EvalTotals, label: &str, answer_metric: bool) -> f64 {
    let Some(stats) = totals.category_stats.get(label) else {
        return 0.0;
    };
    if stats.evaluated == 0 {
        return 0.0;
    }
    let numerator = if answer_metric {
        stats.answer_correct
    } else {
        stats.retrieval_correct
    };
    numerator as f64 / stats.evaluated as f64 * 100.0
}

fn category_count(totals: &EvalTotals, label: &str) -> usize {
    totals
        .category_stats
        .get(label)
        .map(|stats| stats.evaluated)
        .unwrap_or(0)
}

fn print_timing_summary(timings: &AggregateTimings, evaluated: usize) {
    if evaluated == 0 {
        return;
    }
    let denom = evaluated as u128;
    println!(
        "Avg timings (ms): ingest={} | query={} | pack={} | answer={} | judge={} | total={}",
        timings.ingest_ms / denom,
        timings.query_ms / denom,
        timings.pack_ms / denom,
        timings.answer_ms / denom,
        timings.judge_ms / denom,
        timings.total_ms / denom,
    );
    println!(
        "Avg engine query stages (ms): route={} | embed={} | ann={} | rerank={} | fts={} | card={} | pref={} | graph={} | session={} | fuse={} | hydrate={} | visible={} | other={} | total={}",
        timings.inner_route_ms / denom,
        timings.inner_embed_ms / denom,
        timings.inner_ann_ms / denom,
        timings.inner_rerank_ms / denom,
        timings.inner_fts_ms / denom,
        timings.inner_card_ms / denom,
        timings.inner_preference_ms / denom,
        timings.inner_graph_ms / denom,
        timings.inner_session_ms / denom,
        timings.inner_fuse_ms / denom,
        timings.inner_hydrate_ms / denom,
        (timings.inner_route_ms
            + timings.inner_embed_ms
            + timings.inner_ann_ms
            + timings.inner_rerank_ms
            + timings.inner_fts_ms
            + timings.inner_card_ms
            + timings.inner_preference_ms
            + timings.inner_graph_ms
            + timings.inner_session_ms
            + timings.inner_fuse_ms
            + timings.inner_hydrate_ms)
            / denom,
        timings
            .inner_query_total_ms
            .saturating_sub(
                timings.inner_route_ms
                    + timings.inner_embed_ms
                    + timings.inner_ann_ms
                    + timings.inner_rerank_ms
                    + timings.inner_fts_ms
                    + timings.inner_card_ms
                    + timings.inner_preference_ms
                    + timings.inner_graph_ms
                    + timings.inner_session_ms
                    + timings.inner_fuse_ms
                    + timings.inner_hydrate_ms
            )
            / denom,
        timings.inner_query_total_ms / denom,
    );
    println!(
        "Avg routing diagnostics: routed_sessions={} | cards={} | events={} | shadows={} | facets={} | scenes={} | scoped_ann_attempts={} | scoped_primary_hits={}",
        timings.routed_sessions / denom,
        timings.memory_card_hits / denom,
        timings.temporal_event_hits / denom,
        timings.shadow_question_hits / denom,
        timings.facet_posting_hits / denom,
        timings.mem_scene_hits / denom,
        timings.scoped_ann_attempts / denom,
        timings.scoped_primary_hits / denom,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locomo_dialog_ids_map_to_session_ids() {
        assert_eq!(
            locomo_dialog_id_to_session_id("D12:7").as_deref(),
            Some("session_12")
        );
        assert_eq!(locomo_dialog_id_to_session_id("bad"), None);
    }

    #[test]
    fn normalize_locomo_filters_to_answered_evaluable_questions() {
        let samples = vec![LocomoSample {
            sample_id: Some("sample-1".to_string()),
            conversation: LocomoConversation {
                _speaker_a: Some("Alice".to_string()),
                _speaker_b: Some("Bob".to_string()),
                fields: HashMap::from([
                    (
                        "session_1_date_time".to_string(),
                        serde_json::Value::String("1 Jan 2024".to_string()),
                    ),
                    (
                        "session_1".to_string(),
                        json!([
                            {"speaker": "Alice", "dia_id": "D1:1", "text": "I moved to Berlin."},
                            {"speaker": "Bob", "dia_id": "D1:2", "text": "That is exciting."}
                        ]),
                    ),
                    (
                        "session_2_date_time".to_string(),
                        serde_json::Value::String("2 Jan 2024".to_string()),
                    ),
                    (
                        "session_2".to_string(),
                        json!([
                            {"speaker": "Alice", "dia_id": "D2:1", "text": "I like running."}
                        ]),
                    ),
                ]),
            },
            qa: vec![
                LocomoQuestion {
                    question: "Where did Alice move?".to_string(),
                    answer: Some(serde_json::Value::String("Berlin".to_string())),
                    _adversarial_answer: None,
                    evidence: vec!["D1:1".to_string()],
                    category: Some(1),
                },
                LocomoQuestion {
                    question: "Trick question".to_string(),
                    answer: None,
                    _adversarial_answer: Some(serde_json::Value::String("No answer".to_string())),
                    evidence: vec!["D2:1".to_string()],
                    category: Some(5),
                },
            ],
        }];

        let normalized = normalize_locomo_samples(samples);
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].question_id.as_deref(), Some("sample-1/q1"));
        assert_eq!(normalized[0].answer_session_ids, vec!["session_1"]);
        assert_eq!(normalized[0].haystack_session_ids.len(), 2);
    }


}
