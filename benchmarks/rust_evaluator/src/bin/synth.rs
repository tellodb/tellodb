use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::{Duration, NaiveDate, NaiveDateTime};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(name = "synth", about = "Synthetic dataset generator for Tellodb evaluation")]
struct Args {
    /// Number of distinct entities to simulate
    #[arg(long, default_value_t = 10)]
    entities: usize,

    /// Target number of memories (turns/sessions) per entity
    #[arg(long, default_value_t = 20)]
    memories_per_entity: usize,

    /// Rate of temporal updates (0.0 to 1.0)
    #[arg(long, default_value_t = 0.25)]
    update_rate: f64,

    /// Random seed for deterministic generation
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Phrasing set partition: train or heldout
    #[arg(long, value_enum, default_value_t = PhrasingSet::Train)]
    phrasing_set: PhrasingSet,

    /// Output JSON path (defaults to stdout if not specified)
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum PhrasingSet {
    Train,
    Heldout,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Turn {
    role: String,
    content: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
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

/// Deterministic SplitMix64 PRNG.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, start: usize, end: usize) -> usize {
        assert!(start < end);
        start + (self.next_u64() as usize % (end - start))
    }

    fn choose<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        assert!(!slice.is_empty());
        let idx = self.range(0, slice.len());
        &slice[idx]
    }
}

// 7 Templates with >= 8 phrasings each (disjoint train/heldout sets)
const RESIDENCE_VALUES: &[&str] = &[
    "Seattle",
    "London",
    "Austin",
    "Berlin",
    "Tokyo",
    "Paris",
    "Toronto",
    "Sydney",
    "Singapore",
    "Dublin",
];

const RESIDENCE_TRAIN: &[&str] = &[
    "I moved to {val} recently.",
    "I just relocated my home to {val}.",
    "Living in {val} now, really enjoying the neighborhood.",
    "My new address is in {val}.",
];
const RESIDENCE_HELDOUT: &[&str] = &[
    "I have settled down in {val}.",
    "Currently residing in {val}.",
    "Packed my bags and moved over to {val}.",
    "I am based in {val} these days.",
];

const EMPLOYER_VALUES: &[&str] = &[
    "Acme Corp",
    "Apex Labs",
    "Nova Dynamics",
    "Starlight Systems",
    "Nexus Technologies",
    "Omni Horizon",
    "Vanguard AI",
    "Quantum Leap",
];

const EMPLOYER_TRAIN: &[&str] = &[
    "I joined {val} as my new company.",
    "Started working at {val} this week.",
    "I am employed at {val} now.",
    "Signed my employment contract with {val}.",
];
const EMPLOYER_HELDOUT: &[&str] = &[
    "I have transitioned to {val} for work.",
    "My current employer is {val}.",
    "Working as a full-timer at {val} nowadays.",
    "I recently came on board at {val}.",
];

const JOB_TITLE_VALUES: &[&str] = &[
    "Software Engineer",
    "Systems Architect",
    "Product Manager",
    "Data Scientist",
    "DevOps Specialist",
    "Research Scientist",
    "Engineering Director",
    "Security Analyst",
];

const JOB_TITLE_TRAIN: &[&str] = &[
    "My job title is {val}.",
    "I was promoted to {val}.",
    "Taking on the role of {val}.",
    "Working professionally as a {val}.",
];
const JOB_TITLE_HELDOUT: &[&str] = &[
    "I hold the position of {val}.",
    "Serving in the capacity of {val}.",
    "My official designation is {val}.",
    "Leading tasks as a {val}.",
];

const PET_VALUES: &[&str] = &[
    "1 dog",
    "2 cats",
    "3 rescue dogs",
    "1 golden retriever",
    "2 rabbits",
    "4 parakeets",
    "1 adopted cat",
    "3 ferrets",
];

const PET_TRAIN: &[&str] = &[
    "I have {val} at home.",
    "We adopted {val} recently.",
    "Caring for {val} in my household.",
    "My home now has {val}.",
];
const PET_HELDOUT: &[&str] = &[
    "Currently taking care of {val}.",
    "Welcomed {val} into our family.",
    "Living alongside {val}.",
    "Our apartment is home to {val}.",
];

const PARTNER_VALUES: &[&str] =
    &["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Casey", "Riley", "Cameron", "Jamie", "Quinn"];

const PARTNER_TRAIN: &[&str] = &[
    "My partner is {val}.",
    "I went out for dinner with my partner, {val}.",
    "Spending quality time with my spouse {val}.",
    "My partner's name is {val}.",
];
const PARTNER_HELDOUT: &[&str] = &[
    "My significant other is {val}.",
    "My fiancé {val} and I had a great time.",
    "Enjoying our anniversary with my partner {val}.",
    "Sharing life together with {val}.",
];

const FAVORITE_X_CATEGORIES: &[(&str, &[&str])] = &[
    ("coffee", &["Cortado", "Espresso", "Cold brew", "Flat white", "Pour over", "Matcha latte"]),
    (
        "color",
        &[
            "Emerald green",
            "Midnight blue",
            "Amber orange",
            "Crimson red",
            "Lavender",
            "Slate gray",
        ],
    ),
    ("cuisine", &["Ethiopian", "Japanese", "Mexican", "Italian", "Thai", "Vietnamese"]),
];

const FAVORITE_X_TRAIN: &[&str] = &[
    "My absolute favorite {item} is {val}.",
    "I really love {val} the most when it comes to {item}.",
    "{val} is definitely my top pick for {item}.",
    "Whenever I choose {item}, my favorite is {val}.",
];
const FAVORITE_X_HELDOUT: &[&str] = &[
    "Nothing beats {val} as my preferred {item}.",
    "If you ask me about my favorite {item}, it is {val}.",
    "My go-to choice for {item} has always been {val}.",
    "Hands down, {val} is my number one {item}.",
];

const ACCUMULATING_ITEMS: &[&str] = &[
    "marathons completed",
    "books read this year",
    "countries visited",
    "conference talks delivered",
    "patents filed",
];

const ACCUMULATING_TRAIN: &[&str] = &[
    "I just hit a new milestone: {val} {item}.",
    "Updating my count: I have now completed {val} {item}.",
    "Reached {val} total {item} today.",
    "My current total for {item} has reached {val}.",
];
const ACCUMULATING_HELDOUT: &[&str] = &[
    "Marked down another one, bringing my total {item} to {val}.",
    "Tallying it up: that makes {val} {item} so far.",
    "Proud to share that my record of {item} is now {val}.",
    "Standing at {val} {item} in aggregate now.",
];

const DISTRACTORS: &[(&str, &str)] = &[
    (
        "Can you help me summarize the main differences between TCP and UDP?",
        "Certainly! TCP is a connection-oriented, reliable protocol with congestion control and retransmission, while UDP is connectionless, lightweight, and suitable for latency-sensitive streaming.",
    ),
    (
        "What is a good recipe for homemade vegetable soup?",
        "A classic vegetable soup combines diced carrots, celery, onions, potatoes, and tomatoes simmered in seasoned vegetable broth with herbs like thyme and bay leaf.",
    ),
    (
        "How does quicksort achieve an average O(N log N) time complexity?",
        "Quicksort chooses a pivot, partitions the array into elements smaller and larger than the pivot, and recursively sorts each half. On average, the recursion tree has depth log N with O(N) work per level.",
    ),
    (
        "Can you recommend some classic science fiction novels to read?",
        "Consider 'Dune' by Frank Herbert, 'Foundation' by Isaac Asimov, 'Neuromancer' by William Gibson, and 'Hyperion' by Dan Simmons.",
    ),
    (
        "What is the principle of conservation of energy in physics?",
        "The law of conservation of energy states that energy cannot be created or destroyed, only transformed from one form to another in an isolated system.",
    ),
];

// Temporal state tracking for an entity
#[derive(Clone)]
struct TemporalFact {
    value: String,
    session_id: String,
    parsed_date: NaiveDateTime,
}

#[derive(Clone)]
struct AccumulatingFact {
    item: String,
    count: usize,
    session_id: String,
}

/// (attribute name, value history, current-value questions, as-of questions)
/// with one question per phrasing set.
type AttributeQuestions<'a> = (&'a str, &'a [TemporalFact], [&'a str; 2], [&'a str; 2]);

fn favorite_values_history(history: &[(String, TemporalFact)]) -> Vec<TemporalFact> {
    history.iter().map(|(_, fact)| fact.clone()).collect()
}

fn format_date(dt: NaiveDateTime) -> String {
    dt.format("%Y/%m/%d (%a) %H:%M").to_string()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut rng = Rng::new(args.seed);

    let mut instances = Vec::new();

    let phrasing = args.phrasing_set;
    let res_phrasings =
        if phrasing == PhrasingSet::Train { RESIDENCE_TRAIN } else { RESIDENCE_HELDOUT };
    let emp_phrasings =
        if phrasing == PhrasingSet::Train { EMPLOYER_TRAIN } else { EMPLOYER_HELDOUT };
    let job_phrasings =
        if phrasing == PhrasingSet::Train { JOB_TITLE_TRAIN } else { JOB_TITLE_HELDOUT };
    let pet_phrasings = if phrasing == PhrasingSet::Train { PET_TRAIN } else { PET_HELDOUT };
    let partner_phrasings =
        if phrasing == PhrasingSet::Train { PARTNER_TRAIN } else { PARTNER_HELDOUT };
    let fav_phrasings =
        if phrasing == PhrasingSet::Train { FAVORITE_X_TRAIN } else { FAVORITE_X_HELDOUT };
    let acc_phrasings =
        if phrasing == PhrasingSet::Train { ACCUMULATING_TRAIN } else { ACCUMULATING_HELDOUT };

    let base_start_date =
        NaiveDate::from_ymd_opt(2023, 1, 1).unwrap().and_hms_opt(9, 0, 0).unwrap();

    for entity_idx in 0..args.entities {
        let entity_id = format!("entity_{:04}", entity_idx);

        let mut current_dt = base_start_date + Duration::days((entity_idx * 7) as i64);

        let mut residence_history: Vec<TemporalFact> = Vec::new();
        let mut employer_history: Vec<TemporalFact> = Vec::new();
        let mut job_history: Vec<TemporalFact> = Vec::new();
        let mut pet_history: Vec<TemporalFact> = Vec::new();
        let mut partner_history: Vec<TemporalFact> = Vec::new();
        let mut _favorite_history: Vec<(String, TemporalFact)> = Vec::new();
        let mut acc_history: Vec<AccumulatingFact> = Vec::new();

        let mut haystack_dates = Vec::new();
        let mut haystack_sessions = Vec::new();
        let mut haystack_session_ids = Vec::new();

        let initial_res = *rng.choose(RESIDENCE_VALUES);
        let initial_emp = *rng.choose(EMPLOYER_VALUES);
        let initial_job = *rng.choose(JOB_TITLE_VALUES);
        let initial_pet = *rng.choose(PET_VALUES);
        let initial_partner = *rng.choose(PARTNER_VALUES);

        let (fav_item, fav_values) = *rng.choose(FAVORITE_X_CATEGORIES);
        let initial_fav = *rng.choose(fav_values);

        let acc_item = *rng.choose(ACCUMULATING_ITEMS);
        let mut acc_count = rng.range(1, 4);

        let total_sessions = args.memories_per_entity;
        for s_idx in 0..total_sessions {
            let session_id = format!("{}_sess_{:03}", entity_id, s_idx);
            let session_date_str = format_date(current_dt);

            let mut session_turns = Vec::new();

            let action_roll = rng.next_f64();
            if s_idx == 0 {
                // Initial baseline facts in the first session
                let pattern = *rng.choose(res_phrasings);
                let text = pattern.replace("{val}", initial_res);
                session_turns.push(Turn {
                    role: "user".to_string(),
                    content: serde_json::Value::String(text),
                });
                session_turns.push(Turn {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(format!(
                        "Understood, noted that you are in {}.",
                        initial_res
                    )),
                });

                residence_history.push(TemporalFact {
                    value: initial_res.to_string(),
                    session_id: session_id.clone(),
                    parsed_date: current_dt,
                });
            } else if s_idx == 1 {
                let pattern = *rng.choose(emp_phrasings);
                let text = pattern.replace("{val}", initial_emp);
                session_turns.push(Turn {
                    role: "user".to_string(),
                    content: serde_json::Value::String(text),
                });
                session_turns.push(Turn {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(format!(
                        "Congratulations on working at {}!",
                        initial_emp
                    )),
                });

                employer_history.push(TemporalFact {
                    value: initial_emp.to_string(),
                    session_id: session_id.clone(),
                    parsed_date: current_dt,
                });
            } else if s_idx == 2 {
                let pattern = *rng.choose(job_phrasings);
                let text = pattern.replace("{val}", initial_job);
                session_turns.push(Turn {
                    role: "user".to_string(),
                    content: serde_json::Value::String(text),
                });
                session_turns.push(Turn {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(format!(
                        "Noted your role as {}.",
                        initial_job
                    )),
                });

                job_history.push(TemporalFact {
                    value: initial_job.to_string(),
                    session_id: session_id.clone(),
                    parsed_date: current_dt,
                });
            } else if s_idx == 3 {
                let pattern = *rng.choose(pet_phrasings);
                let text = pattern.replace("{val}", initial_pet);
                session_turns.push(Turn {
                    role: "user".to_string(),
                    content: serde_json::Value::String(text),
                });
                session_turns.push(Turn {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(format!(
                        "Animals bring so much joy! Glad you have {}.",
                        initial_pet
                    )),
                });

                pet_history.push(TemporalFact {
                    value: initial_pet.to_string(),
                    session_id: session_id.clone(),
                    parsed_date: current_dt,
                });
            } else if s_idx == 4 {
                let pattern = *rng.choose(partner_phrasings);
                let text = pattern.replace("{val}", initial_partner);
                session_turns.push(Turn {
                    role: "user".to_string(),
                    content: serde_json::Value::String(text),
                });
                session_turns.push(Turn {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(format!(
                        "It's great to hear about you and {}.",
                        initial_partner
                    )),
                });

                partner_history.push(TemporalFact {
                    value: initial_partner.to_string(),
                    session_id: session_id.clone(),
                    parsed_date: current_dt,
                });
            } else if s_idx == 5 {
                let pattern = *rng.choose(fav_phrasings);
                let text = pattern.replace("{item}", fav_item).replace("{val}", initial_fav);
                session_turns.push(Turn {
                    role: "user".to_string(),
                    content: serde_json::Value::String(text),
                });
                session_turns.push(Turn {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(format!(
                        "{} is an excellent favorite {}.",
                        initial_fav, fav_item
                    )),
                });

                _favorite_history.push((
                    fav_item.to_string(),
                    TemporalFact {
                        value: initial_fav.to_string(),
                        session_id: session_id.clone(),
                        parsed_date: current_dt,
                    },
                ));
            } else if s_idx == 6 {
                let pattern = *rng.choose(acc_phrasings);
                let text =
                    pattern.replace("{item}", acc_item).replace("{val}", &acc_count.to_string());
                session_turns.push(Turn {
                    role: "user".to_string(),
                    content: serde_json::Value::String(text),
                });
                session_turns.push(Turn {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(format!(
                        "Impressive progress reaching {} {}!",
                        acc_count, acc_item
                    )),
                });

                acc_history.push(AccumulatingFact {
                    item: acc_item.to_string(),
                    count: acc_count,
                    session_id: session_id.clone(),
                });
            } else if action_roll < args.update_rate {
                // Temporal update
                let update_type = rng.range(0, 4);
                match update_type {
                    0 => {
                        let new_val = *rng.choose(RESIDENCE_VALUES);
                        let pattern = *rng.choose(res_phrasings);
                        let text = pattern.replace("{val}", new_val);
                        session_turns.push(Turn {
                            role: "user".to_string(),
                            content: serde_json::Value::String(text),
                        });
                        session_turns.push(Turn {
                            role: "assistant".to_string(),
                            content: serde_json::Value::String(format!(
                                "Noted your move to {}!",
                                new_val
                            )),
                        });
                        residence_history.push(TemporalFact {
                            value: new_val.to_string(),
                            session_id: session_id.clone(),
                            parsed_date: current_dt,
                        });
                    }
                    1 => {
                        let new_val = *rng.choose(EMPLOYER_VALUES);
                        let pattern = *rng.choose(emp_phrasings);
                        let text = pattern.replace("{val}", new_val);
                        session_turns.push(Turn {
                            role: "user".to_string(),
                            content: serde_json::Value::String(text),
                        });
                        session_turns.push(Turn {
                            role: "assistant".to_string(),
                            content: serde_json::Value::String(format!(
                                "Best of luck at your new workplace {}!",
                                new_val
                            )),
                        });
                        employer_history.push(TemporalFact {
                            value: new_val.to_string(),
                            session_id: session_id.clone(),
                            parsed_date: current_dt,
                        });
                    }
                    2 => {
                        let new_val = *rng.choose(JOB_TITLE_VALUES);
                        let pattern = *rng.choose(job_phrasings);
                        let text = pattern.replace("{val}", new_val);
                        session_turns.push(Turn {
                            role: "user".to_string(),
                            content: serde_json::Value::String(text),
                        });
                        session_turns.push(Turn {
                            role: "assistant".to_string(),
                            content: serde_json::Value::String(format!(
                                "Congratulations on the role of {}!",
                                new_val
                            )),
                        });
                        job_history.push(TemporalFact {
                            value: new_val.to_string(),
                            session_id: session_id.clone(),
                            parsed_date: current_dt,
                        });
                    }
                    _ => {
                        acc_count += rng.range(1, 4);
                        let pattern = *rng.choose(acc_phrasings);
                        let text = pattern
                            .replace("{item}", acc_item)
                            .replace("{val}", &acc_count.to_string());
                        session_turns.push(Turn {
                            role: "user".to_string(),
                            content: serde_json::Value::String(text),
                        });
                        session_turns.push(Turn {
                            role: "assistant".to_string(),
                            content: serde_json::Value::String(format!(
                                "Great milestone reaching {} {}!",
                                acc_count, acc_item
                            )),
                        });
                        acc_history.push(AccumulatingFact {
                            item: acc_item.to_string(),
                            count: acc_count,
                            session_id: session_id.clone(),
                        });
                    }
                }
            } else {
                // Distractor conversation
                let (q, a) = *rng.choose(DISTRACTORS);
                session_turns.push(Turn {
                    role: "user".to_string(),
                    content: serde_json::Value::String(q.to_string()),
                });
                session_turns.push(Turn {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(a.to_string()),
                });
            }

            haystack_dates.push(session_date_str);
            haystack_session_ids.push(session_id);
            haystack_sessions.push(session_turns);

            // Increment time between sessions (from 8 hours to 5 days)
            let delta_hours = rng.range(8, 120) as i64;
            let delta_mins = rng.range(0, 60) as i64;
            current_dt = current_dt + Duration::hours(delta_hours) + Duration::minutes(delta_mins);
        }

        let evaluation_now_dt = current_dt + Duration::days(2);
        let evaluation_now_str = format_date(evaluation_now_dt);

        // 1-2. Current-value and value-as-of questions for every tracked
        // attribute. As-of questions target a random update in the chain, not
        // always the first one.
        let attributes: [AttributeQuestions; 6] = [
            (
                "residence",
                &residence_history,
                ["What city do I live in right now?", "Where am I currently residing?"],
                ["Where was I living as of {date}?", "Where did I reside around {date}?"],
            ),
            (
                "employer",
                &employer_history,
                ["Where do I work these days?", "Which company employs me at present?"],
                ["Where was I working as of {date}?", "Who was my employer around {date}?"],
            ),
            (
                "job_title",
                &job_history,
                ["What is my job title right now?", "What role do I currently hold?"],
                ["What was my job title as of {date}?", "What role did I have around {date}?"],
            ),
            (
                "pet",
                &pet_history,
                ["What pet do I have now?", "Which animal currently lives with me?"],
                ["What pet did I have as of {date}?", "Which animal lived with me around {date}?"],
            ),
            (
                "partner",
                &partner_history,
                ["Who is my partner now?", "Who am I currently with?"],
                ["Who was my partner as of {date}?", "Who was I with around {date}?"],
            ),
            (
                "favorite",
                &favorite_values_history(&_favorite_history),
                ["What is my favorite {item} right now?", "Which {item} do I currently like best?"],
                [
                    "What was my favorite {item} as of {date}?",
                    "Which {item} did I like best around {date}?",
                ],
            ),
        ];
        let phrasing_idx = usize::from(phrasing != PhrasingSet::Train);
        for (attribute, history, current_qs, as_of_qs) in attributes {
            let Some(latest) = history.last() else {
                continue;
            };
            instances.push(Instance {
                question_id: Some(format!("synth_{}_{}_current_val", entity_id, attribute)),
                entity_id: Some(entity_id.clone()),
                question_type: Some("current-value".to_string()),
                question_date: Some(evaluation_now_str.clone()),
                question: current_qs[phrasing_idx].replace("{item}", fav_item),
                haystack_dates: haystack_dates.clone(),
                haystack_sessions: haystack_sessions.clone(),
                haystack_session_ids: haystack_session_ids.clone(),
                answer_session_ids: vec![latest.session_id.clone()],
                answer: Some(serde_json::Value::String(latest.value.clone())),
            });

            if history.len() > 1 {
                let idx = rng.range(0, history.len() - 1);
                let (prior, next_update) = (&history[idx], &history[idx + 1]);
                let mid_secs = (next_update.parsed_date - prior.parsed_date).num_seconds() / 2;
                let query_target_dt = prior.parsed_date + Duration::seconds(mid_secs);
                let date = query_target_dt.format("%Y/%m/%d").to_string();
                instances.push(Instance {
                    question_id: Some(format!("synth_{}_{}_value_as_of", entity_id, attribute)),
                    entity_id: Some(entity_id.clone()),
                    question_type: Some("value-as-of-date".to_string()),
                    question_date: Some(format_date(query_target_dt)),
                    question: as_of_qs[phrasing_idx]
                        .replace("{date}", &date)
                        .replace("{item}", fav_item),
                    haystack_dates: haystack_dates.clone(),
                    haystack_sessions: haystack_sessions.clone(),
                    haystack_session_ids: haystack_session_ids.clone(),
                    answer_session_ids: vec![prior.session_id.clone()],
                    answer: Some(serde_json::Value::String(prior.value.clone())),
                });
            }
        }

        // 3. Count-over-time question (e.g. accumulating items)
        if let Some(latest_acc) = acc_history.last() {
            let question = if phrasing == PhrasingSet::Train {
                format!("What is my total count of {}?", latest_acc.item)
            } else {
                format!("What does my running tally of {} stand at?", latest_acc.item)
            };

            instances.push(Instance {
                question_id: Some(format!("synth_{}_count_over_time", entity_id)),
                entity_id: Some(entity_id.clone()),
                question_type: Some("count-over-time".to_string()),
                question_date: Some(evaluation_now_str.clone()),
                question,
                haystack_dates: haystack_dates.clone(),
                haystack_sessions: haystack_sessions.clone(),
                haystack_session_ids: haystack_session_ids.clone(),
                answer_session_ids: vec![latest_acc.session_id.clone()],
                answer: Some(serde_json::Value::Number(serde_json::Number::from(latest_acc.count))),
            });
        }

        // 4. Never-mentioned question (abstention)
        let unmentioned_q = if phrasing == PhrasingSet::Train {
            *rng.choose(&[
                "What is my car's license plate number?",
                "What is my gym locker combination?",
                "What is my mother's maiden name?",
                "What is my home Wi-Fi password?",
            ])
        } else {
            *rng.choose(&[
                "What is my frequent flyer membership number?",
                "What brand of wristwatch do I wear?",
                "What is my bank account PIN?",
                "What high school did I attend?",
            ])
        };

        instances.push(Instance {
            question_id: Some(format!("synth_{}_never_mentioned", entity_id)),
            entity_id: Some(entity_id.clone()),
            question_type: Some("never-mentioned".to_string()),
            question_date: Some(evaluation_now_str.clone()),
            question: unmentioned_q.to_string(),
            haystack_dates: haystack_dates.clone(),
            haystack_sessions: haystack_sessions.clone(),
            haystack_session_ids: haystack_session_ids.clone(),
            answer_session_ids: vec![],
            answer: None,
        });
    }

    let json_bytes = serde_json::to_vec_pretty(&instances)
        .context("Failed to serialize generated instances to JSON")?;

    match args.output {
        Some(path) => {
            let mut file = File::create(&path)
                .with_context(|| format!("Failed to create output file {}", path.display()))?;
            file.write_all(&json_bytes)?;
            eprintln!(
                "Generated {} synthetic instances for {} entities written to {}",
                instances.len(),
                args.entities,
                path.display()
            );
        }
        None => {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            handle.write_all(&json_bytes)?;
        }
    }

    Ok(())
}
