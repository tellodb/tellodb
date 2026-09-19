use axum::{
    body::{to_bytes, Body},
    http::{Method, Request, Response, StatusCode},
    Router,
};
use serde_json::{json, Value};
use tellodb::{
    api::{build_api, AuthConfig, DEFAULT_TEST_API_KEY},
    config::Config,
    engine::build_state,
    features::Features,
    retrieval::lanes::Lanes,
    runtime_paths::RuntimePaths,
    vector_index::VectorConfig,
};
use tempfile::{tempdir, TempDir};
use tower::ServiceExt;

const DISABLED_TEST_FEATURES: &str = "chunks,gist,keywords,fact_companions,atomic_cards,event_companions,relation_companions,memory_cards,session_router,preferences,retrospective_links,derived_links,graph_edges,semantic_dedup,consolidation,metrics,predicate_canon";

async fn test_app() -> (Router, TempDir) {
    let temp = tempdir().expect("temporary test directory");
    let paths = RuntimePaths::from_root(temp.path().to_path_buf());
    let mut config = Config {
        features: Features::parse(DISABLED_TEST_FEATURES).expect("test feature list"),
        lanes: Lanes::parse("fts").expect("test lane list"),
        vector: VectorConfig::new(384),
        ..Config::default()
    };
    config.embedding.model_id = "test".to_string();
    config.embedding.dimension = Some(384);
    config.embedding.cache_enabled = false;
    config.embedding.executors = 1;
    config.embedding.batch = 8;
    config.rerank.model = "none".to_string();
    config.rerank.enabled = false;
    config.server.api_key = Some(DEFAULT_TEST_API_KEY.to_string());
    let auth = AuthConfig::from_config(&config).expect("test auth configuration");
    let state = build_state(&paths, auth, config).await.expect("test engine state");
    (build_api(state), temp)
}

fn ingest_payload(
    entity_id: &str,
    memory_id: &str,
    timestamp: u64,
    textual_content: &str,
    kind: Option<&str>,
    fact_key: Option<&str>,
) -> Value {
    json!({
        "entity_id": entity_id,
        "memory_id": memory_id,
        "timestamp": timestamp,
        "session_id": "http-test",
        "turn_index": 0,
        "role": "user",
        "textual_content": textual_content,
        "relations": [],
        "kind": kind,
        "fact_key": fact_key,
        "fact_subject": entity_id,
        "fact_predicate": "lives_in",
        "fact_object": textual_content,
        "index_semantic": false,
        "enable_semantic_dedup": false,
        "enable_consolidation": false,
        "enable_mining": false,
    })
}

fn query_payload(entity_id: &str, textual_query: &str, point_in_time_ms: Option<u64>) -> Value {
    let mut payload = json!({
        "textual_query": textual_query,
        "limit": 10,
        "entity_id": entity_id,
        "enable_neural_rerank": false,
    });
    if let Some(timestamp) = point_in_time_ms {
        payload["point_in_time_ms"] = json!(timestamp);
    }
    payload
}

async fn json_request(
    app: &Router,
    method: Method,
    path: &str,
    payload: Value,
    authenticated: bool,
) -> Response<Body> {
    let mut request =
        Request::builder().method(method).uri(path).header("content-type", "application/json");
    if authenticated {
        request = request.header("x-api-key", DEFAULT_TEST_API_KEY);
    }
    app.clone()
        .oneshot(request.body(Body::from(payload.to_string())).expect("request body"))
        .await
        .expect("router response")
}

async fn response_json(response: Response<Body>) -> Value {
    let bytes = to_bytes(response.into_body(), 16 * 1024 * 1024).await.expect("response body");
    serde_json::from_slice(&bytes).expect("json response")
}

async fn ingest_one(app: &Router, payload: Value) {
    let response = json_request(app, Method::POST, "/ingest", payload, true).await;
    assert_eq!(response.status(), StatusCode::CREATED);
}

async fn query_results(app: &Router, payload: Value) -> Vec<Value> {
    let response = json_request(app, Method::POST, "/query", payload, true).await;
    assert_eq!(response.status(), StatusCode::OK);
    response_json(response).await.as_array().cloned().expect("query array")
}

#[tokio::test]
async fn query_without_api_key_is_401() {
    let (app, _temp) = test_app().await;
    let response =
        json_request(&app, Method::POST, "/query", query_payload("alice", "Denver", None), false)
            .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn ingest_then_query_returns_the_memory() {
    let (app, _temp) = test_app().await;
    ingest_one(
        &app,
        ingest_payload("alice", "memory-one", 1_000, "Ada lives in Denver", None, None),
    )
    .await;

    let results = query_results(&app, query_payload("alice", "Ada lives in Denver", None)).await;
    assert!(results.iter().any(|result| result["memory_id"] == "memory-one"));
}

#[tokio::test]
async fn batch_ingest_and_single_ingest_agree() {
    let (app, _temp) = test_app().await;
    let batch_payload =
        ingest_payload("batch-user", "batch-memory", 1_000, "Ada enjoys green tea", None, None);
    let batch_response = json_request(
        &app,
        Method::POST,
        "/ingest/batch",
        json!({ "items": [batch_payload] }),
        true,
    )
    .await;
    assert_eq!(batch_response.status(), StatusCode::CREATED);

    ingest_one(
        &app,
        ingest_payload("single-user", "single-memory", 1_000, "Ada enjoys green tea", None, None),
    )
    .await;

    let batch_results =
        query_results(&app, query_payload("batch-user", "Ada enjoys green tea", None)).await;
    let single_results =
        query_results(&app, query_payload("single-user", "Ada enjoys green tea", None)).await;
    assert!(batch_results.iter().any(|result| {
        result["memory_id"] == "batch-memory" && result["textual_content"] == "Ada enjoys green tea"
    }));
    assert!(single_results.iter().any(|result| {
        result["memory_id"] == "single-memory"
            && result["textual_content"] == "Ada enjoys green tea"
    }));
}

#[tokio::test]
async fn superseded_fact_is_marked_stale() {
    let (app, _temp) = test_app().await;
    ingest_one(
        &app,
        ingest_payload(
            "alice",
            "fact-old",
            1_000,
            "Alice lives in Denver",
            Some("fact"),
            Some("residence"),
        ),
    )
    .await;
    ingest_one(
        &app,
        ingest_payload(
            "alice",
            "fact-new",
            2_000,
            "Alice lives in Seattle",
            Some("fact"),
            Some("residence"),
        ),
    )
    .await;

    let results = query_results(&app, query_payload("alice", "Alice lives", None)).await;
    let old = results
        .iter()
        .find(|result| result["memory_id"] == "fact-old")
        .expect("superseded fact in query results");
    assert_eq!(old["superseded_by"], "fact-new");
    assert!(old["why_stale"].is_object());
}

#[tokio::test]
async fn point_in_time_query_excludes_later_memories() {
    let (app, _temp) = test_app().await;
    ingest_one(&app, ingest_payload("alice", "before", 1_000, "Ada visited Denver", None, None))
        .await;
    ingest_one(
        &app,
        ingest_payload("alice", "after", 2_000, "Ada visited Denver later", None, None),
    )
    .await;

    let results =
        query_results(&app, query_payload("alice", "Ada visited Denver", Some(1_500))).await;
    assert!(results.iter().any(|result| result["memory_id"] == "before"));
    assert!(!results.iter().any(|result| result["memory_id"] == "after"));
}

#[tokio::test]
async fn oversized_body_is_413() {
    let (app, _temp) = test_app().await;
    let body = Body::from(vec![b'x'; 10 * 1024 * 1024 + 1]);
    let request = Request::builder()
        .method(Method::POST)
        .uri("/ingest")
        .header("content-type", "application/json")
        .header("x-api-key", DEFAULT_TEST_API_KEY)
        .body(body)
        .expect("oversized request");
    let response = app.clone().oneshot(request).await.expect("router response");
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn login_is_rate_limited_after_five_attempts() {
    let (app, _temp) = test_app().await;
    for _ in 0..5 {
        let response = json_request(
            &app,
            Method::POST,
            "/login",
            json!({ "username": "missing-user", "password": "incorrect-password" }),
            false,
        )
        .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
    let response = json_request(
        &app,
        Method::POST,
        "/login",
        json!({ "username": "missing-user", "password": "incorrect-password" }),
        false,
    )
    .await;
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn unknown_ranking_config_key_fails_startup() {
    let temp = tempdir().expect("temporary test directory");
    std::fs::write(temp.path().join("ranking_config.json"), r#"{"graph_weight": 1.0}"#)
        .expect("ranking config");
    let paths = RuntimePaths::from_root(temp.path().to_path_buf());
    let error = match build_state(&paths, AuthConfig::embedded(), Config::default()).await {
        Ok(_) => panic!("unknown ranking key must fail startup"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("ranking config"));
}
