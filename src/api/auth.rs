use axum::http::{
    header::{AUTHORIZATION, CONTENT_TYPE},
    HeaderMap, HeaderName, HeaderValue, Method, StatusCode,
};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::CorsLayer;

use crate::api::{EngineState, PlatformWriteOp};
use crate::platform::{ApiKeyAuth, PublicUser};

pub const DEFAULT_TEST_API_KEY: &str = "XXX1111AAA";

#[derive(Clone)]
pub struct AuthConfig {
    pub api_key: Option<Arc<str>>,
}

impl AuthConfig {
    pub fn from_env() -> Self {
        let api_key = env::var("TEMPORAL_MEMORY_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| env::var("TELLODB_API_KEY").ok().filter(|value| !value.trim().is_empty()));

        let api_key = match api_key {
            Some(key) => key.trim().to_string(),
            None => {
                // In release builds, API key is required for security.
                // In debug builds, fall back to the default test key so local dev works out of the box.
                if cfg!(debug_assertions) {
                    tracing::warn!(
                        "WARNING: Using default test API key '{}'. Set TEMPORAL_MEMORY_API_KEY or TELLODB_API_KEY for production.",
                        DEFAULT_TEST_API_KEY
                    );
                    DEFAULT_TEST_API_KEY.to_string()
                } else {
                    panic!(
                        "FATAL: TEMPORAL_MEMORY_API_KEY or TELLODB_API_KEY must be set in production. \
                         The default test key is not allowed in release builds."
                    );
                }
            }
        };

        Self { api_key: Some(Arc::<str>::from(api_key)) }
    }

    pub fn is_required(&self) -> bool {
        self.api_key.is_some()
    }
}

pub fn request_api_key(headers: &HeaderMap) -> Option<&str> {
    if let Some(value) = headers.get("x-api-key").and_then(|value| value.to_str().ok()) {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return Some(trimmed);
        }
    }

    let value = headers.get(AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    if scheme.eq_ignore_ascii_case("bearer") {
        let trimmed = token.trim();
        if !trimmed.is_empty() {
            return Some(trimmed);
        }
    }
    None
}

pub fn request_bearer_token(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    let trimmed = token.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

pub fn cors_allow_origins() -> Vec<HeaderValue> {
    let configured = env::var("TEMPORAL_MEMORY_CORS_ALLOW_ORIGINS")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var("TELLODB_CORS_ALLOW_ORIGINS").ok().filter(|value| !value.trim().is_empty())
        })
        .unwrap_or_else(|| "https://tellodb.com".to_string());

    let mut origins = configured
        .split(',')
        .filter_map(|origin| {
            let trimmed = origin.trim().trim_end_matches('/');
            if trimmed.is_empty() {
                None
            } else {
                HeaderValue::from_str(trimmed).ok()
            }
        })
        .collect::<Vec<_>>();

    if origins.is_empty() {
        origins.push(HeaderValue::from_static("https://tellodb.com"));
    }

    origins
}

pub fn build_cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(cors_allow_origins())
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers([AUTHORIZATION, CONTENT_TYPE, HeaderName::from_static("x-api-key")])
        .expose_headers([
            HeaderName::from_static("x-tm-total-ms"),
            HeaderName::from_static("x-tm-total-us"),
        ])
}

#[derive(Debug, Clone)]
pub enum RequestPrincipal {
    GlobalApiKey,
    UserApiKey(ApiKeyAuth),
}

pub fn principal_user_id(principal: &RequestPrincipal) -> Option<&str> {
    match principal {
        RequestPrincipal::UserApiKey(auth) => Some(auth.user_id.as_str()),
        RequestPrincipal::GlobalApiKey => None,
    }
}

pub fn principal_namespace_prefix(principal: &RequestPrincipal) -> Option<String> {
    match principal {
        RequestPrincipal::UserApiKey(auth) => Some(format!("{}::", auth.user_id)),
        RequestPrincipal::GlobalApiKey => None,
    }
}

pub fn scope_entity_id(entity_id: &str, prefix: Option<&str>) -> String {
    match prefix {
        Some(p) if !entity_id.starts_with(p) => format!("{}{}", p, entity_id),
        _ => entity_id.to_string(),
    }
}

pub fn authorize_global_api_key(headers: &HeaderMap, auth: &AuthConfig) -> Result<(), StatusCode> {
    let Some(expected) = auth.api_key.as_deref() else {
        return Ok(());
    };
    if request_api_key(headers).is_some_and(|provided| constant_time_eq(provided, expected)) {
        Ok(())
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

/// Compare two strings in constant time to prevent timing side-channel attacks.
/// This protects the API key from character-by-character brute force.
fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut result: u8 = 0;
    for (x, y) in a.bytes().zip(b.bytes()) {
        result |= x ^ y;
    }
    result == 0
}

pub fn authorize_request(
    headers: &HeaderMap,
    state: &EngineState,
) -> Result<RequestPrincipal, StatusCode> {
    if authorize_global_api_key(headers, &state.auth).is_ok() {
        return Ok(RequestPrincipal::GlobalApiKey);
    }

    let provided = request_api_key(headers).ok_or(StatusCode::UNAUTHORIZED)?;
    match state.platform.authenticate_api_key(provided) {
        Ok(Some(auth)) => {
            if auth.cluster_id.is_none() {
                return Err(StatusCode::FORBIDDEN);
            }
            if !is_valid_user_id(&auth.user_id) {
                tracing::warn!("API key returned malformed user_id; rejecting");
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
            Ok(RequestPrincipal::UserApiKey(auth))
        }
        Ok(None) => Err(StatusCode::UNAUTHORIZED),
        Err(err) => {
            tracing::warn!("API key auth lookup failed: {:?}", err);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Defense in depth: server-issued user_ids always start with `usr_` and contain
/// only `[A-Za-z0-9_]`. Reject anything that does not, so a bug or future
/// migration that lets an attacker-controlled value flow into SQL still fails.
pub fn is_valid_user_id(user_id: &str) -> bool {
    let len = user_id.len();
    if len < 8 || len > 128 {
        return false;
    }
    if !user_id.starts_with("usr_") {
        return false;
    }
    user_id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

// ── Rate limiting ──
//
// Per-key token bucket. Each API key (and the global API key) gets a bucket
// sized at 2x the steady-state RPS with a burst of 8x. The bucket is refilled
// lazily on every check. Buckets live in process memory; on restart, every
// key gets a fresh budget. This is intentional: a single-instance engine
// with in-process state can only enforce in-process quotas.

const RPS_PER_KEY: f64 = 20.0;
const BURST_PER_KEY: f64 = 80.0;

struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(now: Instant) -> Self {
        Self { tokens: BURST_PER_KEY, last_refill: now }
    }

    /// Returns true if the request fits in the current budget.
    fn try_consume(&mut self, now: Instant) -> bool {
        let elapsed = now.saturating_duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * RPS_PER_KEY).min(BURST_PER_KEY);
        self.last_refill = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

pub struct RateLimiter {
    buckets: Mutex<HashMap<String, TokenBucket>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self { buckets: Mutex::new(HashMap::new()) }
    }

    /// Returns true if the request is allowed. `key` is the API key string
    /// (or a synthetic "__global__" for the global key). Buckets for unknown
    /// keys are created on first use.
    pub fn allow(&self, key: &str) -> bool {
        let now = Instant::now();
        let mut guard = self.buckets.lock();
        let bucket = guard.entry(key.to_string()).or_insert_with(|| TokenBucket::new(now));
        bucket.try_consume(now)
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

pub fn check_rate_limit(state: &EngineState, headers: &HeaderMap) -> Result<(), StatusCode> {
    // Prefer the API key string for per-key buckets. Fall back to peer IP.
    let key = if let Some(provided) = request_api_key(headers) {
        provided.to_string()
    } else if let Some(addr) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()) {
        format!("ip:{}", addr.split(',').next().unwrap_or("").trim())
    } else {
        "__anonymous__".to_string()
    };
    if state.rate_limiter.allow(&key) {
        Ok(())
    } else {
        tracing::warn!(key = %&key[..key.len().min(20)], "rate limit exceeded");
        Err(StatusCode::TOO_MANY_REQUESTS)
    }
}

pub fn record_usage_for_principal(
    state: &EngineState,
    principal: &RequestPrincipal,
    endpoint: &str,
) {
    if let RequestPrincipal::UserApiKey(auth) = principal {
        if let Err(err) = state.platform_write_tx.try_send(PlatformWriteOp::Usage {
            user_id: auth.user_id.clone(),
            endpoint: endpoint.to_string(),
        }) {
            tracing::warn!(
                "failed to queue usage write for user={} key={} endpoint={}: {:?}",
                auth.user_id,
                auth.key_id,
                endpoint,
                err
            );
        }
    }
}

pub fn session_user_from_headers(
    state: &EngineState,
    headers: &HeaderMap,
) -> Result<PublicUser, StatusCode> {
    let token = request_bearer_token(headers).ok_or(StatusCode::UNAUTHORIZED)?;
    match state.platform.resolve_session(token) {
        Ok(Some(user)) => Ok(user),
        Ok(None) => Err(StatusCode::UNAUTHORIZED),
        Err(err) => {
            tracing::warn!("session auth lookup failed: {:?}", err);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_equal_strings() {
        assert!(constant_time_eq("hello", "hello"));
    }

    #[test]
    fn constant_time_eq_different_lengths() {
        assert!(!constant_time_eq("hello", "world!"));
    }

    #[test]
    fn constant_time_eq_same_length_different_content() {
        assert!(!constant_time_eq("hello", "hxllo"));
    }

    #[test]
    fn constant_time_eq_empty_strings() {
        assert!(constant_time_eq("", ""));
    }

    #[test]
    fn constant_time_eq_unicode() {
        assert!(constant_time_eq("héllo", "héllo"));
        assert!(!constant_time_eq("héllo", "hello"));
    }

    #[test]
    fn request_api_key_from_x_api_key_header() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("my-api-key"));
        assert_eq!(request_api_key(&headers), Some("my-api-key"));
    }

    #[test]
    fn request_api_key_from_authorization_bearer() {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer my-api-key"));
        assert_eq!(request_api_key(&headers), Some("my-api-key"));
    }

    #[test]
    fn request_api_key_missing_header() {
        let headers = HeaderMap::new();
        assert_eq!(request_api_key(&headers), None);
    }

    #[test]
    fn request_api_key_empty_value() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static(""));
        assert_eq!(request_api_key(&headers), None);
    }

    #[test]
    fn request_api_key_whitespace_only() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("   "));
        assert_eq!(request_api_key(&headers), None);
    }

    #[test]
    fn request_api_key_prefers_x_api_key_over_bearer() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("from-header"));
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer from-bearer"));
        assert_eq!(request_api_key(&headers), Some("from-header"));
    }

    #[test]
    fn request_bearer_token_valid() {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer my-token"));
        assert_eq!(request_bearer_token(&headers), Some("my-token"));
    }

    #[test]
    fn request_bearer_token_wrong_scheme() {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Basic my-token"));
        assert_eq!(request_bearer_token(&headers), None);
    }

    #[test]
    fn request_bearer_token_missing_token() {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer"));
        assert_eq!(request_bearer_token(&headers), None);
    }

    #[test]
    fn request_bearer_token_empty_token() {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer   "));
        assert_eq!(request_bearer_token(&headers), None);
    }

    #[test]
    fn request_bearer_token_missing_header() {
        let headers = HeaderMap::new();
        assert_eq!(request_bearer_token(&headers), None);
    }

    #[test]
    fn scope_entity_id_with_prefix() {
        assert_eq!(scope_entity_id("abc", Some("ns::")), "ns::abc");
    }

    #[test]
    fn scope_entity_id_without_prefix() {
        assert_eq!(scope_entity_id("abc", None), "abc");
    }

    #[test]
    fn scope_entity_id_already_prefixed() {
        assert_eq!(scope_entity_id("ns::abc", Some("ns::")), "ns::abc");
    }

    #[test]
    fn scope_entity_id_empty_entity_id() {
        assert_eq!(scope_entity_id("", Some("ns::")), "ns::");
    }

    #[test]
    fn principal_user_id_user_api_key() {
        let auth = ApiKeyAuth {
            user_id: "user-1".to_string(),
            key_id: "key-1".to_string(),
            cluster_id: None,
        };
        let principal = RequestPrincipal::UserApiKey(auth);
        assert_eq!(principal_user_id(&principal), Some("user-1"));
    }

    #[test]
    fn principal_user_id_global_api_key() {
        let principal = RequestPrincipal::GlobalApiKey;
        assert_eq!(principal_user_id(&principal), None);
    }

    #[test]
    fn authorize_global_api_key_matching_key() {
        let config = AuthConfig { api_key: Some(Arc::from("secret")) };
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("secret"));
        assert_eq!(authorize_global_api_key(&headers, &config), Ok(()));
    }

    #[test]
    fn authorize_global_api_key_wrong_key() {
        let config = AuthConfig { api_key: Some(Arc::from("secret")) };
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("wrong"));
        assert_eq!(authorize_global_api_key(&headers, &config), Err(StatusCode::UNAUTHORIZED));
    }

    #[test]
    fn authorize_global_api_key_no_auth_config() {
        let config = AuthConfig { api_key: None };
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("anything"));
        assert_eq!(authorize_global_api_key(&headers, &config), Ok(()));
    }

    #[test]
    fn cors_allow_origins_default() {
        std::env::remove_var("TEMPORAL_MEMORY_CORS_ALLOW_ORIGINS");
        std::env::remove_var("TELLODB_CORS_ALLOW_ORIGINS");
        let origins = cors_allow_origins();
        assert_eq!(origins.len(), 1);
        assert_eq!(origins[0], "https://tellodb.com");
    }

    #[test]
    fn cors_allow_origins_from_env() {
        std::env::remove_var("TEMPORAL_MEMORY_CORS_ALLOW_ORIGINS");
        std::env::set_var(
            "TELLODB_CORS_ALLOW_ORIGINS",
            "http://localhost:3000,http://example.com",
        );
        let origins = cors_allow_origins();
        std::env::remove_var("TELLODB_CORS_ALLOW_ORIGINS");
        assert_eq!(origins.len(), 2);
        assert_eq!(origins[0], "http://localhost:3000");
        assert_eq!(origins[1], "http://example.com");
    }

    #[test]
    fn cors_allow_origins_empty_falls_back_to_default() {
        std::env::remove_var("TEMPORAL_MEMORY_CORS_ALLOW_ORIGINS");
        std::env::set_var("TELLODB_CORS_ALLOW_ORIGINS", "");
        let origins = cors_allow_origins();
        std::env::remove_var("TELLODB_CORS_ALLOW_ORIGINS");
        assert_eq!(origins.len(), 1);
        assert_eq!(origins[0], "https://tellodb.com");
    }

    #[test]
    fn cors_allow_origins_trims_trailing_slashes() {
        std::env::remove_var("TEMPORAL_MEMORY_CORS_ALLOW_ORIGINS");
        std::env::set_var(
            "TELLODB_CORS_ALLOW_ORIGINS",
            "http://localhost:3000/,http://example.com/",
        );
        let origins = cors_allow_origins();
        std::env::remove_var("TELLODB_CORS_ALLOW_ORIGINS");
        assert_eq!(origins.len(), 2);
        assert_eq!(origins[0], "http://localhost:3000");
        assert_eq!(origins[1], "http://example.com");
    }

    #[test]
    fn constant_time_eq_case_sensitive() {
        assert!(!constant_time_eq("Secret", "secret"));
    }
}
