use axum::http::{
    header::{AUTHORIZATION, CONTENT_TYPE},
    HeaderMap, HeaderName, HeaderValue, Method, StatusCode,
};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::fmt::Write as _;
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
    pub fn from_config(config: &crate::config::Config) -> anyhow::Result<Self> {
        let api_key = match config.server.api_key.clone() {
            Some(key) => key,
            None if cfg!(debug_assertions) => {
                tracing::warn!(
                    "Using default test API key '{}'. Set TELLODB_API_KEY for production.",
                    DEFAULT_TEST_API_KEY
                );
                DEFAULT_TEST_API_KEY.to_string()
            }
            None => anyhow::bail!("TELLODB_API_KEY must be set to serve the HTTP API"),
        };
        Ok(Self { api_key: Some(Arc::<str>::from(api_key)) })
    }

    /// Server auth from `TELLODB_API_KEY`, with the legacy name accepted during migration. Debug
    /// builds fall back to the test key; release builds require a key.
    pub fn from_env() -> anyhow::Result<Self> {
        Self::from_config(&crate::config::Config::from_env()?)
    }

    /// For in-process use (embedded API, CLI, stdio MCP), where no request
    /// ever carries a key: a random key nobody knows.
    pub fn embedded() -> Self {
        let mut key = String::with_capacity(64);
        for _ in 0..4 {
            let _ = write!(key, "{:016x}", rand::random::<u64>());
        }
        Self { api_key: Some(Arc::<str>::from(key)) }
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

pub fn parse_cors_allow_origins(raw: Option<&str>) -> Vec<HeaderValue> {
    let configured = raw.map(str::trim).filter(|v| !v.is_empty()).unwrap_or("https://tellodb.com");

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

pub fn build_cors_layer(origins: &[String]) -> CorsLayer {
    let configured = (!origins.is_empty()).then(|| origins.join(","));
    CorsLayer::new()
        .allow_origin(parse_cors_allow_origins(configured.as_deref()))
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
    if !(8..=128).contains(&len) {
        return false;
    }
    if !user_id.starts_with("usr_") {
        return false;
    }
    user_id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

// ── Rate limiting ──
//
// Token buckets held in process memory. Every request is charged to its
// client address; requests carrying a user API key are additionally charged
// to that key. The global (operator) key is exempt: it runs benchmarks and
// admin jobs whose request rate would otherwise trip the limiter mid-run.
// Idle buckets are pruned so rotating made-up keys cannot grow memory without
// bound, and the per-address bucket still limits such a client.

const RPS_PER_KEY: f64 = 20.0;
const BURST_PER_KEY: f64 = 80.0;
const RPS_PER_ADDR: f64 = 50.0;
const BURST_PER_ADDR: f64 = 200.0;
const RPS_PER_AUTH: f64 = 0.1;
const BURST_PER_AUTH: f64 = 5.0;
/// Prune idle buckets once the map holds this many entries.
const PRUNE_THRESHOLD: usize = 4096;

struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
    rps: f64,
    burst: f64,
}

impl TokenBucket {
    fn new(now: Instant, rps: f64, burst: f64) -> Self {
        Self { tokens: burst, last_refill: now, rps, burst }
    }

    /// Returns true if the request fits in the current budget.
    fn try_consume(&mut self, now: Instant) -> bool {
        let elapsed = now.saturating_duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.rps).min(self.burst);
        self.last_refill = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// A bucket idle long enough to be full again carries no state.
    fn is_refilled(&self, now: Instant) -> bool {
        self.tokens + now.saturating_duration_since(self.last_refill).as_secs_f64() * self.rps
            >= self.burst
    }
}

pub struct RateLimiter {
    buckets: Mutex<HashMap<String, TokenBucket>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self { buckets: Mutex::new(HashMap::new()) }
    }

    /// Returns true if the request is allowed for `key` (per-key limits).
    pub fn allow(&self, key: &str) -> bool {
        self.allow_with(key, RPS_PER_KEY, BURST_PER_KEY)
    }

    fn allow_with(&self, key: &str, rps: f64, burst: f64) -> bool {
        let now = Instant::now();
        let mut guard = self.buckets.lock();
        if guard.len() >= PRUNE_THRESHOLD && !guard.contains_key(key) {
            // A bucket that has refilled is equivalent to a new one.
            guard.retain(|_, b| !b.is_refilled(now));
        }
        let bucket =
            guard.entry(key.to_string()).or_insert_with(|| TokenBucket::new(now, rps, burst));
        bucket.try_consume(now)
    }

    pub fn len(&self) -> usize {
        self.buckets.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

/// Client address for rate limiting. `x-forwarded-for` is client-controlled,
/// so it is only used behind a trusted proxy (`TELLODB_TRUST_PROXY=1`).
pub fn client_address(
    headers: &HeaderMap,
    peer: Option<std::net::SocketAddr>,
    trust_proxy: bool,
) -> String {
    if trust_proxy {
        if let Some(forwarded) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()) {
            if let Some(first) =
                forwarded.split(',').next().map(str::trim).filter(|s| !s.is_empty())
            {
                return first.to_string();
            }
        }
    }
    peer.map_or_else(|| "unknown".to_string(), |addr| addr.ip().to_string())
}

pub fn check_rate_limit(
    state: &EngineState,
    headers: &HeaderMap,
    peer: Option<std::net::SocketAddr>,
) -> Result<(), StatusCode> {
    let provided = request_api_key(headers);
    if authorize_global_api_key(headers, &state.auth).is_ok() && state.auth.is_required() {
        return Ok(());
    }
    let addr = client_address(headers, peer, state.config.server.trust_proxy);
    let allowed =
        state.rate_limiter.allow_with(&format!("addr:{addr}"), RPS_PER_ADDR, BURST_PER_ADDR)
            && match provided {
                Some(key) => state.rate_limiter.allow(&format!("key:{key}")),
                None => true,
            };
    if allowed {
        Ok(())
    } else {
        tracing::warn!(addr = %addr, "rate limit exceeded");
        Err(StatusCode::TOO_MANY_REQUESTS)
    }
}

pub fn check_auth_rate_limit(
    state: &EngineState,
    headers: &HeaderMap,
    peer: Option<std::net::SocketAddr>,
) -> Result<(), StatusCode> {
    check_auth_rate_limit_with_limiter(
        &state.rate_limiter,
        headers,
        peer,
        state.config.server.trust_proxy,
    )
}

fn check_auth_rate_limit_with_limiter(
    rate_limiter: &RateLimiter,
    headers: &HeaderMap,
    peer: Option<std::net::SocketAddr>,
    trust_proxy: bool,
) -> Result<(), StatusCode> {
    let addr = client_address(headers, peer, trust_proxy);
    if rate_limiter.allow_with(&format!("auth:{addr}"), RPS_PER_AUTH, BURST_PER_AUTH) {
        Ok(())
    } else {
        tracing::warn!(addr = %addr, "authentication rate limit exceeded");
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
    fn rate_limiter_prunes_idle_buckets() {
        let limiter = RateLimiter::new();
        for i in 0..PRUNE_THRESHOLD {
            limiter.allow(&format!("key:{i}"));
        }
        assert_eq!(limiter.len(), PRUNE_THRESHOLD);
        // Wait until the one-token-spent buckets have refilled, then a new key prunes them.
        std::thread::sleep(std::time::Duration::from_millis(100));
        limiter.allow("key:new");
        assert!(limiter.len() < PRUNE_THRESHOLD, "idle buckets should be pruned");
    }

    #[test]
    fn rate_limiter_limits_burst_per_key() {
        let limiter = RateLimiter::new();
        let allowed = (0..200).filter(|_| limiter.allow("key:k")).count();
        assert!((80..=82).contains(&allowed), "allowed {allowed}");
    }

    #[test]
    fn auth_rate_limit_rejects_sixth_rapid_attempt() {
        let rate_limiter = RateLimiter::new();
        let headers = HeaderMap::new();
        let peer = Some("127.0.0.1:8080".parse().unwrap());

        for _ in 0..5 {
            assert_eq!(
                check_auth_rate_limit_with_limiter(&rate_limiter, &headers, peer, false),
                Ok(())
            );
        }
        assert_eq!(
            check_auth_rate_limit_with_limiter(&rate_limiter, &headers, peer, false),
            Err(StatusCode::TOO_MANY_REQUESTS)
        );
    }

    #[test]
    fn forwarded_for_is_ignored_without_trusted_proxy() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", HeaderValue::from_static("1.2.3.4"));
        let peer: std::net::SocketAddr = "10.0.0.7:5555".parse().unwrap();
        assert_eq!(client_address(&headers, Some(peer), false), "10.0.0.7");
    }

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
        let origins = parse_cors_allow_origins(None);
        assert_eq!(origins.len(), 1);
        assert_eq!(origins[0], "https://tellodb.com");
    }

    #[test]
    fn cors_allow_origins_from_env() {
        let origins = parse_cors_allow_origins(Some("http://localhost:3000,http://example.com"));
        assert_eq!(origins.len(), 2);
        assert_eq!(origins[0], "http://localhost:3000");
        assert_eq!(origins[1], "http://example.com");
    }

    #[test]
    fn cors_allow_origins_empty_falls_back_to_default() {
        let origins = parse_cors_allow_origins(Some(""));
        assert_eq!(origins.len(), 1);
        assert_eq!(origins[0], "https://tellodb.com");
    }

    #[test]
    fn cors_allow_origins_trims_trailing_slashes() {
        let origins = parse_cors_allow_origins(Some("http://localhost:3000/,http://example.com/"));
        assert_eq!(origins.len(), 2);
        assert_eq!(origins[0], "http://localhost:3000");
        assert_eq!(origins[1], "http://example.com");
    }

    #[test]
    fn constant_time_eq_case_sensitive() {
        assert!(!constant_time_eq("Secret", "secret"));
    }
}
