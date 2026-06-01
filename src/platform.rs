use anyhow::{anyhow, Context, Result};
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::Argon2;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

const MIN_USERNAME_LENGTH: usize = 3;
const MIN_PASSWORD_LENGTH: usize = 8;
const USER_ID_RANDOM_LENGTH: usize = 20;
const TOKEN_RANDOM_LENGTH: usize = 48;
const API_KEY_PREFIX_LENGTH: usize = 12;
const MAX_STABLE_FACTS: usize = 64;
const MAX_ACTIVITY_FACTS: usize = 120;
const MIN_SENTENCE_LENGTH: usize = 10;
const MAX_STABLE_SENTENCE_LENGTH: usize = 180;
const MAX_ACTIVITY_SENTENCE_LENGTH: usize = 220;

#[derive(Clone)]
pub struct PlatformStore {
    pool: Pool<SqliteConnectionManager>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UserRecord {
    user_id: String,
    username: String,
    password_hash: String,
    created_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProfileActivityRecord {
    fact: String,
    source: String,
    timestamp_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PublicUser {
    pub user_id: String,
    pub username: String,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PublicApiKey {
    pub key_id: String,
    pub name: String,
    pub key_prefix: String,
    pub created_at_ms: u64,
    pub last_used_ms: Option<u64>,
    pub disabled: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct UsageStats {
    pub request_count: u64,
    pub ingest_count: u64,
    pub query_count: u64,
    pub temporal_query_count: u64,
    pub reset_count: u64,
    pub health_count: u64,
    pub version_count: u64,
    pub last_request_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UserProfile {
    pub user_id: String,
    pub stable_facts: Vec<String>,
    pub recent_activity: Vec<ProfileActivity>,
    pub last_updated_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfileActivity {
    pub fact: String,
    pub source: String,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ApiKeyAuth {
    pub user_id: String,
    pub key_id: String,
    /// Present when the key belongs to a fractional (shared) cluster.
    /// Handlers MUST use this as a mandatory entity_id prefix to enforce
    /// per-tenant data isolation on the shared Aletheia engine.
    pub cluster_id: Option<String>,
}

impl PlatformStore {
    pub fn new(path: &str) -> Result<Self> {
        let manager = SqliteConnectionManager::file(path);
        let pool =
            Pool::new(manager).context("failed to create platform database connection pool")?;

        let conn = pool
            .get()
            .context("failed to acquire database connection for schema initialization")?;
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                password_hash TEXT,
                created_at_ms INTEGER
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT,
                created_at_ms INTEGER,
                expires_at_ms INTEGER
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY,
                user_id TEXT,
                name TEXT,
                key_prefix TEXT,
                key_hash TEXT UNIQUE,
                created_at_ms INTEGER,
                last_used_ms INTEGER,
                disabled BOOLEAN,
                cluster_id TEXT
            );

            CREATE TABLE IF NOT EXISTS usage (
                user_id TEXT PRIMARY KEY,
                request_count INTEGER DEFAULT 0,
                ingest_count INTEGER DEFAULT 0,
                query_count INTEGER DEFAULT 0,
                temporal_query_count INTEGER DEFAULT 0,
                reset_count INTEGER DEFAULT 0,
                health_count INTEGER DEFAULT 0,
                version_count INTEGER DEFAULT 0,
                last_request_ms INTEGER
            );

            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                stable_facts TEXT,
                recent_activity TEXT,
                last_updated_ms INTEGER
            );
        ",
        )
        .context("failed to initialize platform database schema")?;

        Ok(Self { pool })
    }

    fn get_conn(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>> {
        self.pool.get().context("Failed to get platform connection")
    }

    pub fn create_user(&self, username: &str, password: &str) -> Result<PublicUser> {
        let username = username.trim().to_ascii_lowercase();
        if username.is_empty() || username.len() < MIN_USERNAME_LENGTH {
            return Err(anyhow!("username must be at least {} characters", MIN_USERNAME_LENGTH));
        }
        if password.len() < MIN_PASSWORD_LENGTH {
            return Err(anyhow!("password must be at least {} characters", MIN_PASSWORD_LENGTH));
        }

        let created_at_ms = now_ms()?;
        let user_id = format!("usr_{}", random_token(USER_ID_RANDOM_LENGTH));
        let password_hash = hash_password(password)?;

        let conn = self.get_conn()?;
        let res = conn.execute(
            "INSERT INTO users (user_id, username, password_hash, created_at_ms) VALUES (?1, ?2, ?3, ?4)",
            params![user_id, username, password_hash, created_at_ms],
        );

        match res {
            Ok(_) => Ok(PublicUser { user_id, username, created_at_ms }),
            Err(rusqlite::Error::SqliteFailure(e, Some(msg)))
                if e.code == rusqlite::ErrorCode::ConstraintViolation && msg.contains("UNIQUE") =>
            {
                Err(anyhow!("username already exists"))
            }
            Err(e) => Err(e).context("failed to insert user into database"),
        }
    }

    pub fn login(&self, username: &str, password: &str) -> Result<PublicUser> {
        let username = username.trim().to_ascii_lowercase();
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare("SELECT user_id, username, password_hash, created_at_ms FROM users WHERE username = ?1")
            .context("failed to prepare login statement")?;
        let user: UserRecord = stmt
            .query_row(params![username], |row| {
                Ok(UserRecord {
                    user_id: row.get(0)?,
                    username: row.get(1)?,
                    password_hash: row.get(2)?,
                    created_at_ms: row.get(3)?,
                })
            })
            .map_err(|_| anyhow!("invalid credentials"))?;

        if !verify_password(&user.password_hash, password) {
            return Err(anyhow!("invalid credentials"));
        }
        Ok(PublicUser {
            user_id: user.user_id,
            username: user.username,
            created_at_ms: user.created_at_ms,
        })
    }

    pub fn create_session(&self, user_id: &str, ttl_secs: u64) -> Result<String> {
        let now = now_ms()?;
        let token = format!("sess_{}", random_token(TOKEN_RANDOM_LENGTH));
        let expires_at_ms = now.saturating_add(ttl_secs.saturating_mul(1000));

        let conn = self.get_conn()?;
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at_ms, expires_at_ms) VALUES (?1, ?2, ?3, ?4)",
            params![token, user_id, now, expires_at_ms],
        ).context("failed to insert session into database")?;
        Ok(token)
    }

    pub fn resolve_session(&self, token: &str) -> Result<Option<PublicUser>> {
        let conn = self.get_conn()?;
        let mut stmt = conn
            .prepare(
                "
            SELECT u.user_id, u.username, u.created_at_ms, s.expires_at_ms 
            FROM sessions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.token = ?1
        ",
            )
            .context("failed to prepare session resolution statement")?;

        let res = stmt.query_row(params![token], |row| {
            let expires_at_ms: u64 = row.get(3)?;
            let now = now_ms().unwrap_or(u64::MAX);
            if now >= expires_at_ms {
                return Err(rusqlite::Error::QueryReturnedNoRows);
            }
            Ok(PublicUser {
                user_id: row.get(0)?,
                username: row.get(1)?,
                created_at_ms: row.get(2)?,
            })
        });

        match res {
            Ok(user) => Ok(Some(user)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn create_api_key(&self, user_id: &str, name: &str) -> Result<(PublicApiKey, String)> {
        self.create_api_key_for_cluster(user_id, name, None).context("failed to create API key")
    }

    pub fn create_api_key_for_cluster(
        &self,
        user_id: &str,
        name: &str,
        cluster_id: Option<&str>,
    ) -> Result<(PublicApiKey, String)> {
        let now = now_ms()?;
        let key_id = format!("key_{}", random_token(USER_ID_RANDOM_LENGTH));
        let raw_key = format!("ak_{}", random_token(TOKEN_RANDOM_LENGTH));
        let key_hash = sha256_hex(&raw_key);
        let key_prefix = raw_key.chars().take(API_KEY_PREFIX_LENGTH).collect::<String>();

        let conn = self.get_conn()?;
        conn.execute(
            "INSERT INTO api_keys (key_id, user_id, name, key_prefix, key_hash, created_at_ms, disabled, cluster_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![key_id, user_id, name.trim(), key_prefix, key_hash, now, false, cluster_id],
        ).context("failed to insert API key into database")?;

        Ok((
            PublicApiKey {
                key_id,
                name: name.trim().to_string(),
                key_prefix,
                created_at_ms: now,
                last_used_ms: None,
                disabled: false,
            },
            raw_key,
        ))
    }

    pub fn inject_api_key(
        &self,
        key_id: &str,
        user_id: &str,
        name: &str,
        raw_key: &str,
        cluster_id: Option<&str>,
    ) -> Result<()> {
        let now = now_ms()?;
        let key_hash = sha256_hex(raw_key);
        let key_prefix = raw_key.chars().take(API_KEY_PREFIX_LENGTH).collect::<String>();

        let conn = self.get_conn()?;
        conn.execute(
            "INSERT OR REPLACE INTO api_keys (key_id, user_id, name, key_prefix, key_hash, created_at_ms, disabled, cluster_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![key_id, user_id, name.trim(), key_prefix, key_hash, now, false, cluster_id],
        ).context("failed to inject API key into database")?;
        Ok(())
    }

    pub fn list_api_keys(&self, user_id: &str) -> Result<Vec<PublicApiKey>> {
        let conn = self.get_conn()?;
        let mut stmt = conn
            .prepare(
                "
            SELECT key_id, name, key_prefix, created_at_ms, last_used_ms, disabled 
            FROM api_keys 
            WHERE user_id = ?1 
            ORDER BY created_at_ms DESC
        ",
            )
            .context("failed to prepare list API keys statement")?;

        let rows = stmt
            .query_map(params![user_id], |row| {
                Ok(PublicApiKey {
                    key_id: row.get(0)?,
                    name: row.get(1)?,
                    key_prefix: row.get(2)?,
                    created_at_ms: row.get(3)?,
                    last_used_ms: row.get(4)?,
                    disabled: row.get(5)?,
                })
            })
            .context("failed to query API keys")?;

        let mut out = Vec::new();
        for row in rows {
            out.push(row.context("failed to read API key row")?);
        }
        Ok(out)
    }

    pub fn revoke_api_key(&self, user_id: &str, key_id: &str) -> Result<()> {
        let conn = self.get_conn()?;
        let count = conn
            .execute(
                "UPDATE api_keys SET disabled = 1 WHERE key_id = ?1 AND user_id = ?2",
                params![key_id, user_id],
            )
            .context("failed to execute API key revocation")?;
        if count == 0 {
            return Err(anyhow!("api key not found"));
        }
        Ok(())
    }

    pub fn admin_revoke_api_key(&self, key_id: &str) -> Result<()> {
        let conn = self.get_conn()?;
        let count = conn
            .execute("UPDATE api_keys SET disabled = 1 WHERE key_id = ?1", params![key_id])
            .context("failed to execute admin API key revocation")?;
        if count == 0 {
            return Err(anyhow!("api key not found"));
        }
        Ok(())
    }

    pub fn authenticate_api_key(&self, raw_key: &str) -> Result<Option<ApiKeyAuth>> {
        let key_hash = sha256_hex(raw_key);
        let conn = self.get_conn()?;
        let mut stmt = conn
            .prepare(
                "
            SELECT user_id, key_id, cluster_id FROM api_keys 
            WHERE key_hash = ?1 AND disabled = 0
        ",
            )
            .context("failed to prepare API key authentication statement")?;

        let res = stmt.query_row(params![key_hash], |row| {
            Ok(ApiKeyAuth { user_id: row.get(0)?, key_id: row.get(1)?, cluster_id: row.get(2)? })
        });

        match res {
            Ok(auth) => {
                self.touch_key(&auth.key_id)?;
                Ok(Some(auth))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn record_usage(&self, user_id: &str, endpoint: &str) -> Result<()> {
        self.record_usage_n(user_id, endpoint, 1)
    }

    pub fn record_usage_n(&self, user_id: &str, endpoint: &str, count: u64) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let conn = self.get_conn()?;

        conn.execute("INSERT OR IGNORE INTO usage (user_id) VALUES (?1)", params![user_id])
            .context("failed to initialize usage record")?;

        let now = now_ms()?;
        let mut query = format!(
            "UPDATE usage SET request_count = request_count + ?, last_request_ms = {}",
            now
        );
        match endpoint {
            "ingest" => query.push_str(", ingest_count = ingest_count + ?"),
            "query" => query.push_str(", query_count = query_count + ?"),
            "temporal_query" => query.push_str(", temporal_query_count = temporal_query_count + ?"),
            "reset" => query.push_str(", reset_count = reset_count + ?"),
            "health" => query.push_str(", health_count = health_count + ?"),
            "version" => query.push_str(", version_count = version_count + ?"),
            _ => {}
        }
        query.push_str(" WHERE user_id = ?");

        if matches!(
            endpoint,
            "ingest" | "query" | "temporal_query" | "reset" | "health" | "version"
        ) {
            conn.execute(&query, params![count, count, user_id])
                .context("failed to update endpoint-specific usage counters")?;
        } else {
            conn.execute("UPDATE usage SET request_count = request_count + ?, last_request_ms = ? WHERE user_id = ?", params![count, now, user_id])
                .context("failed to update generic usage counters")?;
        }

        Ok(())
    }

    pub fn usage_stats(&self, user_id: &str) -> Result<UsageStats> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare("
            SELECT request_count, ingest_count, query_count, temporal_query_count, reset_count, health_count, version_count, last_request_ms 
            FROM usage WHERE user_id = ?1
        ").context("failed to prepare usage stats statement")?;

        let res = stmt.query_row(params![user_id], |row| {
            Ok(UsageStats {
                request_count: row.get(0)?,
                ingest_count: row.get(1)?,
                query_count: row.get(2)?,
                temporal_query_count: row.get(3)?,
                reset_count: row.get(4)?,
                health_count: row.get(5)?,
                version_count: row.get(6)?,
                last_request_ms: row.get(7)?,
            })
        });

        match res {
            Ok(stats) => Ok(stats),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(UsageStats {
                request_count: 0,
                ingest_count: 0,
                query_count: 0,
                temporal_query_count: 0,
                reset_count: 0,
                health_count: 0,
                version_count: 0,
                last_request_ms: None,
            }),
            Err(e) => Err(e.into()),
        }
    }

    pub fn total_usage_stats(&self) -> Result<UsageStats> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare("
            SELECT SUM(request_count), SUM(ingest_count), SUM(query_count), SUM(temporal_query_count),
                   SUM(reset_count), SUM(health_count), SUM(version_count), MAX(last_request_ms)
            FROM usage
        ").context("failed to prepare total usage stats statement")?;

        let res = stmt.query_row([], |row| {
            Ok(UsageStats {
                request_count: row.get::<_, Option<i64>>(0)?.unwrap_or(0) as u64,
                ingest_count: row.get::<_, Option<i64>>(1)?.unwrap_or(0) as u64,
                query_count: row.get::<_, Option<i64>>(2)?.unwrap_or(0) as u64,
                temporal_query_count: row.get::<_, Option<i64>>(3)?.unwrap_or(0) as u64,
                reset_count: row.get::<_, Option<i64>>(4)?.unwrap_or(0) as u64,
                health_count: row.get::<_, Option<i64>>(5)?.unwrap_or(0) as u64,
                version_count: row.get::<_, Option<i64>>(6)?.unwrap_or(0) as u64,
                last_request_ms: row.get::<_, Option<i64>>(7)?.map(|v| v as u64),
            })
        });

        match res {
            Ok(stats) => Ok(stats),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(UsageStats {
                request_count: 0,
                ingest_count: 0,
                query_count: 0,
                temporal_query_count: 0,
                reset_count: 0,
                health_count: 0,
                version_count: 0,
                last_request_ms: None,
            }),
            Err(e) => Err(e.into()),
        }
    }

    pub fn user_profile(&self, user_id: &str) -> Result<UserProfile> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare("SELECT stable_facts, recent_activity, last_updated_ms FROM user_profiles WHERE user_id = ?1")
            .context("failed to prepare user profile query statement")?;

        let res = stmt.query_row(params![user_id], |row| {
            let stable_facts_raw: Option<String> = row.get(0)?;
            let recent_activity_raw: Option<String> = row.get(1)?;

            let stable_facts =
                stable_facts_raw.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            let recent_activity: Vec<ProfileActivityRecord> =
                recent_activity_raw.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();

            Ok(UserProfile {
                user_id: user_id.to_string(),
                stable_facts,
                recent_activity: recent_activity
                    .into_iter()
                    .map(|a| ProfileActivity {
                        fact: a.fact,
                        source: a.source,
                        timestamp_ms: a.timestamp_ms,
                    })
                    .collect(),
                last_updated_ms: row.get(2)?,
            })
        });

        match res {
            Ok(profile) => Ok(profile),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(UserProfile {
                user_id: user_id.to_string(),
                stable_facts: Vec::new(),
                recent_activity: Vec::new(),
                last_updated_ms: 0,
            }),
            Err(e) => Err(e.into()),
        }
    }

    pub fn update_profile_from_text(
        &self,
        user_id: &str,
        text: &str,
        timestamp_ms: u64,
        source: &str,
    ) -> Result<()> {
        let stable_candidates = extract_profile_stable_facts(text);
        let activity_candidates = extract_profile_activity_facts(text);
        if stable_candidates.is_empty() && activity_candidates.is_empty() {
            return Ok(());
        }

        let mut conn = self.get_conn()?;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .context("failed to begin profile update transaction")?;

        let mut current_profile = {
            let mut stmt = tx.prepare(
                "SELECT stable_facts, recent_activity, last_updated_ms FROM user_profiles WHERE user_id = ?1"
            ).context("failed to prepare profile read statement")?;
            let res = stmt.query_row(params![user_id], |row| {
                let stable_facts_raw: Option<String> = row.get(0)?;
                let recent_activity_raw: Option<String> = row.get(1)?;

                let stable_facts = stable_facts_raw
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
                let recent_activity: Vec<ProfileActivityRecord> = recent_activity_raw
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();

                Ok(UserProfile {
                    user_id: user_id.to_string(),
                    stable_facts,
                    recent_activity: recent_activity
                        .into_iter()
                        .map(|a| ProfileActivity {
                            fact: a.fact,
                            source: a.source,
                            timestamp_ms: a.timestamp_ms,
                        })
                        .collect(),
                    last_updated_ms: row.get(2)?,
                })
            });

            match res {
                Ok(profile) => profile,
                Err(rusqlite::Error::QueryReturnedNoRows) => UserProfile {
                    user_id: user_id.to_string(),
                    stable_facts: Vec::new(),
                    recent_activity: Vec::new(),
                    last_updated_ms: 0,
                },
                Err(e) => return Err(e.into()),
            }
        };

        for fact in stable_candidates {
            insert_unique_front(&mut current_profile.stable_facts, fact, MAX_STABLE_FACTS);
        }
        for fact in activity_candidates {
            let item = ProfileActivity { fact, source: source.to_string(), timestamp_ms };
            if let Some(idx) = current_profile
                .recent_activity
                .iter()
                .position(|existing| existing.fact == item.fact && existing.source == item.source)
            {
                current_profile.recent_activity.remove(idx);
            }
            current_profile.recent_activity.insert(0, item);
            if current_profile.recent_activity.len() > MAX_ACTIVITY_FACTS {
                current_profile.recent_activity.truncate(MAX_ACTIVITY_FACTS);
            }
        }
        current_profile.last_updated_ms = timestamp_ms.max(current_profile.last_updated_ms);

        let stable_facts_json = serde_json::to_string(&current_profile.stable_facts)
            .context("failed to serialize stable facts")?;
        let recent_activity_json = serde_json::to_string(&current_profile.recent_activity)
            .context("failed to serialize recent activity")?;

        tx.execute(
            "INSERT OR REPLACE INTO user_profiles (user_id, stable_facts, recent_activity, last_updated_ms) VALUES (?1, ?2, ?3, ?4)",
            params![user_id, stable_facts_json, recent_activity_json, current_profile.last_updated_ms],
        ).context("failed to upsert user profile")?;

        tx.commit().context("failed to commit profile update transaction")?;
        Ok(())
    }

    fn touch_key(&self, key_id: &str) -> Result<()> {
        let conn = self.get_conn()?;
        conn.execute(
            "UPDATE api_keys SET last_used_ms = ?1 WHERE key_id = ?2",
            params![now_ms()?, key_id],
        )
        .context("failed to update API key last_used timestamp")?;
        Ok(())
    }
}

fn now_ms() -> Result<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH).context("clock before epoch")?.as_millis()
        as u64)
}

fn hash_password(password: &str) -> Result<String> {
    let salt_bytes = random_bytes_16();
    let salt = SaltString::encode_b64(&salt_bytes)
        .map_err(|e| anyhow!("failed to encode password salt: {}", e))?;
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| anyhow!("failed to hash password: {}", e))
}

fn verify_password(password_hash: &str, password: &str) -> bool {
    let Ok(parsed_hash) = PasswordHash::new(password_hash) else {
        return false;
    };
    Argon2::default().verify_password(password.as_bytes(), &parsed_hash).is_ok()
}

fn random_token(length: usize) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut out = String::with_capacity(length);
    for _ in 0..length {
        let idx = (rand::random::<u8>() as usize) % CHARS.len();
        out.push(CHARS[idx] as char);
    }
    out
}

fn random_bytes_16() -> [u8; 16] {
    rand::random::<[u8; 16]>()
}

fn sha256_hex(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

fn normalize_fact(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut first = true;
    for word in text.split_whitespace() {
        if !first {
            result.push(' ');
        }
        result.push_str(word);
        first = false;
    }
    result
}

fn split_sentences(text: &str) -> Vec<String> {
    let parts: Vec<&str> = text.split(['.', '!', '?', '\n']).collect();
    let mut result = Vec::with_capacity(parts.len());
    for s in parts {
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            result.push(trimmed.to_string());
        }
    }
    result
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn is_first_person_sentence(lower: &str) -> bool {
    contains_any(
        lower,
        &[
            " i ", " i'm ", " i’m ", " i am ", " i'd ", " i’d ", " i have ", " i’ve ", " i've ",
            " my ",
        ],
    )
}

fn extract_profile_stable_facts(text: &str) -> Vec<String> {
    let stable_markers = [
        "i am",
        "i'm",
        "i’m",
        "i have",
        "i've",
        "i’ve",
        "i use",
        "i prefer",
        "i like",
        "i love",
        "i work",
        "i live",
        "i own",
        "usually",
        "typically",
        "currently",
    ];
    let sentences = split_sentences(text);
    let mut out = Vec::with_capacity(sentences.len());
    for sentence in sentences {
        let normalized = normalize_fact(sentence.as_str());
        if normalized.len() < MIN_SENTENCE_LENGTH || normalized.len() > MAX_STABLE_SENTENCE_LENGTH {
            continue;
        }
        let lower = format!(" {} ", normalized.to_ascii_lowercase());
        if is_first_person_sentence(lower.as_str()) && contains_any(lower.as_str(), &stable_markers)
        {
            insert_unique_front(&mut out, normalized, MAX_STABLE_FACTS);
        }
    }
    out
}

fn extract_profile_activity_facts(text: &str) -> Vec<String> {
    let activity_markers = [
        "today",
        "yesterday",
        "last ",
        "this ",
        "in january",
        "in february",
        "in march",
        "in april",
        "in may",
        "in june",
        "in july",
        "in august",
        "in september",
        "in october",
        "in november",
        "in december",
        "i went",
        "i bought",
        "i attended",
        "i started",
        "i finished",
        "i did",
    ];
    let sentences = split_sentences(text);
    let mut out = Vec::with_capacity(sentences.len());
    for sentence in sentences {
        let normalized = normalize_fact(sentence.as_str());
        if normalized.len() < MIN_SENTENCE_LENGTH || normalized.len() > MAX_ACTIVITY_SENTENCE_LENGTH
        {
            continue;
        }
        let lower = format!(" {} ", normalized.to_ascii_lowercase());
        if is_first_person_sentence(lower.as_str())
            && contains_any(lower.as_str(), &activity_markers)
        {
            insert_unique_front(&mut out, normalized, MAX_ACTIVITY_FACTS);
        }
    }
    out
}

fn insert_unique_front(items: &mut Vec<String>, value: String, max_len: usize) {
    if let Some(idx) = items.iter().position(|existing| existing == &value) {
        items.remove(idx);
    }
    items.insert(0, value);
    if items.len() > max_len {
        items.truncate(max_len);
    }
}

#[cfg(test)]
fn insert_activity_unique_front(
    items: &mut Vec<ProfileActivityRecord>,
    value: ProfileActivityRecord,
    max_len: usize,
) {
    if let Some(idx) = items
        .iter()
        .position(|existing| existing.fact == value.fact && existing.source == value.source)
    {
        items.remove(idx);
    }
    items.insert(0, value);
    if items.len() > max_len {
        items.truncate(max_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn make_store() -> (PlatformStore, NamedTempFile) {
        let temp = NamedTempFile::new().unwrap();
        let store = PlatformStore::new(temp.path().to_str().unwrap()).unwrap();
        (store, temp)
    }

    fn create_test_user(store: &PlatformStore, username: &str, password: &str) -> PublicUser {
        store.create_user(username, password).unwrap()
    }

    // ── User lifecycle ──────────────────────────────────────────────

    #[test]
    fn create_user_success() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "alice", "password123");
        assert!(user.user_id.starts_with("usr_"));
        assert_eq!(user.username, "alice");
        assert!(user.created_at_ms > 0);
    }

    #[test]
    fn create_user_duplicate_username_errors() {
        let (store, _tmp) = make_store();
        create_test_user(&store, "alice", "password123");
        let err = store.create_user("alice", "password123").unwrap_err();
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn create_user_duplicate_username_case_insensitive_errors() {
        let (store, _tmp) = make_store();
        create_test_user(&store, "Alice", "password123");
        let err = store.create_user("alice", "password123").unwrap_err();
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn login_with_correct_credentials() {
        let (store, _tmp) = make_store();
        create_test_user(&store, "alice", "password123");
        let user = store.login("alice", "password123").unwrap();
        assert_eq!(user.username, "alice");
    }

    #[test]
    fn login_with_incorrect_password_errors() {
        let (store, _tmp) = make_store();
        create_test_user(&store, "alice", "password123");
        let err = store.login("alice", "wrongpass").unwrap_err();
        assert!(err.to_string().contains("invalid credentials"));
    }

    #[test]
    fn login_with_nonexistent_user_errors() {
        let (store, _tmp) = make_store();
        create_test_user(&store, "dummy", "password123");
        let err = store.login("nobody", "password123").unwrap_err();
        assert!(err.to_string().contains("invalid credentials"));
    }

    #[test]
    fn login_case_insensitive_username() {
        let (store, _tmp) = make_store();
        create_test_user(&store, "Alice", "password123");
        let user = store.login("ALICE", "password123").unwrap();
        assert_eq!(user.username, "alice");
    }

    #[test]
    fn create_user_short_username_errors() {
        let (store, _tmp) = make_store();
        let err = store.create_user("ab", "password123").unwrap_err();
        assert!(err.to_string().contains("at least 3 characters"));
    }

    #[test]
    fn create_user_short_password_errors() {
        let (store, _tmp) = make_store();
        let err = store.create_user("alice", "short").unwrap_err();
        assert!(err.to_string().contains("at least 8 characters"));
    }

    #[test]
    fn create_user_empty_username_errors() {
        let (store, _tmp) = make_store();
        let err = store.create_user("", "password123").unwrap_err();
        assert!(err.to_string().contains("at least 3 characters"));
    }

    // ── Session management ──────────────────────────────────────────

    #[test]
    fn create_and_resolve_session() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "bob", "password123");
        let token = store.create_session(&user.user_id, 3600).unwrap();
        assert!(token.starts_with("sess_"));
        assert_eq!(token.len(), 5 + 48);
        let resolved = store.resolve_session(&token).unwrap().unwrap();
        assert_eq!(resolved.user_id, user.user_id);
    }

    #[test]
    fn resolve_session_expired_token_returns_none() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "bob", "password123");
        let token = store.create_session(&user.user_id, 0).unwrap();
        let resolved = store.resolve_session(&token).unwrap();
        assert!(resolved.is_none());
    }

    #[test]
    fn resolve_session_invalid_token_returns_none() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "bob", "password123");
        store.create_session(&user.user_id, 3600).unwrap();
        let resolved = store.resolve_session("sess_nonexistent").unwrap();
        assert!(resolved.is_none());
    }

    #[test]
    fn resolve_session_empty_token_returns_none() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "bob", "password123");
        store.create_session(&user.user_id, 3600).unwrap();
        let resolved = store.resolve_session("").unwrap();
        assert!(resolved.is_none());
    }

    // ── API key management ──────────────────────────────────────────

    #[test]
    fn create_and_authenticate_api_key() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "carol", "password123");
        let (public, raw) = store.create_api_key(&user.user_id, "test-key").unwrap();
        assert!(public.key_id.starts_with("key_"));
        assert_eq!(public.name, "test-key");
        assert!(!public.disabled);
        assert!(raw.starts_with("ak_"));
        let auth = store.authenticate_api_key(&raw).unwrap().unwrap();
        assert_eq!(auth.user_id, user.user_id);
        assert_eq!(auth.key_id, public.key_id);
        assert!(auth.cluster_id.is_none());
    }

    #[test]
    fn inject_and_authenticate_api_key() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "carol", "password123");
        store
            .inject_api_key(
                "key_injected",
                &user.user_id,
                "injected-key",
                "ak_injected_raw_token_value_xyz",
                None,
            )
            .unwrap();
        let auth = store.authenticate_api_key("ak_injected_raw_token_value_xyz").unwrap().unwrap();
        assert_eq!(auth.user_id, user.user_id);
        assert_eq!(auth.key_id, "key_injected");
    }

    #[test]
    fn authenticate_api_key_invalid_key_returns_none() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "carol", "password123");
        store.create_api_key(&user.user_id, "dummy").unwrap();
        let result = store.authenticate_api_key("ak_invalid_key_here").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn list_api_keys_returns_user_keys() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "carol", "password123");
        store.create_api_key(&user.user_id, "key-a").unwrap();
        store.create_api_key(&user.user_id, "key-b").unwrap();
        let keys = store.list_api_keys(&user.user_id).unwrap();
        assert_eq!(keys.len(), 2);
        let names: Vec<&str> = keys.iter().map(|k| k.name.as_str()).collect();
        assert!(names.contains(&"key-a"));
        assert!(names.contains(&"key-b"));
    }

    #[test]
    fn list_api_keys_other_user_not_visible() {
        let (store, _tmp) = make_store();
        let user_a = create_test_user(&store, "alice", "password123");
        let user_b = create_test_user(&store, "bob", "password123");
        store.create_api_key(&user_a.user_id, "alice-key").unwrap();
        let keys_b = store.list_api_keys(&user_b.user_id).unwrap();
        assert!(keys_b.is_empty());
    }

    #[test]
    fn revoke_api_key_disables_key() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "carol", "password123");
        let (public, raw) = store.create_api_key(&user.user_id, "test-key").unwrap();
        store.revoke_api_key(&user.user_id, &public.key_id).unwrap();
        let result = store.authenticate_api_key(&raw).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn revoke_api_key_wrong_user_errors() {
        let (store, _tmp) = make_store();
        let user_a = create_test_user(&store, "alice", "password123");
        let user_b = create_test_user(&store, "bob", "password123");
        let (public, _raw) = store.create_api_key(&user_a.user_id, "key").unwrap();
        let err = store.revoke_api_key(&user_b.user_id, &public.key_id).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn revoke_api_key_nonexistent_errors() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "alice", "password123");
        let err = store.revoke_api_key(&user.user_id, "key_nonexistent").unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn list_api_keys_after_revoke_still_shows_disabled() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "carol", "password123");
        let (public, _raw) = store.create_api_key(&user.user_id, "test-key").unwrap();
        store.revoke_api_key(&user.user_id, &public.key_id).unwrap();
        let keys = store.list_api_keys(&user.user_id).unwrap();
        assert_eq!(keys.len(), 1);
        assert!(keys[0].disabled);
    }

    #[test]
    fn authenticate_api_key_updates_last_used() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "carol", "password123");
        let (_public, raw) = store.create_api_key(&user.user_id, "key").unwrap();
        store.authenticate_api_key(&raw).unwrap().unwrap();
        let auth = store.authenticate_api_key(&raw).unwrap().unwrap();
        assert_eq!(auth.user_id, user.user_id);
    }

    // ── Cluster-scoped keys ─────────────────────────────────────────

    #[test]
    fn create_api_key_for_cluster_sets_cluster_id() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "dave", "password123");
        let (public, raw) = store
            .create_api_key_for_cluster(&user.user_id, "cluster-key", Some("tenant-42"))
            .unwrap();
        assert!(public.key_id.starts_with("key_"));
        assert!(raw.starts_with("ak_"));
        let auth = store.authenticate_api_key(&raw).unwrap().unwrap();
        assert_eq!(auth.cluster_id.as_deref(), Some("tenant-42"));
    }

    #[test]
    fn create_api_key_for_cluster_without_cluster_is_none() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "dave", "password123");
        let (_public, raw) =
            store.create_api_key_for_cluster(&user.user_id, "regular-key", None).unwrap();
        let auth = store.authenticate_api_key(&raw).unwrap().unwrap();
        assert!(auth.cluster_id.is_none());
    }

    // ── Usage tracking ──────────────────────────────────────────────

    #[test]
    fn record_usage_increments_request_count() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "eve", "password123");
        store.record_usage(&user.user_id, "ingest").unwrap();
        let stats = store.usage_stats(&user.user_id).unwrap();
        assert_eq!(stats.request_count, 1);
        assert_eq!(stats.ingest_count, 1);
    }

    #[test]
    fn record_usage_n_with_multiple_counts() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "eve", "password123");
        store.record_usage_n(&user.user_id, "query", 5).unwrap();
        let stats = store.usage_stats(&user.user_id).unwrap();
        assert_eq!(stats.request_count, 5);
        assert_eq!(stats.query_count, 5);
    }

    #[test]
    fn record_usage_different_endpoints() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "faythe", "password123");
        store.record_usage(&user.user_id, "ingest").unwrap();
        store.record_usage(&user.user_id, "query").unwrap();
        store.record_usage(&user.user_id, "temporal_query").unwrap();
        store.record_usage(&user.user_id, "reset").unwrap();
        store.record_usage(&user.user_id, "health").unwrap();
        store.record_usage(&user.user_id, "version").unwrap();
        let stats = store.usage_stats(&user.user_id).unwrap();
        assert_eq!(stats.request_count, 6);
        assert_eq!(stats.ingest_count, 1);
        assert_eq!(stats.query_count, 1);
        assert_eq!(stats.temporal_query_count, 1);
        assert_eq!(stats.reset_count, 1);
        assert_eq!(stats.health_count, 1);
        assert_eq!(stats.version_count, 1);
        assert!(stats.last_request_ms.is_some());
    }

    #[test]
    fn record_usage_unknown_endpoint_only_increments_request_count() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "grace", "password123");
        store.record_usage(&user.user_id, "unknown_endpoint").unwrap();
        let stats = store.usage_stats(&user.user_id).unwrap();
        assert_eq!(stats.request_count, 1);
        assert_eq!(stats.ingest_count, 0);
        assert_eq!(stats.query_count, 0);
        assert_eq!(stats.temporal_query_count, 0);
        assert_eq!(stats.reset_count, 0);
        assert_eq!(stats.health_count, 0);
        assert_eq!(stats.version_count, 0);
    }

    #[test]
    fn usage_stats_zero_for_new_user() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "heidi", "password123");
        let _other = create_test_user(&store, "other", "password123");
        store.record_usage(&_other.user_id, "ingest").unwrap();
        let stats = store.usage_stats(&user.user_id).unwrap();
        assert_eq!(stats.request_count, 0);
        assert_eq!(stats.ingest_count, 0);
        assert_eq!(stats.last_request_ms, None);
    }

    #[test]
    fn record_usage_n_zero_does_nothing() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "ivan", "password123");
        let _other = create_test_user(&store, "other", "password123");
        store.record_usage(&_other.user_id, "ingest").unwrap();
        store.record_usage_n(&user.user_id, "query", 0).unwrap();
        let stats = store.usage_stats(&user.user_id).unwrap();
        assert_eq!(stats.request_count, 0);
    }

    #[test]
    fn usage_stats_are_separate_per_user() {
        let (store, _tmp) = make_store();
        let user_a = create_test_user(&store, "alice", "password123");
        let user_b = create_test_user(&store, "bob", "password123");
        store.record_usage(&user_a.user_id, "ingest").unwrap();
        store.record_usage_n(&user_b.user_id, "query", 3).unwrap();
        let stats_a = store.usage_stats(&user_a.user_id).unwrap();
        let stats_b = store.usage_stats(&user_b.user_id).unwrap();
        assert_eq!(stats_a.request_count, 1);
        assert_eq!(stats_a.ingest_count, 1);
        assert_eq!(stats_b.request_count, 3);
        assert_eq!(stats_b.query_count, 3);
    }

    // ── Profile extraction ──────────────────────────────────────────

    #[test]
    fn extract_profile_stable_facts_finds_first_person_stable() {
        let text = "I am a software engineer. I live in New York. I like Rust programming.";
        let facts = extract_profile_stable_facts(text);
        assert!(!facts.is_empty());
        assert!(facts.iter().any(|f| f.to_ascii_lowercase().contains("software engineer")));
    }

    #[test]
    fn extract_profile_stable_facts_ignores_short_sentences() {
        let text = "I am hi.";
        let facts = extract_profile_stable_facts(text);
        assert!(facts.is_empty());
    }

    #[test]
    fn extract_profile_stable_facts_ignores_long_sentences() {
        let text = format!("I am {} years old.", "very ".repeat(50));
        let facts = extract_profile_stable_facts(&text);
        assert!(facts.is_empty());
    }

    #[test]
    fn extract_profile_stable_facts_no_first_person_returns_empty() {
        let text = "The sky is blue. Usually it rains in April.";
        let facts = extract_profile_stable_facts(text);
        assert!(facts.is_empty());
    }

    #[test]
    fn extract_profile_activity_facts_finds_activity() {
        let text = "I went to the store yesterday. I bought a new laptop.";
        let facts = extract_profile_activity_facts(text);
        assert!(!facts.is_empty());
        assert!(facts.iter().any(|f| f.to_ascii_lowercase().contains("yesterday")));
    }

    #[test]
    fn extract_profile_activity_facts_no_first_person_returns_empty() {
        let text = "Yesterday was a good day.";
        let facts = extract_profile_activity_facts(text);
        assert!(facts.is_empty());
    }

    #[test]
    fn extract_profile_activity_facts_ignores_out_of_range_length() {
        let text = "I went.";
        let facts = extract_profile_activity_facts(text);
        assert!(facts.is_empty());
    }

    #[test]
    fn update_profile_from_text_stores_stable_and_activity_facts() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "mallory", "password123");
        let text = "I am a chef. I bought a new knife today.";
        store.update_profile_from_text(&user.user_id, text, 1000, "chat").unwrap();
        let profile = store.user_profile(&user.user_id).unwrap();
        assert!(!profile.stable_facts.is_empty());
        assert!(!profile.recent_activity.is_empty());
        assert_eq!(profile.last_updated_ms, 1000);
    }

    #[test]
    fn update_profile_from_text_no_candidates_does_nothing() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "mallory", "password123");
        let _other = create_test_user(&store, "other", "password123");
        store.update_profile_from_text(&_other.user_id, "I am a tester.", 100, "chat").unwrap();
        store.update_profile_from_text(&user.user_id, "The sky is blue.", 1000, "chat").unwrap();
        let profile = store.user_profile(&user.user_id).unwrap();
        assert!(profile.stable_facts.is_empty());
        assert_eq!(profile.last_updated_ms, 0);
    }

    #[test]
    fn user_profile_returns_empty_for_new_user() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "nancy", "password123");
        let _other = create_test_user(&store, "other", "password123");
        store.update_profile_from_text(&_other.user_id, "I am a tester.", 100, "chat").unwrap();
        let profile = store.user_profile(&user.user_id).unwrap();
        assert!(profile.stable_facts.is_empty());
        assert!(profile.recent_activity.is_empty());
        assert_eq!(profile.last_updated_ms, 0);
    }

    // ── Profile helpers ─────────────────────────────────────────────

    #[test]
    fn insert_unique_front_removes_duplicates() {
        let mut items = vec!["a".to_string(), "b".to_string()];
        insert_unique_front(&mut items, "a".to_string(), 10);
        assert_eq!(items, vec!["a", "b"]);
    }

    #[test]
    fn insert_unique_front_adds_new_items_at_front() {
        let mut items = vec!["b".to_string()];
        insert_unique_front(&mut items, "a".to_string(), 10);
        assert_eq!(items, vec!["a", "b"]);
    }

    #[test]
    fn insert_unique_front_truncates_at_max_len() {
        let mut items: Vec<String> = (0..5).map(|i| format!("item-{}", i)).collect();
        insert_unique_front(&mut items, "new".to_string(), 3);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], "new");
    }

    #[test]
    fn insert_activity_unique_front_removes_duplicate_fact_and_source() {
        let mut items = vec![ProfileActivityRecord {
            fact: "hello".to_string(),
            source: "chat".to_string(),
            timestamp_ms: 100,
        }];
        let dup = ProfileActivityRecord {
            fact: "hello".to_string(),
            source: "chat".to_string(),
            timestamp_ms: 200,
        };
        insert_activity_unique_front(&mut items, dup, 10);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].timestamp_ms, 200);
    }

    #[test]
    fn insert_activity_unique_front_different_source_not_duplicate() {
        let mut items = vec![ProfileActivityRecord {
            fact: "hello".to_string(),
            source: "chat".to_string(),
            timestamp_ms: 100,
        }];
        let different = ProfileActivityRecord {
            fact: "hello".to_string(),
            source: "email".to_string(),
            timestamp_ms: 200,
        };
        insert_activity_unique_front(&mut items, different, 10);
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn insert_activity_unique_front_truncates_at_max_len() {
        let mut items: Vec<ProfileActivityRecord> = (0..5)
            .map(|i| ProfileActivityRecord {
                fact: format!("fact-{}", i),
                source: "test".to_string(),
                timestamp_ms: i as u64,
            })
            .collect();
        let new = ProfileActivityRecord {
            fact: "new".to_string(),
            source: "test".to_string(),
            timestamp_ms: 999,
        };
        insert_activity_unique_front(&mut items, new, 3);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].fact, "new");
    }

    #[test]
    fn is_first_person_sentence_detects_variants() {
        assert!(is_first_person_sentence(" i am here "));
        assert!(is_first_person_sentence(" i'm here "));
        assert!(is_first_person_sentence(" i have a car "));
        assert!(is_first_person_sentence(" my name is bob "));
        assert!(!is_first_person_sentence(" the cat is here "));
        assert!(!is_first_person_sentence(" imagination runs wild "));
    }

    #[test]
    fn contains_any_returns_true_when_needle_found() {
        assert!(contains_any("hello world", &["world", "foo"]));
        assert!(!contains_any("hello world", &["foo", "bar"]));
    }

    #[test]
    fn split_sentences_splits_on_punctuation() {
        let sentences = split_sentences("Hello world. How are you? I'm fine!");
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world");
    }

    #[test]
    fn split_sentences_splits_on_newline() {
        let sentences = split_sentences("Line one.\\nLine two.");
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn split_sentences_skips_empty() {
        let sentences = split_sentences("Hello...world!");
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn normalize_fact_collapses_whitespace() {
        let result = normalize_fact("  hello   world  ");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn normalize_fact_empty_string() {
        let result = normalize_fact("");
        assert_eq!(result, "");
    }

    // ── Helpers ─────────────────────────────────────────────────────

    #[test]
    fn sha256_hex_consistent_output() {
        let a = sha256_hex("hello");
        let b = sha256_hex("hello");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn sha256_hex_different_inputs() {
        let a = sha256_hex("hello");
        let b = sha256_hex("world");
        assert_ne!(a, b);
    }

    #[test]
    fn random_token_correct_length() {
        let token = random_token(48);
        assert_eq!(token.len(), 48);
        let short = random_token(8);
        assert_eq!(short.len(), 8);
    }

    #[test]
    fn random_token_alphanumeric() {
        let token = random_token(100);
        assert!(token.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn random_bytes_16_produces_16_bytes() {
        let bytes = random_bytes_16();
        assert_eq!(bytes.len(), 16);
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn create_user_username_with_special_characters() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "user-name_123", "password123");
        assert_eq!(user.username, "user-name_123");
    }

    #[test]
    fn create_user_username_with_leading_trailing_spaces() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "  spaced_user  ", "password123");
        assert_eq!(user.username, "spaced_user");
    }

    #[test]
    fn create_user_password_with_special_characters() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "special_pw", "p@ssw0rd!🔥#$%");
        assert_eq!(user.username, "special_pw");
    }

    #[test]
    fn api_key_name_with_spaces() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "oscar", "password123");
        let (public, _raw) = store.create_api_key(&user.user_id, "  my key  ").unwrap();
        assert_eq!(public.name, "my key");
    }

    #[test]
    fn multiple_sessions_for_same_user() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "peggy", "password123");
        let t1 = store.create_session(&user.user_id, 3600).unwrap();
        let t2 = store.create_session(&user.user_id, 3600).unwrap();
        assert_ne!(t1, t2);
        assert!(store.resolve_session(&t1).unwrap().is_some());
        assert!(store.resolve_session(&t2).unwrap().is_some());
    }

    #[test]
    fn concurrent_create_user_unique_ids() {
        let (store, _tmp) = make_store();
        let a = create_test_user(&store, "user_a", "password123");
        let b = create_test_user(&store, "user_b", "password123");
        assert_ne!(a.user_id, b.user_id);
    }

    #[test]
    fn api_key_prefix_is_first_12_chars_of_raw() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "wendy", "password123");
        let (public, raw) = store.create_api_key(&user.user_id, "key").unwrap();
        let expected_prefix: String = raw.chars().take(12).collect();
        assert_eq!(public.key_prefix, expected_prefix);
    }

    #[test]
    fn update_profile_from_text_updates_timestamp() {
        let (store, _tmp) = make_store();
        let user = create_test_user(&store, "xavier", "password123");
        let text = "I am a designer. I finished a project today.";
        store.update_profile_from_text(&user.user_id, text, 500, "chat").unwrap();
        store.update_profile_from_text(&user.user_id, text, 1000, "chat").unwrap();
        let profile = store.user_profile(&user.user_id).unwrap();
        assert_eq!(profile.last_updated_ms, 1000);
    }
}
