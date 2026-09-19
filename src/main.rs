use std::env;
use tokio::net::TcpListener;
use tokio::signal;
use tokio::time::{interval, Duration};
use tracing::{error, info};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use tellodb::db::{Engine, Memory, Query};
use tellodb::{api, init_tracing_with_default_level, runtime_paths::RuntimePaths};

const USAGE: &str = "\
tellodb — temporal memory engine for AI agents

USAGE:
    tellodb [--data-dir DIR] [COMMAND]

COMMANDS:
    serve                 HTTP API server (default)
    mcp [--entity ID]     Model Context Protocol server over stdio
    doctor                Report models, device, settings and tenant health
    ingest --entity ID [--session ID] [FILE]
                          Store memories: JSON lines ({\"text\", \"role\", \"session_id\",
                          \"turn_index\", \"timestamp_ms\", \"kind\"}) or plain text lines,
                          from FILE or stdin
    query --entity ID [--limit N] [--as-of MS] TEXT
                          Search memories; prints JSON
    help                  Show this message

Configuration is read from TELLODB_* environment variables (see README).
";

struct Cli {
    command: String,
    options: std::collections::HashMap<String, String>,
    positional: Vec<String>,
}

fn parse_cli() -> anyhow::Result<Cli> {
    let mut args = env::args().skip(1).peekable();
    let mut command = None;
    let mut options = std::collections::HashMap::new();
    let mut positional = Vec::new();
    while let Some(arg) = args.next() {
        if let Some(flag) = arg.strip_prefix("--") {
            let (name, value) = match flag.split_once('=') {
                Some((name, value)) => (name.to_string(), value.to_string()),
                None if flag == "help" => ("help".to_string(), String::new()),
                None => {
                    let value =
                        args.next().ok_or_else(|| anyhow::anyhow!("--{flag} needs a value"))?;
                    (flag.to_string(), value)
                }
            };
            options.insert(name, value);
        } else if command.is_none() {
            command = Some(arg);
        } else {
            positional.push(arg);
        }
    }
    if options.contains_key("help") {
        command = Some("help".to_string());
    }
    Ok(Cli { command: command.unwrap_or_else(|| "serve".to_string()), options, positional })
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let cli = parse_cli()?;
    if cli.command == "help" || cli.command == "-h" {
        print!("{USAGE}");
        return Ok(());
    }
    if let Some(dir) = cli.options.get("data-dir") {
        env::set_var("TELLODB_DATA_DIR", dir);
    }
    // The server logs its progress; one-shot commands and MCP only warnings.
    init_tracing_with_default_level(if cli.command == "serve" {
        tracing::Level::INFO
    } else {
        tracing::Level::WARN
    });
    let paths = RuntimePaths::from_env()?;

    match cli.command.as_str() {
        "serve" => serve(&paths).await,
        "mcp" => {
            let engine = Engine::from_paths(&paths, "default").await?;
            let entity = cli.options.get("entity").cloned().unwrap_or_else(|| "user".to_string());
            tellodb::mcp_stdio::McpServer::new(engine, entity).run().await
        }
        "doctor" => {
            let engine = Engine::from_paths(&paths, "default").await?;
            let report = tellodb::doctor::report(engine.state(), &paths)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        "ingest" => ingest_command(&paths, &cli).await,
        "query" => {
            let entity = cli.options.get("entity").cloned();
            let text = cli.positional.join(" ");
            anyhow::ensure!(!text.trim().is_empty(), "query text is required\n\n{USAGE}");
            let mut query = Query::new(text);
            query.entity_id = entity;
            if let Some(limit) = cli.options.get("limit") {
                query.limit = limit.parse()?;
            }
            if let Some(as_of) = cli.options.get("as-of") {
                query.as_of_ms = Some(as_of.parse()?);
            }
            let engine = Engine::from_paths(&paths, "default").await?;
            let hits = engine.query(query).await?;
            println!("{}", serde_json::to_string_pretty(&hits)?);
            Ok(())
        }
        other => anyhow::bail!("unknown command `{other}`\n\n{USAGE}"),
    }
}

async fn ingest_command(paths: &RuntimePaths, cli: &Cli) -> anyhow::Result<()> {
    use tokio::io::AsyncBufReadExt;
    let entity = cli
        .options
        .get("entity")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("--entity is required\n\n{USAGE}"))?;
    let session = cli.options.get("session").cloned();
    let reader: Box<dyn tokio::io::AsyncRead + Unpin> = match cli.positional.first() {
        Some(path) if path != "-" => Box::new(tokio::fs::File::open(path).await?),
        _ => Box::new(tokio::io::stdin()),
    };
    let mut lines = tokio::io::BufReader::new(reader).lines();
    let mut memories = Vec::new();
    let mut turn = 0u32;
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let mut memory = match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(value) if value.is_object() => {
                let mut value = value;
                value["entity_id"] = serde_json::Value::String(entity.clone());
                serde_json::from_value::<Memory>(value)?
            }
            _ => Memory::new(entity.clone(), line),
        };
        if memory.session_id.is_none() {
            if let Some(session) = &session {
                memory.session_id = Some(session.clone());
                memory.turn_index.get_or_insert(turn);
            }
        }
        turn += 1;
        memories.push(memory);
    }
    let engine = Engine::from_paths(paths, "default").await?;
    let mut total = tellodb::db::IngestReport::default();
    for batch in memories.chunks(64) {
        let report = engine.ingest(batch.to_vec()).await?;
        total.memories += report.memories;
        total.expanded += report.expanded;
        total.embedded += report.embedded;
        total.total_ms += report.total_ms;
    }
    engine.checkpoint()?;
    println!("{}", serde_json::to_string_pretty(&total)?);
    Ok(())
}

async fn serve(paths: &RuntimePaths) -> anyhow::Result<()> {
    info!("Starting tellodb server...");
    let auth = api::AuthConfig::from_env()?;
    let state = tellodb::engine::build_state(paths, auth).await?;
    let tenant_manager = state.tenant_manager.clone();
    info!("API key auth enabled on all routes (TEMPORAL_MEMORY_API_KEY or TELLODB_API_KEY).");

    // Local-first default: only reachable from this machine unless a host is
    // configured explicitly (containers set TEMPORAL_MEMORY_HOST=0.0.0.0).
    let host = env::var("TEMPORAL_MEMORY_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("PORT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| env::var("TEMPORAL_MEMORY_PORT").ok().filter(|value| !value.trim().is_empty()))
        .unwrap_or_else(|| "3000".to_string());
    let bind_address = format!("{}:{}", host, port);
    let app = api::build_api(state);

    let maintenance = tenant_manager.clone();
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(300));
        loop {
            ticker.tick().await;
            let tenants = maintenance.all_tenants();
            let _ = tokio::task::spawn_blocking(move || {
                let now_ms = match std::time::SystemTime::now()
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                {
                    Ok(duration) => duration.as_millis() as u64,
                    Err(_) => return,
                };
                for tenant in tenants {
                    if let Err(error) = tenant.checkpoint() {
                        error!(error = ?error, "WAL checkpoint failed");
                    }
                    if let Err(error) = tenant.expire_records(now_ms) {
                        error!(error = ?error, "Lifecycle expiration sweep failed");
                    }
                }
            })
            .await;
        }
    });

    info!(address = %bind_address, "Memory Engine live");
    let listener = TcpListener::bind(&bind_address).await?;
    axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>())
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Checkpointing tenant databases...");
    let tenants = tenant_manager.clone();
    tokio::task::spawn_blocking(move || {
        for tenant in tenants.all_tenants() {
            if let Err(err) = tenant.checkpoint() {
                error!(error = ?err, "Final WAL checkpoint failed");
            }
        }
    })
    .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Shutdown signal received (Ctrl+C)");
        }
        _ = terminate => {
            info!("Shutdown signal received (SIGTERM)");
        }
    }

    info!("Shutting down...");
}
