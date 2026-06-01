use thiserror::Error;

/// Strategic error type using `thiserror` for library-level error handling.
#[derive(Error, Debug)]
pub enum AletheiaError {
    #[error("Storage error: {0}")]
    Storage(#[from] rusqlite::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Strategic result type alias for the project — see [`AletheiaError`].
pub type AletheiaResult<T> = Result<T, AletheiaError>;
