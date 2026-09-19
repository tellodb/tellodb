use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("storage: {0}")]
    Storage(#[from] rusqlite::Error),
    #[error("vector index: {0}")]
    Vector(String),
    #[error("embedding model: {0}")]
    Embedding(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid request: {0}")]
    BadRequest(String),
    #[error("unauthorized")]
    Unauthorized,
    #[error("rate limited")]
    RateLimited,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type EngineResult<T> = Result<T, EngineError>;

impl EngineError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Other(anyhow::anyhow!(message.into()))
    }

    pub fn from_http_status(status: u16) -> Self {
        match status {
            400 => Self::BadRequest("bad request".to_string()),
            401 => Self::Unauthorized,
            404 => Self::NotFound("resource not found".to_string()),
            429 => Self::RateLimited,
            _ => Self::internal(format!("request failed with status {status}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EngineError;

    #[test]
    fn http_statuses_map_to_domain_categories() {
        assert!(matches!(EngineError::from_http_status(400), EngineError::BadRequest(_)));
        assert!(matches!(EngineError::from_http_status(401), EngineError::Unauthorized));
        assert!(matches!(EngineError::from_http_status(404), EngineError::NotFound(_)));
        assert!(matches!(EngineError::from_http_status(429), EngineError::RateLimited));
        assert!(matches!(EngineError::from_http_status(500), EngineError::Other(_)));
    }
}
