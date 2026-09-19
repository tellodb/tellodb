use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::error::EngineError;

impl IntoResponse for EngineError {
    fn into_response(self) -> Response {
        let status = match &self {
            EngineError::NotFound(_) => StatusCode::NOT_FOUND,
            EngineError::BadRequest(_) => StatusCode::BAD_REQUEST,
            EngineError::Unauthorized => StatusCode::UNAUTHORIZED,
            EngineError::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            EngineError::Storage(_)
            | EngineError::Vector(_)
            | EngineError::Embedding(_)
            | EngineError::Other(_) => {
                tracing::error!(error = ?self, "request failed");
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        (status, Json(json!({"error": self.to_string()}))).into_response()
    }
}
