use axum::http::HeaderMap;
use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::IntoResponse,
};

use crate::api::auth::session_user_from_headers;
use crate::api::types::*;
use crate::api::EngineState;

const SESSION_TTL_SECONDS: u64 = 60 * 60 * 24 * 30;

pub async fn platform_signup_handler(
    State(state): State<EngineState>,
    Json(payload): Json<PlatformSignupPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let user = state
        .platform
        .create_user(payload.username.as_str(), payload.password.as_str())
        .map_err(|err| {
            let msg = err.to_string();
            tracing::warn!("platform signup failed: {}", msg);
            if msg.contains("exists") || msg.contains("at least") {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        })?;
    let token =
        state.platform.create_session(&user.user_id, SESSION_TTL_SECONDS).map_err(|err| {
            tracing::warn!("platform signup session create failed: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok((StatusCode::CREATED, Json(PlatformAuthResponse { token, user })))
}

pub async fn platform_login_handler(
    State(state): State<EngineState>,
    Json(payload): Json<PlatformLoginPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let user = state.platform.login(payload.username.as_str(), payload.password.as_str()).map_err(
        |err| {
            let msg = err.to_string();
            if msg.contains("invalid credentials") {
                StatusCode::UNAUTHORIZED
            } else {
                tracing::warn!("platform login failed: {}", msg);
                StatusCode::INTERNAL_SERVER_ERROR
            }
        },
    )?;
    let token =
        state.platform.create_session(&user.user_id, SESSION_TTL_SECONDS).map_err(|err| {
            tracing::warn!("platform login session create failed: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok((StatusCode::OK, Json(PlatformAuthResponse { token, user })))
}

pub async fn platform_me_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, StatusCode> {
    let user = session_user_from_headers(&state, &headers)?;
    Ok((StatusCode::OK, Json(user)))
}

pub async fn platform_create_api_key_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(payload): Json<PlatformCreateApiKeyPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let user = session_user_from_headers(&state, &headers)?;
    let (api_key, key) =
        state.platform.create_api_key(&user.user_id, payload.name.as_str()).map_err(|err| {
            tracing::warn!("platform create api key failed: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok((StatusCode::CREATED, Json(PlatformApiKeyCreateResponse { api_key, key })))
}

pub async fn platform_list_api_keys_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, StatusCode> {
    let user = session_user_from_headers(&state, &headers)?;
    let api_keys = state.platform.list_api_keys(&user.user_id).map_err(|err| {
        tracing::warn!("platform list api keys failed: {:?}", err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok((StatusCode::OK, Json(PlatformApiKeyListResponse { api_keys })))
}

pub async fn platform_revoke_api_key_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Path(key_id): Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let user = session_user_from_headers(&state, &headers)?;
    state.platform.revoke_api_key(&user.user_id, key_id.as_str()).map_err(|err| {
        let msg = err.to_string();
        if msg.contains("not found") {
            StatusCode::NOT_FOUND
        } else {
            tracing::warn!("platform revoke api key failed: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        }
    })?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn platform_stats_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, StatusCode> {
    let user = session_user_from_headers(&state, &headers)?;
    let usage = state.platform.usage_stats(&user.user_id).map_err(|err| {
        tracing::warn!("platform usage stats failed: {:?}", err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok((StatusCode::OK, Json(PlatformStatsResponse { usage })))
}

pub async fn platform_profile_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, StatusCode> {
    let user = session_user_from_headers(&state, &headers)?;
    let profile = state.platform.user_profile(&user.user_id).map_err(|err| {
        tracing::warn!("platform profile failed: {:?}", err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok((StatusCode::OK, Json(PlatformProfileResponse { profile })))
}
