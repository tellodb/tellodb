use axum::{
    extract::{ConnectInfo, Json, Path, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Extension,
};

use crate::api::auth::{self, session_user_from_headers};
use crate::api::types::{
    PlatformApiKeyCreateResponse, PlatformApiKeyListResponse, PlatformAuthResponse,
    PlatformCreateApiKeyPayload, PlatformLoginPayload, PlatformProfileResponse,
    PlatformSignupPayload, PlatformStatsResponse,
};
use crate::api::EngineState;
use crate::error::{EngineError, EngineResult};

const SESSION_TTL_SECONDS: u64 = 60 * 60 * 24 * 30;

pub async fn platform_signup_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    peer: Option<Extension<ConnectInfo<std::net::SocketAddr>>>,
    Json(payload): Json<PlatformSignupPayload>,
) -> EngineResult<impl IntoResponse> {
    auth::check_auth_rate_limit(
        &state,
        &headers,
        peer.map(|Extension(ConnectInfo(address))| address),
    )?;
    let user = state
        .platform
        .create_user(payload.username.as_str(), payload.password.as_str())
        .map_err(|err| {
            let msg = err.to_string();
            tracing::warn!("platform signup failed: {}", msg);
            if msg.contains("exists") || msg.contains("at least") || msg.contains("at most") {
                EngineError::bad_request(msg)
            } else {
                EngineError::Other(err)
            }
        })?;
    let token =
        state.platform.create_session(&user.user_id, SESSION_TTL_SECONDS).map_err(|err| {
            tracing::warn!("platform signup session create failed: {:?}", err);
            EngineError::Other(err)
        })?;
    Ok((StatusCode::CREATED, Json(PlatformAuthResponse { token, user })))
}

pub async fn platform_login_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    peer: Option<Extension<ConnectInfo<std::net::SocketAddr>>>,
    Json(payload): Json<PlatformLoginPayload>,
) -> EngineResult<impl IntoResponse> {
    auth::check_auth_rate_limit(
        &state,
        &headers,
        peer.map(|Extension(ConnectInfo(address))| address),
    )?;
    let user = state.platform.login(payload.username.as_str(), payload.password.as_str()).map_err(
        |err| {
            let msg = err.to_string();
            if msg.contains("invalid credentials") {
                EngineError::Unauthorized
            } else if msg.contains("at most") {
                EngineError::bad_request(msg)
            } else {
                tracing::warn!("platform login failed: {}", msg);
                EngineError::Other(err)
            }
        },
    )?;
    let token =
        state.platform.create_session(&user.user_id, SESSION_TTL_SECONDS).map_err(|err| {
            tracing::warn!("platform login session create failed: {:?}", err);
            EngineError::Other(err)
        })?;
    Ok((StatusCode::OK, Json(PlatformAuthResponse { token, user })))
}

pub async fn platform_logout_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> EngineResult<impl IntoResponse> {
    let token = auth::request_bearer_token(&headers).ok_or(EngineError::Unauthorized)?;
    state.platform.delete_session(token).map_err(|error| {
        tracing::warn!(error = ?error, "platform logout failed");
        EngineError::Other(error)
    })?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn platform_me_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> EngineResult<impl IntoResponse> {
    let user = session_user_from_headers(&state, &headers)?;
    Ok((StatusCode::OK, Json(user)))
}

pub async fn platform_create_api_key_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(payload): Json<PlatformCreateApiKeyPayload>,
) -> EngineResult<impl IntoResponse> {
    let user = session_user_from_headers(&state, &headers)?;
    let (api_key, key) =
        state.platform.create_api_key(&user.user_id, payload.name.as_str()).map_err(|err| {
            tracing::warn!("platform create api key failed: {:?}", err);
            EngineError::Other(err)
        })?;
    Ok((StatusCode::CREATED, Json(PlatformApiKeyCreateResponse { api_key, key })))
}

pub async fn platform_list_api_keys_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> EngineResult<impl IntoResponse> {
    let user = session_user_from_headers(&state, &headers)?;
    let api_keys = state.platform.list_api_keys(&user.user_id).map_err(|err| {
        tracing::warn!("platform list api keys failed: {:?}", err);
        EngineError::Other(err)
    })?;
    Ok((StatusCode::OK, Json(PlatformApiKeyListResponse { api_keys })))
}

pub async fn platform_revoke_api_key_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Path(key_id): Path<String>,
) -> EngineResult<impl IntoResponse> {
    let user = session_user_from_headers(&state, &headers)?;
    state.platform.revoke_api_key(&user.user_id, key_id.as_str()).map_err(|err| {
        let msg = err.to_string();
        if msg.contains("not found") {
            EngineError::NotFound(msg)
        } else {
            tracing::warn!("platform revoke api key failed: {:?}", err);
            EngineError::Other(err)
        }
    })?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn platform_stats_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> EngineResult<impl IntoResponse> {
    let user = session_user_from_headers(&state, &headers)?;
    let usage = state.platform.usage_stats(&user.user_id).map_err(|err| {
        tracing::warn!("platform usage stats failed: {:?}", err);
        EngineError::Other(err)
    })?;
    Ok((StatusCode::OK, Json(PlatformStatsResponse { usage })))
}

pub async fn platform_profile_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> EngineResult<impl IntoResponse> {
    let user = session_user_from_headers(&state, &headers)?;
    let profile = state.platform.user_profile(&user.user_id).map_err(|err| {
        tracing::warn!("platform profile failed: {:?}", err);
        EngineError::Other(err)
    })?;
    Ok((StatusCode::OK, Json(PlatformProfileResponse { profile })))
}
