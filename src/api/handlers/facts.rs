//! `GET /facts/current` and `GET /facts/history`: what a fact's value is now,
//! what it was at a point in time, and how it changed.

use crate::api::auth::{
    principal_namespace_prefix, principal_user_id, record_usage_for_principal, scope_entity_id,
    RequestPrincipal,
};
use crate::api::EngineState;
use axum::extract::{Extension, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
pub struct FactParams {
    /// Whose fact; defaults to the caller's namespace.
    pub entity_id: Option<String>,
    /// Fact slot, e.g. `residence`.
    pub fact_key: String,
    /// For `current`: the value that held at this time instead of now.
    pub as_of_ms: Option<u64>,
}

/// Resolves the caller's tenant and the entity the request is about.
fn scope(
    state: &EngineState,
    principal: &RequestPrincipal,
    requested: Option<&str>,
) -> Result<(std::sync::Arc<crate::storage::TenantStore>, String), StatusCode> {
    let ns_prefix = principal_namespace_prefix(principal);
    let tenant_id = principal_user_id(principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|err| {
        tracing::error!(error = ?err, tenant_id, "fact lookup tenant open failed");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    let entity_id = requested
        .filter(|e| !e.trim().is_empty())
        .map(|e| scope_entity_id(e, ns_prefix.as_deref()))
        .or_else(|| ns_prefix.as_deref().map(|p| p.trim_end_matches(':').to_string()))
        .ok_or(StatusCode::BAD_REQUEST)?;
    Ok((tenant, entity_id))
}

pub async fn current_fact_handler(
    State(state): State<EngineState>,
    Extension(principal): Extension<RequestPrincipal>,
    Query(params): Query<FactParams>,
) -> Result<impl IntoResponse, StatusCode> {
    let (tenant, entity_id) = scope(&state, &principal, params.entity_id.as_deref())?;
    let history = tenant.fact_history(&entity_id, &params.fact_key).map_err(|err| {
        tracing::error!(error = ?err, "fact history read failed");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    // As-of: the version whose validity interval covers that instant.
    let version = match params.as_of_ms {
        Some(as_of) => history.iter().rev().find(|v| {
            v.valid_from_ms <= as_of && v.valid_to_ms.map_or(true, |valid_to| as_of < valid_to)
        }),
        None => history.iter().find(|v| v.is_current),
    };
    record_usage_for_principal(&state, &principal, "facts_current");
    Ok((
        StatusCode::OK,
        Json(json!({
            "entity_id": entity_id,
            "fact_key": params.fact_key,
            "as_of_ms": params.as_of_ms,
            "value": version.map(|v| v.object.clone()),
            "version": version,
            "versions": history.len(),
        })),
    ))
}

pub async fn fact_history_handler(
    State(state): State<EngineState>,
    Extension(principal): Extension<RequestPrincipal>,
    Query(params): Query<FactParams>,
) -> Result<impl IntoResponse, StatusCode> {
    let (tenant, entity_id) = scope(&state, &principal, params.entity_id.as_deref())?;
    let history = tenant.fact_history(&entity_id, &params.fact_key).map_err(|err| {
        tracing::error!(error = ?err, "fact history read failed");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    record_usage_for_principal(&state, &principal, "facts_history");
    Ok((
        StatusCode::OK,
        Json(json!({
            "entity_id": entity_id,
            "fact_key": params.fact_key,
            "history": history,
        })),
    ))
}
