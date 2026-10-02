//! Personal access token management API.
//!
//! Lets a logged-in portal user mint, list and revoke their own tokens, and
//! lets an admin see every token with its owner and last-used time. These
//! tokens authenticate programmatic clients such as the built-in MCP search
//! server. The secret is shown exactly once, at creation; only its SHA-256 hash
//! is stored.

use super::auth::{check_api_rate_limit, AdminUser, AuthUser};
use super::routes::ApiState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use claudear_storage::{ApiTokenRow, ApiTokenWithOwner};
use rand::RngExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Human-visible token prefix, also stored for display: `cldr_` + 8 hex chars.
const TOKEN_PREFIX_LEN: usize = "cldr_".len() + 8;

#[derive(Deserialize)]
pub struct CreateTokenRequest {
    pub name: String,
    /// Optional expiry as an RFC3339 / SQLite datetime string.
    #[serde(default)]
    pub expires_at: Option<String>,
}

/// Response for a freshly created token. `secret` is returned only here.
#[derive(Serialize)]
pub struct CreatedTokenResponse {
    #[serde(flatten)]
    pub token: ApiTokenRow,
    pub secret: String,
}

/// Generate a new token secret: `cldr_` followed by 48 hex chars (24 bytes).
fn generate_secret() -> String {
    let mut bytes = [0u8; 24];
    rand::rng().fill(&mut bytes);
    format!("cldr_{}", hex::encode(bytes))
}

fn hash_secret(secret: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(secret.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// `POST /api/tokens` — create a token for the current user.
pub async fn create_token_handler(
    user: AuthUser,
    State(state): State<ApiState>,
    Json(body): Json<CreateTokenRequest>,
) -> Result<(StatusCode, Json<CreatedTokenResponse>), StatusCode> {
    if !check_api_rate_limit(user.id) {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    let name = body.name.trim();
    if name.is_empty() || name.len() > 100 {
        return Err(StatusCode::BAD_REQUEST);
    }

    let secret = generate_secret();
    let hash = hash_secret(&secret);
    let prefix = &secret[..TOKEN_PREFIX_LEN];

    let token = state
        .tracker
        .create_api_token(user.id, name, &hash, prefix, body.expires_at.as_deref())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok((
        StatusCode::CREATED,
        Json(CreatedTokenResponse { token, secret }),
    ))
}

/// `GET /api/tokens` — list the current user's tokens (no secrets).
pub async fn list_tokens_handler(
    user: AuthUser,
    State(state): State<ApiState>,
) -> Result<Json<Vec<ApiTokenRow>>, StatusCode> {
    let tokens = state
        .tracker
        .list_api_tokens(user.id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(tokens))
}

/// `DELETE /api/tokens/{id}` — revoke a token. Users may revoke their own;
/// admins may revoke any.
pub async fn revoke_token_handler(
    user: AuthUser,
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> StatusCode {
    // Admins can revoke any token; everyone else is scoped to their own.
    let scope = if user.role == "admin" {
        None
    } else {
        Some(user.id)
    };
    match state.tracker.delete_api_token(&id, scope) {
        Ok(true) => StatusCode::NO_CONTENT,
        Ok(false) => StatusCode::NOT_FOUND,
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// `GET /api/tokens/all` — admin view of every token with its owner.
pub async fn list_all_tokens_handler(
    _admin: AdminUser,
    State(state): State<ApiState>,
) -> Result<Json<Vec<ApiTokenWithOwner>>, StatusCode> {
    let tokens = state
        .tracker
        .list_all_api_tokens()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(tokens))
}

/// `GET /api/users/{id}/tokens` — admin: list a specific user's tokens.
pub async fn list_user_tokens_handler(
    _admin: AdminUser,
    State(state): State<ApiState>,
    Path(user_id): Path<i64>,
) -> Result<Json<Vec<ApiTokenRow>>, StatusCode> {
    let tokens = state
        .tracker
        .list_api_tokens(user_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(tokens))
}

/// `POST /api/users/{id}/tokens` — admin: mint a token for a specific user.
pub async fn create_user_token_handler(
    admin: AdminUser,
    State(state): State<ApiState>,
    Path(user_id): Path<i64>,
    Json(body): Json<CreateTokenRequest>,
) -> Result<(StatusCode, Json<CreatedTokenResponse>), StatusCode> {
    if !check_api_rate_limit(admin.0.id) {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    let name = body.name.trim();
    if name.is_empty() || name.len() > 100 {
        return Err(StatusCode::BAD_REQUEST);
    }

    // The target user must exist.
    if state
        .tracker
        .get_user_by_id(user_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .is_none()
    {
        return Err(StatusCode::NOT_FOUND);
    }

    let secret = generate_secret();
    let hash = hash_secret(&secret);
    let prefix = &secret[..TOKEN_PREFIX_LEN];

    let token = state
        .tracker
        .create_api_token(user_id, name, &hash, prefix, body.expires_at.as_deref())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok((
        StatusCode::CREATED,
        Json(CreatedTokenResponse { token, secret }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_has_prefix_and_hashes_stably() {
        let secret = generate_secret();
        assert!(secret.starts_with("cldr_"));
        assert_eq!(secret.len(), "cldr_".len() + 48);
        assert_eq!(&secret[..TOKEN_PREFIX_LEN], &secret[..13]);
        assert_eq!(hash_secret(&secret), hash_secret(&secret));
        assert_ne!(hash_secret(&secret), hash_secret(&generate_secret()));
    }

    #[test]
    fn prefix_is_ascii_safe_to_slice() {
        // Secret is hex + ascii prefix, so byte slicing never splits a char.
        let secret = generate_secret();
        assert!(secret.is_char_boundary(TOKEN_PREFIX_LEN));
    }
}
