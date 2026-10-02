//! Cloudflare Access JWT verification.
//!
//! When `[mcp_server].require_cloudflare_access` is enabled, every `/mcp`
//! request must carry a `Cf-Access-Jwt-Assertion` header holding a JWT that
//! Cloudflare Access signs after the user authenticates (including through
//! WARP). Verifying it here means a request that did not transit the Cloudflare
//! tunnel/Access cannot reach the tool surface even if it reaches the origin
//! port directly.
//!
//! The token is validated against the team's public keys published at
//! `https://<team_domain>/cdn-cgi/access/certs` (JWKS), with the configured
//! audience (AUD) and issuer (`https://<team_domain>`) enforced. The key set is
//! cached and refetched when it goes stale or a token presents an unknown key
//! id (Cloudflare rotates keys).
//!
//! Machine clients authenticate non-interactively with a Cloudflare Access
//! **service token**, sending `CF-Access-Client-Id` and `CF-Access-Client-Secret`
//! to the Cloudflare hostname. Cloudflare validates those at the edge and still
//! forwards a signed `Cf-Access-Jwt-Assertion` to the origin, so the same
//! verification path covers both interactive (WARP) users and service accounts;
//! only the identity claim differs (`common_name`/`sub` instead of `email`).

use crate::config::CloudflareAccessConfig;
use jsonwebtoken::{decode, decode_header, jwk::JwkSet, DecodingKey, Validation};
use serde::Deserialize;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// How long a fetched JWKS is trusted before a refresh is forced.
const JWKS_TTL: Duration = Duration::from_secs(3600);

/// Claims we care about from a Cloudflare Access JWT.
#[derive(Debug, Clone, Deserialize)]
pub struct CfAccessClaims {
    /// Authenticated user's email (interactive Access identity). Empty for
    /// service-token principals.
    #[serde(default)]
    pub email: String,
    /// Service-token name, present when a Cloudflare Access service token
    /// (`CF-Access-Client-Id`/`Secret`) was used instead of a user login.
    #[serde(default)]
    pub common_name: String,
    /// Subject claim; a stable identifier for either principal type.
    #[serde(default)]
    pub sub: String,
}

impl CfAccessClaims {
    /// A human-friendly principal label for logging: the email for users, the
    /// service-token name for machines, else the subject.
    pub fn principal(&self) -> &str {
        if !self.email.is_empty() {
            &self.email
        } else if !self.common_name.is_empty() {
            &self.common_name
        } else {
            &self.sub
        }
    }
}

struct CachedJwks {
    keys: JwkSet,
    fetched_at: Instant,
}

/// Verifies Cloudflare Access JWTs against a team's JWKS, caching the key set.
pub struct CfAccessVerifier {
    team_domain: String,
    audience: String,
    certs_url: String,
    issuer: String,
    http: reqwest::Client,
    cache: RwLock<Option<CachedJwks>>,
}

impl CfAccessVerifier {
    pub fn new(cfg: &CloudflareAccessConfig) -> Self {
        let team_domain = cfg.team_domain.trim().trim_end_matches('/').to_string();
        Self {
            certs_url: format!("https://{team_domain}/cdn-cgi/access/certs"),
            issuer: format!("https://{team_domain}"),
            audience: cfg.audience.trim().to_string(),
            team_domain,
            http: reqwest::Client::new(),
            cache: RwLock::new(None),
        }
    }

    /// True when the verifier has enough configuration to run.
    pub fn is_configured(&self) -> bool {
        !self.team_domain.is_empty() && !self.audience.is_empty()
    }

    /// Verify a raw `Cf-Access-Jwt-Assertion` token, returning its claims.
    pub async fn verify(&self, token: &str) -> Result<CfAccessClaims, String> {
        if !self.is_configured() {
            return Err("Cloudflare Access is not configured (team_domain/audience)".to_string());
        }

        let header = decode_header(token).map_err(|e| format!("bad token header: {e}"))?;
        let kid = header
            .kid
            .ok_or_else(|| "token missing key id (kid)".to_string())?;

        // Try the cached key set, refetching once if the kid is unknown or stale.
        let mut key = self.find_key(&kid, false).await?;
        if key.is_none() {
            key = self.find_key(&kid, true).await?;
        }
        let jwk = key.ok_or_else(|| format!("no matching Cloudflare key for kid {kid}"))?;

        let decoding_key =
            DecodingKey::from_jwk(&jwk).map_err(|e| format!("invalid Cloudflare key: {e}"))?;

        let mut validation = Validation::new(header.alg);
        validation.set_audience(&[&self.audience]);
        validation.set_issuer(&[&self.issuer]);

        let data = decode::<CfAccessClaims>(token, &decoding_key, &validation)
            .map_err(|e| format!("token verification failed: {e}"))?;
        Ok(data.claims)
    }

    /// Look up a key by id from the cache, optionally forcing a refetch first.
    async fn find_key(
        &self,
        kid: &str,
        force_refresh: bool,
    ) -> Result<Option<jsonwebtoken::jwk::Jwk>, String> {
        if force_refresh || self.cache_is_stale().await {
            self.refresh().await?;
        }
        let guard = self.cache.read().await;
        Ok(guard.as_ref().and_then(|c| c.keys.find(kid)).cloned())
    }

    async fn cache_is_stale(&self) -> bool {
        match &*self.cache.read().await {
            Some(c) => c.fetched_at.elapsed() >= JWKS_TTL,
            None => true,
        }
    }

    async fn refresh(&self) -> Result<(), String> {
        let keys: JwkSet = self
            .http
            .get(&self.certs_url)
            .send()
            .await
            .map_err(|e| format!("failed to fetch Cloudflare certs: {e}"))?
            .error_for_status()
            .map_err(|e| format!("Cloudflare certs request failed: {e}"))?
            .json()
            .await
            .map_err(|e| format!("failed to parse Cloudflare certs: {e}"))?;
        *self.cache.write().await = Some(CachedJwks {
            keys,
            fetched_at: Instant::now(),
        });
        Ok(())
    }
}
