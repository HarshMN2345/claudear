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
use tokio::sync::{Mutex, RwLock};

/// How long a fetched JWKS is trusted before a refresh is forced.
const JWKS_TTL: Duration = Duration::from_secs(3600);

/// Minimum spacing between outbound JWKS fetches. Bounds the work an
/// unauthenticated caller can trigger by presenting tokens with unknown key ids
/// (this runs before bearer auth).
const MIN_REFRESH_INTERVAL: Duration = Duration::from_secs(60);

/// Timeout for the JWKS fetch so a slow Cloudflare response cannot pile up.
const FETCH_TIMEOUT: Duration = Duration::from_secs(10);

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
    /// Single-flight gate for refreshes; holds the last fetch-attempt time so
    /// concurrent callers coalesce and a cooldown can be enforced.
    refresh_gate: Mutex<Option<Instant>>,
}

impl CfAccessVerifier {
    pub fn new(cfg: &CloudflareAccessConfig) -> Self {
        let team_domain = cfg.team_domain.trim().trim_end_matches('/').to_string();
        Self {
            certs_url: format!("https://{team_domain}/cdn-cgi/access/certs"),
            issuer: format!("https://{team_domain}"),
            audience: cfg.audience.trim().to_string(),
            team_domain,
            http: reqwest::Client::builder()
                .timeout(FETCH_TIMEOUT)
                .build()
                .unwrap_or_default(),
            cache: RwLock::new(None),
            refresh_gate: Mutex::new(None),
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

        // Fast path: a known key from a non-stale cache needs no fetch.
        let mut jwk = self.cached_fresh_key(&kid).await;
        if jwk.is_none() {
            // Unknown or stale: attempt a bounded, coalesced refresh, then retry.
            self.refresh_if_allowed().await;
            jwk = self.cached_key(&kid).await;
        }
        let jwk = jwk.ok_or_else(|| format!("no matching Cloudflare key for kid {kid}"))?;

        let decoding_key =
            DecodingKey::from_jwk(&jwk).map_err(|e| format!("invalid Cloudflare key: {e}"))?;

        let mut validation = Validation::new(header.alg);
        // jsonwebtoken only requires `exp` by default and validates aud/iss only
        // when present. Require them so a signed token without aud/iss cannot pass
        // without proving it belongs to the configured Access application.
        validation.set_required_spec_claims(&["exp", "aud", "iss"]);
        validation.set_audience(&[&self.audience]);
        validation.set_issuer(&[&self.issuer]);

        let data = decode::<CfAccessClaims>(token, &decoding_key, &validation)
            .map_err(|e| format!("token verification failed: {e}"))?;
        Ok(data.claims)
    }

    /// A key from the cache only if the cache is within its TTL.
    async fn cached_fresh_key(&self, kid: &str) -> Option<jsonwebtoken::jwk::Jwk> {
        let guard = self.cache.read().await;
        match guard.as_ref() {
            Some(c) if c.fetched_at.elapsed() < JWKS_TTL => c.keys.find(kid).cloned(),
            _ => None,
        }
    }

    /// A key from the cache regardless of age.
    async fn cached_key(&self, kid: &str) -> Option<jsonwebtoken::jwk::Jwk> {
        self.cache
            .read()
            .await
            .as_ref()
            .and_then(|c| c.keys.find(kid).cloned())
    }

    /// Refresh the JWKS, coalescing concurrent callers and enforcing a cooldown
    /// so bogus assertions (valid-looking tokens with unknown key ids) cannot
    /// drive unbounded outbound fetches, since this runs before bearer auth. A
    /// failure is logged and swallowed so a Cloudflare hiccup is not amplified
    /// into a per-request error storm; the subsequent key lookup simply misses.
    async fn refresh_if_allowed(&self) {
        // Serialize refreshers; whoever waited may find the work already done.
        let mut last_attempt = self.refresh_gate.lock().await;

        // Cooldown is the only gate here: this is reached only when the wanted
        // key is absent from a fresh cache (an unknown/rotated kid) or the cache
        // is stale, and in both cases we want to refetch — just not more than
        // once per MIN_REFRESH_INTERVAL. A coalesced second caller that skips on
        // cooldown still reads the cache the first caller populated, because
        // verify() does an any-age lookup afterwards.
        if let Some(t) = *last_attempt {
            if t.elapsed() < MIN_REFRESH_INTERVAL {
                return;
            }
        }

        *last_attempt = Some(Instant::now());
        match self.fetch_jwks().await {
            Ok(keys) => {
                *self.cache.write().await = Some(CachedJwks {
                    keys,
                    fetched_at: Instant::now(),
                });
            }
            Err(e) => {
                tracing::warn!(component = "cf_access", error = %e, "JWKS refresh failed");
            }
        }
    }

    async fn fetch_jwks(&self) -> Result<JwkSet, String> {
        self.http
            .get(&self.certs_url)
            .send()
            .await
            .map_err(|e| format!("failed to fetch Cloudflare certs: {e}"))?
            .error_for_status()
            .map_err(|e| format!("Cloudflare certs request failed: {e}"))?
            .json()
            .await
            .map_err(|e| format!("failed to parse Cloudflare certs: {e}"))
    }

    /// Seed the JWKS cache directly, bypassing the network, for tests. Also marks
    /// a recent refresh so an unknown-kid lookup does not trigger a real fetch.
    #[cfg(test)]
    async fn seed_cache_for_test(&self, jwks_json: &str) {
        let keys: JwkSet = serde_json::from_str(jwks_json).unwrap();
        *self.cache.write().await = Some(CachedJwks {
            keys,
            fetched_at: Instant::now(),
        });
        *self.refresh_gate.lock().await = Some(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
    use serde::Serialize;
    use std::time::{SystemTime, UNIX_EPOCH};

    const KID: &str = "test-kid";
    const N_B64U: &str = "yLRaUXCSXU1OGkgaW4mvOsneI31gRVz0-wHGUpWR1k6x0QsbQp7qaYi1ypOaEnJrro_8cpYmbxblkcyqDPWweXAHJjXwCXBlJ67nEByt5Ni6WIodAShztX0djx4oqyms8YLuaJSWl3_RaMJFGUMdm4Py_JG-Mq38Jt10Gh-GmN1BgSZ2PIGws8wNrt7AM3-ndvBef7ggXAnY4D34T8bI28dcU2K82hpNdJXJ9R0JU-GcFZJGa1ntVX8Lp_fpq0jhUaZvcmxrZMAYGVCrJW1y8EeViqBsIHEND1GA7-PCXiiM_4fIib5v17BtU_7mmgDTLkgZtHjq8KAAwYyh8pRqhw";
    const E_B64U: &str = "AQAB";
    const PRIV_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDItFpRcJJdTU4a
SBpbia86yd4jfWBFXPT7AcZSlZHWTrHRCxtCnuppiLXKk5oScmuuj/xyliZvFuWR
zKoM9bB5cAcmNfAJcGUnrucQHK3k2LpYih0BKHO1fR2PHiirKazxgu5olJaXf9Fo
wkUZQx2bg/L8kb4yrfwm3XQaH4aY3UGBJnY8gbCzzA2u3sAzf6d28F5/uCBcCdjg
PfhPxsjbx1xTYrzaGk10lcn1HQlT4ZwVkkZrWe1Vfwun9+mrSOFRpm9ybGtkwBgZ
UKslbXLwR5WKoGwgcQ0PUYDv48JeKIz/h8iJvm/XsG1T/uaaANMuSBm0eOrwoADB
jKHylGqHAgMBAAECggEAJOqbhEhO+69o8seZZYXtP8R6wg9hIlEHVJYOewk84mzm
IxvGX1ooptG5EnJU0BjQurKMTi1VE3DkOA2rp6eXVrbm8b3REYNlb8epg5qq16GP
oRKCZECjC9pgEf+LnnQTdfbN0FmoW2RsybaWDB/+taivDIroL501+JYaMWXkFYCk
95pIpECBy4F4bNUKtuNJAqE0CF0HhPCSY6UiFRD4gxpKWHZFWb5RdZUvVaPerzwb
txln1WHzx+mWZT/2k0YqK2ogy51azNl6uwAz52GkcBidbwBuVHMR1FDlI1LH6Xlc
3gILNs0Vm/1Ahm+PwPsLT7iR9ZWtt6KQ9dW/sEPSGQKBgQD/8qQdRrzBXOZFdS/4
xI1oquxSTPVwzSuJWWDiyhILzgrnEmxKKafPbIdbTqFfeeumzc9gj8O/epySLaR9
LxDVc+0AkxYtkYQ6nbX+8cFcKAiz1x9Mf0CUiMM6iVtfW7j76oyJPUbNan/sqxxr
955bgBxVMBJf+bKCxX1SG29zzwKBgQDIvtQP1H+gmnpz0Bqxedc+qSPrtw3oyBFp
BF+EJBbs8PbhvKz47WIE0D4nPAHysRLUqqtmlDmCLc6O86e6L5oORlomPLOfM9hY
8aEsVTGTxwX8xpauvKaYVIx7QSWeoGv4bxoe+AXsagyiF9suHe8635+h2vWLq1mp
AIPeuMvzyQKBgFl3+xBU0tSQ4dmzzjIamwfUf8mBJ2boAWkAulJsqoQ/4SXHFd2S
1Bs459PuF5DlcI+db/lkJ9v+Q08B73bnBe5nmJhT0jPZoyxORvk4jwvk3q3m7AT0
kqGZcQ08SJl72Z0N71Rl/CMAMHmNkuDW7R81GDJbHIE6KsF1wYn7FymXAoGAB9zp
o4EYSqsiVrz0/rSeCLdJT+dIpTCI9gsUzrE3MKqzkN36DHoH19ZsSM8h6GalLS1O
L2No6T9wEstaa4GH0D1TNKI2CutV8w3r2TexDG/EPUVuC4QaJmdRZVaE6bSw5fc8
F7BxUvRIcGTs0d6cSzsNHqLb8U+R4HvDroqgenkCgYEAg+asFSLxqQt+6NAYB7uJ
tL+l70KcE5q+L+xza0c9oyfg2Cvm0vB9QZavmGTPY7cJPBXDiFsNyAS2gQiw7Vey
n+7mIzY6zICzzi4Zgyj3hixDCl/7tJKRFHTtUXMeYeCLbqhJ2UQNuoUoW3t9ryq4
LylW2Cn3jMwQSP7PPLXmTZU=
-----END PRIVATE KEY-----
";

    #[derive(Serialize)]
    struct TestClaims {
        aud: String,
        iss: String,
        exp: usize,
        #[serde(skip_serializing_if = "String::is_empty")]
        email: String,
        #[serde(skip_serializing_if = "String::is_empty")]
        common_name: String,
    }

    fn jwks_json() -> String {
        format!(
            r#"{{"keys":[{{"kty":"RSA","use":"sig","alg":"RS256","kid":"{KID}","n":"{N_B64U}","e":"{E_B64U}"}}]}}"#
        )
    }

    fn verifier() -> CfAccessVerifier {
        CfAccessVerifier::new(&CloudflareAccessConfig {
            team_domain: "team.cloudflareaccess.com".to_string(),
            audience: "aud-tag".to_string(),
        })
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn sign(aud: &str, iss: &str, exp: usize, kid: &str) -> String {
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some(kid.to_string());
        let claims = TestClaims {
            aud: aud.to_string(),
            iss: iss.to_string(),
            exp,
            email: "dev@example.com".to_string(),
            common_name: String::new(),
        };
        encode(
            &header,
            &claims,
            &EncodingKey::from_rsa_pem(PRIV_PEM.as_bytes()).unwrap(),
        )
        .unwrap()
    }

    const ISS: &str = "https://team.cloudflareaccess.com";

    #[tokio::test]
    async fn verifies_a_valid_token() {
        let v = verifier();
        v.seed_cache_for_test(&jwks_json()).await;
        let tok = sign("aud-tag", ISS, (now_secs() + 3600) as usize, KID);
        let claims = v.verify(&tok).await.expect("valid token");
        assert_eq!(claims.email, "dev@example.com");
        assert_eq!(claims.principal(), "dev@example.com");
    }

    #[tokio::test]
    async fn rejects_wrong_audience() {
        let v = verifier();
        v.seed_cache_for_test(&jwks_json()).await;
        let tok = sign("other-aud", ISS, (now_secs() + 3600) as usize, KID);
        assert!(v.verify(&tok).await.is_err());
    }

    #[tokio::test]
    async fn rejects_wrong_issuer() {
        let v = verifier();
        v.seed_cache_for_test(&jwks_json()).await;
        let tok = sign(
            "aud-tag",
            "https://evil.example.com",
            (now_secs() + 3600) as usize,
            KID,
        );
        assert!(v.verify(&tok).await.is_err());
    }

    #[tokio::test]
    async fn rejects_expired_token() {
        let v = verifier();
        v.seed_cache_for_test(&jwks_json()).await;
        // Well past the default 60s leeway.
        let tok = sign("aud-tag", ISS, (now_secs() - 3600) as usize, KID);
        assert!(v.verify(&tok).await.is_err());
    }

    #[tokio::test]
    async fn rejects_unknown_kid() {
        let v = verifier();
        v.seed_cache_for_test(&jwks_json()).await; // cache holds only KID
        let tok = sign("aud-tag", ISS, (now_secs() + 3600) as usize, "rotated-kid");
        // Cooldown (seeded) blocks a refetch, so the unknown key simply misses.
        assert!(v.verify(&tok).await.is_err());
    }

    #[tokio::test]
    async fn rejects_token_missing_aud_or_iss() {
        #[derive(Serialize)]
        struct BareClaims {
            exp: usize,
        }
        let v = verifier();
        v.seed_cache_for_test(&jwks_json()).await;
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some(KID.to_string());
        // Correctly signed, exp valid, but no aud/iss claims at all.
        let tok = encode(
            &header,
            &BareClaims {
                exp: (now_secs() + 3600) as usize,
            },
            &EncodingKey::from_rsa_pem(PRIV_PEM.as_bytes()).unwrap(),
        )
        .unwrap();
        assert!(v.verify(&tok).await.is_err());
    }

    #[tokio::test]
    async fn rejects_when_not_configured() {
        let v = CfAccessVerifier::new(&CloudflareAccessConfig::default());
        assert!(!v.is_configured());
        assert!(v.verify("anything").await.is_err());
    }

    #[test]
    fn principal_prefers_email_then_common_name_then_sub() {
        let email = CfAccessClaims {
            email: "a@b.c".into(),
            common_name: "svc".into(),
            sub: "s".into(),
        };
        assert_eq!(email.principal(), "a@b.c");
        let svc = CfAccessClaims {
            email: String::new(),
            common_name: "svc".into(),
            sub: "s".into(),
        };
        assert_eq!(svc.principal(), "svc");
        let sub = CfAccessClaims {
            email: String::new(),
            common_name: String::new(),
            sub: "s".into(),
        };
        assert_eq!(sub.principal(), "s");
    }
}
