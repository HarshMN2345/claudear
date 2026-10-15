//! The process-wide rustls crypto provider.
//!
//! axum-server and rustls-acme are built without a provider of their own, and
//! reqwest uses whichever provider is installed when a client is built, so
//! ring is installed once per process. Installing it from `main` covers the
//! binary and nothing else — every unit test that builds a client runs in a
//! process that never saw `main` — so the install lives here, next to the
//! client factory that library code goes through.

use std::sync::Once;

static INSTALL: Once = Once::new();

/// Install ring as the process-wide provider, once.
///
/// Safe to call from anywhere and at any time: repeat calls are no-ops, and a
/// provider another caller installed first is left alone.
pub fn ensure_crypto_provider() {
    INSTALL.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Build a reqwest client, with the provider installed first.
///
/// A client built with `reqwest::Client::new` before anything installed ring
/// would fall back to reqwest's own provider instead.
pub fn client() -> reqwest::Client {
    ensure_crypto_provider();
    reqwest::Client::new()
}

/// Start building a reqwest client, with the provider installed first.
pub fn client_builder() -> reqwest::ClientBuilder {
    ensure_crypto_provider();
    reqwest::Client::builder()
}
