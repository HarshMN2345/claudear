//! The process-wide rustls crypto provider.
//!
//! reqwest, axum-server and rustls-acme are built without a provider of their
//! own. reqwest falls back to aws-lc only when another crate in the build turns
//! on its default TLS, so a client built before a provider is installed can
//! panic with "No provider set". Installing ring from `main` covers the binary
//! and nothing else — every unit test that builds a client runs in a process
//! that never saw `main` — so the install lives here, next to the client
//! factory that library code goes through.

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
/// Clients claudear builds itself go through here or [`client_builder`];
/// calling `reqwest::Client::new` directly, before ring is installed, can
/// panic. abnegate-http's clients cannot, because that crate turns on
/// reqwest's aws-lc fallback wherever it is built.
pub fn client() -> reqwest::Client {
    ensure_crypto_provider();
    reqwest::Client::new()
}

/// Start building a reqwest client, with the provider installed first.
pub fn client_builder() -> reqwest::ClientBuilder {
    ensure_crypto_provider();
    reqwest::Client::builder()
}
