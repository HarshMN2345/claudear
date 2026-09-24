//! Graceful shutdown for the daemon: stop taking new work, then drain in-flight runs.

use std::time::Duration;

/// How long shutdown waits for in-flight runs to finish.
pub const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
