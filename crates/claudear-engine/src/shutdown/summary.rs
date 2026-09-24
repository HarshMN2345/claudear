use super::{Outcome, Reason};

/// How a shutdown went, for choosing the exit code.
#[derive(Debug)]
#[must_use]
pub struct Summary {
    /// What started the shutdown.
    pub reason: Reason,
    /// How the drain ended.
    pub outcome: Outcome,
    /// The error of the service whose failure started the shutdown.
    pub error: Option<anyhow::Error>,
}
