use super::Reason;
use futures::Stream;
use std::io;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::signal::unix::{self, Signal, SignalKind};

/// The stop signals, as a stream of shutdown reasons: SIGINT and SIGHUP read as
/// [`Reason::Interrupted`], SIGTERM as [`Reason::Terminated`].
///
/// SIGHUP is only heard when it wasn't already ignored at startup, so a closed terminal still
/// leaves a daemon started under `nohup` running.
#[derive(Debug)]
pub struct Signals {
    listeners: Vec<(Signal, Reason)>,
}

impl Signals {
    /// Installs the handlers inside the current Tokio runtime. From then on these signals no
    /// longer kill the process, so the caller decides how to exit.
    pub fn listen() -> io::Result<Self> {
        let mut listeners = vec![
            (unix::signal(SignalKind::interrupt())?, Reason::Interrupted),
            (unix::signal(SignalKind::terminate())?, Reason::Terminated),
        ];
        if !ignored(disposition(libc::SIGHUP)?) {
            listeners.push((unix::signal(SignalKind::hangup())?, Reason::Interrupted));
        }
        Ok(Self { listeners })
    }
}

impl Stream for Signals {
    type Item = Reason;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Reason>> {
        for (listener, reason) in self.listeners.iter_mut() {
            if listener.poll_recv(context).is_ready() {
                return Poll::Ready(Some(*reason));
            }
        }
        Poll::Pending
    }
}

/// Whether `handler` is the ignore disposition, which `nohup` sets for SIGHUP.
fn ignored(handler: libc::sighandler_t) -> bool {
    handler == libc::SIG_IGN
}

/// Reads the handler currently installed for `signal` without changing it.
fn disposition(signal: libc::c_int) -> io::Result<libc::sighandler_t> {
    let mut action = MaybeUninit::<libc::sigaction>::uninit();
    // SAFETY: with a null new action, sigaction only writes the current action into `action`.
    if unsafe { libc::sigaction(signal, std::ptr::null(), action.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the successful sigaction call above initialised `action`.
    Ok(unsafe { action.assume_init() }.sa_sigaction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signals_can_drive_a_shutdown() {
        fn stream<S: Stream<Item = Reason> + Unpin>() {}

        stream::<Signals>();
    }

    #[test]
    fn only_an_ignored_hangup_is_left_alone() {
        assert!(ignored(libc::SIG_IGN));
        assert!(!ignored(libc::SIG_DFL));
    }

    #[test]
    fn disposition_reads_without_changing_the_handler() {
        let before = disposition(libc::SIGHUP).expect("SIGHUP is a valid signal");

        assert_eq!(disposition(libc::SIGHUP).unwrap(), before);
    }

    #[test]
    fn disposition_rejects_an_invalid_signal() {
        let error = disposition(0).expect_err("0 is not a signal sigaction accepts");

        assert_eq!(error.raw_os_error(), Some(libc::EINVAL));
    }
}
