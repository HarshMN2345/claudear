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
        receive(&mut self.listeners, |listener| listener.poll_recv(context))
    }
}

/// Yields the reason of the first listener whose signal arrived.
///
/// A closed listener can never deliver its signal, so it is dropped rather than taken for one,
/// and the signals end once no listener is left.
fn receive<Listener>(
    listeners: &mut Vec<(Listener, Reason)>,
    mut poll: impl FnMut(&mut Listener) -> Poll<Option<()>>,
) -> Poll<Option<Reason>> {
    let mut index = 0;
    while let Some((listener, reason)) = listeners.get_mut(index) {
        match poll(listener) {
            Poll::Ready(Some(())) => return Poll::Ready(Some(*reason)),
            Poll::Ready(None) => {
                listeners.remove(index);
            }
            Poll::Pending => index += 1,
        }
    }
    if listeners.is_empty() {
        Poll::Ready(None)
    } else {
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

    const IDLE: Poll<Option<()>> = Poll::Pending;
    const DELIVERED: Poll<Option<()>> = Poll::Ready(Some(()));
    const CLOSED: Poll<Option<()>> = Poll::Ready(None);

    fn received(listeners: &mut Vec<(Poll<Option<()>>, Reason)>) -> Poll<Option<Reason>> {
        receive(listeners, |listener| *listener)
    }

    #[test]
    fn delivered_signal_yields_its_reason() {
        let mut listeners = vec![(IDLE, Reason::Interrupted), (DELIVERED, Reason::Terminated)];

        assert_eq!(
            received(&mut listeners),
            Poll::Ready(Some(Reason::Terminated))
        );
        assert_eq!(listeners.len(), 2, "every listener keeps listening");
    }

    #[test]
    fn closed_listener_is_dropped_instead_of_read_as_a_signal() {
        let mut listeners = vec![(CLOSED, Reason::Interrupted), (IDLE, Reason::Terminated)];

        assert_eq!(received(&mut listeners), Poll::Pending);
        assert_eq!(listeners, [(IDLE, Reason::Terminated)]);
    }

    #[test]
    fn signal_behind_a_closed_listener_is_still_delivered() {
        let mut listeners = vec![
            (CLOSED, Reason::Interrupted),
            (DELIVERED, Reason::Terminated),
        ];

        assert_eq!(
            received(&mut listeners),
            Poll::Ready(Some(Reason::Terminated))
        );
    }

    #[test]
    fn signals_end_once_every_listener_has_closed() {
        let mut listeners = vec![(CLOSED, Reason::Interrupted), (CLOSED, Reason::Terminated)];

        assert_eq!(received(&mut listeners), Poll::Ready(None));
        assert!(listeners.is_empty());
    }

    #[test]
    fn signals_without_listeners_have_ended() {
        assert_eq!(received(&mut Vec::new()), Poll::Ready(None));
    }

    #[test]
    fn only_an_ignored_hangup_is_left_alone() {
        assert!(ignored(libc::SIG_IGN));
        assert!(!ignored(libc::SIG_DFL));
    }

    #[test]
    fn disposition_reads_an_ignored_signal_as_ignored() {
        let handler = disposition(libc::SIGPIPE).expect("SIGPIPE is a valid signal");

        assert!(
            ignored(handler),
            "the Rust runtime ignores SIGPIPE before main"
        );
    }

    #[test]
    fn disposition_rejects_an_invalid_signal() {
        let error = disposition(0).expect_err("0 is not a signal sigaction accepts");

        assert_eq!(error.raw_os_error(), Some(libc::EINVAL));
    }
}
