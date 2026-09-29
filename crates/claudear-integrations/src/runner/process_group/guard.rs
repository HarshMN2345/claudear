use super::marker::Marker;
use super::{Drain, Registry};
use std::future::Future;
use std::io;
use std::process::ExitStatus;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::sync::watch;

#[cfg(unix)]
const EXIT_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// How long processes a run left behind get to exit on SIGTERM before they are
/// killed.
const TERMINATE_GRACE: Duration = Duration::from_secs(5);

/// The process group an agent CLI leads, killed with everything left running in
/// it once the CLI exits ([`Self::wait`]), on [`Self::finish`] or on drop.
/// Finishing or dropping the guard also kills every process that left the group
/// but still carries the run's marker in its environment. Processes in the group
/// inherit the CLI's stdout and stderr, so until they die a reader never sees
/// EOF.
pub struct Guard<'a> {
    id: Option<u32>,
    marker: Option<Marker>,
    registry: &'a Registry,
    killed: watch::Sender<bool>,
}

impl<'a> Guard<'a> {
    /// Spawn `command` as the leader of a new process group, with a fresh
    /// marker in its environment, both registered in `registry` until the guard
    /// finishes or drops.
    pub fn spawn(command: &mut Command, registry: &'a Registry) -> io::Result<(Child, Self)> {
        #[cfg(unix)]
        command.process_group(0);
        let marker = Marker::new();
        marker.apply(command);
        registry.track(marker);
        let child = command
            .kill_on_drop(true)
            .spawn()
            .inspect_err(|_| registry.release(marker))?;
        let id = child.id();
        if let Some(id) = id {
            registry.insert(id);
        }
        let guard = Self {
            id,
            marker: Some(marker),
            registry,
            killed: watch::Sender::new(false),
        };
        Ok((child, guard))
    }

    /// Wait for the CLI to exit, killing its group before reaping it: until then
    /// the leader keeps the group id reserved, so the kill cannot reach another
    /// group that reused it.
    pub async fn wait(&mut self, child: &mut Child) -> io::Result<ExitStatus> {
        #[cfg(unix)]
        if let Some(id) = self.id {
            match Self::exited(id).await {
                Ok(()) => self.kill_group(),
                Err(error) => tracing::warn!(
                    component = "runner",
                    process_group = id,
                    error = %error,
                    "Failed to watch agent CLI for exit; killing its group once it is reaped"
                ),
            }
        }
        let status = child.wait().await;
        self.kill_group();
        status
    }

    /// Kill the group, then give marked processes that left it a grace period
    /// to exit on SIGTERM before killing whatever is left.
    pub async fn finish(&mut self) {
        self.kill_group();
        // Left in place until the sweep ends, so dropping a cancelled finish
        // still sweeps.
        if let Some(marker) = self.marker {
            let terminate = tokio::task::spawn_blocking(move || marker.terminate(TERMINATE_GRACE));
            if let Err(error) = terminate.await {
                tracing::warn!(
                    component = "runner",
                    marker = %marker,
                    error = %error,
                    "Failed to terminate processes an agent run left behind, killing them instead"
                );
                marker.kill();
            }
            self.registry.release(marker);
            self.marker = None;
        }
    }

    /// Kill the group alone, so marked processes that left it still get their
    /// SIGTERM grace from [`Self::finish`].
    fn kill_group(&mut self) {
        if let Some(id) = self.id.take() {
            self.registry.kill(id);
        }
        self.killed.send_replace(true);
    }

    /// A [`Drain`] for a reader of the CLI's output, whose grace and cutoff
    /// start when the group is killed or dropped.
    pub fn drain(&self, grace: Duration, cutoff: Duration) -> Drain {
        Drain::new(self.after_kill(grace), self.after_kill(cutoff))
    }

    fn after_kill(&self, delay: Duration) -> impl Future<Output = ()> + Send + 'static {
        let mut killed = self.killed.subscribe();
        async move {
            let _ = killed.wait_for(|killed| *killed).await;
            tokio::time::sleep(delay).await;
        }
    }
}

#[cfg(unix)]
impl Guard<'_> {
    async fn exited(id: u32) -> io::Result<()> {
        let mut child_exits =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::child())?;
        while !Self::has_exited(id)? {
            // Polling covers a SIGCHLD that never arrives, e.g. one swallowed by a
            // ptrace-based tool such as cargo-tarpaulin.
            tokio::select! {
                _ = child_exits.recv() => {}
                () = tokio::time::sleep(EXIT_POLL_INTERVAL) => {}
            }
        }
        Ok(())
    }

    /// Whether child `id` has exited, leaving it unreaped.
    fn has_exited(id: u32) -> io::Result<bool> {
        let options = libc::WEXITED | libc::WNOHANG | libc::WNOWAIT;
        loop {
            // SAFETY: siginfo_t is plain data, valid when zeroed.
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            // SAFETY: `info` is a valid siginfo_t for waitid to fill in.
            if unsafe { libc::waitid(libc::P_PID, id as libc::id_t, &mut info, options) } == 0 {
                // SAFETY: waitid filled in `info`, leaving si_pid 0 while the child runs.
                return Ok(unsafe { info.si_pid() } != 0);
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.kill_group();
        if let Some(marker) = self.marker.take() {
            marker.kill();
            self.registry.release(marker);
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::runner::process_group::tests::{
        assert_killed, background_pid, exits_within, group_command, is_running, BACKGROUND_SLEEP,
        EXIT_DEADLINE,
    };
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    use crate::runner::process_group::tests::{
        assert_perl_environment_readable, kill, stop_if_running, ESCAPED_SLEEP,
    };

    #[tokio::test]
    async fn test_failed_spawn_leaves_nothing_for_shutdown_to_wait_on() {
        let registry = Registry::new();
        let directory = tempfile::tempdir().unwrap();

        let spawned = Guard::spawn(
            &mut Command::new(directory.path().join("claude")),
            &registry,
        );

        assert!(spawned.is_err(), "a missing CLI must fail to spawn");
        tokio::time::timeout(EXIT_DEADLINE, registry.emptied())
            .await
            .expect("a failed spawn must not leave the registry waiting on its run");
    }

    #[tokio::test]
    async fn test_wait_kills_the_group_and_returns_the_leader_status() {
        let registry = Registry::new();
        let (mut leader, mut guard) =
            Guard::spawn(&mut group_command("sleep 300 & echo $!; exit 3"), &registry).unwrap();
        let background = background_pid(&mut leader).await;

        let status = tokio::time::timeout(EXIT_DEADLINE, guard.wait(&mut leader))
            .await
            .expect("wait must return once the leader exits")
            .unwrap();

        assert_eq!(status.code(), Some(3));
        assert!(
            exits_within(background),
            "background process {background} outlived its leader"
        );
        guard.finish().await;
        assert!(registry.is_empty(), "a finished guard must unregister");
    }

    #[tokio::test]
    async fn test_wait_leaves_a_running_leader_alone() {
        let registry = Registry::new();
        let (mut leader, mut guard) =
            Guard::spawn(&mut group_command(BACKGROUND_SLEEP), &registry).unwrap();
        let background = background_pid(&mut leader).await;

        let early = tokio::time::timeout(Duration::from_millis(200), guard.wait(&mut leader)).await;

        assert!(early.is_err(), "wait returned while the leader was running");
        assert!(
            leader.try_wait().unwrap().is_none(),
            "wait killed a running leader"
        );
        assert!(is_running(background));
        drop(guard);
        assert_killed(&mut leader).await;
    }

    #[tokio::test]
    async fn test_drop_kills_the_group_and_unregisters_it() {
        let registry = Registry::new();
        let (mut leader, guard) =
            Guard::spawn(&mut group_command(BACKGROUND_SLEEP), &registry).unwrap();
        let background = background_pid(&mut leader).await;

        drop(guard);

        assert_killed(&mut leader).await;
        assert!(
            exits_within(background),
            "background process {background} survived the guard's drop"
        );
        assert!(registry.is_empty(), "a dropped guard must unregister");
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[tokio::test]
    async fn test_drop_kills_a_marked_process_in_a_session_of_its_own() {
        assert_perl_environment_readable();
        let registry = Registry::new();
        let (mut leader, guard) =
            Guard::spawn(&mut group_command(ESCAPED_SLEEP), &registry).unwrap();
        let escaped = background_pid(&mut leader).await;

        drop(guard);

        let exited = exits_within(escaped);
        if !exited {
            kill(escaped);
        }
        assert!(
            exited,
            "process {escaped} that left the group outlived the guard's drop"
        );
        assert_killed(&mut leader).await;
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[tokio::test]
    async fn test_finish_terminates_a_marked_process_in_a_session_of_its_own() {
        assert_perl_environment_readable();
        let registry = Registry::new();
        let (mut leader, mut guard) =
            Guard::spawn(&mut group_command(ESCAPED_SLEEP), &registry).unwrap();
        let escaped = background_pid(&mut leader).await;

        guard.finish().await;

        let running = stop_if_running(escaped);
        assert!(
            !running,
            "process {escaped} that left the group outlived finish"
        );
        assert!(registry.is_empty(), "a finished guard must unregister");
        assert_killed(&mut leader).await;
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[tokio::test]
    async fn test_finish_kills_a_marked_process_when_its_sweep_fails() {
        assert_perl_environment_readable();
        let registry = Registry::new();
        let (mut leader, mut guard) =
            Guard::spawn(&mut group_command(ESCAPED_SLEEP), &registry).unwrap();
        let escaped = background_pid(&mut leader).await;
        let stopped = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let handle = stopped.handle().clone();
        stopped.shutdown_background();
        let finish = tokio::time::timeout(EXIT_DEADLINE, guard.finish());

        {
            // A runtime that has shut down cancels blocking work, so the sweep
            // never runs.
            let _context = handle.enter();
            finish.await.expect(
                "finish hung: spawn_blocking on a shut-down runtime no longer resolves as cancelled",
            );
        }

        let exited = exits_within(escaped);
        if !exited {
            kill(escaped);
        }
        assert!(
            exited,
            "process {escaped} that left the group outlived a finish whose sweep failed"
        );
        assert!(registry.is_empty(), "a finished guard must unregister");
        assert_killed(&mut leader).await;
    }

    #[tokio::test]
    async fn test_kill_all_reaches_a_guarded_group() {
        let registry = Registry::new();
        let (mut leader, _guard) =
            Guard::spawn(&mut group_command(BACKGROUND_SLEEP), &registry).unwrap();
        let background = background_pid(&mut leader).await;

        registry.kill_all();

        assert_killed(&mut leader).await;
        assert!(
            exits_within(background),
            "background process {background} survived kill_all"
        );
    }
}
