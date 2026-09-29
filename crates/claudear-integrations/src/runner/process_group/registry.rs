use super::marker::Marker;
use super::signal::Signal;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

static GLOBAL: Registry = Registry::new();

const EMPTIED_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Process groups of agent CLIs that are still running, and the markers of runs
/// whose processes may have left them. Each CLI leads its own group, out of
/// reach of the terminal's signals, so claudear passes them on itself and kills
/// whatever is left before it exits.
#[derive(Debug, Default)]
pub struct Registry {
    ids: Mutex<BTreeSet<u32>>,
    markers: Mutex<BTreeSet<Marker>>,
    interrupted: AtomicBool,
    killed: AtomicBool,
}

impl Registry {
    pub const fn new() -> Self {
        Self {
            ids: Mutex::new(BTreeSet::new()),
            markers: Mutex::new(BTreeSet::new()),
            interrupted: AtomicBool::new(false),
            killed: AtomicBool::new(false),
        }
    }

    /// The registry every agent run registers its process group in.
    pub fn global() -> &'static Self {
        &GLOBAL
    }

    /// Register group `id`, interrupting it straight away if the registry has
    /// been interrupted, or killing it if every group already was: a run can
    /// spawn its CLI after claudear was told to stop.
    pub(super) fn insert(&self, id: u32) {
        let mut ids = self.ids();
        if self.killed.load(Ordering::SeqCst) {
            Self::send(id, Signal::Kill);
            return;
        }
        ids.insert(id);
        if self.interrupted.load(Ordering::SeqCst) {
            Self::send(id, Signal::Interrupt);
        }
    }

    /// Kill group `id` unless [`Self::kill_all`] already did, so a group id the
    /// OS has since reused is never signalled.
    pub(super) fn kill(&self, id: u32) {
        if self.ids().remove(&id) {
            Self::send(id, Signal::Kill);
        }
    }

    /// Track `marker` until its run has swept the processes that carry it, so
    /// [`Self::kill_all`] still reaches them meanwhile.
    pub(super) fn track(&self, marker: Marker) {
        self.markers().insert(marker);
    }

    pub(super) fn release(&self, marker: Marker) {
        self.markers().remove(&marker);
    }

    /// Send every group, including any registered later, the SIGINT a terminal
    /// Ctrl-C would, so each CLI can clean up the shells it started in sessions
    /// of their own.
    pub fn interrupt_all(&self) {
        let ids = self.ids();
        self.interrupted.store(true, Ordering::SeqCst);
        for &id in ids.iter() {
            Self::send(id, Signal::Interrupt);
        }
    }

    /// Whether [`Self::interrupt_all`] has been called, after which nothing
    /// should start new agent runs.
    pub fn is_interrupted(&self) -> bool {
        self.interrupted.load(Ordering::SeqCst)
    }

    /// Kill every group, including any registered later, and every process
    /// that carries a tracked marker.
    pub fn kill_all(&self) {
        let ids = {
            let mut ids = self.ids();
            self.killed.store(true, Ordering::SeqCst);
            std::mem::take(&mut *ids)
        };
        for id in ids {
            Self::send(id, Signal::Kill);
        }
        let markers = std::mem::take(&mut *self.markers());
        for marker in markers {
            marker.kill();
        }
    }

    /// Interrupt every group, give their runs up to `grace` to end, then kill
    /// whatever is left.
    pub async fn shutdown(&self, grace: Duration) {
        self.interrupt_all();
        let _ = tokio::time::timeout(grace, self.emptied()).await;
        self.kill_all();
    }

    /// Resolves once every group has been killed and every marker swept, by its
    /// run or by [`Self::kill_all`].
    pub async fn emptied(&self) {
        while !self.is_empty() {
            tokio::time::sleep(EMPTIED_POLL_INTERVAL).await;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.ids().is_empty() && self.markers().is_empty()
    }

    fn ids(&self) -> MutexGuard<'_, BTreeSet<u32>> {
        self.ids.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn markers(&self) -> MutexGuard<'_, BTreeSet<Marker>> {
        self.markers.lock().unwrap_or_else(PoisonError::into_inner)
    }

    #[cfg(unix)]
    fn send(id: u32, signal: Signal) {
        // Group 0 would be claudear's own.
        let Ok(group @ 1..) = libc::pid_t::try_from(id) else {
            return;
        };
        // SAFETY: killpg takes no pointers and only sends a signal.
        if unsafe { libc::killpg(group, signal.number()) } == 0 {
            return;
        }
        let error = std::io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::ESRCH) => {}
            // macOS reports EPERM for a group whose members are all zombies.
            #[cfg(target_os = "macos")]
            Some(libc::EPERM) => tracing::debug!(
                component = "runner",
                process_group = id,
                "Agent process group has only zombies left"
            ),
            _ => tracing::warn!(
                component = "runner",
                process_group = id,
                signal = ?signal,
                error = %error,
                "Failed to signal agent process group"
            ),
        }
    }

    #[cfg(not(unix))]
    fn send(_id: u32, _signal: Signal) {}
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::runner::process_group::tests::{
        assert_interrupted, assert_killed, exits_within, group_command, is_running, spawn_group,
        BACKGROUND_SLEEP, EXIT_DEADLINE, INTERRUPTIBLE_LEADER,
    };
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    use crate::runner::process_group::tests::{
        assert_perl_environment_readable, background_pid, kill, stop_if_running, ESCAPED_SLEEP,
    };
    use crate::runner::process_group::Guard;

    const INTERRUPT_IGNORING_LEADER: &str = "trap '' INT; sleep 300 & echo $!; exec sleep 300";

    /// Like [`ESCAPED_SLEEP`], but the background process ignores SIGTERM.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    const TERMINATE_IGNORING_ESCAPED_SLEEP: &str = concat!(
        r#"perl -e 'use POSIX; $SIG{TERM} = "IGNORE"; $| = 1; setsid() or die; "#,
        r#"print "$$\n"; sleep 300' & exec sleep 300"#,
    );

    #[tokio::test]
    async fn test_kill_all_kills_every_registered_group() {
        let registry = Registry::new();
        let (mut first, first_background) = spawn_group(BACKGROUND_SLEEP).await;
        let (mut second, second_background) = spawn_group(BACKGROUND_SLEEP).await;
        registry.insert(first.id().unwrap());
        registry.insert(second.id().unwrap());

        registry.kill_all();

        for (leader, background) in [
            (&mut first, first_background),
            (&mut second, second_background),
        ] {
            assert_killed(leader).await;
            assert!(
                exits_within(background),
                "background process {background} survived kill_all"
            );
        }
    }

    #[tokio::test]
    async fn test_kill_all_empties_the_registry() {
        let registry = Registry::new();
        let (mut leader, background) = spawn_group(BACKGROUND_SLEEP).await;
        registry.insert(leader.id().unwrap());

        registry.kill_all();

        assert_killed(&mut leader).await;
        assert!(exits_within(background));
        assert!(
            registry.is_empty(),
            "a killed group must not be signalled again once the OS reuses its id"
        );
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[tokio::test]
    async fn test_kill_all_kills_marked_processes_of_every_run() {
        assert_perl_environment_readable();
        let registry = Registry::new();
        let (mut first, _first_guard) =
            Guard::spawn(&mut group_command(ESCAPED_SLEEP), &registry).unwrap();
        let (mut second, _second_guard) =
            Guard::spawn(&mut group_command(ESCAPED_SLEEP), &registry).unwrap();
        let escaped = [
            background_pid(&mut first).await,
            background_pid(&mut second).await,
        ];

        registry.kill_all();

        let survivors: Vec<u32> = escaped
            .into_iter()
            .filter(|&pid| !exits_within(pid))
            .collect();
        for &pid in &survivors {
            kill(pid);
        }
        assert!(
            survivors.is_empty(),
            "processes {survivors:?} that left their runs' groups survived kill_all"
        );
        assert_killed(&mut first).await;
        assert_killed(&mut second).await;
    }

    #[tokio::test]
    async fn test_kill_all_lets_shutdown_stop_waiting_on_guarded_runs() {
        let registry = Registry::new();
        let (mut leader, _guard) =
            Guard::spawn(&mut group_command(BACKGROUND_SLEEP), &registry).unwrap();

        registry.kill_all();

        tokio::time::timeout(EXIT_DEADLINE, registry.emptied())
            .await
            .expect("emptied must resolve once kill_all has swept every run");
        assert_killed(&mut leader).await;
    }

    #[tokio::test]
    async fn test_runs_spawned_after_kill_all_are_killed() {
        let registry = Registry::new();
        registry.kill_all();

        let (mut leader, _guard) =
            Guard::spawn(&mut group_command(BACKGROUND_SLEEP), &registry).unwrap();

        assert_killed(&mut leader).await;
    }

    #[tokio::test]
    async fn test_kill_signals_only_registered_groups() {
        let registry = Registry::new();
        let (mut leader, background) = spawn_group(BACKGROUND_SLEEP).await;
        let id = leader.id().unwrap();

        registry.kill(id);
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            leader.try_wait().unwrap().is_none(),
            "an unregistered group must not be signalled"
        );

        registry.insert(id);
        registry.kill(id);
        assert_killed(&mut leader).await;
        assert!(
            exits_within(background),
            "background process {background} survived kill"
        );
    }

    #[tokio::test]
    async fn test_interrupt_all_interrupts_without_forgetting_the_group() {
        let registry = Registry::new();
        let (mut leader, background) = spawn_group(INTERRUPTIBLE_LEADER).await;
        registry.insert(leader.id().unwrap());

        registry.interrupt_all();

        assert_interrupted(&mut leader).await;
        assert!(registry.is_interrupted());
        registry.kill_all();
        assert!(
            exits_within(background),
            "kill_all must still reach whatever ignored the interrupt"
        );
    }

    #[tokio::test]
    async fn test_groups_registered_after_interrupt_all_are_interrupted() {
        let registry = Registry::new();
        registry.interrupt_all();
        let (mut leader, background) = spawn_group(INTERRUPTIBLE_LEADER).await;

        registry.insert(leader.id().unwrap());

        assert_interrupted(&mut leader).await;
        registry.kill_all();
        assert!(exits_within(background));
    }

    #[tokio::test]
    async fn test_shutdown_kills_what_ignores_the_interrupt() {
        let registry = Registry::new();
        let (mut leader, background) = spawn_group(INTERRUPT_IGNORING_LEADER).await;
        registry.insert(leader.id().unwrap());

        registry.shutdown(Duration::from_millis(200)).await;

        assert_killed(&mut leader).await;
        assert!(exits_within(background));
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_shutdown_returns_once_interrupted_runs_end() {
        let registry = Registry::new();
        let (mut leader, background) = spawn_group(INTERRUPTIBLE_LEADER).await;
        let id = leader.id().unwrap();
        registry.insert(id);
        let run_ends_on_interrupt = async {
            assert_interrupted(&mut leader).await;
            registry.kill(id);
        };

        let shutdown = tokio::time::timeout(EXIT_DEADLINE, async {
            tokio::join!(registry.shutdown(2 * EXIT_DEADLINE), run_ends_on_interrupt)
        })
        .await;

        assert!(
            shutdown.is_ok(),
            "shutdown waited out its grace after the run ended"
        );
        assert!(exits_within(background));
    }

    #[tokio::test]
    async fn test_emptied_waits_for_every_group_to_be_killed() {
        let registry = Registry::new();
        let (mut leader, background) = spawn_group(BACKGROUND_SLEEP).await;
        let id = leader.id().unwrap();
        registry.insert(id);

        let early = tokio::time::timeout(Duration::from_millis(200), registry.emptied()).await;
        assert!(early.is_err(), "emptied resolved while a group was live");
        assert!(is_running(background));

        registry.kill(id);

        tokio::time::timeout(EXIT_DEADLINE, registry.emptied())
            .await
            .expect("emptied must resolve once the last group is killed");
        assert_killed(&mut leader).await;
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[tokio::test]
    async fn test_emptied_waits_for_a_run_still_sweeping() {
        assert_perl_environment_readable();
        let registry = Registry::new();
        let (mut leader, mut guard) = Guard::spawn(
            &mut group_command(TERMINATE_IGNORING_ESCAPED_SLEEP),
            &registry,
        )
        .unwrap();
        let escaped = background_pid(&mut leader).await;

        let ((), emptied_while_sweeping) = tokio::join!(guard.finish(), async {
            tokio::time::timeout(Duration::from_millis(500), registry.emptied())
                .await
                .is_ok()
        });

        let running = stop_if_running(escaped);
        assert!(
            !emptied_while_sweeping,
            "emptied resolved while a run was still sweeping the processes it left behind"
        );
        assert!(
            !running,
            "process {escaped} that ignores SIGTERM outlived the sweep's grace"
        );
        tokio::time::timeout(EXIT_DEADLINE, registry.emptied())
            .await
            .expect("emptied must resolve once the run's sweep finishes");
        assert_killed(&mut leader).await;
    }
}
