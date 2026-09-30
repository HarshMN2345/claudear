use super::signal::Signal;
use std::fmt;
use std::time::{Duration, Instant};
use tokio::process::Command;
use uuid::Uuid;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "linux")]
use linux as platform;
#[cfg(target_os = "macos")]
use macos as platform;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    use super::Signal;

    pub(super) fn find(_entry: &[u8]) -> Vec<u32> {
        Vec::new()
    }

    pub(super) fn signal(_entry: &[u8], _signal: Signal) -> usize {
        0
    }
}

const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Tags the environment of an agent run's CLI, which every process it starts
/// inherits, so the run can find those that left its process group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Marker(Uuid);

impl fmt::Display for Marker {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0.hyphenated())
    }
}

impl Marker {
    pub(super) const VARIABLE: &'static str = "CLAUDEAR_RUN";

    pub(super) fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub(super) fn apply(self, command: &mut Command) {
        command.env(Self::VARIABLE, self.to_string());
    }

    /// Processes of claudear's effective user, other than claudear itself,
    /// whose own environment holds this marker's exact entry.
    pub(super) fn find(self) -> Vec<u32> {
        platform::find(self.entry().as_bytes())
    }

    /// SIGTERM every marked process, give them up to `grace` to exit, then
    /// kill whatever is left. Blocks until done.
    pub(super) fn terminate(self, grace: Duration) {
        self.send(Signal::Terminate);
        let start = Instant::now();
        while start.elapsed() < grace && !self.find().is_empty() {
            std::thread::sleep(POLL_INTERVAL);
        }
        self.kill();
    }

    /// SIGKILL every marked process, twice to catch children forked during the
    /// first pass.
    pub(super) fn kill(self) {
        self.send(Signal::Kill);
        self.send(Signal::Kill);
    }

    fn send(self, signal: Signal) {
        let processes = platform::signal(self.entry().as_bytes(), signal);
        if processes > 0 {
            tracing::debug!(
                component = "runner",
                marker = %self,
                signal = ?signal,
                processes,
                "Signalled processes an agent run left outside its process group"
            );
        }
    }

    fn entry(self) -> String {
        format!("{}={self}", Self::VARIABLE)
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests {
    use super::*;
    use crate::runner::process_group::tests::{
        assert_perl_environment_readable, exits_within, is_running, kill, marked, perl, spawn,
        stop, stop_if_running, ESCAPED_SLEEP, EXIT_DEADLINE,
    };
    use std::process::{Child, Command, Stdio};

    const SLEEP: &str = r#"$| = 1; print "$$\n"; sleep 300"#;

    /// Moves to a session of its own and prints its pid; on SIGTERM it creates
    /// the file named by its first argument before it exits.
    const CLEANING_SESSION_SLEEP: &str = concat!(
        r#"use POSIX; $SIG{TERM} = sub { open(my $f, ">", $ARGV[0]) or die; exit 0 }; "#,
        r#"$| = 1; setsid() or die; print "$$\n"; sleep 300"#,
    );

    const TERMINATE_IGNORING_SESSION_SLEEP: &str = concat!(
        r#"use POSIX; $SIG{TERM} = "IGNORE"; $| = 1; setsid() or die; "#,
        r#"print "$$\n"; sleep 300"#,
    );

    /// Prints its own pid, then that of a child in its session which drops the
    /// variable named by the first argument before it execs and prints.
    const SESSION_SLEEP_WITH_UNMARKED_CHILD: &str = concat!(
        r#"use POSIX; $| = 1; setsid() or die; print "$$\n"; "#,
        r#"defined(my $pid = fork) or die; "#,
        r#"if (!$pid) { delete $ENV{$ARGV[0]}; exec $^X, "-e", '$| = 1; print "$$\n"; sleep 300' } "#,
        r#"sleep 300"#,
    );

    const GRACE: Duration = Duration::from_millis(500);

    /// Whether `child` stops within [`EXIT_DEADLINE`], killing it otherwise so
    /// a failing test leaves nothing running.
    fn stops(mut child: Child) -> bool {
        let stopped = exits_within(child.id());
        if !stopped {
            let _ = child.kill();
        }
        child.wait().unwrap();
        stopped
    }

    #[test]
    fn test_kill_ends_a_marked_process_in_a_session_of_its_own() {
        assert_perl_environment_readable();
        let marker = Marker::new();
        let (shell, [escaped]) = spawn(
            Command::new("sh")
                .args(["-c", ESCAPED_SLEEP])
                .env(Marker::VARIABLE, marker.to_string())
                .stdout(Stdio::piped()),
        );

        marker.kill();

        let exited = exits_within(escaped);
        if !exited {
            kill(escaped);
        }
        stop(shell);
        assert!(
            exited,
            "process {escaped} carries the marker in a session of its own but outlived kill"
        );
    }

    #[test]
    fn test_kill_spares_near_misses_of_the_marker() {
        assert_perl_environment_readable();
        let marker = Marker::new();
        let near_misses: [(Child, [u32; 1]); 5] = [
            spawn(&mut perl(SLEEP)),
            spawn(perl(SLEEP).env(Marker::VARIABLE, format!("{marker}x"))),
            spawn(&mut marked(SLEEP, Marker::new())),
            spawn(perl(SLEEP).env(format!("X{}", Marker::VARIABLE), marker.to_string())),
            spawn(perl(SLEEP).arg(format!("{}={marker}", Marker::VARIABLE))),
        ];
        let (exact, [_]) = spawn(marked(SLEEP, marker).arg(""));

        marker.kill();

        let stopped = stops(exact);
        std::thread::sleep(Duration::from_millis(200));
        let killed: Vec<u32> = near_misses
            .into_iter()
            .filter_map(|(mut child, [pid])| {
                let running = child.try_wait().unwrap().is_none();
                stop(child);
                (!running).then_some(pid)
            })
            .collect();
        assert!(
            stopped,
            "the process carrying the exact entry outlived kill"
        );
        assert!(
            killed.is_empty(),
            "processes {killed:?} without the exact entry in their environment were killed"
        );
    }

    #[test]
    fn test_terminate_does_not_wait_out_its_grace_on_a_zombie() {
        assert_perl_environment_readable();
        let marker = Marker::new();
        let (mut child, [pid]) = spawn(&mut marked(SLEEP, marker));
        child.kill().unwrap();
        let exited = exits_within(pid);

        let start = Instant::now();
        marker.terminate(EXIT_DEADLINE);
        let elapsed = start.elapsed();
        child.wait().unwrap();

        assert!(exited, "process {pid} outlived SIGKILL");
        assert!(
            elapsed < EXIT_DEADLINE / 2,
            "terminate took {elapsed:?} waiting on zombie {pid}, which nothing may reap"
        );
    }

    #[test]
    fn test_terminate_lets_a_process_clean_up_before_it_stops() {
        assert_perl_environment_readable();
        let directory = tempfile::tempdir().unwrap();
        let cleaned = directory.path().join("cleaned");
        let marker = Marker::new();
        let (child, [_]) = spawn(marked(CLEANING_SESSION_SLEEP, marker).arg(&cleaned));

        let start = Instant::now();
        marker.terminate(EXIT_DEADLINE);
        let elapsed = start.elapsed();

        assert!(stops(child), "the process outlived terminate");
        assert!(
            cleaned.exists(),
            "the process was stopped before it could clean up on SIGTERM"
        );
        assert!(
            elapsed < EXIT_DEADLINE / 2,
            "terminate took {elapsed:?} although the process stopped on SIGTERM"
        );
    }

    #[test]
    fn test_terminate_kills_a_process_that_ignores_sigterm_after_the_grace() {
        assert_perl_environment_readable();
        let marker = Marker::new();
        let (child, [_]) = spawn(&mut marked(TERMINATE_IGNORING_SESSION_SLEEP, marker));

        let start = Instant::now();
        marker.terminate(GRACE);
        let elapsed = start.elapsed();

        assert!(
            stops(child),
            "a process that ignores SIGTERM outlived terminate"
        );
        assert!(
            elapsed >= GRACE,
            "terminate killed after {elapsed:?}, before the grace ran out"
        );
    }

    #[test]
    fn test_kill_ends_a_marked_process_in_claudears_own_session() {
        assert_perl_environment_readable();
        let marker = Marker::new();
        let (child, [_]) = spawn(&mut marked(SLEEP, marker));

        marker.kill();

        assert!(
            stops(child),
            "a marked process in claudear's session outlived kill"
        );
    }

    #[test]
    fn test_kill_spares_unmarked_processes() {
        assert_perl_environment_readable();
        let marker = Marker::new();
        let (unmarked, [unmarked_pid]) = spawn(&mut perl(SLEEP));
        let (session, [_, sibling]) =
            spawn(marked(SESSION_SLEEP_WITH_UNMARKED_CHILD, marker).arg(Marker::VARIABLE));

        marker.kill();

        let stopped = stops(session);
        std::thread::sleep(Duration::from_millis(200));
        let unmarked_running = is_running(unmarked_pid);
        let sibling_running = stop_if_running(sibling);
        stop(unmarked);

        assert!(stopped, "the marked process outlived kill");
        assert!(
            unmarked_running,
            "unmarked process {unmarked_pid} in claudear's session was killed"
        );
        assert!(
            sibling_running,
            "unmarked process {sibling} in a marked process's session was killed"
        );
    }
}
