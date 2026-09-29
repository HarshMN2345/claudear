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
        stop, stop_if_running, ESCAPED_SLEEP, EXIT_DEADLINE, SESSION_SLEEP,
    };
    use std::os::unix::process::ExitStatusExt;
    use std::process::{Child, Command, ExitStatus, Stdio};

    const SLEEP: &str = r#"$| = 1; print "$$\n"; sleep 300"#;

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

    /// Wait for `child`, killing it if it outlives [`EXIT_DEADLINE`] so a
    /// failing test leaves nothing running.
    fn exit_status(mut child: Child) -> ExitStatus {
        let exited = exits_within(child.id());
        if !exited {
            let _ = child.kill();
        }
        let status = child.wait().unwrap();
        assert!(exited, "process {} outlived the sweep", child.id());
        status
    }

    #[test]
    fn test_find_includes_a_marked_process_in_a_session_of_its_own() {
        assert_perl_environment_readable();
        let marker = Marker::new();
        let (shell, [escaped]) = spawn(
            Command::new("sh")
                .args(["-c", ESCAPED_SLEEP])
                .env(Marker::VARIABLE, marker.to_string())
                .stdout(Stdio::piped()),
        );

        let found = marker.find();
        kill(escaped);
        stop(shell);

        assert!(
            found.contains(&escaped),
            "process {escaped} carries the marker but was not found in {found:?}"
        );
    }

    #[test]
    fn test_find_ignores_unmarked_and_near_miss_processes() {
        assert_perl_environment_readable();
        let marker = Marker::new();
        let (unmarked, [_]) = spawn(&mut perl(SLEEP));
        let (suffixed, [_]) = spawn(perl(SLEEP).env(Marker::VARIABLE, format!("{marker}x")));
        let (other, [_]) = spawn(&mut marked(SLEEP, Marker::new()));
        let (prefixed, [_]) =
            spawn(perl(SLEEP).env(format!("X{}", Marker::VARIABLE), marker.to_string()));
        let (exact, [pid]) = spawn(&mut marked(SLEEP, marker));

        let found = marker.find();
        for child in [unmarked, suffixed, other, prefixed, exact] {
            stop(child);
        }

        assert_eq!(found, [pid], "only the exact entry may match");
    }

    #[test]
    fn test_find_ignores_a_marked_zombie() {
        assert_perl_environment_readable();
        let marker = Marker::new();
        let (mut child, [pid]) = spawn(&mut marked(SLEEP, marker));
        let alive = marker.find();

        child.kill().unwrap();
        let exited = exits_within(pid);
        let dead = marker.find();
        child.wait().unwrap();

        assert!(alive.contains(&pid), "process {pid} must be found alive");
        assert!(exited, "process {pid} outlived SIGKILL");
        assert!(
            !dead.contains(&pid),
            "zombie {pid} must not be found, or terminate waits out its grace on it"
        );
    }

    #[test]
    fn test_terminate_ends_a_process_that_honours_sigterm() {
        assert_perl_environment_readable();
        let marker = Marker::new();
        let (child, [_]) = spawn(&mut marked(SESSION_SLEEP, marker));

        let start = Instant::now();
        marker.terminate(EXIT_DEADLINE);
        let elapsed = start.elapsed();

        let status = exit_status(child);
        assert_eq!(
            status.signal(),
            Some(libc::SIGTERM),
            "the process exited with {status} instead of on SIGTERM"
        );
        assert!(
            elapsed < EXIT_DEADLINE / 2,
            "terminate took {elapsed:?} although the process exited on SIGTERM"
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

        let status = exit_status(child);
        assert_eq!(
            status.signal(),
            Some(libc::SIGKILL),
            "the process exited with {status} instead of being killed"
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

        let status = exit_status(child);
        assert_eq!(
            status.signal(),
            Some(libc::SIGKILL),
            "the process exited with {status} instead of being killed"
        );
    }

    #[test]
    fn test_kill_spares_unmarked_processes() {
        assert_perl_environment_readable();
        let marker = Marker::new();
        let (unmarked, [unmarked_pid]) = spawn(&mut perl(SLEEP));
        let (session, [leader, sibling]) =
            spawn(marked(SESSION_SLEEP_WITH_UNMARKED_CHILD, marker).arg(Marker::VARIABLE));
        let found = marker.find();
        if found != [leader] {
            kill(sibling);
            stop(unmarked);
            stop(session);
            panic!("only process {leader} carries the marker, but {found:?} were found");
        }

        marker.kill();

        let status = exit_status(session);
        std::thread::sleep(Duration::from_millis(200));
        let unmarked_running = is_running(unmarked_pid);
        let sibling_running = stop_if_running(sibling);
        stop(unmarked);

        assert_eq!(
            status.signal(),
            Some(libc::SIGKILL),
            "the marked process exited with {status} instead of being killed"
        );
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
