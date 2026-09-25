#![cfg(unix)]

mod sandbox;

use claudear::ipc::{IpcClient, IpcData, IpcResponse};
use claudear::shutdown;
use sandbox::Sandbox;
use std::fs;
use std::io;
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::ExitStatus;
use std::time::Duration;
use tokio::net::UnixStream;
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout, Instant};

const STARTUP_WAIT: Duration = Duration::from_secs(40);
const STOP_MARGIN: Duration = Duration::from_secs(4);
const EXIT_WAIT: Duration = Duration::from_secs(20);
const HANGUP_WAIT: Duration = Duration::from_secs(2);
const SUSPEND_WAIT: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(50);
const DAEMON_POLL_INTERVAL: Duration = Duration::from_secs(3600);
const STOP_SIGNALS: [libc::c_int; 3] = [libc::SIGINT, libc::SIGTERM, libc::SIGHUP];
const DAEMON: &str = "daemon";
const STOP: &str = "stop";
const STOPPED: &str = "Daemon stopped.";
const SOCKET_CLOSED: &str = "The daemon closed its control socket";
const FORCED_NOTICE: &str = "Shutdown forced";

impl Sandbox {
    /// The daemon's command, with the stop signals at their default disposition: the daemon
    /// keeps ignoring any it inherits ignored, and a test runner that a non-interactive shell
    /// starts in the background inherits SIGINT ignored.
    fn daemon(&self) -> Command {
        let poll_interval = DAEMON_POLL_INTERVAL.as_millis().to_string();
        let mut command = self.command(
            DAEMON,
            &[
                "start",
                "--foreground",
                "--poll",
                "--poll-interval",
                &poll_interval,
                "--no-webhooks",
                "--no-dashboard",
            ],
        );
        // SAFETY: the hook runs between fork and exec and only calls set_disposition, which is
        // async-signal-safe.
        unsafe { command.pre_exec(|| set_disposition(&STOP_SIGNALS, libc::SIG_DFL)) };
        command
    }

    async fn start(&self) -> Child {
        self.launch(self.daemon()).await
    }

    /// Starts the daemon with SIGHUP ignored, as `nohup` starts it.
    async fn start_under_nohup(&self) -> Child {
        let mut command = self.daemon();
        // SAFETY: the hook runs between fork and exec and only calls set_disposition, which is
        // async-signal-safe.
        unsafe { command.pre_exec(|| set_disposition(&[libc::SIGHUP], libc::SIG_IGN)) };
        self.launch(command).await
    }

    async fn launch(&self, mut command: Command) -> Child {
        let mut daemon = command.spawn().expect("spawn the daemon");
        self.wait_until_ready(&mut daemon).await;
        daemon
    }

    async fn wait_until_ready(&self, daemon: &mut Child) {
        let deadline = Instant::now() + STARTUP_WAIT;
        loop {
            if let Some(status) = daemon.try_wait().expect("check on the daemon") {
                panic!(
                    "the daemon exited during startup with {status}\n{}",
                    self.diagnostics()
                );
            }
            if UnixStream::connect(self.socket()).await.is_ok() {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "the daemon did not accept IPC connections within {}s\n{}",
                STARTUP_WAIT.as_secs(),
                self.diagnostics()
            );
            sleep(POLL_INTERVAL).await;
        }
    }

    async fn stop(&self) {
        let deadline = shutdown::EXIT_TIMEOUT + STOP_MARGIN;
        let stop = timeout(deadline, self.command(STOP, &["stop"]).status())
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "claudear stop did not return within {}s\n{}",
                    deadline.as_secs(),
                    self.diagnostics()
                )
            })
            .expect("run claudear stop");
        assert!(
            stop.success(),
            "claudear stop failed with {stop}\n{}",
            self.diagnostics()
        );
    }

    async fn stop_with(&self, daemon: &mut Child, signal: libc::c_int) -> ExitStatus {
        send(daemon, signal);
        self.wait_for_exit(daemon).await
    }

    async fn wait_for_exit(&self, daemon: &mut Child) -> ExitStatus {
        timeout(EXIT_WAIT, daemon.wait())
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "the daemon did not exit within {}s\n{}",
                    EXIT_WAIT.as_secs(),
                    self.diagnostics()
                )
            })
            .expect("wait for the daemon")
    }

    /// Sends `signals` while the daemon is suspended, so it receives them together and the
    /// drain the first one starts cannot finish before the next one arrives.
    async fn send_together(&self, daemon: &Child, signals: &[libc::c_int]) {
        send(daemon, libc::SIGSTOP);
        self.wait_until_suspended(daemon).await;
        for &signal in signals {
            send(daemon, signal);
        }
        send(daemon, libc::SIGCONT);
    }

    async fn wait_until_suspended(&self, daemon: &Child) {
        let pid = pid_of(daemon);
        let deadline = Instant::now() + SUSPEND_WAIT;
        loop {
            let mut status: libc::c_int = 0;
            // SAFETY: WNOHANG keeps waitpid from blocking, and WUNTRACED reports the stop
            // without reaping the daemon, which the test still owns.
            let changed =
                unsafe { libc::waitpid(pid, &mut status, libc::WUNTRACED | libc::WNOHANG) };
            if changed == pid && libc::WIFSTOPPED(status) {
                return;
            }
            assert_eq!(
                changed,
                0,
                "waitpid reported status {status} for the daemon\n{}",
                self.diagnostics()
            );
            assert!(
                Instant::now() < deadline,
                "the daemon was not suspended within {}s\n{}",
                SUSPEND_WAIT.as_secs(),
                self.diagnostics()
            );
            sleep(POLL_INTERVAL).await;
        }
    }

    async fn assert_running(&self) {
        let status = IpcClient::with_socket_path(self.socket())
            .status()
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "the daemon did not answer a status request: {error}\n{}",
                    self.diagnostics()
                )
            });
        assert!(
            matches!(&status, IpcResponse::Ok(IpcData::State(state)) if state.running),
            "unexpected status {status:?}\n{}",
            self.diagnostics()
        );
    }

    fn runtime_directory(&self) -> PathBuf {
        if cfg!(target_os = "macos") {
            // SAFETY: getuid has no preconditions and cannot fail.
            let uid = unsafe { libc::getuid() };
            self.path().join(format!("claudear-{uid}"))
        } else {
            self.path().to_path_buf()
        }
    }

    fn socket(&self) -> PathBuf {
        self.runtime_directory().join("claudear.sock")
    }

    fn pid_file(&self) -> PathBuf {
        self.runtime_directory().join("claudear.pid")
    }

    fn read(&self, name: &str) -> String {
        fs::read_to_string(self.output(name)).unwrap_or_default()
    }

    fn assert_files_removed(&self) {
        assert!(
            !self.socket().exists(),
            "the socket file was left behind\n{}",
            self.diagnostics()
        );
        assert!(
            !self.pid_file().exists(),
            "the PID file was left behind\n{}",
            self.diagnostics()
        );
    }

    fn log(&self) -> String {
        self.log_files()
            .map(|path| fs::read_to_string(path).unwrap_or_default())
            .collect()
    }
}

fn pid_of(daemon: &Child) -> libc::pid_t {
    let id = daemon.id().expect("the daemon has not been reaped");
    libc::pid_t::try_from(id).expect("the PID fits in pid_t")
}

fn is_alive(pid: libc::pid_t) -> bool {
    // SAFETY: signal 0 only checks that the process exists; nothing is delivered.
    unsafe { libc::kill(pid, 0) == 0 }
}

fn send(daemon: &Child, signal: libc::c_int) {
    // SAFETY: kill only sends `signal` to the daemon this test spawned and still owns.
    let sent = unsafe { libc::kill(pid_of(daemon), signal) };
    assert_eq!(
        sent,
        0,
        "send signal {signal} to the daemon: {}",
        io::Error::last_os_error()
    );
}

fn set_disposition(signals: &[libc::c_int], disposition: libc::sighandler_t) -> io::Result<()> {
    for &signal in signals {
        // SAFETY: callers pass SIG_IGN or SIG_DFL, so no handler is installed, and signal is
        // async-signal-safe.
        if unsafe { libc::signal(signal, disposition) } == libc::SIG_ERR {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

fn assert_exited_cleanly(status: ExitStatus, sandbox: &Sandbox) {
    assert_eq!(
        status.code(),
        Some(0),
        "the daemon exited with {status}\n{}",
        sandbox.diagnostics()
    );
}

#[tokio::test]
async fn daemon_without_an_http_server_keeps_running() {
    let sandbox = Sandbox::new();
    let mut daemon = sandbox.start().await;

    sandbox.assert_running().await;
    assert!(
        daemon.try_wait().expect("check on the daemon").is_none(),
        "the daemon exited after startup\n{}",
        sandbox.diagnostics()
    );
}

#[tokio::test]
async fn stop_returns_once_the_daemon_has_exited() {
    let sandbox = Sandbox::new();
    let mut daemon = sandbox.start().await;
    let pid = pid_of(&daemon);
    let exited = tokio::spawn(async move { daemon.wait().await });

    sandbox.stop().await;

    assert!(
        !is_alive(pid),
        "claudear stop returned before the daemon exited\n{}",
        sandbox.diagnostics()
    );
    assert!(
        sandbox.read(STOP).contains(STOPPED),
        "claudear stop did not report the exit\n{}",
        sandbox.diagnostics()
    );
    let status = timeout(EXIT_WAIT, exited)
        .await
        .expect("the daemon was reaped")
        .expect("the reaper task finished")
        .expect("wait for the daemon");
    assert_exited_cleanly(status, &sandbox);
    sandbox.assert_files_removed();
}

#[tokio::test]
async fn stop_without_the_pid_file_reports_only_the_closed_socket() {
    let sandbox = Sandbox::new();
    let mut daemon = sandbox.start().await;
    fs::remove_file(sandbox.pid_file()).expect("remove the PID file");

    sandbox.stop().await;

    let output = sandbox.read(STOP);
    assert!(
        !output.contains(STOPPED),
        "claudear stop claimed the daemon had exited without knowing its PID\n{}",
        sandbox.diagnostics()
    );
    assert!(
        output.contains(SOCKET_CLOSED),
        "claudear stop did not report the closed socket\n{}",
        sandbox.diagnostics()
    );
    let status = sandbox.wait_for_exit(&mut daemon).await;
    assert_exited_cleanly(status, &sandbox);
    sandbox.assert_files_removed();
}

#[tokio::test]
async fn sigterm_drains_and_exits_cleanly() {
    let sandbox = Sandbox::new();
    let mut daemon = sandbox.start().await;

    let status = sandbox.stop_with(&mut daemon, libc::SIGTERM).await;

    assert_exited_cleanly(status, &sandbox);
    sandbox.assert_files_removed();
}

#[tokio::test]
async fn sigint_drains_and_exits_cleanly() {
    let sandbox = Sandbox::new();
    let mut daemon = sandbox.start().await;

    let status = sandbox.stop_with(&mut daemon, libc::SIGINT).await;

    assert_exited_cleanly(status, &sandbox);
    sandbox.assert_files_removed();
}

#[tokio::test]
async fn sighup_drains_and_exits_cleanly() {
    let sandbox = Sandbox::new();
    let mut daemon = sandbox.start().await;

    let status = sandbox.stop_with(&mut daemon, libc::SIGHUP).await;

    assert_exited_cleanly(status, &sandbox);
    sandbox.assert_files_removed();
}

#[tokio::test]
async fn sighup_under_nohup_leaves_the_daemon_running() {
    let sandbox = Sandbox::new();
    let mut daemon = sandbox.start_under_nohup().await;

    send(&daemon, libc::SIGHUP);

    if let Ok(status) = timeout(HANGUP_WAIT, daemon.wait()).await {
        panic!(
            "the daemon exited after an ignored SIGHUP with {status:?}\n{}",
            sandbox.diagnostics()
        );
    }
    sandbox.assert_running().await;
    let status = sandbox.stop_with(&mut daemon, libc::SIGTERM).await;
    assert_exited_cleanly(status, &sandbox);
    sandbox.assert_files_removed();
}

#[tokio::test]
async fn second_signal_forces_the_exit_after_flushing_the_log() {
    let sandbox = Sandbox::new();
    let mut daemon = sandbox.start().await;

    sandbox
        .send_together(&daemon, &[libc::SIGINT, libc::SIGTERM])
        .await;
    let status = sandbox.wait_for_exit(&mut daemon).await;

    assert_eq!(
        status.signal(),
        Some(libc::SIGINT),
        "a forced daemon must die of SIGINT, so a calling shell stops too, but it exited with {status}\n{}",
        sandbox.diagnostics()
    );
    assert!(
        sandbox.log().contains(FORCED_NOTICE),
        "the forced shutdown never reached the log file\n{}",
        sandbox.diagnostics()
    );
    sandbox.assert_files_removed();
}
