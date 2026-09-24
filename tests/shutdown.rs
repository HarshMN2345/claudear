#![cfg(unix)]

use claudear::ipc::{IpcClient, IpcData, IpcResponse};
use claudear::shutdown;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::time::Duration;
use tempfile::TempDir;
use tokio::net::UnixStream;
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout, Instant};

const STARTUP_WAIT: Duration = Duration::from_secs(40);
const STOP_MARGIN: Duration = Duration::from_secs(4);
const EXIT_WAIT: Duration = Duration::from_secs(20);
const HANGUP_WAIT: Duration = Duration::from_secs(2);
const POLL_INTERVAL: Duration = Duration::from_millis(50);
const DAEMON_POLL_INTERVAL: Duration = Duration::from_secs(3600);
const TAIL_LINES: usize = 40;
const UNREACHABLE_URL: &str = "http://127.0.0.1:1";
const DAEMON: &str = "daemon";
const STOP: &str = "stop";
const STOPPED: &str = "Daemon stopped.";
const SOCKET_CLOSED: &str = "The daemon closed its control socket";

struct Sandbox {
    root: TempDir,
}

impl Sandbox {
    fn new() -> Self {
        let root = tempfile::Builder::new()
            .prefix("claudear")
            .tempdir_in("/tmp")
            .expect("create a sandbox under /tmp");
        fs::write(root.path().join("claudear.toml"), config(root.path()))
            .expect("write the sandbox config");
        Self { root }
    }

    fn path(&self) -> &Path {
        self.root.path()
    }

    fn command(&self, name: &str, args: &[&str]) -> Command {
        let output = File::create(self.output(name)).expect("create the output file");
        let mut command = Command::new(env!("CARGO_BIN_EXE_claudear"));
        command
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", self.path())
            .env("TMPDIR", self.path())
            .env("XDG_RUNTIME_DIR", self.path())
            .current_dir(self.path())
            .arg("--config")
            .arg(self.path().join("claudear.toml"))
            .arg("--log-dir")
            .arg(self.path().join("logs"))
            .args(args)
            .stdin(Stdio::null())
            .stdout(output.try_clone().expect("share the output file"))
            .stderr(output)
            .kill_on_drop(true);
        command
    }

    fn daemon(&self) -> Command {
        let poll_interval = DAEMON_POLL_INTERVAL.as_millis().to_string();
        self.command(
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
        )
    }

    async fn start(&self) -> Child {
        self.launch(self.daemon()).await
    }

    /// Starts the daemon with `disposition` for SIGHUP, which it inherits like the SIG_IGN that
    /// `nohup` sets.
    async fn start_with_hangup(&self, disposition: libc::sighandler_t) -> Child {
        let mut command = self.daemon();
        // SAFETY: the hook runs between fork and exec and only calls set_hangup, which is
        // async-signal-safe.
        unsafe { command.pre_exec(move || set_hangup(disposition)) };
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

    fn output(&self, name: &str) -> PathBuf {
        self.path().join(format!("{name}.out"))
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

    fn diagnostics(&self) -> String {
        let logs = fs::read_dir(self.path().join("logs"))
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path());
        [self.output(DAEMON), self.output(STOP)]
            .into_iter()
            .chain(logs)
            .filter(|path| path.is_file())
            .map(|path| format!("{}:\n{}", path.display(), tail(&path)))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn config(root: &Path) -> String {
    let root = root.display();
    format!(
        r#"workspace = "{root}/workspace"
db_path = "{root}/claudear.db"
storage_dir = "{root}/storage"
known_orgs = []
auto_discover_paths = []

[code_index]
enabled = false

[regression]
enabled = false

[issues.jira]
enabled = true
base_url = "{UNREACHABLE_URL}"
email = "shutdown@example.com"
api_token = "unused"
project_keys = ["SHUTDOWN"]
"#
    )
}

fn tail(path: &Path) -> String {
    let content = fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = content.lines().collect();
    lines[lines.len().saturating_sub(TAIL_LINES)..].join("\n")
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

fn set_hangup(disposition: libc::sighandler_t) -> io::Result<()> {
    // SAFETY: callers pass SIG_IGN or SIG_DFL, so no handler is installed, and signal is
    // async-signal-safe.
    if unsafe { libc::signal(libc::SIGHUP, disposition) } == libc::SIG_ERR {
        return Err(io::Error::last_os_error());
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
    let mut daemon = sandbox.start_with_hangup(libc::SIG_DFL).await;

    let status = sandbox.stop_with(&mut daemon, libc::SIGHUP).await;

    assert_exited_cleanly(status, &sandbox);
    sandbox.assert_files_removed();
}

#[tokio::test]
async fn sighup_under_nohup_leaves_the_daemon_running() {
    let sandbox = Sandbox::new();
    let mut daemon = sandbox.start_with_hangup(libc::SIG_IGN).await;

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
