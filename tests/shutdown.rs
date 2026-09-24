#![cfg(unix)]

use claudear::ipc::{IpcClient, IpcData, IpcResponse};
use claudear::shutdown;
use std::fs::{self, File};
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
const POLL_INTERVAL: Duration = Duration::from_millis(50);
const TAIL_LINES: usize = 40;
const UNREACHABLE_URL: &str = "http://127.0.0.1:1";
const DAEMON: &str = "daemon";
const STOP: &str = "stop";

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

    async fn start(&self) -> Child {
        let mut daemon = self
            .command(
                DAEMON,
                &[
                    "start",
                    "--foreground",
                    "--poll",
                    "--poll-interval",
                    "3600000",
                    "--no-webhooks",
                    "--no-dashboard",
                ],
            )
            .spawn()
            .expect("spawn the daemon");
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

    fn runtime_dir(&self) -> PathBuf {
        if cfg!(target_os = "macos") {
            // SAFETY: getuid has no preconditions and cannot fail.
            let uid = unsafe { libc::getuid() };
            self.path().join(format!("claudear-{uid}"))
        } else {
            self.path().to_path_buf()
        }
    }

    fn socket(&self) -> PathBuf {
        self.runtime_dir().join("claudear.sock")
    }

    fn pid_file(&self) -> PathBuf {
        self.runtime_dir().join("claudear.pid")
    }

    fn output(&self, name: &str) -> PathBuf {
        self.path().join(format!("{name}.out"))
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

    let status = IpcClient::with_socket_path(sandbox.socket())
        .status()
        .await
        .expect("the daemon answers a status request");

    assert!(
        matches!(&status, IpcResponse::Ok(IpcData::State(state)) if state.running),
        "unexpected status {status:?}"
    );
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
    let deadline = shutdown::EXIT_TIMEOUT + STOP_MARGIN;

    let stop = timeout(deadline, sandbox.command(STOP, &["stop"]).status())
        .await
        .unwrap_or_else(|_| {
            panic!(
                "claudear stop did not return within {}s\n{}",
                deadline.as_secs(),
                sandbox.diagnostics()
            )
        })
        .expect("run claudear stop");

    assert!(
        stop.success(),
        "claudear stop failed with {stop}\n{}",
        sandbox.diagnostics()
    );
    assert!(
        !is_alive(pid),
        "claudear stop returned before the daemon exited\n{}",
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
async fn sigterm_drains_and_exits_cleanly() {
    let sandbox = Sandbox::new();
    let mut daemon = sandbox.start().await;

    // SAFETY: kill only sends SIGTERM to the daemon this test spawned and still owns.
    let sent = unsafe { libc::kill(pid_of(&daemon), libc::SIGTERM) };
    assert_eq!(sent, 0, "send SIGTERM to the daemon");
    let status = timeout(EXIT_WAIT, daemon.wait())
        .await
        .unwrap_or_else(|_| {
            panic!(
                "the daemon did not exit within {}s of SIGTERM\n{}",
                EXIT_WAIT.as_secs(),
                sandbox.diagnostics()
            )
        })
        .expect("wait for the daemon");

    assert_exited_cleanly(status, &sandbox);
    sandbox.assert_files_removed();
}
