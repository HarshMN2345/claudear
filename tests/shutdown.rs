#![cfg(all(unix, feature = "sqlite"))]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Generous, because CI runs the tests under `cargo tarpaulin`'s ptrace.
const DEADLINE: Duration = Duration::from_secs(60);

const POLL_INTERVAL: Duration = Duration::from_millis(50);

const RUNTIME_FILES: [&str; 2] = ["claudear.pid", "claudear.sock"];

/// Stands in for Jira: reports every request and leaves it unanswered unless
/// the test answers it, so whatever claudear asks Jira waits until then.
struct Jira {
    address: SocketAddr,
    requests: mpsc::Receiver<Request>,
}

impl Jira {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, requests) = mpsc::channel();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let mut head = [0; 1024];
                let length = stream.read(&mut head).unwrap_or(0);
                let head = String::from_utf8_lossy(&head[..length]).into_owned();
                if sender.send(Request { head, stream }).is_err() {
                    return;
                }
            }
        });
        Self { address, requests }
    }

    fn wait_for_request(&self, claudear: &mut Claudear) -> Request {
        let start = Instant::now();
        while start.elapsed() < DEADLINE {
            if let Ok(request) = self.requests.recv_timeout(POLL_INTERVAL) {
                return request;
            }
            claudear.assert_running();
        }
        panic!("claudear never asked Jira anything\n{}", claudear.output());
    }
}

/// A request to the [`Jira`] stub, whose connection stays open until answered.
struct Request {
    head: String,
    stream: TcpStream,
}

impl Request {
    fn answer_not_found(mut self) {
        let response = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        self.stream.write_all(response).unwrap();
    }
}

/// A claudear home whose only source is a [`Jira`] stub, with its IPC socket
/// and PID file kept apart from any claudear running on this machine.
struct Sandbox {
    directory: TempDir,
    jira: Jira,
}

impl Sandbox {
    fn new() -> Self {
        // The IPC socket lives under TMPDIR, and macOS caps socket paths at
        // 104 bytes, which its per-user temp dir can nearly use up alone.
        let directory = tempfile::Builder::new()
            .prefix("claudear")
            .tempdir_in("/tmp")
            .unwrap();
        for name in ["home", "logs", "runtime", "workspace"] {
            std::fs::create_dir(directory.path().join(name)).unwrap();
        }
        let jira = Jira::start();
        let root = directory.path().display();
        std::fs::write(
            directory.path().join("claudear.toml"),
            format!(
                r#"db_path = "{root}/claudear.db"
workspace = "{root}/workspace"

[issues.jira]
enabled = true
base_url = "http://{address}"
email = "claudear@example.com"
api_token = "token"
project_keys = ["TEST"]
"#,
                address = jira.address,
            ),
        )
        .unwrap();
        Self { directory, jira }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.directory.path().join(name)
    }

    fn command(&self, arguments: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_claudear"));
        command
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", self.path("home"))
            .env("TMPDIR", self.path("runtime"))
            .env("XDG_RUNTIME_DIR", self.path("runtime"))
            .arg("--config")
            .arg(self.path("claudear.toml"))
            .arg("--log-dir")
            .arg(self.path("logs"))
            .args(arguments)
            .stdin(Stdio::null());
        command
    }

    fn spawn(&self, arguments: &[&str]) -> Claudear {
        let output = self.path("output.log");
        let stdout = std::fs::File::create(&output).unwrap();
        let stderr = stdout.try_clone().unwrap();
        let child = self
            .command(arguments)
            .stdout(stdout)
            .stderr(stderr)
            .spawn()
            .unwrap();
        Claudear { child, output }
    }

    /// Start the daemon the way `claudear start` does by default: detached
    /// into a session of its own, with no controlling terminal.
    fn daemonize(&self, arguments: &[&str]) -> Daemon {
        let output = self.command(arguments).output().unwrap();
        assert!(output.status.success(), "claudear start failed: {output:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let id = stderr
            .lines()
            .find_map(|line| line.strip_prefix("Daemon started (PID: "))
            .and_then(|rest| rest.strip_suffix(')'))
            .and_then(|id| id.parse().ok())
            .unwrap_or_else(|| panic!("claudear start did not report its PID: {stderr}"));
        Daemon { id }
    }

    /// The daemon's PID file and IPC socket, found under
    /// `$XDG_RUNTIME_DIR` on Linux and `$TMPDIR/claudear-<uid>` on macOS.
    fn runtime_files(&self) -> Vec<PathBuf> {
        let runtime = self.path("runtime");
        let mut directories = vec![runtime.clone()];
        directories.extend(
            std::fs::read_dir(&runtime)
                .unwrap()
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.is_dir()),
        );
        directories
            .iter()
            .flat_map(|directory| RUNTIME_FILES.map(|name| directory.join(name)))
            .filter(|path| path.exists())
            .collect()
    }

    fn logs(&self) -> String {
        std::fs::read_dir(self.path("logs"))
            .unwrap()
            .flatten()
            .map(|entry| std::fs::read_to_string(entry.path()).unwrap_or_default())
            .collect()
    }
}

/// A claudear this test spawned, killed if the test fails before it exits.
struct Claudear {
    child: Child,
    output: PathBuf,
}

impl Claudear {
    fn output(&self) -> String {
        std::fs::read_to_string(&self.output).unwrap_or_default()
    }

    fn assert_running(&mut self) {
        if let Some(status) = self.child.try_wait().unwrap() {
            panic!("claudear exited early with {status}\n{}", self.output());
        }
    }

    fn wait_until_serving(&mut self, port: u16) {
        let start = Instant::now();
        while start.elapsed() < DEADLINE {
            self.assert_running();
            if serves_http(port) {
                return;
            }
            std::thread::sleep(POLL_INTERVAL);
        }
        panic!("claudear never served HTTP on {port}\n{}", self.output());
    }

    fn wait_for_output(&mut self, text: &str) {
        let start = Instant::now();
        while start.elapsed() < DEADLINE {
            if self.output().contains(text) {
                return;
            }
            self.assert_running();
            std::thread::sleep(POLL_INTERVAL);
        }
        panic!("claudear never printed {text:?}\n{}", self.output());
    }

    fn terminate(&self) {
        terminate(self.child.id());
    }

    fn wait(&mut self) -> ExitStatus {
        let start = Instant::now();
        while start.elapsed() < DEADLINE {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            std::thread::sleep(POLL_INTERVAL);
        }
        panic!("claudear outlived SIGTERM\n{}", self.output());
    }
}

impl Drop for Claudear {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A daemonized claudear, which is not this test's child.
struct Daemon {
    id: u32,
}

impl Daemon {
    fn wait_until_serving(&self, port: u16) {
        let start = Instant::now();
        while start.elapsed() < DEADLINE {
            assert!(is_running(self.id), "the daemon exited early");
            if serves_http(port) {
                return;
            }
            std::thread::sleep(POLL_INTERVAL);
        }
        panic!("the daemon never served HTTP on {port}");
    }

    fn terminate(&self) {
        terminate(self.id);
        let start = Instant::now();
        while start.elapsed() < DEADLINE {
            if !is_running(self.id) {
                return;
            }
            std::thread::sleep(POLL_INTERVAL);
        }
        panic!("the daemon outlived SIGTERM");
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        if is_running(self.id) {
            // SAFETY: kill takes no pointers and only sends a signal.
            unsafe { libc::kill(self.id as libc::pid_t, libc::SIGKILL) };
        }
    }
}

fn terminate(id: u32) {
    // SAFETY: kill takes no pointers and only sends a signal.
    let result = unsafe { libc::kill(id as libc::pid_t, libc::SIGTERM) };
    assert_eq!(result, 0, "{}", std::io::Error::last_os_error());
}

/// Zombies count as exited: an orphaned daemon is reaped by whoever adopts
/// it, which may never get to it (e.g. a container without an init process).
fn is_running(id: u32) -> bool {
    let output = Command::new("ps")
        .args(["-o", "stat=", "-p", &id.to_string()])
        .output()
        .unwrap();
    let stat = String::from_utf8_lossy(&output.stdout);
    let stat = stat.trim();
    !stat.is_empty() && !stat.starts_with('Z')
}

/// claudear answers HTTP only once its services run, by which point it has
/// installed its signal handlers.
fn serves_http(port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    stream.set_read_timeout(Some(DEADLINE)).unwrap();
    let request = b"GET /api/health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    if stream.write_all(request).is_err() {
        return false;
    }
    let mut response = [0; 5];
    stream.read_exact(&mut response).is_ok() && &response == b"HTTP/"
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn assert_drained(status: ExitStatus, claudear: &Claudear) {
    assert!(
        status.success(),
        "claudear exited with {status} instead of draining\n{}",
        claudear.output()
    );
}

#[test]
fn test_terminate_drains_start_in_the_foreground() {
    let sandbox = Sandbox::new();
    let port = free_port();
    let mut claudear = sandbox.spawn(&["start", "--foreground", "--port", &port.to_string()]);
    claudear.wait_until_serving(port);

    claudear.terminate();
    let status = claudear.wait();

    assert_drained(status, &claudear);
    assert!(
        sandbox.runtime_files().is_empty(),
        "claudear left {:?} behind",
        sandbox.runtime_files()
    );
}

#[test]
fn test_terminate_drains_the_daemon() {
    let sandbox = Sandbox::new();
    let port = free_port();
    let daemon = sandbox.daemonize(&["start", "--port", &port.to_string()]);
    daemon.wait_until_serving(port);

    daemon.terminate();

    assert!(
        sandbox.runtime_files().is_empty(),
        "the daemon left {:?} behind",
        sandbox.runtime_files()
    );
    let logs = sandbox.logs();
    assert!(
        logs.contains("Claude Watcher stopped gracefully"),
        "the daemon did not drain its watcher:\n{logs}"
    );
}

#[test]
fn test_terminate_drains_webhook_mode() {
    let sandbox = Sandbox::new();
    let port = free_port();
    let mut claudear = sandbox.spawn(&["webhook", &port.to_string()]);
    claudear.wait_until_serving(port);

    claudear.terminate();
    let status = claudear.wait();

    assert_drained(status, &claudear);
}

#[test]
fn test_terminate_drains_poll_mode() {
    let sandbox = Sandbox::new();
    let port = free_port();
    let mut claudear = sandbox.spawn(&["poll", "--port", &port.to_string()]);
    claudear.wait_until_serving(port);

    claudear.terminate();
    let status = claudear.wait();

    assert_drained(status, &claudear);
}

#[test]
fn test_terminate_interrupts_a_trigger() {
    let sandbox = Sandbox::new();
    let mut claudear = sandbox.spawn(&["trigger", "jira", "TEST-1"]);
    let request = sandbox.jira.wait_for_request(&mut claudear);
    assert!(
        request.head.contains("TEST-1"),
        "unexpected request: {}",
        request.head
    );

    claudear.terminate();
    claudear.wait_for_output("Interrupted, stopping agent runs");
    request.answer_not_found();
    let status = claudear.wait();

    assert_eq!(
        status.signal(),
        Some(libc::SIGINT),
        "trigger exited with {status} instead of force-quitting once its runs were interrupted\n{}",
        claudear.output()
    );
}

#[test]
fn test_runtime_files_are_found_while_the_daemon_runs() {
    let sandbox = Sandbox::new();
    let port = free_port();
    let mut claudear = sandbox.spawn(&["start", "--foreground", "--port", &port.to_string()]);
    claudear.wait_until_serving(port);

    let found: Vec<_> = sandbox
        .runtime_files()
        .iter()
        .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
        .map(str::to_owned)
        .collect();

    assert_eq!(
        found, RUNTIME_FILES,
        "the leftover checks would pass vacuously"
    );
}
