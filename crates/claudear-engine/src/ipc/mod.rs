//! Inter-process communication for the watcher daemon.
//!
//! Platform-specific transport details (Unix domain sockets vs TCP) live in
//! the [`transport`] module.  This file exposes the high-level helpers that the
//! rest of the crate uses for daemon lifecycle management.

mod client;
mod lock;
mod protocol;
mod server;
pub(crate) mod transport;

pub use client::{print_response, IpcClient};
pub use lock::Lock;
pub use protocol::{IpcCommand, IpcData, IpcResponse, WatcherState};
pub use server::IpcServer;

use std::path::{Path, PathBuf};

/// Returns a private runtime directory for IPC files, scoped to the current user.
///
/// * **Linux** — prefers `XDG_RUNTIME_DIR` (already user-private, mode 0700).
/// * **Windows** — uses `claudear` in the per-user local application data folder.
/// * **macOS / other** — creates `claudear-{uid}` under the system temp dir with
///   mode 0700.
fn ipc_runtime_dir() -> PathBuf {
    #[cfg(windows)]
    {
        // No fallback to %TEMP%: it can be shared between accounts, and the port file there
        // would hand the daemon secret to all of them.
        let dir = dirs::data_local_dir()
            .expect("Windows has a per-user local application data folder")
            .join("claudear");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::error!("Failed to create IPC runtime dir {:?}: {}", dir, e);
        }
        dir
    }

    #[cfg(not(windows))]
    {
        // On Linux, XDG_RUNTIME_DIR is already user-private
        if !cfg!(target_os = "macos") {
            if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
                return PathBuf::from(xdg);
            }
        }

        // Fallback: create a user-scoped subdirectory in the temp dir
        let uid = unsafe { libc::getuid() };
        let dir = std::env::temp_dir().join(format!("claudear-{}", uid));
        if !dir.exists() {
            if let Err(e) = std::fs::create_dir_all(&dir) {
                tracing::error!(
                    "Failed to create IPC runtime dir {:?}: {} — falling back to world-readable temp dir",
                    dir,
                    e
                );
                return std::env::temp_dir();
            }
            if let Err(e) = claudear_core::platform::set_dir_permissions_secure(&dir) {
                tracing::warn!("Failed to set IPC runtime dir permissions: {}", e);
            }
        }
        dir
    }
}

/// Default socket path for the IPC server (Unix) or port file path (Windows).
pub fn default_socket_path() -> PathBuf {
    #[cfg(windows)]
    {
        ipc_runtime_dir().join("claudear.port")
    }
    #[cfg(not(windows))]
    {
        ipc_runtime_dir().join("claudear.sock")
    }
}

/// Default PID file path.
pub fn default_pid_path() -> PathBuf {
    ipc_runtime_dir().join("claudear.pid")
}

/// Default path of the file the running `claudear start` daemon holds its [`Lock`] on.
pub fn default_lock_path() -> PathBuf {
    ipc_runtime_dir().join("claudear.lock")
}

/// Check if a watcher daemon is running.
pub fn is_daemon_running() -> bool {
    is_accepting(&default_socket_path())
}

/// Whether a listener is accepting connections on the socket at `socket_path`.
fn is_accepting(socket_path: &Path) -> bool {
    transport::check_connection(socket_path)
}

/// Read the PID stored in `pid_path`, if it holds one.
fn read_pid_file(pid_path: &Path) -> Option<u32> {
    std::fs::read_to_string(pid_path).ok()?.trim().parse().ok()
}

/// Write the current process PID to `pid_path`.
fn write_pid_file(pid_path: &Path) -> std::io::Result<()> {
    std::fs::write(pid_path, std::process::id().to_string())
}

/// A temporary directory for IPC endpoints in tests. On Unix it lives under `/tmp`, so socket
/// paths inside it stay within the Unix socket path limit however long `TMPDIR` is.
#[cfg(test)]
fn short_temporary_directory() -> tempfile::TempDir {
    let mut builder = tempfile::Builder::new();
    builder.prefix("claudear-ipc");
    #[cfg(unix)]
    let directory = builder.tempdir_in("/tmp");
    #[cfg(not(unix))]
    let directory = builder.tempdir();
    directory.expect("create a temporary directory for IPC endpoints")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn an_endpoint_round_trips_a_message_on_this_platform() {
        use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

        let dir = short_temporary_directory();
        let endpoint = dir.path().join("round-trip");

        let listener = transport::bind(&endpoint).expect("bind");
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            let (reader, mut writer) = stream.into_split();
            let mut reader = BufReader::new(reader);
            let mut line = String::new();
            reader.read_line(&mut line).await.expect("read");
            writer
                .write_all(format!("echo:{}", line.trim()).as_bytes())
                .await
                .expect("write");
        });

        let mut client = transport::connect(&endpoint).await.expect("connect");
        client.write_all(b"hello\n").await.expect("send");

        let mut echoed = String::new();
        client.read_to_string(&mut echoed).await.expect("receive");

        assert_eq!(echoed, "echo:hello");
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn an_endpoint_reports_whether_anyone_is_listening() {
        let dir = short_temporary_directory();
        let endpoint = dir.path().join("liveness");

        assert!(
            !transport::check_connection(&endpoint),
            "nothing exists yet"
        );

        let listener = transport::bind(&endpoint).expect("bind");

        assert!(
            transport::check_connection(&endpoint),
            "a bound endpoint answers"
        );

        drop(listener);

        assert!(
            !transport::check_connection(&endpoint),
            "a dropped listener stops answering"
        );
    }

    /// The secret is what stands in for a socket's owner-only mode, so an
    /// unrelated local process presenting the wrong one must get nowhere.
    #[test]
    fn a_wrong_secret_is_never_accepted() {
        let secret = "0123456789abcdef";

        assert!(transport::secret_matches(secret, secret));
        assert!(!transport::secret_matches("", secret));
        assert!(!transport::secret_matches("0123456789abcdee", secret));
        assert!(!transport::secret_matches("0123456789abcdef0", secret));
    }

    #[cfg(windows)]
    #[test]
    fn test_ipc_runtime_dir_is_in_the_per_user_local_application_data_folder() {
        let local_application_data = dirs::data_local_dir().expect("local application data");
        assert_eq!(
            ipc_runtime_dir(),
            local_application_data.join("claudear"),
            "the port file and its secret must never land in a directory other accounts can read"
        );
    }

    #[test]
    fn test_ipc_runtime_dir_is_absolute() {
        let dir = ipc_runtime_dir();
        assert!(
            dir.is_absolute(),
            "Expected ipc_runtime_dir to return an absolute path, got: {:?}",
            dir
        );
    }

    #[test]
    fn test_read_pid_file_ignores_missing_and_malformed_files() {
        let directory = short_temporary_directory();
        let pid_path = directory.path().join("claudear.pid");
        assert_eq!(read_pid_file(&pid_path), None);

        std::fs::write(&pid_path, "not a pid").unwrap();
        assert_eq!(read_pid_file(&pid_path), None);

        std::fs::write(&pid_path, "42\n").unwrap();
        assert_eq!(read_pid_file(&pid_path), Some(42));
    }

    #[test]
    fn test_write_pid_file_stores_own_pid() {
        let directory = short_temporary_directory();
        let pid_path = directory.path().join("claudear.pid");

        write_pid_file(&pid_path).expect("write_pid_file should succeed");

        assert_eq!(read_pid_file(&pid_path), Some(std::process::id()));
    }

    #[tokio::test]
    async fn test_is_accepting_only_while_a_listener_is_bound() {
        let directory = short_temporary_directory();
        let socket_path = directory.path().join("claudear.sock");
        assert!(!is_accepting(&socket_path));

        let listener = transport::bind(&socket_path).unwrap();
        assert!(is_accepting(&socket_path));

        drop(listener);
        assert!(socket_path.exists(), "dropping a listener leaves its file");
        assert!(!is_accepting(&socket_path));
    }
}
