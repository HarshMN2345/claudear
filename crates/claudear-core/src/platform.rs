//! Platform abstraction layer.
//!
//! Centralises all OS-specific logic so that the rest of the codebase can stay
//! platform-agnostic.  Every function is a no-op or a sensible default on
//! platforms where the underlying primitive does not exist.

use std::path::Path;

/// Set restrictive **file** permissions (Unix `0o600` — owner read/write only).
///
/// On Windows this is a no-op; NTFS ACLs are inherited from the parent
/// directory and are typically sufficient for single-user machines.
pub fn set_file_permissions_secure(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

/// Set restrictive **directory** permissions (Unix `0o700` — owner only).
///
/// On Windows this is a no-op.
pub fn set_dir_permissions_secure(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

/// Check whether a command exists on the system PATH.
///
/// Uses `which` on Unix and `where` on Windows.
pub fn command_exists(binary: &str) -> bool {
    #[cfg(not(windows))]
    let cmd = "which";
    #[cfg(windows)]
    let cmd = "where";

    std::process::Command::new(cmd)
        .arg(binary)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_exists_known_binary() {
        #[cfg(not(windows))]
        assert!(command_exists("ls"));
        #[cfg(windows)]
        assert!(command_exists("cmd"));
    }

    #[test]
    fn test_command_exists_nonexistent() {
        assert!(!command_exists("__nonexistent_binary_12345__"));
    }

    /// The outcome that matters is that a secret written through these helpers
    /// is not readable by anyone else, however the platform expresses that.
    #[test]
    fn a_secured_file_is_not_readable_by_other_users() {
        let dir = tempfile::tempdir().expect("temp dir");
        let secret = dir.path().join("token");
        std::fs::write(&secret, "s3cret").expect("write");

        set_file_permissions_secure(&secret).expect("secure the file");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&secret)
                .expect("metadata")
                .permissions()
                .mode();
            assert_eq!(
                mode & 0o077,
                0,
                "group and other must hold no bits, got {:o}",
                mode
            );
        }

        assert_eq!(std::fs::read_to_string(&secret).expect("read"), "s3cret");
    }

    #[test]
    fn a_secured_directory_is_not_traversable_by_other_users() {
        let parent = tempfile::tempdir().expect("temp dir");
        let dir = parent.path().join("private");
        std::fs::create_dir(&dir).expect("create");

        set_dir_permissions_secure(&dir).expect("secure the dir");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&dir)
                .expect("metadata")
                .permissions()
                .mode();
            assert_eq!(
                mode & 0o077,
                0,
                "group and other must hold no bits, got {:o}",
                mode
            );
        }

        std::fs::write(dir.join("inside"), "ok").expect("the owner can still write");
    }
}
