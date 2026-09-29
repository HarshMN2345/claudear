use super::Signal;
use std::fs::{self, DirEntry};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::fs::MetadataExt;

const PROCESSES: &str = "/proc";

pub(super) fn find(entry: &[u8]) -> Vec<u32> {
    marked(entry).unwrap_or_default()
}

pub(super) fn signal(entry: &[u8], signal: Signal) -> usize {
    match marked(entry) {
        Ok(pids) => pids
            .into_iter()
            .filter(|&pid| deliver(pid, entry, signal))
            .count(),
        Err(error) => {
            tracing::warn!(
                component = "runner",
                signal = ?signal,
                error = %error,
                "Failed to list processes an agent run left behind"
            );
            0
        }
    }
}

fn marked(entry: &[u8]) -> io::Result<Vec<u32>> {
    let own = std::process::id();
    // SAFETY: geteuid takes no pointers and cannot fail.
    let user = unsafe { libc::geteuid() };
    let pids = fs::read_dir(PROCESSES)?
        .filter_map(Result::ok)
        .filter_map(|process| {
            let pid = pid(&process)?;
            let matched = pid != own && owned_by(&process, user) && carries(pid, entry);
            matched.then_some(pid)
        })
        .collect();
    Ok(pids)
}

fn pid(process: &DirEntry) -> Option<u32> {
    process.file_name().to_str()?.parse().ok()
}

fn owned_by(process: &DirEntry, user: libc::uid_t) -> bool {
    process
        .metadata()
        .is_ok_and(|metadata| metadata.uid() == user)
}

fn carries(pid: u32, entry: &[u8]) -> bool {
    fs::read(format!("{PROCESSES}/{pid}/environ")).is_ok_and(|environment| {
        environment
            .split(|&byte| byte == 0)
            .any(|variable| variable == entry)
    })
}

/// Signal `pid` if it carries `entry`, checked while holding a pidfd where one
/// is available so the signal cannot reach a process that took over the pid
/// after the check.
fn deliver(pid: u32, entry: &[u8], signal: Signal) -> bool {
    let Ok(target) = libc::pid_t::try_from(pid) else {
        return false;
    };
    let delivered = match open(target) {
        Ok(pidfd) => if_carries(pid, entry, || send(&pidfd, signal)).or_else(|error| {
            match error.raw_os_error() {
                Some(libc::ENOSYS | libc::EPERM) => fallback(target, entry, signal),
                _ => Err(error),
            }
        }),
        Err(error) if error.raw_os_error() == Some(libc::ESRCH) => Ok(false),
        Err(_) => fallback(target, entry, signal),
    };
    match delivered {
        Ok(delivered) => delivered,
        Err(error) if error.raw_os_error() == Some(libc::ESRCH) => false,
        Err(error) => {
            tracing::warn!(
                component = "runner",
                pid,
                signal = ?signal,
                error = %error,
                "Failed to signal a process an agent run left behind"
            );
            false
        }
    }
}

/// Signal `pid` if it carries `entry` without a pidfd, which leaves the pid
/// free to be reused between the check and the signal.
fn fallback(pid: libc::pid_t, entry: &[u8], signal: Signal) -> io::Result<bool> {
    if_carries(pid.cast_unsigned(), entry, || kill(pid, signal))
}

fn if_carries(pid: u32, entry: &[u8], sender: impl FnOnce() -> io::Result<()>) -> io::Result<bool> {
    if !carries(pid, entry) {
        return Ok(false);
    }
    sender()?;
    Ok(true)
}

fn open(pid: libc::pid_t) -> io::Result<OwnedFd> {
    // SAFETY: pidfd_open takes no pointers.
    let pidfd = checked(unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) })?;
    // SAFETY: pidfd_open returned a new descriptor, which nothing else owns.
    Ok(unsafe { OwnedFd::from_raw_fd(pidfd as RawFd) })
}

fn send(pidfd: &OwnedFd, signal: Signal) -> io::Result<()> {
    // SAFETY: the pidfd stays open for the call, and a null siginfo has the
    // kernel fill in what kill would.
    checked(unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            pidfd.as_raw_fd(),
            signal.number(),
            std::ptr::null::<libc::siginfo_t>(),
            0,
        )
    })?;
    Ok(())
}

fn kill(pid: libc::pid_t, signal: Signal) -> io::Result<()> {
    // SAFETY: kill takes no pointers and only sends a signal.
    let result = unsafe { libc::kill(pid, signal.number()) };
    checked(result.into())?;
    Ok(())
}

fn checked(result: libc::c_long) -> io::Result<libc::c_long> {
    if result == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{fallback, Signal};
    use crate::runner::process_group::marker::Marker;
    use crate::runner::process_group::tests::{
        assert_perl_environment_readable, exits_within, is_running, marked, perl, spawn, stop,
        SESSION_SLEEP,
    };

    #[test]
    fn test_fallback_kills_only_a_marked_process() {
        assert_perl_environment_readable();
        let marker = Marker::new();
        let entry = marker.entry();
        let (marked_child, [marked_pid]) = spawn(&mut marked(SESSION_SLEEP, marker));
        let (unmarked, [unmarked_pid]) = spawn(&mut perl(SESSION_SLEEP));

        for pid in [marked_pid, unmarked_pid] {
            fallback(pid.cast_signed(), entry.as_bytes(), Signal::Kill).unwrap();
        }
        let exited = exits_within(marked_pid);
        let running = is_running(unmarked_pid);
        stop(marked_child);
        stop(unmarked);

        assert!(exited, "marked process {marked_pid} outlived SIGKILL");
        assert!(running, "unmarked process {unmarked_pid} was killed");
    }
}
