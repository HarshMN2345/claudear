use super::Signal;
use std::io;
use std::ptr;

const PROC_UID_ONLY: u32 = 4;

const SLACK: usize = 64;

pub(super) fn find(entry: &[u8]) -> Vec<u32> {
    let Some(mut arguments) = argument_buffer() else {
        return Vec::new();
    };
    pids()
        .into_iter()
        .filter(|&pid| carries(pid, entry, &mut arguments))
        .map(libc::pid_t::cast_unsigned)
        .collect()
}

pub(super) fn signal(entry: &[u8], signal: Signal) -> usize {
    let Some(mut arguments) = argument_buffer() else {
        return 0;
    };
    pids()
        .into_iter()
        .filter(|&pid| carries(pid, entry, &mut arguments) && send(pid, signal))
        .count()
}

/// Every process of claudear's effective user, other than claudear itself.
fn pids() -> Vec<libc::pid_t> {
    let mut pids = list().unwrap_or_else(|error| {
        tracing::warn!(
            component = "runner",
            error = %error,
            "Failed to list processes to find those an agent run left behind"
        );
        Vec::new()
    });
    let own = std::process::id();
    pids.retain(|&pid| pid > 0 && pid.cast_unsigned() != own);
    pids
}

fn list() -> io::Result<Vec<libc::pid_t>> {
    // SAFETY: a null buffer only asks proc_listpids for the byte size it needs.
    let estimate = unsafe { libc::proc_listpids(PROC_UID_ONLY, user(), ptr::null_mut(), 0) };
    let mut pids = vec![0; listed(estimate)? + SLACK];
    loop {
        let filled = fill(&mut pids)?;
        if filled < pids.len() {
            pids.truncate(filled);
            return Ok(pids);
        }
        pids.resize(pids.len() * 2, 0);
    }
}

/// Fill `pids` with those of claudear's effective user, returning how many it
/// holds.
fn fill(pids: &mut [libc::pid_t]) -> io::Result<usize> {
    let bytes = libc::c_int::try_from(size_of_val(pids)).map_err(io::Error::other)?;
    // SAFETY: `pids` is valid for `bytes` writable bytes, and proc_listpids
    // writes at most that many.
    let written =
        unsafe { libc::proc_listpids(PROC_UID_ONLY, user(), pids.as_mut_ptr().cast(), bytes) };
    listed(written)
}

/// The number of pids in `bytes` returned by proc_listpids, which returns 0 on
/// failure: listing by claudear's own user always includes claudear.
fn listed(bytes: libc::c_int) -> io::Result<usize> {
    match usize::try_from(bytes) {
        Ok(bytes @ 1..) => Ok(bytes / size_of::<libc::pid_t>()),
        _ => Err(io::Error::last_os_error()),
    }
}

fn user() -> libc::uid_t {
    // SAFETY: geteuid takes no arguments and cannot fail.
    unsafe { libc::geteuid() }
}

/// A buffer that fits the arguments and environment of any process.
fn argument_buffer() -> Option<Vec<u8>> {
    let mut limit = [0; size_of::<libc::c_int>()];
    if let Err(error) = query(&mut [libc::CTL_KERN, libc::KERN_ARGMAX], &mut limit) {
        tracing::warn!(
            component = "runner",
            error = %error,
            "Failed to size process arguments to find those an agent run left behind"
        );
        return None;
    }
    let limit = usize::try_from(libc::c_int::from_ne_bytes(limit)).ok()?;
    Some(vec![0; limit])
}

/// Whether the environment of process `pid` holds `entry`. Reading it fails for
/// zombies, processes that are gone and those of other users, so they never
/// match.
fn carries(pid: libc::pid_t, entry: &[u8], arguments: &mut [u8]) -> bool {
    query(&mut [libc::CTL_KERN, libc::KERN_PROCARGS2, pid], arguments)
        .ok()
        .and_then(|filled| arguments.get(..filled))
        .is_some_and(|arguments| environment(arguments).any(|variable| variable == entry))
}

/// The environment in the KERN_PROCARGS2 layout: argc as a native `c_int`, the
/// executable path, NUL padding, argc arguments, the environment up to an empty
/// string, then the apple strings, which must not match.
fn environment(arguments: &[u8]) -> impl Iterator<Item = &[u8]> {
    let (count, strings) = arguments
        .split_first_chunk()
        .and_then(|(count, strings)| {
            let count = usize::try_from(libc::c_int::from_ne_bytes(*count)).ok()?;
            let end = strings.iter().rposition(|&byte| byte == 0)?;
            Some((count, &strings[..end]))
        })
        .unwrap_or_default();
    let mut strings = strings.split(|&byte| byte == 0);
    let _executable_path = strings.next();
    strings
        .skip_while(|string| string.is_empty())
        .skip(count)
        .take_while(|string| !string.is_empty())
}

/// Read sysctl `name` into `buffer`, returning how many bytes it filled.
fn query(name: &mut [libc::c_int], buffer: &mut [u8]) -> io::Result<usize> {
    let mut filled = buffer.len();
    // SAFETY: `name` is valid for its length in ints and `buffer` for `filled`
    // writable bytes; sysctl writes at most that many, stores how many it wrote
    // in `filled`, and sets nothing.
    let result = unsafe {
        libc::sysctl(
            name.as_mut_ptr(),
            name.len() as libc::c_uint,
            buffer.as_mut_ptr().cast(),
            &mut filled,
            ptr::null_mut(),
            0,
        )
    };
    if result == 0 {
        Ok(filled)
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Send `signal` to process `pid`, reporting whether it was delivered.
fn send(pid: libc::pid_t, signal: Signal) -> bool {
    // SAFETY: kill takes no pointers and only sends a signal.
    if unsafe { libc::kill(pid, signal.number()) } == 0 {
        return true;
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() != Some(libc::ESRCH) {
        tracing::warn!(
            component = "runner",
            pid,
            signal = ?signal,
            error = %error,
            "Failed to signal a process an agent run left behind"
        );
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENTRY: &[u8] = b"CLAUDEAR_RUN=1";

    /// A KERN_PROCARGS2 buffer holding `count`, then each of `strings` followed
    /// by a NUL.
    fn arguments(count: libc::c_int, strings: &[&str]) -> Vec<u8> {
        let mut arguments = count.to_ne_bytes().to_vec();
        for string in strings {
            arguments.extend_from_slice(string.as_bytes());
            arguments.push(0);
        }
        arguments
    }

    fn collect(arguments: &[u8]) -> Vec<&[u8]> {
        environment(arguments).collect()
    }

    #[test]
    fn test_environment_follows_the_padding_and_the_arguments() {
        let arguments = arguments(
            2,
            &[
                "/bin/tool",
                "",
                "",
                "tool",
                "CLAUDEAR_RUN=1",
                "HOME=/",
                "PATH=/bin",
                "",
            ],
        );

        assert_eq!(collect(&arguments), [&b"HOME=/"[..], &b"PATH=/bin"[..]]);
    }

    #[test]
    fn test_environment_counts_empty_arguments() {
        let arguments = arguments(3, &["/bin/tool", "", "tool", "", "", "HOME=/", ""]);

        assert_eq!(collect(&arguments), [&b"HOME=/"[..]]);
    }

    #[test]
    fn test_environment_stops_before_the_apple_strings() {
        let arguments = arguments(
            1,
            &["/bin/tool", "", "tool", "HOME=/", "", "CLAUDEAR_RUN=1", ""],
        );

        assert!(
            !environment(&arguments).any(|variable| variable == ENTRY),
            "apple strings must not match"
        );
    }

    #[test]
    fn test_environment_of_a_process_that_hides_it_is_empty() {
        let arguments = arguments(1, &["/bin/tool", "", "tool"]);

        assert!(collect(&arguments).is_empty());
    }

    #[test]
    fn test_environment_ignores_an_unterminated_string() {
        let mut arguments = arguments(1, &["/bin/tool", "", "tool", "HOME=/"]);
        arguments.extend_from_slice(ENTRY);

        assert_eq!(collect(&arguments), [&b"HOME=/"[..]]);
    }

    #[test]
    fn test_environment_of_a_malformed_buffer_is_empty() {
        let negative = arguments(-1, &["/bin/tool", "", "tool", "CLAUDEAR_RUN=1", ""]);

        for malformed in [&[][..], &[1, 0][..], &negative[..]] {
            assert!(
                collect(malformed).is_empty(),
                "{malformed:?} must hold no environment"
            );
        }
    }

    #[test]
    fn test_pids_excludes_claudear_itself() {
        let own = libc::pid_t::try_from(std::process::id()).unwrap();

        let pids = pids();

        assert!(
            !pids.is_empty(),
            "claudear's user must have other processes"
        );
        assert!(!pids.contains(&own), "claudear must never signal itself");
    }
}
