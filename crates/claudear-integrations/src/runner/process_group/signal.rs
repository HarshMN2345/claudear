#[derive(Clone, Copy, Debug)]
pub(super) enum Signal {
    Interrupt,
    Terminate,
    Kill,
}

impl Signal {
    #[cfg(unix)]
    pub(super) fn number(self) -> libc::c_int {
        match self {
            Self::Interrupt => libc::SIGINT,
            Self::Terminate => libc::SIGTERM,
            Self::Kill => libc::SIGKILL,
        }
    }
}
