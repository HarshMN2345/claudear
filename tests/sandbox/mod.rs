use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tempfile::TempDir;
use tokio::process::Command;

const CONFIG: &str = "claudear.toml";
const DATABASE: &str = "claudear.db";
const LOGS: &str = "logs";
const OUTPUT_EXTENSION: &str = "out";
const TAIL_LINES: usize = 40;
const UNREACHABLE_URL: &str = "http://127.0.0.1:1";

/// A home, config and database for the claudear binary, whose only source is a Jira nothing
/// listens for and whose retries are due at once. It lives under /tmp to keep the daemon's
/// socket path short.
pub struct Sandbox {
    root: TempDir,
}

impl Sandbox {
    pub fn new() -> Self {
        let root = tempfile::Builder::new()
            .prefix("claudear")
            .tempdir_in("/tmp")
            .expect("create a sandbox under /tmp");
        let sandbox = Self { root };
        fs::write(
            sandbox.path().join(CONFIG),
            config(sandbox.path(), &sandbox.database()),
        )
        .expect("write the sandbox config");
        sandbox
    }

    pub fn path(&self) -> &Path {
        self.root.path()
    }

    pub fn database(&self) -> PathBuf {
        self.path().join(DATABASE)
    }

    /// Claudear with `arguments`, run inside the sandbox with its stdout and stderr going to the
    /// output file for `name`.
    pub fn command(&self, name: &str, arguments: &[&str]) -> Command {
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
            .arg(self.path().join(CONFIG))
            .arg("--log-dir")
            .arg(self.path().join(LOGS))
            .args(arguments)
            .stdin(Stdio::null())
            .stdout(output.try_clone().expect("share the output file"))
            .stderr(output)
            .kill_on_drop(true);
        command
    }

    pub fn output(&self, name: &str) -> PathBuf {
        self.path().join(format!("{name}.{OUTPUT_EXTENSION}"))
    }

    pub fn log_files(&self) -> impl Iterator<Item = PathBuf> {
        files_in(&self.path().join(LOGS))
    }

    /// The tail of every command's output and every log file, for failure messages.
    pub fn diagnostics(&self) -> String {
        let mut outputs: Vec<PathBuf> = files_in(self.path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == OUTPUT_EXTENSION)
            })
            .collect();
        outputs.sort();
        outputs
            .into_iter()
            .chain(self.log_files())
            .map(|path| format!("{}:\n{}", path.display(), tail(&path)))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn config(root: &Path, database: &Path) -> String {
    let root = root.display();
    let database = database.display();
    format!(
        r#"workspace = "{root}/workspace"
db_path = "{database}"
storage_dir = "{root}/storage"
known_orgs = []
auto_discover_paths = []

[code_index]
enabled = false

[regression]
enabled = false

[retry]
base_delay_ms = 0
max_delay_ms = 0

[issues.jira]
enabled = true
base_url = "{UNREACHABLE_URL}"
email = "sandbox@example.com"
api_token = "unused"
project_keys = ["SANDBOX"]
"#
    )
}

fn files_in(directory: &Path) -> impl Iterator<Item = PathBuf> {
    fs::read_dir(directory)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
}

fn tail(path: &Path) -> String {
    let content = fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = content.lines().collect();
    lines[lines.len().saturating_sub(TAIL_LINES)..].join("\n")
}
