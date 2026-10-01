//! Reads Discord support forum threads from the Appwrite project the threads bot
//! syncs them into, and ranks them for the support digest.
//!
//! Claudear only reads that project; it needs no access to the forum, and
//! nothing is ever posted back to it.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;

use chrono::{DateTime, Duration, Utc};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};

use claudear_config::{Config, SupportDigestConfig};
use claudear_core::error::{Error, Result};
use claudear_integrations::reports::{
    is_solved, SupportDigest, SupportMessage, SupportStatus, SupportThread,
};

/// Rows per Appwrite list request (the API caps a page at 100 rows and an
/// `equal` query at 100 values).
const PAGE_LIMIT: usize = 100;

/// Most rows read from one table per scan.
const MAX_ROWS: usize = 10_000;

/// How many likely-resolved threads a digest lists.
const RESOLVED_LIMIT: usize = 5;

/// HTTP timeout for Appwrite requests.
const HTTP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Extension, on the database file name, of the file that keeps the last sent
/// digest across restarts.
const SENT_EXTENSION: &str = "support_digest_sent.json";

#[derive(Deserialize)]
struct ThreadRow {
    #[serde(rename = "$id")]
    id: String,
    title: String,
    author: String,
    author_id: Option<String>,
    tags: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct MessageRow {
    #[serde(rename = "threadId")]
    thread_id: String,
    author: String,
    author_id: Option<String>,
    message: String,
    timestamp: String,
}

#[derive(Deserialize)]
struct AuthorRow {
    discord_id: String,
}

/// Reads the threads project and builds [`SupportDigest`]s.
pub struct SupportDigestOrchestrator {
    http: reqwest::Client,
    config: SupportDigestConfig,
    /// Needs-reply threads listed in the last digest that was sent.
    sent: Mutex<HashSet<String>>,
    /// Where `sent` is saved; `None` for an in-memory database.
    sent_path: Option<PathBuf>,
}

impl SupportDigestOrchestrator {
    /// Build from config, or `None` when the digest is disabled.
    pub fn from_config(config: &Config) -> Option<Self> {
        let cfg = &config.reports.support_digest;
        if !cfg.enabled {
            return None;
        }
        let http = match reqwest::Client::builder().timeout(HTTP_TIMEOUT).build() {
            Ok(http) => http,
            Err(e) => {
                tracing::warn!(error = %e, "Support digest HTTP client failed; skipping");
                return None;
            }
        };
        let sent_path = (config.db_path.as_os_str() != ":memory:")
            .then(|| config.db_path.with_extension(SENT_EXTENSION));
        let sent = sent_path.as_deref().map(load_sent).unwrap_or_default();
        Some(Self {
            http,
            config: cfg.clone(),
            sent: Mutex::new(sent),
            sent_path,
        })
    }

    /// Rank the forum's open threads. Threads missing from the last sent digest
    /// are marked new.
    pub async fn collect(&self) -> Result<SupportDigest> {
        let now = Utc::now();
        let since = now - Duration::days(self.config.days);

        // Rows synced before `is_resolved` existed leave it null; count them open.
        let rows: Vec<ThreadRow> = self
            .list_rows(
                "threads",
                vec![
                    json!({"method": "or", "values": [
                        {"method": "equal", "attribute": "is_resolved", "values": [false]},
                        {"method": "isNull", "attribute": "is_resolved"},
                    ]}),
                    json!({"method": "greaterThan", "attribute": "last_activity", "values": [since.to_rfc3339()]}),
                ],
            )
            .await?;
        let rows: Vec<ThreadRow> = rows
            .into_iter()
            .filter(|row| !is_solved(&row.title))
            .collect();

        let mut by_thread: HashMap<String, Vec<MessageRow>> = HashMap::new();
        for chunk in rows.chunks(PAGE_LIMIT) {
            let ids: Vec<&str> = chunk.iter().map(|row| row.id.as_str()).collect();
            let page: Vec<MessageRow> = self
                .list_rows(
                    "messages",
                    vec![json!({"method": "equal", "attribute": "threadId", "values": ids})],
                )
                .await?;
            for message in page {
                by_thread
                    .entry(message.thread_id.clone())
                    .or_default()
                    .push(message);
            }
        }

        let threads: Vec<SupportThread> = rows
            .into_iter()
            .filter_map(|row| {
                let owner_id = row.author_id.clone().unwrap_or_default();
                let mut messages: Vec<SupportMessage> = by_thread
                    .remove(&row.id)?
                    .into_iter()
                    .filter_map(|message| {
                        let timestamp = DateTime::parse_from_rfc3339(&message.timestamp)
                            .ok()?
                            .with_timezone(&Utc);
                        // Older rows have no author id; match the poster by name.
                        let author_id = match message.author_id.filter(|id| !id.is_empty()) {
                            Some(id) => id,
                            None if message.author == row.author => owner_id.clone(),
                            None => format!("name:{}", message.author),
                        };
                        Some(SupportMessage {
                            author_id,
                            author: message.author,
                            content: message.message,
                            timestamp,
                        })
                    })
                    .collect();
                messages.sort_by_key(|message| message.timestamp);
                Some(SupportThread {
                    url: format!(
                        "https://discord.com/channels/{}/{}",
                        self.config.guild_id, row.id
                    ),
                    id: row.id,
                    title: row.title,
                    owner_id,
                    tags: row.tags.unwrap_or_default(),
                    messages,
                })
            })
            .collect();

        let team = self.team().await?;
        let mut needs_reply = Vec::new();
        let mut likely_resolved = Vec::new();
        let mut waiting_on_user = 0;
        for entry in threads
            .iter()
            .filter_map(|thread| thread.triage(&team, now))
        {
            match entry.status {
                SupportStatus::NeedsReply => needs_reply.push(entry),
                SupportStatus::LikelyResolved => likely_resolved.push(entry),
                SupportStatus::WaitingOnUser => waiting_on_user += 1,
            }
        }
        needs_reply.sort_by_key(|entry| std::cmp::Reverse(entry.score));
        likely_resolved.sort_by_key(|entry| std::cmp::Reverse(entry.waiting_hours));
        let needs_reply_total = needs_reply.len();
        // The Discord message lists at most this many; anything cut here is never
        // marked sent, so it still counts as new later.
        needs_reply.truncate(
            self.config
                .max_entries
                .min(claudear_integrations::notifier::SUPPORT_DIGEST_MAX_ENTRIES),
        );
        likely_resolved.truncate(RESOLVED_LIMIT);

        let sent = self.sent.lock().unwrap();
        for entry in &mut needs_reply {
            entry.is_new = !sent.contains(&entry.thread_id);
        }
        drop(sent);

        Ok(SupportDigest {
            days: self.config.days,
            needs_reply,
            needs_reply_total,
            likely_resolved,
            waiting_on_user,
        })
    }

    /// Remember which threads a sent digest listed, on disk too so a restart
    /// does not announce them again.
    pub fn mark_sent(&self, digest: &SupportDigest) {
        let sent: HashSet<String> = digest
            .needs_reply
            .iter()
            .map(|entry| entry.thread_id.clone())
            .collect();
        if let Some(path) = &self.sent_path {
            let json = serde_json::to_string(&sent).unwrap_or_default();
            if let Err(e) = std::fs::write(path, json) {
                tracing::warn!(path = %path.display(), error = %e, "Failed to save support digest state");
            }
        }
        *self.sent.lock().unwrap() = sent;
    }

    /// Configured team user ids plus thread authors holding a team role. The
    /// authors table only has people who opened a thread, so most staff need
    /// listing in `team_user_ids`.
    async fn team(&self) -> Result<HashSet<String>> {
        let mut team: HashSet<String> = self.config.team_user_ids.iter().cloned().collect();
        if !self.config.team_roles.is_empty() {
            let authors: Vec<AuthorRow> = self
                .list_rows(
                    "authors",
                    vec![json!({"method": "contains", "attribute": "roles", "values": self.config.team_roles})],
                )
                .await?;
            team.extend(authors.into_iter().map(|author| author.discord_id));
        }
        Ok(team)
    }

    /// Every row of a table matching `queries`, paged by cursor up to [`MAX_ROWS`].
    async fn list_rows<T: DeserializeOwned>(
        &self,
        table: &str,
        queries: Vec<Value>,
    ) -> Result<Vec<T>> {
        #[derive(Deserialize)]
        struct Page {
            rows: Vec<Value>,
        }

        let url = format!(
            "{}/tablesdb/{}/tables/{}/rows",
            self.config.endpoint.trim_end_matches('/'),
            self.config.database_id,
            table
        );
        let mut rows = Vec::new();
        let mut cursor: Option<String> = None;
        while rows.len() < MAX_ROWS {
            let mut params: Vec<(&str, String)> = queries
                .iter()
                .chain(&[json!({"method": "limit", "values": [PAGE_LIMIT]})])
                .map(|query| ("queries[]", query.to_string()))
                .collect();
            if let Some(cursor) = &cursor {
                params.push((
                    "queries[]",
                    json!({"method": "cursorAfter", "values": [cursor]}).to_string(),
                ));
            }
            params.push(("total", "false".to_string()));
            let url = url::Url::parse_with_params(&url, &params)
                .map_err(|e| Error::config(format!("Invalid support digest endpoint: {e}")))?;

            let mut request = self
                .http
                .get(url)
                .header("X-Appwrite-Project", &self.config.project_id);
            if let Some(key) = &self.config.api_key {
                request = request.header("X-Appwrite-Key", key.expose());
            }
            let response = request.send().await?;
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if !status.is_success() {
                return Err(Error::network(format!(
                    "Failed to list {table} rows ({status}): {body}"
                )));
            }
            let page: Page = serde_json::from_str(&body)
                .map_err(|e| Error::Other(format!("Invalid {table} rows: {e}")))?;

            let full = page.rows.len() == PAGE_LIMIT;
            cursor = page
                .rows
                .last()
                .and_then(|row| row.get("$id"))
                .and_then(Value::as_str)
                .map(String::from);
            for row in page.rows {
                rows.push(
                    serde_json::from_value(row)
                        .map_err(|e| Error::Other(format!("Invalid {table} row: {e}")))?,
                );
            }
            if !full || cursor.is_none() {
                break;
            }
        }
        Ok(rows)
    }
}

/// Threads listed in the last sent digest, or none when the file is missing or
/// unreadable.
fn load_sent(path: &std::path::Path) -> HashSet<String> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|json| serde_json::from_str(&json).ok())
        .unwrap_or_default()
}
