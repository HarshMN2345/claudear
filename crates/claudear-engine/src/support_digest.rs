//! Collects Discord support forum threads and ranks them for the support digest.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use chrono::{DateTime, Duration, Utc};

use claudear_config::{Config, SupportDigestConfig};
use claudear_core::error::{Error, Result};
use claudear_integrations::discord::{DiscordClient, DiscordMessage, DiscordThread};
use claudear_integrations::reports::{
    is_solved, SupportDigest, SupportMessage, SupportStatus, SupportThread,
};

/// Discord message page size (the API caps a single page at 100).
const PAGE_LIMIT: usize = 100;

/// How many likely-resolved threads a digest lists.
const RESOLVED_LIMIT: usize = 5;

/// Pause between thread reads; the Discord client does not retry 429s.
const READ_DELAY: std::time::Duration = std::time::Duration::from_millis(250);

/// Most message pages read per thread (1000 messages). Longer threads are
/// ranked on their newest messages.
const MAX_PAGES: usize = 10;

/// How long a team role lookup is trusted before it is read again.
const ROLE_TTL: std::time::Duration = std::time::Duration::from_secs(6 * 3600);

/// File, next to the database, that keeps the last sent digest across restarts.
const SENT_FILE: &str = "support_digest_sent.json";

/// Reads the support forum and builds [`SupportDigest`]s.
pub struct SupportDigestOrchestrator {
    client: DiscordClient,
    config: SupportDigestConfig,
    /// Whether an author holds a team role, and when that was looked up.
    team_roles: Mutex<HashMap<String, (bool, Instant)>>,
    /// Needs-reply threads listed in the last digest that was sent.
    sent: Mutex<HashSet<String>>,
    /// Where `sent` is saved; `None` for an in-memory database.
    sent_path: Option<PathBuf>,
}

impl SupportDigestOrchestrator {
    /// Build from config, or `None` when the digest is disabled or missing a bot
    /// token or channel. The bot token falls back to the merged notifier/issue
    /// Discord config.
    pub fn from_config(config: &Config) -> Option<Self> {
        let cfg = &config.reports.support_digest;
        if !cfg.enabled {
            return None;
        }
        // Treat blank strings (from copied example config) as unset so the
        // notifier/issue Discord fallback actually applies.
        let nonempty = |s: &str| (!s.trim().is_empty()).then(|| s.to_string());
        let token = cfg
            .bot_token
            .as_ref()
            .and_then(|s| nonempty(s.expose()))
            .or_else(|| {
                config
                    .discord_merged()
                    .bot_token
                    .as_ref()
                    .and_then(|s| nonempty(s.expose()))
            });
        let Some(token) = token.filter(|_| !cfg.channel_id.trim().is_empty()) else {
            tracing::warn!("Support digest enabled but missing bot_token or channel_id; skipping");
            return None;
        };
        let sent_path = (config.db_path.as_os_str() != ":memory:")
            .then(|| config.db_path.with_file_name(SENT_FILE));
        let sent = sent_path.as_deref().map(load_sent).unwrap_or_default();
        match DiscordClient::new(token) {
            Ok(client) => Some(Self {
                client,
                config: cfg.clone(),
                team_roles: Mutex::default(),
                sent: Mutex::new(sent),
                sent_path,
            }),
            Err(e) => {
                tracing::warn!(error = %e, "Support digest Discord client failed; skipping");
                None
            }
        }
    }

    /// Rank the forum's open threads. Threads missing from the last sent digest
    /// are marked new.
    pub async fn collect(&self) -> Result<SupportDigest> {
        let now = Utc::now();
        let since = now - Duration::days(self.config.days);

        let forum = self.client.get_channel(&self.config.channel_id).await?;
        let guild_id = forum.guild_id.clone().ok_or_else(|| {
            Error::config("reports.support_digest.channel_id is not a guild channel")
        })?;
        let tag_names: HashMap<&str, &str> = forum
            .available_tags
            .iter()
            .map(|tag| (tag.id.as_str(), tag.name.as_str()))
            .collect();

        // Unanswered posts auto-archive, so recently archived threads count too.
        let mut candidates: Vec<DiscordThread> = self
            .client
            .list_active_threads(&guild_id, None, None)
            .await?
            .into_iter()
            .filter(|thread| thread.parent_id.as_deref() == Some(forum.id.as_str()))
            .collect();
        candidates.extend(
            self.client
                .list_public_archived_threads_since(&forum.id, since)
                .await?,
        );
        let mut seen = HashSet::new();
        candidates.retain(|thread| seen.insert(thread.id.clone()));

        let mut threads = Vec::new();
        for thread in candidates {
            let tags: Vec<String> = thread
                .applied_tags
                .iter()
                .filter_map(|id| tag_names.get(id.as_str()))
                .map(|name| name.to_string())
                .collect();
            if is_solved(&thread.name, &tags, &self.config.solved_tags) {
                continue;
            }
            let Some(owner_id) = thread.owner_id.clone() else {
                continue;
            };
            let messages = match self.thread_messages(&thread.id).await {
                Ok(messages) => messages,
                Err(e) => {
                    tracing::warn!(thread = %thread.id, error = %e, "Failed to read support thread; skipping");
                    continue;
                }
            };
            let messages: Vec<SupportMessage> = messages
                .into_iter()
                .filter_map(|message| {
                    let author = message.author.filter(|author| !author.bot)?;
                    let timestamp = DateTime::parse_from_rfc3339(&message.timestamp)
                        .ok()?
                        .with_timezone(&Utc);
                    Some(SupportMessage {
                        author_id: author.id,
                        author: author.username,
                        content: message.content,
                        timestamp,
                    })
                })
                .collect();
            if messages
                .last()
                .is_none_or(|message| message.timestamp < since)
            {
                continue;
            }
            threads.push(SupportThread {
                url: format!("https://discord.com/channels/{}/{}", guild_id, thread.id),
                id: thread.id,
                title: thread.name,
                owner_id,
                tags,
                messages,
            });
        }

        let team = self.team(&guild_id, &threads).await;
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
        needs_reply.sort_by(|a, b| b.score.cmp(&a.score));
        likely_resolved.sort_by(|a, b| b.waiting_hours.cmp(&a.waiting_hours));
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

    /// A thread's messages, oldest first, paging back up to [`MAX_PAGES`].
    async fn thread_messages(&self, thread_id: &str) -> Result<Vec<DiscordMessage>> {
        tokio::time::sleep(READ_DELAY).await;
        // The first page comes newest first; older pages oldest first.
        let mut messages: Vec<DiscordMessage> = self
            .client
            .list_channel_messages(thread_id, PAGE_LIMIT)
            .await?
            .into_iter()
            .rev()
            .collect();
        let mut page_len = messages.len();
        for _ in 1..MAX_PAGES {
            if page_len < PAGE_LIMIT {
                break;
            }
            let Some(oldest) = messages.first().map(|message| message.id.clone()) else {
                break;
            };
            tokio::time::sleep(READ_DELAY).await;
            let older = self
                .client
                .list_channel_messages_before(thread_id, &oldest, PAGE_LIMIT)
                .await?;
            page_len = older.len();
            messages.splice(0..0, older);
        }
        Ok(messages)
    }

    /// Configured team user ids plus repliers holding a team role. A failed role
    /// lookup counts as community and is retried on the next scan.
    async fn team(&self, guild_id: &str, threads: &[SupportThread]) -> HashSet<String> {
        let mut team: HashSet<String> = self.config.team_user_ids.iter().cloned().collect();
        if self.config.team_role_ids.is_empty() {
            return team;
        }

        let repliers: HashSet<&str> = threads
            .iter()
            .flat_map(|thread| {
                thread
                    .messages
                    .iter()
                    .filter(|message| message.author_id != thread.owner_id)
                    .map(|message| message.author_id.as_str())
            })
            .collect();

        for user_id in repliers {
            let cached = self
                .team_roles
                .lock()
                .unwrap()
                .get(user_id)
                .filter(|(_, at)| at.elapsed() < ROLE_TTL)
                .map(|(is_team, _)| *is_team);
            let is_team = match cached {
                Some(is_team) => is_team,
                None => match self.client.get_member_roles(guild_id, user_id).await {
                    Ok(roles) => {
                        let is_team = roles.is_some_and(|roles| {
                            roles
                                .iter()
                                .any(|role| self.config.team_role_ids.contains(role))
                        });
                        self.team_roles
                            .lock()
                            .unwrap()
                            .insert(user_id.to_string(), (is_team, Instant::now()));
                        is_team
                    }
                    Err(e) => {
                        tracing::debug!(user = %user_id, error = %e, "Failed to read guild member roles");
                        false
                    }
                },
            };
            if is_team {
                team.insert(user_id.to_string());
            }
        }

        team
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
