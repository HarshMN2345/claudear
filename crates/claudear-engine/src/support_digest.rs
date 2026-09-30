//! Collects Discord support forum threads and ranks them for the support digest.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use chrono::{DateTime, Duration, Utc};

use claudear_config::{Config, SupportDigestConfig};
use claudear_core::error::{Error, Result};
use claudear_integrations::discord::{DiscordClient, DiscordThread};
use claudear_integrations::reports::{
    is_solved, SupportDigest, SupportMessage, SupportStatus, SupportThread,
};

/// Discord message page size (the API caps a single page at 100).
const PAGE_LIMIT: usize = 100;

/// How many likely-resolved threads a digest lists.
const RESOLVED_LIMIT: usize = 5;

/// Pause between thread reads; the Discord client does not retry 429s.
const READ_DELAY: std::time::Duration = std::time::Duration::from_millis(250);

/// Reads the support forum and builds [`SupportDigest`]s.
pub struct SupportDigestOrchestrator {
    client: DiscordClient,
    config: SupportDigestConfig,
    /// Whether an author holds a team role, looked up once per process.
    team_roles: Mutex<HashMap<String, bool>>,
    /// Needs-reply threads listed in the last digest that was sent.
    sent: Mutex<HashSet<String>>,
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
        match DiscordClient::new(token) {
            Ok(client) => Some(Self {
                client,
                config: cfg.clone(),
                team_roles: Mutex::default(),
                sent: Mutex::default(),
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
            tokio::time::sleep(READ_DELAY).await;
            let messages = match self
                .client
                .list_channel_messages(&thread.id, PAGE_LIMIT)
                .await
            {
                Ok(messages) => messages,
                Err(e) => {
                    tracing::warn!(thread = %thread.id, error = %e, "Failed to read support thread; skipping");
                    continue;
                }
            };
            // Discord returns newest first.
            let messages: Vec<SupportMessage> = messages
                .into_iter()
                .rev()
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
        needs_reply.truncate(self.config.max_entries);
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

    /// Remember which threads a sent digest listed.
    pub fn mark_sent(&self, digest: &SupportDigest) {
        *self.sent.lock().unwrap() = digest
            .needs_reply
            .iter()
            .map(|entry| entry.thread_id.clone())
            .collect();
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
            let cached = self.team_roles.lock().unwrap().get(user_id).copied();
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
                            .insert(user_id.to_string(), is_team);
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
