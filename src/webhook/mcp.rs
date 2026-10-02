//! Built-in MCP (Model Context Protocol) search server.
//!
//! Exposes the existing semantic search services - code search and the Discord
//! knowledgebase - as MCP tools over the Streamable HTTP transport, mounted at
//! `POST /mcp` on the same HTTP server as the webhooks and dashboard.
//!
//! This is a minimal, dependency-free implementation of the subset of the MCP
//! JSON-RPC surface that a tool-only server needs: `initialize`, `tools/list`,
//! `tools/call`, and `ping`. Notifications (requests without an `id`, e.g.
//! `notifications/initialized`) are accepted and acknowledged with `202`.
//!
//! Only tools backed by a real, query-string search index are exposed. Today
//! that means code and Discord; other sources (HelpScout, Sentry, Linear) are
//! ingested as issues but have no clean query-search primitive, so they are not
//! surfaced here. Adding a new tool is a matter of extending `tool_definitions`
//! and `call_tool`.

use super::server::AppState;
use crate::config::McpSearchServerConfig;
use crate::knowledgebase::DiscordSearchService;
use crate::repo::code_index::CodeSearchService;
use crate::storage::FixAttemptTracker;
use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Json, Response},
};
use serde_json::{json, Value};
use std::sync::Arc;

/// Protocol versions this handler implements. The newest (first) is advertised
/// when a client requests a version we do not recognise.
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];
const DEFAULT_PROTOCOL_VERSION: &str = SUPPORTED_PROTOCOL_VERSIONS[0];

/// Everything an MCP request needs from the server, borrowed from `AppState`.
///
/// Kept separate from `AppState` so the whole request path can be exercised in
/// tests without constructing the full webhook server state.
struct McpContext<'a> {
    cfg: &'a McpSearchServerConfig,
    code_search: Option<&'a CodeSearchService>,
    discord_search: Option<&'a DiscordSearchService>,
    tracker: &'a dyn FixAttemptTracker,
}

/// Outcome of handling a POST: either a bare 202 ack (for notifications) or a
/// JSON-RPC body to return with 200.
enum PostOutcome {
    Accepted,
    Json(Value),
}

/// Reject `GET /mcp`: this server does not offer the optional server-initiated
/// SSE stream, only request/response over POST.
pub(crate) async fn mcp_get_handler() -> Response {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        Json(json!({
            "jsonrpc": "2.0",
            "id": Value::Null,
            "error": {
                "code": -32000,
                "message": "This MCP endpoint only supports POST (no SSE stream)."
            }
        })),
    )
        .into_response()
}

/// Handle a single JSON-RPC request posted to `/mcp`.
pub(crate) async fn mcp_post_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let cfg = &state.config.mcp_server;

    // DNS-rebinding guard: a browser always sends `Origin`, so reject any origin
    // not explicitly allow-listed. CLI / server-to-server clients send none and
    // are allowed through.
    let origin = headers
        .get(header::ORIGIN)
        .and_then(|value| value.to_str().ok());
    if !origin_allowed(&cfg.allowed_origins, origin) {
        return (
            StatusCode::FORBIDDEN,
            Json(rpc_error(
                Value::Null,
                -32001,
                "Origin not allowed".to_string(),
            )),
        )
            .into_response();
    }

    let ctx = McpContext {
        cfg,
        code_search: state.code_search_service.as_deref(),
        discord_search: state.discord_search_service.as_deref(),
        tracker: state.tracker.as_ref(),
    };

    match process_post(&ctx, &body).await {
        PostOutcome::Accepted => StatusCode::ACCEPTED.into_response(),
        PostOutcome::Json(value) => Json(value).into_response(),
    }
}

/// Core request processing, independent of axum extractors so it can be tested
/// directly against its JSON output.
async fn process_post(ctx: &McpContext<'_>, body: &[u8]) -> PostOutcome {
    let request: Value = match serde_json::from_slice(body) {
        Ok(value) => value,
        Err(e) => {
            return PostOutcome::Json(rpc_error(
                Value::Null,
                -32700,
                format!("Parse error: {e}"),
            ));
        }
    };

    // JSON-RPC batching was removed in the 2025-06-18 spec; we only accept a
    // single request object and reject arrays explicitly.
    if request.is_array() {
        return PostOutcome::Json(rpc_error(
            Value::Null,
            -32600,
            "Batch requests are not supported".to_string(),
        ));
    }

    let method = request.get("method").and_then(Value::as_str).unwrap_or("");
    let id = request.get("id").cloned();
    let params = request.get("params").cloned().unwrap_or(Value::Null);

    // A request without an `id` is a notification: do the work (nothing, for the
    // notifications we expect) and acknowledge without a JSON-RPC body.
    let Some(id) = id else {
        return PostOutcome::Accepted;
    };

    let response = match method {
        "initialize" => rpc_result(id, initialize_result(&params)),
        "ping" => rpc_result(id, json!({})),
        "tools/list" => rpc_result(id, json!({ "tools": tool_definitions(ctx.cfg) })),
        "tools/call" => match call_tool(ctx, &params).await {
            Ok(result) => rpc_result(id, result),
            Err(ToolError::Protocol { code, message }) => rpc_error(id, code, message),
        },
        other => rpc_error(id, -32601, format!("Method not found: {other}")),
    };

    PostOutcome::Json(response)
}

/// Whether a request's `Origin` is acceptable. No origin (non-browser client)
/// is always allowed; a present origin must be in the allow-list.
fn origin_allowed(allowed: &[String], origin: Option<&str>) -> bool {
    match origin {
        None => true,
        Some(origin) => allowed.iter().any(|a| a == origin),
    }
}

/// Resolve the protocol version to advertise: echo the client's request when we
/// support it, otherwise fall back to our newest supported version so the client
/// can decide whether to proceed.
fn negotiate_version(requested: Option<&str>) -> &'static str {
    requested
        .and_then(|v| SUPPORTED_PROTOCOL_VERSIONS.iter().copied().find(|s| *s == v))
        .unwrap_or(DEFAULT_PROTOCOL_VERSION)
}

fn initialize_result(params: &Value) -> Value {
    let requested = params
        .get("protocolVersion")
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty());

    json!({
        "protocolVersion": negotiate_version(requested),
        "capabilities": { "tools": {} },
        "serverInfo": {
            "name": "claudear-search",
            "version": env!("CARGO_PKG_VERSION"),
        },
    })
}

/// Build the list of exposed tools based on configuration.
fn tool_definitions(cfg: &McpSearchServerConfig) -> Vec<Value> {
    let mut tools = Vec::new();

    if cfg.expose_code {
        tools.push(json!({
            "name": "code_search",
            "description": "Semantic search over indexed source code. Returns the \
                most relevant code chunks for a natural-language or code query, \
                each with its file path, line range and similarity score.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural-language or code query." },
                    "repo": { "type": "string", "description": "Optional repository name (e.g. 'org/repo') to scope the search." },
                    "limit": { "type": "integer", "description": "Max results to return.", "minimum": 1 }
                },
                "required": ["query"]
            }
        }));

        tools.push(json!({
            "name": "find_symbol",
            "description": "Find code symbols (functions, classes, structs, traits, \
                etc.) by name via exact substring match. Returns each match's \
                kind, file, line range and signature.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Symbol name or substring to match." },
                    "kind": {
                        "type": "string",
                        "description": "Optional symbol kind filter.",
                        "enum": ["function", "class", "method", "struct", "impl", "interface", "trait", "enum", "module", "constant"]
                    },
                    "repo": { "type": "string", "description": "Optional repository name to scope the search." },
                    "limit": { "type": "integer", "description": "Max results to return.", "minimum": 1 }
                },
                "required": ["name"]
            }
        }));
    }

    if cfg.expose_discord {
        tools.push(json!({
            "name": "discord_search",
            "description": "Semantic search over the indexed Discord knowledgebase. \
                Returns the most relevant past conversations with participants, \
                timestamps and jump links.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural-language query." },
                    "channel_id": { "type": "string", "description": "Optional Discord channel id to scope the search." },
                    "limit": { "type": "integer", "description": "Max results to return.", "minimum": 1 }
                },
                "required": ["query"]
            }
        }));
    }

    tools
}

/// A JSON-RPC protocol-level error. Tool *execution* failures are not protocol
/// errors - they are returned as a successful result with `isError: true`.
enum ToolError {
    Protocol { code: i64, message: String },
}

async fn call_tool(ctx: &McpContext<'_>, params: &Value) -> Result<Value, ToolError> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::Protocol {
            code: -32602,
            message: "Missing tool name".to_string(),
        })?;
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    match name {
        "code_search" if ctx.cfg.expose_code => Ok(code_search(ctx, &args).await),
        "find_symbol" if ctx.cfg.expose_code => Ok(find_symbol(ctx, &args)),
        "discord_search" if ctx.cfg.expose_discord => Ok(discord_search(ctx, &args).await),
        other => Err(ToolError::Protocol {
            code: -32602,
            message: format!("Unknown tool: {other}"),
        }),
    }
}

async fn code_search(ctx: &McpContext<'_>, args: &Value) -> Value {
    let Some(service) = ctx.code_search else {
        return tool_error("Code search is unavailable (code indexing or embeddings not configured).");
    };
    let Some(query) = str_arg(args, "query") else {
        return tool_error("Missing required argument: query");
    };

    let repo_id = match resolve_repo(ctx, args) {
        Ok(id) => id,
        Err(msg) => return tool_error(&msg),
    };
    let limit = ctx.cfg.resolve_limit(usize_arg(args, "limit"));

    match service.search(query, repo_id, limit).await {
        Ok(results) => tool_text(format_code_results(&results)),
        Err(e) => tool_error(&format!("Code search failed: {e}")),
    }
}

fn find_symbol(ctx: &McpContext<'_>, args: &Value) -> Value {
    let Some(service) = ctx.code_search else {
        return tool_error("Symbol search is unavailable (code indexing not configured).");
    };
    let Some(name) = str_arg(args, "name") else {
        return tool_error("Missing required argument: name");
    };

    let kind = match str_arg(args, "kind") {
        Some(k) => match claudear_core::types::SymbolKind::from_str_loose(k) {
            Some(kind) => Some(kind),
            None => return tool_error(&format!("Unknown symbol kind: {k}")),
        },
        None => None,
    };
    let repo_id = match resolve_repo(ctx, args) {
        Ok(id) => id,
        Err(msg) => return tool_error(&msg),
    };

    match service.find_symbol(name, kind, repo_id) {
        Ok(mut symbols) => {
            // The storage layer caps symbol lookups generously (up to 100); apply
            // the configured limit here so this tool is bounded like the others.
            symbols.truncate(ctx.cfg.resolve_limit(usize_arg(args, "limit")));
            tool_text(format_symbols(&symbols))
        }
        Err(e) => tool_error(&format!("Symbol search failed: {e}")),
    }
}

async fn discord_search(ctx: &McpContext<'_>, args: &Value) -> Value {
    let Some(service) = ctx.discord_search else {
        return tool_error("Discord search is unavailable (Discord knowledgebase not configured).");
    };
    let Some(query) = str_arg(args, "query") else {
        return tool_error("Missing required argument: query");
    };

    let channel_id = str_arg(args, "channel_id");
    let limit = ctx.cfg.resolve_limit(usize_arg(args, "limit"));

    match service.search(query, channel_id, limit).await {
        Ok(results) if results.is_empty() => {
            tool_text("No relevant Discord discussions found.".to_string())
        }
        Ok(results) => tool_text(crate::knowledgebase::format_discord_search_context(&results)),
        Err(e) => tool_error(&format!("Discord search failed: {e}")),
    }
}

// --- argument helpers -------------------------------------------------------

fn str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn usize_arg(args: &Value, key: &str) -> Option<usize> {
    args.get(key).and_then(Value::as_u64).map(|v| v as usize)
}

/// Resolve an optional repo scope from the arguments into a repo id.
///
/// Accepts either `repo_id` (integer) directly, or `repo` (name) which is
/// looked up in the index. Returns `Ok(None)` when neither is supplied, and
/// `Err` with a user-facing message when a given name is not indexed.
fn resolve_repo(ctx: &McpContext<'_>, args: &Value) -> Result<Option<i64>, String> {
    if let Some(id) = args.get("repo_id").and_then(Value::as_i64) {
        return Ok(Some(id));
    }
    let Some(name) = str_arg(args, "repo") else {
        return Ok(None);
    };
    match ctx.tracker.get_indexed_repo(name) {
        Ok(Some(repo)) => Ok(Some(repo.id)),
        Ok(None) => Err(format!("Repository '{name}' is not indexed.")),
        Err(e) => Err(format!("Failed to look up repository '{name}': {e}")),
    }
}

// --- result formatting ------------------------------------------------------

fn format_code_results(results: &[claudear_core::types::CodeSearchResult]) -> String {
    use std::fmt::Write;

    if results.is_empty() {
        return "No matching code found.".to_string();
    }

    let mut out = format!("Found {} code match(es):\n", results.len());
    for (i, result) in results.iter().enumerate() {
        let chunk = &result.chunk;
        let _ = write!(
            out,
            "\n### {}. {}:{}-{} ({}, {:.0}% match)",
            i + 1,
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
            chunk.language,
            result.score * 100.0,
        );
        if let Some(symbol) = chunk.symbol_name.as_ref().filter(|s| !s.is_empty()) {
            let _ = write!(out, " - `{symbol}`");
        }
        // Truncate long chunks on a UTF-8 boundary to keep responses bounded.
        let text = truncate_on_boundary(&chunk.chunk_text, 1500);
        let _ = write!(out, "\n```{}\n{}\n```\n", chunk.language, text);
    }
    out
}

fn format_symbols(symbols: &[claudear_core::types::CodeSymbol]) -> String {
    use std::fmt::Write;

    if symbols.is_empty() {
        return "No matching symbols found.".to_string();
    }

    let mut out = format!("Found {} symbol(s):\n", symbols.len());
    for symbol in symbols {
        let _ = write!(
            out,
            "\n- **{}** ({}) - {}:{}-{}",
            symbol.symbol_name,
            symbol.symbol_kind,
            symbol.file_path,
            symbol.start_line,
            symbol.end_line,
        );
        if let Some(parent) = symbol.parent_symbol.as_ref().filter(|p| !p.is_empty()) {
            let _ = write!(out, " (in `{parent}`)");
        }
        if let Some(sig) = symbol.signature.as_ref().filter(|s| !s.is_empty()) {
            let _ = write!(out, "\n  `{}`", truncate_on_boundary(sig, 300));
        }
    }
    out
}

fn truncate_on_boundary(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…\n(truncated)", &text[..end])
}

// --- JSON-RPC / MCP envelope helpers ----------------------------------------

fn rpc_result(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn rpc_error(id: Value, code: i64, message: String) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// Build a successful `tools/call` result carrying a single text block.
fn tool_text(text: String) -> Value {
    json!({ "content": [{ "type": "text", "text": text }] })
}

/// Build a `tools/call` result flagged as a tool execution error. Per the MCP
/// spec these are reported in the result (not as a JSON-RPC error) so the model
/// can see and react to them.
fn tool_error(message: &str) -> Value {
    json!({
        "content": [{ "type": "text", "text": message }],
        "isError": true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::SqliteTracker;
    use claudear_core::types::{CodeChunk, CodeSearchResult, CodeSymbol, Language, SymbolKind};

    fn cfg(expose_code: bool, expose_discord: bool) -> McpSearchServerConfig {
        McpSearchServerConfig {
            enabled: true,
            expose_code,
            expose_discord,
            ..Default::default()
        }
    }

    /// Build a context with no live search services, backed by an in-memory
    /// tracker. Tool calls therefore report their service as unavailable, which
    /// is enough to exercise the full request/dispatch path.
    fn test_ctx<'a>(
        cfg: &'a McpSearchServerConfig,
        tracker: &'a dyn FixAttemptTracker,
    ) -> McpContext<'a> {
        McpContext {
            cfg,
            code_search: None,
            discord_search: None,
            tracker,
        }
    }

    async fn post(ctx: &McpContext<'_>, request: Value) -> Option<Value> {
        match process_post(ctx, request.to_string().as_bytes()).await {
            PostOutcome::Accepted => None,
            PostOutcome::Json(value) => Some(value),
        }
    }

    fn tool_names(tools: &[Value]) -> Vec<String> {
        tools
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect()
    }

    // --- pure helpers -------------------------------------------------------

    #[test]
    fn tool_definitions_respect_toggles() {
        let all = tool_names(&tool_definitions(&cfg(true, true)));
        assert_eq!(all, vec!["code_search", "find_symbol", "discord_search"]);

        let code_only = tool_names(&tool_definitions(&cfg(true, false)));
        assert_eq!(code_only, vec!["code_search", "find_symbol"]);

        let discord_only = tool_names(&tool_definitions(&cfg(false, true)));
        assert_eq!(discord_only, vec!["discord_search"]);

        assert!(tool_definitions(&cfg(false, false)).is_empty());
    }

    #[test]
    fn tool_definitions_have_valid_schemas() {
        for tool in tool_definitions(&cfg(true, true)) {
            assert!(tool["name"].is_string());
            assert!(tool["description"].is_string());
            assert_eq!(tool["inputSchema"]["type"], "object");
            assert!(tool["inputSchema"]["required"].is_array());
        }
    }

    #[test]
    fn negotiate_version_only_accepts_supported() {
        assert_eq!(negotiate_version(Some("2025-06-18")), "2025-06-18");
        assert_eq!(negotiate_version(Some("2024-11-05")), "2024-11-05");
        // Unknown or missing versions fall back to the newest supported one.
        assert_eq!(negotiate_version(Some("1999-01-01")), DEFAULT_PROTOCOL_VERSION);
        assert_eq!(negotiate_version(None), DEFAULT_PROTOCOL_VERSION);
    }

    #[test]
    fn origin_allowed_blocks_unknown_browser_origins() {
        let allowed = vec!["https://app.example.com".to_string()];
        // No Origin header (CLI/server client) is always allowed.
        assert!(origin_allowed(&allowed, None));
        assert!(origin_allowed(&allowed, Some("https://app.example.com")));
        assert!(!origin_allowed(&allowed, Some("https://evil.example.com")));
        // Empty allow-list blocks every browser origin.
        assert!(!origin_allowed(&[], Some("https://app.example.com")));
        assert!(origin_allowed(&[], None));
    }

    #[test]
    fn rpc_envelopes_are_well_formed() {
        let ok = rpc_result(json!(1), json!({ "x": 2 }));
        assert_eq!(ok["jsonrpc"], "2.0");
        assert_eq!(ok["id"], 1);
        assert_eq!(ok["result"]["x"], 2);
        assert!(ok.get("error").is_none());

        let err = rpc_error(json!("abc"), -32601, "nope".to_string());
        assert_eq!(err["id"], "abc");
        assert_eq!(err["error"]["code"], -32601);
        assert_eq!(err["error"]["message"], "nope");
        assert!(err.get("result").is_none());
    }

    #[test]
    fn tool_result_helpers_set_error_flag() {
        let ok = tool_text("hi".to_string());
        assert_eq!(ok["content"][0]["type"], "text");
        assert_eq!(ok["content"][0]["text"], "hi");
        assert!(ok.get("isError").is_none());

        let err = tool_error("boom");
        assert_eq!(err["isError"], true);
        assert_eq!(err["content"][0]["text"], "boom");
    }

    #[test]
    fn str_arg_trims_and_rejects_blank() {
        let args = json!({ "query": "  hello  ", "blank": "   ", "n": 3 });
        assert_eq!(str_arg(&args, "query"), Some("hello"));
        assert_eq!(str_arg(&args, "blank"), None);
        assert_eq!(str_arg(&args, "missing"), None);
        assert_eq!(str_arg(&args, "n"), None);
    }

    #[test]
    fn usize_arg_parses_only_unsigned() {
        let args = json!({ "limit": 5, "neg": -1, "s": "7" });
        assert_eq!(usize_arg(&args, "limit"), Some(5));
        assert_eq!(usize_arg(&args, "neg"), None);
        assert_eq!(usize_arg(&args, "s"), None);
        assert_eq!(usize_arg(&args, "missing"), None);
    }

    #[test]
    fn truncate_respects_utf8_boundaries() {
        assert_eq!(truncate_on_boundary("short", 10), "short");
        // "a" + "é"×10: 'a' at byte 0, then each 'é' spans 2 bytes. Byte offset 4
        // lands inside the second 'é', so truncation must step back to offset 3 -
        // a naive `&text[..4]` would panic. (Offset 5 is already a boundary and
        // would not exercise this path.)
        let s = "a".to_string() + &"é".repeat(10);
        let out = truncate_on_boundary(&s, 4);
        assert!(out.ends_with("(truncated)"));
        // Stepped back to the boundary at byte 3: "a" + one "é".
        assert!(out.starts_with("aé…"));
    }

    fn chunk() -> CodeChunk {
        CodeChunk {
            id: Some(1),
            repo_id: 1,
            file_path: "src/lib.rs".to_string(),
            chunk_type: "function".to_string(),
            symbol_name: Some("do_thing".to_string()),
            language: Language::Rust,
            start_line: 10,
            end_line: 20,
            chunk_text: "fn do_thing() {}".to_string(),
            context_text: String::new(),
            file_hash: "h".to_string(),
            content_hash: None,
        }
    }

    #[test]
    fn format_code_results_renders_matches() {
        assert_eq!(format_code_results(&[]), "No matching code found.");

        let results = vec![CodeSearchResult {
            chunk: chunk(),
            score: 0.9321,
        }];
        let out = format_code_results(&results);
        assert!(out.contains("src/lib.rs:10-20"));
        assert!(out.contains("93% match"));
        assert!(out.contains("do_thing"));
        assert!(out.contains("fn do_thing()"));
    }

    #[test]
    fn format_symbols_renders_matches() {
        assert_eq!(format_symbols(&[]), "No matching symbols found.");

        let symbols = vec![CodeSymbol {
            id: Some(1),
            repo_id: 1,
            file_path: "src/lib.rs".to_string(),
            symbol_name: "do_thing".to_string(),
            symbol_kind: SymbolKind::Function,
            parent_symbol: Some("Thing".to_string()),
            language: Language::Rust,
            start_line: 10,
            end_line: 20,
            signature: Some("fn do_thing()".to_string()),
        }];
        let out = format_symbols(&symbols);
        assert!(out.contains("**do_thing** (function)"));
        assert!(out.contains("src/lib.rs:10-20"));
        assert!(out.contains("in `Thing`"));
        assert!(out.contains("fn do_thing()"));
    }

    // --- request-path (observable response) tests ---------------------------

    #[tokio::test]
    async fn initialize_returns_capabilities_and_negotiated_version() {
        let cfg = cfg(true, true);
        let tracker = SqliteTracker::in_memory().unwrap();
        let ctx = test_ctx(&cfg, &tracker);

        let resp = post(
            &ctx,
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": { "protocolVersion": "2025-06-18" }
            }),
        )
        .await
        .expect("initialize returns a body");

        assert_eq!(resp["id"], 1);
        assert_eq!(resp["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(resp["result"]["serverInfo"]["name"], "claudear-search");
        assert!(resp["result"]["capabilities"]["tools"].is_object());
    }

    #[tokio::test]
    async fn initialize_downgrades_unknown_version() {
        let cfg = cfg(true, true);
        let tracker = SqliteTracker::in_memory().unwrap();
        let ctx = test_ctx(&cfg, &tracker);

        let resp = post(
            &ctx,
            json!({ "id": 1, "method": "initialize", "params": { "protocolVersion": "3000-01-01" } }),
        )
        .await
        .unwrap();
        assert_eq!(resp["result"]["protocolVersion"], DEFAULT_PROTOCOL_VERSION);
    }

    #[tokio::test]
    async fn tools_list_reflects_exposure_toggles() {
        let cfg = cfg(true, false);
        let tracker = SqliteTracker::in_memory().unwrap();
        let ctx = test_ctx(&cfg, &tracker);

        let resp = post(&ctx, json!({ "id": 2, "method": "tools/list" }))
            .await
            .unwrap();
        let names = tool_names(resp["result"]["tools"].as_array().unwrap());
        assert_eq!(names, vec!["code_search", "find_symbol"]);
    }

    #[tokio::test]
    async fn ping_returns_empty_result() {
        let cfg = cfg(true, true);
        let tracker = SqliteTracker::in_memory().unwrap();
        let ctx = test_ctx(&cfg, &tracker);

        let resp = post(&ctx, json!({ "id": 9, "method": "ping" }))
            .await
            .unwrap();
        assert_eq!(resp["result"], json!({}));
    }

    #[tokio::test]
    async fn notification_is_acknowledged_without_body() {
        let cfg = cfg(true, true);
        let tracker = SqliteTracker::in_memory().unwrap();
        let ctx = test_ctx(&cfg, &tracker);

        // No `id` => notification => 202 with no JSON body.
        let resp = post(&ctx, json!({ "method": "notifications/initialized" })).await;
        assert!(resp.is_none());
    }

    #[tokio::test]
    async fn unknown_method_returns_method_not_found() {
        let cfg = cfg(true, true);
        let tracker = SqliteTracker::in_memory().unwrap();
        let ctx = test_ctx(&cfg, &tracker);

        let resp = post(&ctx, json!({ "id": 3, "method": "does/not/exist" }))
            .await
            .unwrap();
        assert_eq!(resp["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn parse_error_and_batch_are_rejected() {
        let cfg = cfg(true, true);
        let tracker = SqliteTracker::in_memory().unwrap();
        let ctx = test_ctx(&cfg, &tracker);

        let PostOutcome::Json(parse) = process_post(&ctx, b"{ not json").await else {
            panic!("expected JSON body");
        };
        assert_eq!(parse["error"]["code"], -32700);

        let PostOutcome::Json(batch) = process_post(&ctx, b"[{}]").await else {
            panic!("expected JSON body");
        };
        assert_eq!(batch["error"]["code"], -32600);
    }

    #[tokio::test]
    async fn tool_call_reports_unavailable_service_as_error_result() {
        let cfg = cfg(true, true);
        let tracker = SqliteTracker::in_memory().unwrap();
        let ctx = test_ctx(&cfg, &tracker);

        let resp = post(
            &ctx,
            json!({
                "id": 4,
                "method": "tools/call",
                "params": { "name": "code_search", "arguments": { "query": "foo" } }
            }),
        )
        .await
        .unwrap();

        // Service unavailable is a tool-execution error, not a JSON-RPC error.
        assert!(resp.get("error").is_none());
        assert_eq!(resp["result"]["isError"], true);
        assert!(resp["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("unavailable"));
    }

    #[tokio::test]
    async fn tool_call_rejects_unknown_or_disabled_tool() {
        let cfg = cfg(true, false); // discord disabled
        let tracker = SqliteTracker::in_memory().unwrap();
        let ctx = test_ctx(&cfg, &tracker);

        let resp = post(
            &ctx,
            json!({
                "id": 5,
                "method": "tools/call",
                "params": { "name": "discord_search", "arguments": { "query": "x" } }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp["error"]["code"], -32602);
    }
}
