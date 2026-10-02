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
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use serde_json::{json, Value};
use std::sync::Arc;

/// Protocol version advertised when the client does not request one.
const DEFAULT_PROTOCOL_VERSION: &str = "2025-06-18";

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
    body: axum::body::Bytes,
) -> Response {
    let request: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(e) => {
            return Json(rpc_error(
                Value::Null,
                -32700,
                format!("Parse error: {e}"),
            ))
            .into_response();
        }
    };

    // JSON-RPC batching was removed in the 2025-06-18 spec; we only accept a
    // single request object and reject arrays explicitly.
    if request.is_array() {
        return Json(rpc_error(
            Value::Null,
            -32600,
            "Batch requests are not supported".to_string(),
        ))
        .into_response();
    }

    let method = request.get("method").and_then(Value::as_str).unwrap_or("");
    let id = request.get("id").cloned();
    let params = request.get("params").cloned().unwrap_or(Value::Null);

    // A request without an `id` is a notification: do the work (nothing, for the
    // notifications we expect) and acknowledge without a JSON-RPC body.
    let Some(id) = id else {
        return StatusCode::ACCEPTED.into_response();
    };

    let response = match method {
        "initialize" => rpc_result(id, initialize_result(&params)),
        "ping" => rpc_result(id, json!({})),
        "tools/list" => rpc_result(id, json!({ "tools": tool_definitions(&state) })),
        "tools/call" => match call_tool(&state, &params).await {
            Ok(result) => rpc_result(id, result),
            Err(ToolError::Protocol { code, message }) => rpc_error(id, code, message),
        },
        other => rpc_error(id, -32601, format!("Method not found: {other}")),
    };

    Json(response).into_response()
}

fn initialize_result(params: &Value) -> Value {
    // Echo the client's requested protocol version when present so we negotiate
    // a version both sides understand; otherwise advertise our default.
    let protocol_version = params
        .get("protocolVersion")
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
        .unwrap_or(DEFAULT_PROTOCOL_VERSION);

    json!({
        "protocolVersion": protocol_version,
        "capabilities": { "tools": {} },
        "serverInfo": {
            "name": "claudear-search",
            "version": env!("CARGO_PKG_VERSION"),
        },
    })
}

/// Build the list of exposed tools based on configuration.
fn tool_definitions(state: &AppState) -> Vec<Value> {
    let cfg = &state.config.mcp_server;
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
                    "repo": { "type": "string", "description": "Optional repository name to scope the search." }
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

async fn call_tool(state: &AppState, params: &Value) -> Result<Value, ToolError> {
    let name = params.get("name").and_then(Value::as_str).ok_or_else(|| {
        ToolError::Protocol {
            code: -32602,
            message: "Missing tool name".to_string(),
        }
    })?;
    let args = params.get("arguments").cloned().unwrap_or(json!({}));
    let cfg = &state.config.mcp_server;

    match name {
        "code_search" if cfg.expose_code => Ok(code_search(state, &args).await),
        "find_symbol" if cfg.expose_code => Ok(find_symbol(state, &args)),
        "discord_search" if cfg.expose_discord => Ok(discord_search(state, &args).await),
        other => Err(ToolError::Protocol {
            code: -32602,
            message: format!("Unknown tool: {other}"),
        }),
    }
}

async fn code_search(state: &AppState, args: &Value) -> Value {
    let Some(service) = state.code_search_service.as_ref() else {
        return tool_error("Code search is unavailable (code indexing or embeddings not configured).");
    };
    let Some(query) = str_arg(args, "query") else {
        return tool_error("Missing required argument: query");
    };

    let repo_id = match resolve_repo(state, args) {
        Ok(id) => id,
        Err(msg) => return tool_error(&msg),
    };
    let limit = state
        .config
        .mcp_server
        .resolve_limit(usize_arg(args, "limit"));

    match service.search(query, repo_id, limit).await {
        Ok(results) => tool_text(format_code_results(&results)),
        Err(e) => tool_error(&format!("Code search failed: {e}")),
    }
}

fn find_symbol(state: &AppState, args: &Value) -> Value {
    let Some(service) = state.code_search_service.as_ref() else {
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
    let repo_id = match resolve_repo(state, args) {
        Ok(id) => id,
        Err(msg) => return tool_error(&msg),
    };

    match service.find_symbol(name, kind, repo_id) {
        Ok(symbols) => tool_text(format_symbols(&symbols)),
        Err(e) => tool_error(&format!("Symbol search failed: {e}")),
    }
}

async fn discord_search(state: &AppState, args: &Value) -> Value {
    let Some(service) = state.discord_search_service.as_ref() else {
        return tool_error("Discord search is unavailable (Discord knowledgebase not configured).");
    };
    let Some(query) = str_arg(args, "query") else {
        return tool_error("Missing required argument: query");
    };

    let channel_id = str_arg(args, "channel_id");
    let limit = state
        .config
        .mcp_server
        .resolve_limit(usize_arg(args, "limit"));

    match service.search(query, channel_id, limit).await {
        Ok(results) if results.is_empty() => tool_text("No relevant Discord discussions found.".to_string()),
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
    args.get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
}

/// Resolve an optional repo scope from the arguments into a repo id.
///
/// Accepts either `repo_id` (integer) directly, or `repo` (name) which is
/// looked up in the index. Returns `Ok(None)` when neither is supplied, and
/// `Err` with a user-facing message when a given name is not indexed.
fn resolve_repo(state: &AppState, args: &Value) -> Result<Option<i64>, String> {
    if let Some(id) = args.get("repo_id").and_then(Value::as_i64) {
        return Ok(Some(id));
    }
    let Some(name) = str_arg(args, "repo") else {
        return Ok(None);
    };
    match state.tracker.get_indexed_repo(name) {
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
