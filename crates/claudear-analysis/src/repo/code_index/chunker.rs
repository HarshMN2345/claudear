//! AST-aware semantic chunking of source files.
//!
//! Produces chunks aligned to symbol boundaries (functions, classes, top-level code)
//! suitable for embedding and vector similarity search.

use super::parser::{extract_symbols, parse_file};
use super::types::{CodeChunk, CodeSymbol, Language, SymbolKind};
use claudear_core::error::Result;

/// Maximum chunk size in characters (~1500 tokens).
const MAX_CHUNK_CHARS: usize = 6000;
/// Minimum chunk size — only truly empty/whitespace-only chunks are rejected.
/// Small functions (getters, setters, one-liners) are preserved for full coverage.
const MIN_CHUNK_CHARS: usize = 15;
/// Large class threshold: classes above this line count get split into method chunks.
const LARGE_CLASS_LINES: usize = 100;
/// Maximum characters of import context to include per chunk.
const MAX_IMPORT_CONTEXT_CHARS: usize = 500;
/// Number of lines to carry forward when splitting oversized chunks.
const OVERLAP_LINES: usize = 3;

/// Chunk a source file using its AST, producing semantic chunks for embedding.
pub fn chunk_file(
    source: &str,
    language: Language,
    repo_id: i64,
    file_path: &str,
    file_hash: &str,
) -> Result<(Vec<CodeSymbol>, Vec<CodeChunk>)> {
    // Skip generated/minified files that would pollute the index.
    if is_likely_generated(source, file_path) {
        return Ok((Vec::new(), Vec::new()));
    }

    let tree = parse_file(source, language)?;
    let symbols = extract_symbols(&tree, source.as_bytes(), language, repo_id, file_path);

    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut covered_lines = vec![false; lines.len()];

    // Extract file-level imports for inclusion in chunk context.
    let imports = extract_imports(source, language);

    // Phase 1: Create chunks from top-level container symbols (classes, impls, traits).
    for sym in symbols.iter().filter(|s| is_top_level_container(s)) {
        let start = sym.start_line.saturating_sub(1); // convert to 0-indexed
        let end = sym.end_line.min(lines.len());
        let line_count = end.saturating_sub(start);

        if line_count > LARGE_CLASS_LINES {
            // Large container: skeleton overview + per-method chunks.
            chunks.push(build_skeleton_chunk(
                &lines,
                sym,
                &symbols,
                language,
                repo_id,
                file_path,
                file_hash,
                imports.as_deref(),
            ));

            // Create individual method/function chunks for every child symbol.
            for method in symbols.iter().filter(|s| {
                s.parent_symbol.as_deref() == Some(&sym.symbol_name)
                    && matches!(s.symbol_kind, SymbolKind::Method | SymbolKind::Function)
            }) {
                let m_start = method.start_line.saturating_sub(1);
                let m_end = method.end_line.min(lines.len());
                let chunk = build_symbol_chunk(
                    &lines,
                    method,
                    language,
                    repo_id,
                    file_path,
                    file_hash,
                    imports.as_deref(),
                );
                chunks.push(chunk);
                mark_covered(&mut covered_lines, m_start, m_end);
            }
        } else {
            // Small container: single chunk with all its contents.
            let chunk = build_symbol_chunk(
                &lines,
                sym,
                language,
                repo_id,
                file_path,
                file_hash,
                imports.as_deref(),
            );
            chunks.push(chunk);
        }
        mark_covered(&mut covered_lines, start, end);
    }

    // Phase 2: Create chunks from standalone functions (not inside containers).
    for sym in symbols
        .iter()
        .filter(|s| s.parent_symbol.is_none() && matches!(s.symbol_kind, SymbolKind::Function))
    {
        let start = sym.start_line.saturating_sub(1);
        let end = sym.end_line.min(lines.len());
        if !covered_lines.get(start).copied().unwrap_or(true) {
            let chunk = build_symbol_chunk(
                &lines,
                sym,
                language,
                repo_id,
                file_path,
                file_hash,
                imports.as_deref(),
            );
            chunks.push(chunk);
            mark_covered(&mut covered_lines, start, end);
        }
    }

    // Phase 3: Standalone constants/enums that aren't inside containers.
    for sym in symbols.iter().filter(|s| {
        s.parent_symbol.is_none()
            && matches!(
                s.symbol_kind,
                SymbolKind::Constant | SymbolKind::Enum | SymbolKind::Struct
            )
            && !is_top_level_container(s)
    }) {
        let start = sym.start_line.saturating_sub(1);
        let end = sym.end_line.min(lines.len());
        if !covered_lines.get(start).copied().unwrap_or(true) {
            let chunk = build_symbol_chunk(
                &lines,
                sym,
                language,
                repo_id,
                file_path,
                file_hash,
                imports.as_deref(),
            );
            chunks.push(chunk);
            mark_covered(&mut covered_lines, start, end);
        }
    }

    // Phase 4: Collect uncovered top-level code into contiguous block chunks.
    // Instead of arbitrary batching, group by contiguous non-empty lines
    // separated by blank lines.
    build_top_level_chunks(
        &lines,
        &covered_lines,
        language,
        repo_id,
        file_path,
        file_hash,
        imports.as_deref(),
        &mut chunks,
    );

    // Phase 5: Split any oversized chunks.
    let final_chunks = split_oversized(chunks);

    Ok((symbols, final_chunks))
}

fn is_likely_generated(source: &str, file_path: &str) -> bool {
    abnegate_index::generated(file_path, source)
}

/// Extract file-level import/use statements for enriching chunk context.
///
/// Returns a compact summary of imports (capped at [`MAX_IMPORT_CONTEXT_CHARS`]).
fn extract_imports(source: &str, language: Language) -> Option<String> {
    let mut import_lines = Vec::new();
    let mut in_go_import_block = false;

    for line in source.lines() {
        let trimmed = line.trim();

        // Track Go's `import (...)` block.
        if language == Language::Go {
            if trimmed == "import (" {
                in_go_import_block = true;
                continue;
            }
            if in_go_import_block {
                if trimmed == ")" {
                    in_go_import_block = false;
                } else if !trimmed.is_empty() {
                    import_lines.push(trimmed.to_string());
                }
                continue;
            }
        }

        let is_import = match language {
            Language::Rust => trimmed.starts_with("use ") || trimmed.starts_with("pub use "),
            Language::Python => trimmed.starts_with("import ") || trimmed.starts_with("from "),
            Language::JavaScript | Language::TypeScript | Language::Tsx => {
                trimmed.starts_with("import ") || trimmed.contains("require(")
            }
            Language::Go => trimmed.starts_with("import "),
            Language::Java | Language::Kotlin => {
                trimmed.starts_with("import ") || trimmed.starts_with("package ")
            }
            Language::C | Language::Cpp => trimmed.starts_with("#include"),
            Language::CSharp => trimmed.starts_with("using "),
            Language::Ruby => {
                trimmed.starts_with("require ") || trimmed.starts_with("require_relative ")
            }
            Language::Php => trimmed.starts_with("use ") || trimmed.starts_with("namespace "),
            Language::Swift => trimmed.starts_with("import "),
            Language::Dart => trimmed.starts_with("import ") || trimmed.starts_with("part "),
            _ => false,
        };

        if is_import {
            import_lines.push(trimmed.to_string());
        }
    }

    if import_lines.is_empty() {
        return None;
    }

    // Cap total import context size.
    let mut result = String::new();
    for line in &import_lines {
        if result.len() + line.len() + 1 > MAX_IMPORT_CONTEXT_CHARS {
            result.push_str("// ...\n");
            break;
        }
        result.push_str(line);
        result.push('\n');
    }

    Some(result)
}

fn is_top_level_container(sym: &CodeSymbol) -> bool {
    sym.parent_symbol.is_none()
        && matches!(
            sym.symbol_kind,
            SymbolKind::Class
                | SymbolKind::Struct
                | SymbolKind::Trait
                | SymbolKind::Impl
                | SymbolKind::Interface
                | SymbolKind::Enum
                | SymbolKind::Module
        )
}

fn mark_covered(covered: &mut [bool], start: usize, end: usize) {
    for line in covered.iter_mut().take(end).skip(start) {
        *line = true;
    }
}

fn collect_lines(lines: &[&str], start: usize, end: usize) -> String {
    lines[start..end.min(lines.len())].join("\n")
}

/// Gather preceding comment/attribute lines for a symbol.
///
/// Handles multi-line Python docstrings (`"""` / `'''`): when we encounter a
/// closing delimiter we continue walking back until the matching opening
/// delimiter is found so that the full docstring is captured.
fn gather_leading_context(lines: &[&str], start_0idx: usize) -> usize {
    if start_0idx >= lines.len() {
        return start_0idx;
    }
    let mut ctx_start = start_0idx;
    let mut inside_triple_quote: Option<&str> = None; // "\"\"\"" or "'''"
    while ctx_start > 0 {
        let prev = lines[ctx_start - 1].trim();

        // When inside a multi-line docstring, keep walking until we find
        // the matching opening delimiter.
        if let Some(delim) = inside_triple_quote {
            ctx_start -= 1;
            if prev.contains(delim) {
                inside_triple_quote = None;
            }
            continue;
        }

        if prev.is_empty() {
            break;
        }

        // Check for closing triple-quote delimiter that starts a backwards
        // walk through a multi-line docstring.
        if prev.ends_with("\"\"\"") || prev.starts_with("\"\"\"") {
            let delim = "\"\"\"";
            ctx_start -= 1;
            // If the opening and closing are on the same line, we're done.
            let count = prev.matches(delim).count();
            if count < 2 {
                inside_triple_quote = Some(delim);
            }
            continue;
        }
        if prev.ends_with("'''") || prev.starts_with("'''") {
            let delim = "'''";
            ctx_start -= 1;
            let count = prev.matches(delim).count();
            if count < 2 {
                inside_triple_quote = Some(delim);
            }
            continue;
        }

        if prev.starts_with("//")       // C-style line comments (includes ///)
            || prev.starts_with("#[")    // Rust attributes
            || prev.starts_with("#![")   // Rust inner attributes
            || prev.starts_with("# ")    // Python/Ruby comments
            || prev.starts_with("/**")   // Block doc comment start
            || prev.starts_with("* ")    // Block doc comment continuation
            || prev.starts_with("*/")    // Block doc comment end
            || prev.starts_with("@")
        // Java/Kotlin annotations
        {
            ctx_start -= 1;
        } else {
            break;
        }
    }
    ctx_start
}

/// Build a chunk for a symbol (function, class, method, etc.).
///
/// Unlike the previous version, this never returns None — even small functions
/// (getters, one-liners) are preserved for full search coverage.
fn build_symbol_chunk(
    lines: &[&str],
    sym: &CodeSymbol,
    language: Language,
    repo_id: i64,
    file_path: &str,
    file_hash: &str,
    imports: Option<&str>,
) -> CodeChunk {
    let raw_start = sym.start_line.saturating_sub(1);
    let start = gather_leading_context(lines, raw_start);
    let end = sym.end_line.min(lines.len());

    let text = if end > start {
        collect_lines(lines, start, end)
    } else {
        String::new()
    };

    let chunk_type = match sym.symbol_kind {
        SymbolKind::Function => "function",
        SymbolKind::Method => "function",
        SymbolKind::Class => "class",
        SymbolKind::Struct => "class",
        SymbolKind::Impl => "impl_block",
        SymbolKind::Trait => "class",
        SymbolKind::Interface => "class",
        SymbolKind::Enum => "class",
        SymbolKind::Module => "module_header",
        SymbolKind::Constant => "top_level",
    };

    let prefix = build_context_prefix(
        file_path,
        language,
        chunk_type,
        Some(&sym.symbol_name),
        sym.parent_symbol.as_deref(),
        sym.signature.as_deref(),
        imports,
    );

    CodeChunk {
        id: None,
        repo_id,
        file_path: file_path.to_string(),
        chunk_type: chunk_type.to_string(),
        symbol_name: Some(sym.symbol_name.clone()),
        language,
        start_line: start + 1,
        end_line: end,
        chunk_text: text,
        context_text: prefix,
        file_hash: file_hash.to_string(),
        content_hash: None,
    }
}

/// Build a skeleton chunk for a large class (declaration + method signatures only).
#[allow(clippy::too_many_arguments)]
fn build_skeleton_chunk(
    lines: &[&str],
    sym: &CodeSymbol,
    all_symbols: &[CodeSymbol],
    language: Language,
    repo_id: i64,
    file_path: &str,
    file_hash: &str,
    imports: Option<&str>,
) -> CodeChunk {
    let raw_start = sym.start_line.saturating_sub(1);
    let start = gather_leading_context(lines, raw_start);
    let end = sym.end_line.min(lines.len());

    // Collect leading comments/attributes + first few lines of the class.
    let mut skeleton = String::new();
    if start < raw_start {
        skeleton.push_str(&collect_lines(lines, start, raw_start));
        skeleton.push('\n');
    }
    // First 5 lines of the class body
    let header_end = (raw_start + 5).min(end);
    skeleton.push_str(&collect_lines(lines, raw_start, header_end));
    skeleton.push_str("\n\n    // ... methods ...\n\n");

    for method in all_symbols.iter().filter(|s| {
        s.parent_symbol.as_deref() == Some(&sym.symbol_name)
            && matches!(s.symbol_kind, SymbolKind::Method | SymbolKind::Function)
    }) {
        if let Some(ref sig) = method.signature {
            skeleton.push_str("    ");
            skeleton.push_str(sig);
            skeleton.push_str(" { ... }\n");
        }
    }

    let prefix = build_context_prefix(
        file_path,
        language,
        "class",
        Some(&sym.symbol_name),
        None,
        sym.signature.as_deref(),
        imports,
    );

    CodeChunk {
        id: None,
        repo_id,
        file_path: file_path.to_string(),
        chunk_type: "class".to_string(),
        symbol_name: Some(sym.symbol_name.clone()),
        language,
        start_line: start + 1,
        end_line: end,
        chunk_text: skeleton,
        context_text: prefix,
        file_hash: file_hash.to_string(),
        content_hash: None,
    }
}

/// Group contiguous uncovered lines into semantic top-level blocks.
///
/// Splits on blank lines to create natural groupings rather than
/// arbitrary fixed-size batches.
#[allow(clippy::too_many_arguments)]
fn build_top_level_chunks(
    lines: &[&str],
    covered_lines: &[bool],
    language: Language,
    repo_id: i64,
    file_path: &str,
    file_hash: &str,
    imports: Option<&str>,
    chunks: &mut Vec<CodeChunk>,
) {
    // Collect contiguous blocks of uncovered non-empty lines.
    let mut blocks: Vec<(usize, usize)> = Vec::new(); // (start, end) 0-indexed
    let mut block_start: Option<usize> = None;
    let mut consecutive_blanks = 0;

    for (i, &covered) in covered_lines.iter().enumerate() {
        let is_blank = lines.get(i).unwrap_or(&"").trim().is_empty();

        if !covered && !is_blank {
            if block_start.is_none() {
                block_start = Some(i);
            }
            consecutive_blanks = 0;
        } else if block_start.is_some() {
            if is_blank && !covered {
                consecutive_blanks += 1;
                // Two+ consecutive blank lines = block boundary.
                if consecutive_blanks >= 2 {
                    if let Some(bs) = block_start.take() {
                        let end = i.saturating_sub(consecutive_blanks - 1);
                        if end > bs {
                            blocks.push((bs, end));
                        }
                    }
                    consecutive_blanks = 0;
                }
            } else {
                // Covered line or non-blank covered: end the block.
                if let Some(bs) = block_start.take() {
                    blocks.push((bs, i));
                }
                consecutive_blanks = 0;
            }
        }
    }
    // Flush final block.
    if let Some(bs) = block_start {
        blocks.push((bs, lines.len()));
    }

    let prefix = build_context_prefix(file_path, language, "top_level", None, None, None, imports);

    for (block_start_idx, block_end_idx) in blocks {
        // Collect only uncovered lines within the block.
        let text: String = (block_start_idx..block_end_idx)
            .filter(|&i| !covered_lines.get(i).copied().unwrap_or(true))
            .map(|i| lines.get(i).copied().unwrap_or(""))
            .collect::<Vec<&str>>()
            .join("\n");

        if text.len() >= MIN_CHUNK_CHARS {
            chunks.push(CodeChunk {
                id: None,
                repo_id,
                file_path: file_path.to_string(),
                chunk_type: "top_level".to_string(),
                symbol_name: None,
                language,
                start_line: block_start_idx + 1,
                end_line: block_end_idx,
                chunk_text: text,
                context_text: prefix.clone(),
                file_hash: file_hash.to_string(),
                content_hash: None,
            });
        }
    }
}

/// Build the metadata prefix stored in the DB.
///
/// Includes: file path, language, symbol info, parent, signature, and imports.
/// This prefix is prepended to the code body at embedding time for rich context.
pub fn build_context_prefix(
    file_path: &str,
    language: Language,
    chunk_type: &str,
    symbol_name: Option<&str>,
    parent: Option<&str>,
    signature: Option<&str>,
    imports: Option<&str>,
) -> String {
    let mut ctx = format!("File: {}\nLanguage: {}\n", file_path, language);

    if let Some(name) = symbol_name {
        ctx.push_str(&format!("Symbol: {} ({})\n", name, chunk_type));
    } else {
        ctx.push_str(&format!("Type: {}\n", chunk_type));
    }

    if let Some(p) = parent {
        ctx.push_str(&format!("Parent: {}\n", p));
    }

    if let Some(sig) = signature {
        ctx.push_str(&format!("Signature: {}\n", sig));
    }

    if let Some(imp) = imports {
        if !imp.is_empty() {
            ctx.push_str("Imports:\n");
            ctx.push_str(imp);
        }
    }

    ctx
}

/// Build full embedding text from a chunk's stored context_text and code body.
///
/// This is the preferred path for embedding — uses the rich context prefix
/// (with imports, signature) stored on the chunk rather than reconstructing.
pub fn build_embedding_text(context_text: &str, chunk_text: &str) -> String {
    let mut text = context_text.to_string();
    text.push('\n');
    if chunk_text.len() > MAX_CHUNK_CHARS {
        let mut end = MAX_CHUNK_CHARS;
        while end > 0 && !chunk_text.is_char_boundary(end) {
            end -= 1;
        }
        text.push_str(&chunk_text[..end]);
        text.push_str("\n// ... truncated ...");
    } else {
        text.push_str(chunk_text);
    }
    text
}

/// Split chunks that exceed MAX_CHUNK_CHARS.
///
/// Carries forward the last [`OVERLAP_LINES`] lines from each sub-chunk to the
/// start of the next so that boundary context is preserved for better embedding
/// quality.
fn split_oversized(chunks: Vec<CodeChunk>) -> Vec<CodeChunk> {
    let mut result = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        if chunk.chunk_text.len() <= MAX_CHUNK_CHARS {
            result.push(chunk);
            continue;
        }

        // Split at line boundaries near the limit.
        let lines: Vec<&str> = chunk.chunk_text.lines().collect();
        let mut part_start = 0;
        let mut current_size = 0;
        let base_start_line = chunk.start_line;

        for (i, line) in lines.iter().enumerate() {
            current_size += line.len() + 1; // +1 for newline
            if current_size >= MAX_CHUNK_CHARS || i == lines.len() - 1 {
                let part_end = i + 1;
                let text = lines[part_start..part_end].join("\n");
                if text.len() >= MIN_CHUNK_CHARS {
                    result.push(CodeChunk {
                        id: None,
                        repo_id: chunk.repo_id,
                        file_path: chunk.file_path.clone(),
                        chunk_type: chunk.chunk_type.clone(),
                        symbol_name: chunk.symbol_name.clone(),
                        language: chunk.language,
                        start_line: base_start_line + part_start,
                        end_line: base_start_line + part_end - 1,
                        chunk_text: text,
                        context_text: chunk.context_text.clone(),
                        file_hash: chunk.file_hash.clone(),
                        content_hash: None,
                    });
                }
                // Carry forward the last OVERLAP_LINES lines for context.
                part_start = part_end.saturating_sub(OVERLAP_LINES);
                current_size = lines[part_start..part_end]
                    .iter()
                    .map(|l| l.len() + 1)
                    .sum();
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: build embedding text from individual components.
    fn build_context_text(
        file_path: &str,
        language: Language,
        chunk_type: &str,
        symbol_name: Option<&str>,
        parent: Option<&str>,
        code: &str,
    ) -> String {
        let prefix = build_context_prefix(
            file_path,
            language,
            chunk_type,
            symbol_name,
            parent,
            None,
            None,
        );
        build_embedding_text(&prefix, code)
    }

    #[test]
    fn test_chunk_rust_file() {
        let src = r#"
use std::io;

/// A greeter.
pub struct Greeter {
    name: String,
}

impl Greeter {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string() }
    }

    pub fn greet(&self) -> String {
        format!("Hello, {}!", self.name)
    }
}

fn standalone() {
    println!("I'm standalone");
}
"#;
        let (symbols, chunks) = chunk_file(src, Language::Rust, 1, "greeter.rs", "abc123").unwrap();

        assert!(!symbols.is_empty());
        assert!(!chunks.is_empty());

        // Should have at least: struct chunk, impl chunk or method chunks, standalone fn chunk
        let chunk_types: Vec<&str> = chunks.iter().map(|c| c.chunk_type.as_str()).collect();
        assert!(
            chunk_types.contains(&"function")
                || chunk_types.contains(&"class")
                || chunk_types.contains(&"impl_block"),
            "chunk_types = {:?}",
            chunk_types
        );

        // All chunks should have context_text populated
        for chunk in &chunks {
            assert!(chunk.context_text.contains("File: greeter.rs"));
            assert!(chunk.context_text.contains("Language: Rust"));
            assert_eq!(chunk.file_hash, "abc123");
        }
    }

    #[test]
    fn test_chunk_python_class_and_function() {
        let src = r#"
import os

class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

def main():
    calc = Calculator()
    print(calc.add(1, 2))
"#;
        let (_, chunks) = chunk_file(src, Language::Python, 1, "calc.py", "hash1").unwrap();
        assert!(!chunks.is_empty());

        // Check that the class is chunked
        assert!(chunks
            .iter()
            .any(|c| c.symbol_name.as_deref() == Some("Calculator")));
    }

    #[test]
    fn test_chunk_empty_file() {
        let src = "";
        let (symbols, chunks) = chunk_file(src, Language::Rust, 1, "empty.rs", "h").unwrap();
        assert!(symbols.is_empty());
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_build_context_text_format() {
        let ctx = build_context_text(
            "src/auth.rs",
            Language::Rust,
            "function",
            Some("authenticate"),
            Some("AuthService"),
            "pub fn authenticate(&self) -> bool { true }",
        );
        assert!(ctx.contains("File: src/auth.rs"));
        assert!(ctx.contains("Language: Rust"));
        assert!(ctx.contains("Symbol: authenticate (function)"));
        assert!(ctx.contains("Parent: AuthService"));
        assert!(ctx.contains("pub fn authenticate"));
    }

    #[test]
    fn test_small_functions_preserved() {
        // Small functions should NOT be silently dropped.
        let src = r#"
fn get_name() -> &str {
    "alice"
}

fn get_age() -> u32 {
    30
}
"#;
        let (_, chunks) = chunk_file(src, Language::Rust, 1, "getters.rs", "h").unwrap();
        let fn_names: Vec<&str> = chunks
            .iter()
            .filter_map(|c| c.symbol_name.as_deref())
            .collect();
        assert!(
            fn_names.contains(&"get_name"),
            "Small function get_name should be preserved, got: {:?}",
            fn_names
        );
        assert!(
            fn_names.contains(&"get_age"),
            "Small function get_age should be preserved, got: {:?}",
            fn_names
        );
    }

    #[test]
    fn test_imports_included_in_context() {
        let src = r#"
use std::collections::HashMap;
use std::io::Read;

fn process(data: &HashMap<String, Vec<u8>>) -> String {
    let mut result = String::new();
    for (key, value) in data {
        result.push_str(key);
    }
    result
}
"#;
        let (_, chunks) = chunk_file(src, Language::Rust, 1, "process.rs", "h").unwrap();
        let fn_chunk = chunks
            .iter()
            .find(|c| c.symbol_name.as_deref() == Some("process"))
            .expect("Should have a process function chunk");

        assert!(
            fn_chunk.context_text.contains("Imports:"),
            "Context should include imports section"
        );
        assert!(
            fn_chunk.context_text.contains("HashMap"),
            "Context should include HashMap import"
        );
    }

    #[test]
    fn test_generated_file_skipped() {
        let src = "// Code generated by protoc-gen-go. DO NOT EDIT.\npackage pb\n";
        let (symbols, chunks) = chunk_file(src, Language::Go, 1, "api.pb.go", "h").unwrap();
        assert!(
            symbols.is_empty(),
            "Generated file should produce no symbols"
        );
        assert!(chunks.is_empty(), "Generated file should produce no chunks");
    }

    #[test]
    fn test_minified_file_skipped() {
        // Single very long line simulating minified JS.
        let src = "x".repeat(10000);
        let (symbols, chunks) =
            chunk_file(&src, Language::JavaScript, 1, "app.min.js", "h").unwrap();
        assert!(
            symbols.is_empty(),
            "Minified file should produce no symbols"
        );
        assert!(chunks.is_empty(), "Minified file should produce no chunks");
    }

    #[test]
    fn test_signature_in_context() {
        let src = r#"
pub fn calculate_total(items: &[Item], tax_rate: f64) -> f64 {
    items.iter().map(|i| i.price).sum::<f64>() * (1.0 + tax_rate)
}
"#;
        let (_, chunks) = chunk_file(src, Language::Rust, 1, "pricing.rs", "h").unwrap();
        let fn_chunk = chunks
            .iter()
            .find(|c| c.symbol_name.as_deref() == Some("calculate_total"))
            .expect("Should have calculate_total chunk");

        assert!(
            fn_chunk.context_text.contains("Signature:"),
            "Context should include signature"
        );
    }

    #[test]
    fn test_gather_leading_context_with_comments() {
        let lines = vec!["// This is a comment", "// Another comment", "fn foo() {}"];
        let start = gather_leading_context(&lines, 2);
        assert_eq!(start, 0, "Should walk back past both // comments");
    }

    #[test]
    fn test_gather_leading_context_with_rust_attributes() {
        let lines = vec!["#[derive(Debug)]", "#[allow(dead_code)]", "struct Foo {}"];
        let start = gather_leading_context(&lines, 2);
        assert_eq!(start, 0, "Should walk back past #[...] attributes");
    }

    #[test]
    fn test_gather_leading_context_with_java_annotations() {
        let lines = vec![
            "@Override",
            "@SuppressWarnings(\"unchecked\")",
            "public void foo() {}",
        ];
        let start = gather_leading_context(&lines, 2);
        assert_eq!(start, 0, "Should walk back past @ annotations");
    }

    #[test]
    fn test_gather_leading_context_with_block_docs() {
        let lines = vec![
            "/**",
            " * This is a doc comment.",
            " */",
            "fn documented() {}",
        ];
        let start = gather_leading_context(&lines, 3);
        assert_eq!(start, 0, "Should walk back past /** */ block doc comment");
    }

    #[test]
    fn test_gather_leading_context_at_file_start() {
        let lines = vec!["fn at_start() {}"];
        let start = gather_leading_context(&lines, 0);
        assert_eq!(start, 0, "At line 0, cannot go further back");
    }

    #[test]
    fn test_gather_leading_context_with_gap() {
        let lines = vec!["// First comment", "", "// Second comment", "fn foo() {}"];
        let start = gather_leading_context(&lines, 3);
        // Empty line at index 1 breaks the chain, so only index 2 comment is included
        assert_eq!(start, 2, "Empty line should break context gathering");
    }

    fn make_symbol(kind: SymbolKind, parent: Option<&str>) -> CodeSymbol {
        CodeSymbol {
            id: None,
            repo_id: 1,
            file_path: "test.rs".to_string(),
            symbol_name: "TestSymbol".to_string(),
            symbol_kind: kind,
            parent_symbol: parent.map(|s| s.to_string()),
            language: Language::Rust,
            start_line: 1,
            end_line: 10,
            signature: None,
        }
    }

    #[test]
    fn test_is_top_level_container() {
        // Should return true for container kinds without parent
        assert!(is_top_level_container(&make_symbol(
            SymbolKind::Class,
            None
        )));
        assert!(is_top_level_container(&make_symbol(
            SymbolKind::Struct,
            None
        )));
        assert!(is_top_level_container(&make_symbol(
            SymbolKind::Trait,
            None
        )));
        assert!(is_top_level_container(&make_symbol(SymbolKind::Impl, None)));
        assert!(is_top_level_container(&make_symbol(
            SymbolKind::Interface,
            None
        )));
        assert!(is_top_level_container(&make_symbol(SymbolKind::Enum, None)));
        assert!(is_top_level_container(&make_symbol(
            SymbolKind::Module,
            None
        )));

        // Should return false for non-container kinds
        assert!(!is_top_level_container(&make_symbol(
            SymbolKind::Function,
            None
        )));
        assert!(!is_top_level_container(&make_symbol(
            SymbolKind::Method,
            None
        )));
        assert!(!is_top_level_container(&make_symbol(
            SymbolKind::Constant,
            None
        )));
    }

    #[test]
    fn test_is_top_level_container_with_parent() {
        // Symbols with a parent are never top-level, even if the kind is a container
        assert!(!is_top_level_container(&make_symbol(
            SymbolKind::Class,
            Some("OuterClass")
        )));
        assert!(!is_top_level_container(&make_symbol(
            SymbolKind::Struct,
            Some("Module")
        )));
        assert!(!is_top_level_container(&make_symbol(
            SymbolKind::Impl,
            Some("Parent")
        )));
        assert!(!is_top_level_container(&make_symbol(
            SymbolKind::Trait,
            Some("OuterTrait")
        )));
    }

    #[test]
    fn test_mark_covered() {
        let mut covered = vec![false; 10];
        mark_covered(&mut covered, 2, 5);

        assert!(!covered[0]);
        assert!(!covered[1]);
        assert!(covered[2]);
        assert!(covered[3]);
        assert!(covered[4]);
        assert!(!covered[5]);
        assert!(!covered[9]);
    }

    #[test]
    fn test_collect_lines() {
        let lines = vec!["line0", "line1", "line2", "line3", "line4"];
        assert_eq!(collect_lines(&lines, 1, 3), "line1\nline2");
        assert_eq!(
            collect_lines(&lines, 0, 5),
            "line0\nline1\nline2\nline3\nline4"
        );
        assert_eq!(collect_lines(&lines, 0, 1), "line0");
        // End beyond bounds is clamped
        assert_eq!(collect_lines(&lines, 3, 100), "line3\nline4");
    }

    #[test]
    fn test_split_oversized_within_limit() {
        let chunk = CodeChunk {
            id: None,
            repo_id: 1,
            file_path: "test.rs".to_string(),
            chunk_type: "function".to_string(),
            symbol_name: Some("small_fn".to_string()),
            language: Language::Rust,
            start_line: 1,
            end_line: 5,
            chunk_text: "fn small() {\n    println!(\"hello\");\n}".to_string(),
            context_text: "File: test.rs".to_string(),
            file_hash: "hash1".to_string(),
            content_hash: None,
        };
        let result = split_oversized(vec![chunk.clone()]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].chunk_text, chunk.chunk_text);
    }

    #[test]
    fn test_split_oversized_over_limit() {
        // Create a chunk larger than MAX_CHUNK_CHARS
        let long_line = "x".repeat(200);
        let lines: Vec<String> = (0..50)
            .map(|i| format!("// line {}: {}", i, long_line))
            .collect();
        let big_text = lines.join("\n");
        assert!(big_text.len() > MAX_CHUNK_CHARS);

        let chunk = CodeChunk {
            id: None,
            repo_id: 1,
            file_path: "big.rs".to_string(),
            chunk_type: "top_level".to_string(),
            symbol_name: None,
            language: Language::Rust,
            start_line: 1,
            end_line: 50,
            chunk_text: big_text,
            context_text: "File: big.rs".to_string(),
            file_hash: "hash2".to_string(),
            content_hash: None,
        };
        let result = split_oversized(vec![chunk]);
        // Should produce more than one chunk
        assert!(
            result.len() > 1,
            "Expected oversized chunk to be split into multiple, got {}",
            result.len()
        );
        // Each resulting chunk should be within the limit (or close, since split is at line boundaries)
        for c in &result {
            // The chunk text should be significantly smaller than the original
            assert!(c.chunk_text.len() >= MIN_CHUNK_CHARS);
        }
    }

    #[test]
    fn test_build_context_text() {
        let ctx = build_context_text(
            "src/main.rs",
            Language::Rust,
            "function",
            Some("main"),
            None,
            "fn main() {\n    println!(\"Hello\");\n}",
        );
        assert!(ctx.contains("File: src/main.rs"));
        assert!(ctx.contains("Language: Rust"));
        assert!(ctx.contains("Symbol: main (function)"));
        assert!(!ctx.contains("Parent:"));
        assert!(ctx.contains("fn main()"));
    }

    #[test]
    fn test_build_context_text_with_parent() {
        let ctx = build_context_text(
            "src/server.rs",
            Language::Rust,
            "function",
            Some("handle_request"),
            Some("Server"),
            "fn handle_request(&self) {}",
        );
        assert!(ctx.contains("File: src/server.rs"));
        assert!(ctx.contains("Symbol: handle_request (function)"));
        assert!(ctx.contains("Parent: Server"));
        assert!(ctx.contains("fn handle_request"));
    }

    #[test]
    fn test_build_context_text_truncation() {
        // Code longer than MAX_CHUNK_CHARS should be truncated
        let long_code = "x".repeat(MAX_CHUNK_CHARS + 1000);
        let ctx = build_context_text(
            "src/big.rs",
            Language::Rust,
            "top_level",
            None,
            None,
            &long_code,
        );
        assert!(ctx.contains("File: src/big.rs"));
        assert!(ctx.contains("Type: top_level"));
        assert!(ctx.contains("// ... truncated ..."));
        // The context should be less than the original long_code
        assert!(ctx.len() < long_code.len());
    }

    #[test]
    fn test_build_context_text_no_symbol_name() {
        let ctx = build_context_text(
            "src/lib.rs",
            Language::Rust,
            "top_level",
            None,
            None,
            "use std::io;",
        );
        assert!(ctx.contains("Type: top_level"));
        assert!(!ctx.contains("Symbol:"));
    }

    #[test]
    fn test_chunk_file_simple_rust_function() {
        let src = r#"
/// Adds two numbers.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;
        let (symbols, chunks) = chunk_file(src, Language::Rust, 42, "math.rs", "hash_add").unwrap();

        assert!(!symbols.is_empty(), "Should extract at least one symbol");
        // The function should be chunked as a standalone function
        let fn_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.symbol_name.as_deref() == Some("add"))
            .collect();
        assert!(
            !fn_chunks.is_empty(),
            "Should produce a chunk for the 'add' function"
        );
        let fc = &fn_chunks[0];
        assert_eq!(fc.chunk_type, "function");
        assert_eq!(fc.repo_id, 42);
        assert_eq!(fc.file_path, "math.rs");
        assert_eq!(fc.file_hash, "hash_add");
        assert!(fc.chunk_text.contains("pub fn add"));
        // Leading doc comment should be gathered
        assert!(
            fc.chunk_text.contains("/// Adds two numbers"),
            "Leading doc comment should be included in the chunk text"
        );
    }

    #[test]
    fn test_chunk_file_struct_with_methods() {
        let src = r#"
pub struct Counter {
    value: i32,
    name: String,
    description: Option<String>,
}

impl Counter {
    pub fn new() -> Self {
        Self { value: 0 }
    }

    pub fn increment(&mut self) {
        self.value += 1;
    }

    pub fn get(&self) -> i32 {
        self.value
    }
}
"#;
        let (symbols, chunks) =
            chunk_file(src, Language::Rust, 1, "counter.rs", "cnt_hash").unwrap();

        assert!(!symbols.is_empty());
        assert!(!chunks.is_empty());

        // Struct and impl should be chunked as small containers (under LARGE_CLASS_LINES)
        let struct_chunk = chunks
            .iter()
            .find(|c| c.symbol_name.as_deref() == Some("Counter") && c.chunk_type == "class");
        assert!(
            struct_chunk.is_some(),
            "Should have a chunk for the Counter struct"
        );

        // The impl block should also be chunked
        let impl_chunk = chunks.iter().find(|c| c.chunk_type == "impl_block");
        assert!(
            impl_chunk.is_some(),
            "Should have an impl_block chunk for Counter"
        );
    }

    #[test]
    fn test_chunk_file_empty_source_returns_empty() {
        let (symbols, chunks) = chunk_file("", Language::Rust, 1, "empty.rs", "e").unwrap();
        assert!(symbols.is_empty());
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_extract_imports_rust() {
        let src = "use std::collections::HashMap;\nuse std::io::Read;\n\nfn main() {}";
        let imports = extract_imports(src, Language::Rust);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("use std::collections::HashMap;"));
        assert!(imports.contains("use std::io::Read;"));
    }

    #[test]
    fn test_extract_imports_python() {
        let src = "import os\nfrom pathlib import Path\n\ndef main(): pass";
        let imports = extract_imports(src, Language::Python);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("import os"));
        assert!(imports.contains("from pathlib import Path"));
    }

    #[test]
    fn test_extract_imports_go_block() {
        let src = "package main\n\nimport (\n\t\"fmt\"\n\t\"os\"\n)\n\nfunc main() {}";
        let imports = extract_imports(src, Language::Go);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("\"fmt\""));
        assert!(imports.contains("\"os\""));
    }

    #[test]
    fn test_extract_imports_none_when_empty() {
        let src = "fn main() {}";
        let imports = extract_imports(src, Language::Rust);
        assert!(imports.is_none());
    }

    #[test]
    fn test_is_likely_generated_protobuf() {
        assert!(is_likely_generated("", "api.pb.go"));
        assert!(is_likely_generated("", "types.pb.rs"));
        assert!(is_likely_generated("", "service.grpc.go"));
    }

    #[test]
    fn test_is_likely_generated_minified() {
        assert!(is_likely_generated("", "bundle.min.js"));
        assert!(is_likely_generated("", "styles.min.css"));
    }

    #[test]
    fn test_is_likely_generated_header_marker() {
        let src = "// Code generated by protoc-gen-go. DO NOT EDIT.\npackage pb\n";
        assert!(is_likely_generated(src, "api.go"));
    }

    #[test]
    fn test_is_likely_generated_normal_file() {
        let src = "fn main() {\n    println!(\"hello\");\n}\n";
        assert!(!is_likely_generated(src, "main.rs"));
    }

    #[test]
    fn test_is_likely_generated_dts() {
        assert!(is_likely_generated("", "types.d.ts"));
    }

    #[test]
    fn test_build_embedding_text() {
        let prefix = "File: test.rs\nLanguage: Rust\nSymbol: foo (function)\n";
        let code = "fn foo() { 42 }";
        let result = build_embedding_text(prefix, code);
        assert!(result.contains("File: test.rs"));
        assert!(result.contains("fn foo() { 42 }"));
    }

    #[test]
    fn test_build_embedding_text_truncation() {
        let prefix = "File: big.rs\n";
        let code = "x".repeat(MAX_CHUNK_CHARS + 500);
        let result = build_embedding_text(prefix, &code);
        assert!(result.contains("// ... truncated ..."));
        assert!(result.len() < code.len());
    }

    #[test]
    fn test_build_context_prefix_with_imports_and_signature() {
        let imports = "use std::io;\nuse std::collections::HashMap;\n";
        let ctx = build_context_prefix(
            "src/lib.rs",
            Language::Rust,
            "function",
            Some("process"),
            Some("Handler"),
            Some("pub fn process(&self, data: &[u8]) -> Result<()>"),
            Some(imports),
        );
        assert!(ctx.contains("File: src/lib.rs"));
        assert!(ctx.contains("Symbol: process (function)"));
        assert!(ctx.contains("Parent: Handler"));
        assert!(ctx.contains("Signature: pub fn process"));
        assert!(ctx.contains("Imports:"));
        assert!(ctx.contains("use std::io;"));
    }

    #[test]
    fn test_contiguous_top_level_grouping() {
        // Top-level code that isn't captured as symbols should be grouped
        // by contiguous blocks, not arbitrary fixed-size batches.
        let src = r#"
// Module documentation
// This module handles configuration parsing
// and validation for the application.

// Version info
const VERSION: &str = "1.0";
"#;
        let (_, chunks) = chunk_file(src, Language::Rust, 1, "config.rs", "h").unwrap();
        // The comments are uncovered top-level code — should be in at most one chunk.
        let top_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.chunk_type == "top_level" && c.symbol_name.is_none())
            .collect();
        assert!(
            top_chunks.len() <= 1,
            "Contiguous top-level comments should be in one chunk, got {}",
            top_chunks.len()
        );
    }

    #[test]
    fn test_build_context_text_all_chunk_types() {
        let types = [
            ("function", Some("process_data"), Some("DataProcessor")),
            ("class", Some("MyClass"), None),
            ("impl_block", Some("MyStruct"), None),
            ("top_level", None, None),
            ("module_header", Some("mymod"), None),
        ];

        for (chunk_type, symbol, parent) in &types {
            let ctx = build_context_text(
                "file.rs",
                Language::Rust,
                chunk_type,
                *symbol,
                *parent,
                "code",
            );
            assert!(
                ctx.contains("File: file.rs"),
                "Missing file info for {}",
                chunk_type
            );
            assert!(
                ctx.contains("Language: Rust"),
                "Missing language for {}",
                chunk_type
            );
            if let Some(sym) = symbol {
                assert!(
                    ctx.contains(&format!("Symbol: {} ({})", sym, chunk_type)),
                    "Missing symbol for {}",
                    chunk_type
                );
            }
            if let Some(p) = parent {
                assert!(
                    ctx.contains(&format!("Parent: {}", p)),
                    "Missing parent for {}",
                    chunk_type
                );
            }
        }
    }

    #[test]
    fn test_build_context_text_different_languages() {
        let languages = [
            Language::Rust,
            Language::Python,
            Language::TypeScript,
            Language::Java,
            Language::Go,
        ];
        for lang in &languages {
            let ctx = build_context_text("file.ext", *lang, "function", Some("f"), None, "code");
            assert!(
                ctx.contains(&format!("Language: {}", lang)),
                "Missing language for {}",
                lang
            );
        }
    }

    #[test]
    fn test_build_context_text_truncation_at_char_boundary() {
        // Create a string with multi-byte characters that might split at a bad boundary
        let code = "a".repeat(MAX_CHUNK_CHARS - 10) + &"\u{00e9}".repeat(20); // é is 2 bytes in UTF-8

        let ctx = build_context_text("test.rs", Language::Rust, "top_level", None, None, &code);
        // Should not panic and should be valid UTF-8
        assert!(ctx.contains("// ... truncated ..."));
    }

    #[test]
    fn test_build_context_text_lua() {
        let ctx = build_context_text(
            "plugin.lua",
            Language::Lua,
            "function",
            Some("setup"),
            None,
            "function setup() end",
        );
        assert!(ctx.contains("Language: Lua"));
        assert!(ctx.contains("Symbol: setup (function)"));
        assert!(ctx.contains("function setup() end"));
    }

    #[test]
    fn test_build_context_text_json() {
        let ctx = build_context_text("package.json", Language::Json, "top_level", None, None, "");
        assert!(ctx.contains("Language: JSON"));
        assert!(ctx.contains("Type: top_level"));
    }

    #[test]
    fn test_build_context_text_yaml() {
        let ctx = build_context_text("ci.yaml", Language::Yaml, "top_level", None, None, "");
        assert!(ctx.contains("Language: YAML"));
        assert!(ctx.contains("Type: top_level"));
    }

    #[test]
    fn test_build_context_text_dockerfile() {
        let ctx = build_context_text(
            "Dockerfile",
            Language::Dockerfile,
            "top_level",
            None,
            None,
            "",
        );
        assert!(ctx.contains("Language: Dockerfile"));
    }

    #[test]
    fn test_extract_imports_javascript() {
        let src = "import React from 'react';\nimport { useState } from 'react';\n\nconst App = () => {};";
        let imports = extract_imports(src, Language::JavaScript);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("import React from 'react';"));
        assert!(imports.contains("import { useState } from 'react';"));
    }

    #[test]
    fn test_extract_imports_java() {
        let src = "package com.example;\n\nimport java.util.List;\nimport java.util.Map;\n\npublic class Foo {}";
        let imports = extract_imports(src, Language::Java);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("package com.example;"));
        assert!(imports.contains("import java.util.List;"));
    }

    #[test]
    fn test_extract_imports_c() {
        let src = "#include <stdio.h>\n#include \"myheader.h\"\n\nint main() { return 0; }";
        let imports = extract_imports(src, Language::C);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("#include <stdio.h>"));
        assert!(imports.contains("#include \"myheader.h\""));
    }

    #[test]
    fn test_extract_imports_caps_at_limit() {
        // Many imports should be capped at MAX_IMPORT_CONTEXT_CHARS.
        let imports_src: String = (0..100)
            .map(|i| format!("use crate::module{}::SomeLongTypeName;", i))
            .collect::<Vec<_>>()
            .join("\n");
        let src = format!("{}\n\nfn main() {{}}", imports_src);
        let imports = extract_imports(&src, Language::Rust);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.len() <= MAX_IMPORT_CONTEXT_CHARS + 10); // +10 for trailing "// ...\n"
        assert!(imports.contains("// ..."));
    }

    #[test]
    fn test_chunk_file_large_class_splitting() {
        // Create a Python class with > LARGE_CLASS_LINES (100) lines
        // so that it triggers the skeleton + per-method chunking path.
        let mut src = String::from("import os\n\nclass BigClass:\n");
        // Add methods so total line count > 100
        for i in 0..40 {
            src.push_str(&format!(
                "    def method_{}(self):\n        return {}\n\n",
                i, i
            ));
        }
        // Should be well over 100 lines
        let line_count = src.lines().count();
        assert!(
            line_count > LARGE_CLASS_LINES,
            "Test source should be > {} lines, got {}",
            LARGE_CLASS_LINES,
            line_count
        );

        let (symbols, chunks) =
            chunk_file(&src, Language::Python, 1, "big_class.py", "bighash").unwrap();
        assert!(!symbols.is_empty());
        assert!(!chunks.is_empty());

        // Should have a skeleton chunk for the class
        let class_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.symbol_name.as_deref() == Some("BigClass") && c.chunk_type == "class")
            .collect();
        assert!(
            !class_chunks.is_empty(),
            "Should have a skeleton class chunk for BigClass"
        );

        // The skeleton chunk should contain method signatures placeholder
        let skeleton = &class_chunks[0];
        assert!(
            skeleton.chunk_text.contains("// ... methods ..."),
            "Skeleton should contain method signature placeholder"
        );

        // Should have individual method chunks
        let method_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.chunk_type == "function" && c.symbol_name.is_some())
            .collect();
        assert!(
            !method_chunks.is_empty(),
            "Should have individual method chunks from the large class"
        );
    }

    #[test]
    fn test_build_symbol_chunk_method_type() {
        let sym = CodeSymbol {
            id: None,
            repo_id: 1,
            file_path: "test.rs".to_string(),
            symbol_name: "do_something".to_string(),
            symbol_kind: SymbolKind::Method,
            parent_symbol: Some("MyStruct".to_string()),
            language: Language::Rust,
            start_line: 2,
            end_line: 5,
            signature: Some("fn do_something(&self)".to_string()),
        };
        let lines = vec![
            "impl MyStruct {",
            "    fn do_something(&self) {",
            "        println!(\"hello\");",
            "    }",
            "}",
        ];
        let chunk = build_symbol_chunk(&lines, &sym, Language::Rust, 1, "test.rs", "hash", None);
        assert_eq!(chunk.chunk_type, "function");
        assert_eq!(chunk.symbol_name.as_deref(), Some("do_something"));
        assert!(chunk.context_text.contains("Parent: MyStruct"));
        assert!(chunk
            .context_text
            .contains("Signature: fn do_something(&self)"));
    }

    #[test]
    fn test_build_symbol_chunk_constant_type() {
        let sym = CodeSymbol {
            id: None,
            repo_id: 1,
            file_path: "test.rs".to_string(),
            symbol_name: "MAX_SIZE".to_string(),
            symbol_kind: SymbolKind::Constant,
            parent_symbol: None,
            language: Language::Rust,
            start_line: 1,
            end_line: 1,
            signature: None,
        };
        let lines = vec!["const MAX_SIZE: usize = 100;"];
        let chunk = build_symbol_chunk(&lines, &sym, Language::Rust, 1, "test.rs", "hash", None);
        assert_eq!(chunk.chunk_type, "top_level");
        assert_eq!(chunk.symbol_name.as_deref(), Some("MAX_SIZE"));
    }

    #[test]
    fn test_build_symbol_chunk_module_type() {
        let sym = CodeSymbol {
            id: None,
            repo_id: 1,
            file_path: "test.rs".to_string(),
            symbol_name: "utils".to_string(),
            symbol_kind: SymbolKind::Module,
            parent_symbol: None,
            language: Language::Rust,
            start_line: 1,
            end_line: 3,
            signature: None,
        };
        let lines = vec!["mod utils {", "    pub fn helper() {}", "}"];
        let chunk = build_symbol_chunk(&lines, &sym, Language::Rust, 1, "test.rs", "hash", None);
        assert_eq!(chunk.chunk_type, "module_header");
    }

    #[test]
    fn test_build_symbol_chunk_enum_type() {
        let sym = CodeSymbol {
            id: None,
            repo_id: 1,
            file_path: "test.rs".to_string(),
            symbol_name: "Color".to_string(),
            symbol_kind: SymbolKind::Enum,
            parent_symbol: None,
            language: Language::Rust,
            start_line: 1,
            end_line: 4,
            signature: None,
        };
        let lines = vec!["enum Color {", "    Red,", "    Blue,", "}"];
        let chunk = build_symbol_chunk(&lines, &sym, Language::Rust, 1, "test.rs", "hash", None);
        assert_eq!(chunk.chunk_type, "class");
    }

    #[test]
    fn test_build_symbol_chunk_interface_type() {
        let sym = CodeSymbol {
            id: None,
            repo_id: 1,
            file_path: "test.ts".to_string(),
            symbol_name: "UserService".to_string(),
            symbol_kind: SymbolKind::Interface,
            parent_symbol: None,
            language: Language::TypeScript,
            start_line: 1,
            end_line: 3,
            signature: None,
        };
        let lines = vec!["interface UserService {", "    getUser(): User;", "}"];
        let chunk = build_symbol_chunk(
            &lines,
            &sym,
            Language::TypeScript,
            1,
            "test.ts",
            "hash",
            None,
        );
        assert_eq!(chunk.chunk_type, "class");
    }

    #[test]
    fn test_build_symbol_chunk_trait_type() {
        let sym = CodeSymbol {
            id: None,
            repo_id: 1,
            file_path: "test.rs".to_string(),
            symbol_name: "Drawable".to_string(),
            symbol_kind: SymbolKind::Trait,
            parent_symbol: None,
            language: Language::Rust,
            start_line: 1,
            end_line: 3,
            signature: None,
        };
        let lines = vec!["trait Drawable {", "    fn draw(&self);", "}"];
        let chunk = build_symbol_chunk(&lines, &sym, Language::Rust, 1, "test.rs", "hash", None);
        assert_eq!(chunk.chunk_type, "class");
    }

    #[test]
    fn test_build_symbol_chunk_with_imports() {
        let sym = CodeSymbol {
            id: None,
            repo_id: 1,
            file_path: "test.rs".to_string(),
            symbol_name: "process".to_string(),
            symbol_kind: SymbolKind::Function,
            parent_symbol: None,
            language: Language::Rust,
            start_line: 1,
            end_line: 3,
            signature: Some("fn process() -> bool".to_string()),
        };
        let lines = vec!["fn process() -> bool {", "    true", "}"];
        let imports = "use std::io;\n";
        let chunk = build_symbol_chunk(
            &lines,
            &sym,
            Language::Rust,
            1,
            "test.rs",
            "hash",
            Some(imports),
        );
        assert!(chunk.context_text.contains("Imports:"));
        assert!(chunk.context_text.contains("use std::io;"));
    }

    #[test]
    fn test_build_symbol_chunk_empty_range() {
        // When end <= start, the text should be empty
        let sym = CodeSymbol {
            id: None,
            repo_id: 1,
            file_path: "test.rs".to_string(),
            symbol_name: "empty".to_string(),
            symbol_kind: SymbolKind::Function,
            parent_symbol: None,
            language: Language::Rust,
            start_line: 5,
            end_line: 1, // end before start
            signature: None,
        };
        let lines = vec!["line1", "line2"];
        let chunk = build_symbol_chunk(&lines, &sym, Language::Rust, 1, "test.rs", "hash", None);
        assert!(chunk.chunk_text.is_empty());
    }

    #[test]
    fn test_build_skeleton_chunk_with_methods() {
        // Build a class symbol with child methods
        let class_sym = CodeSymbol {
            id: None,
            repo_id: 1,
            file_path: "test.py".to_string(),
            symbol_name: "Calculator".to_string(),
            symbol_kind: SymbolKind::Class,
            parent_symbol: None,
            language: Language::Python,
            start_line: 1,
            end_line: 15,
            signature: Some("class Calculator".to_string()),
        };

        let method1 = CodeSymbol {
            id: None,
            repo_id: 1,
            file_path: "test.py".to_string(),
            symbol_name: "add".to_string(),
            symbol_kind: SymbolKind::Method,
            parent_symbol: Some("Calculator".to_string()),
            language: Language::Python,
            start_line: 3,
            end_line: 5,
            signature: Some("def add(self, a, b)".to_string()),
        };

        let method2 = CodeSymbol {
            id: None,
            repo_id: 1,
            file_path: "test.py".to_string(),
            symbol_name: "subtract".to_string(),
            symbol_kind: SymbolKind::Method,
            parent_symbol: Some("Calculator".to_string()),
            language: Language::Python,
            start_line: 7,
            end_line: 9,
            signature: Some("def subtract(self, a, b)".to_string()),
        };

        // A function not belonging to Calculator - should be excluded
        let other_fn = CodeSymbol {
            id: None,
            repo_id: 1,
            file_path: "test.py".to_string(),
            symbol_name: "standalone".to_string(),
            symbol_kind: SymbolKind::Function,
            parent_symbol: None,
            language: Language::Python,
            start_line: 11,
            end_line: 13,
            signature: Some("def standalone()".to_string()),
        };

        let all_symbols = vec![class_sym.clone(), method1, method2, other_fn];

        let mut lines = Vec::new();
        for i in 0..15 {
            lines.push(format!("# line {}", i));
        }
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();

        let chunk = build_skeleton_chunk(
            &line_refs,
            &class_sym,
            &all_symbols,
            Language::Python,
            1,
            "test.py",
            "hash",
            None,
        );

        assert_eq!(chunk.chunk_type, "class");
        assert_eq!(chunk.symbol_name.as_deref(), Some("Calculator"));
        // Should contain method signatures
        assert!(
            chunk.chunk_text.contains("def add(self, a, b)"),
            "Skeleton should include add method signature. Got: {}",
            chunk.chunk_text
        );
        assert!(
            chunk.chunk_text.contains("def subtract(self, a, b)"),
            "Skeleton should include subtract method signature"
        );
        // Should NOT contain standalone function (different parent)
        assert!(
            !chunk.chunk_text.contains("def standalone()"),
            "Skeleton should not include functions from other parents"
        );
        assert!(chunk.chunk_text.contains("// ... methods ..."));
    }

    #[test]
    fn test_build_top_level_chunks_basic() {
        let lines = vec![
            "// This is a top-level comment",
            "// describing the module",
            "",
            "// Another block",
            "// of comments",
        ];
        let covered_lines = vec![false, false, false, false, false];
        let mut chunks = Vec::new();

        build_top_level_chunks(
            &lines,
            &covered_lines,
            Language::Rust,
            1,
            "test.rs",
            "hash",
            None,
            &mut chunks,
        );

        assert!(
            !chunks.is_empty(),
            "Should produce at least one top-level chunk"
        );
        for c in &chunks {
            assert_eq!(c.chunk_type, "top_level");
            assert!(c.symbol_name.is_none());
        }
    }

    #[test]
    fn test_build_top_level_chunks_split_on_double_blank() {
        // Two consecutive blank lines should split blocks
        let lines = vec![
            "// Block 1 line 1",
            "// Block 1 line 2",
            "",
            "",
            "// Block 2 line 1",
            "// Block 2 line 2",
        ];
        let covered_lines = vec![false, false, false, false, false, false];
        let mut chunks = Vec::new();

        build_top_level_chunks(
            &lines,
            &covered_lines,
            Language::Rust,
            1,
            "test.rs",
            "hash",
            None,
            &mut chunks,
        );

        assert!(
            chunks.len() >= 2,
            "Double blank line should split into at least 2 blocks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_build_top_level_chunks_covered_lines_end_block() {
        // A covered line in the middle should end the current block
        let lines = vec![
            "// uncovered line 1",
            "// uncovered line 2",
            "fn covered_fn() {}", // covered
            "// uncovered line 3",
            "// uncovered line 4",
        ];
        let covered_lines = vec![false, false, true, false, false];
        let mut chunks = Vec::new();

        build_top_level_chunks(
            &lines,
            &covered_lines,
            Language::Rust,
            1,
            "test.rs",
            "hash",
            None,
            &mut chunks,
        );

        // Should produce 2 blocks: lines 0-1 and lines 3-4
        assert!(
            chunks.len() >= 2,
            "Covered line should split into separate blocks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_build_top_level_chunks_all_covered() {
        let lines = vec!["fn a() {}", "fn b() {}"];
        let covered_lines = vec![true, true];
        let mut chunks = Vec::new();

        build_top_level_chunks(
            &lines,
            &covered_lines,
            Language::Rust,
            1,
            "test.rs",
            "hash",
            None,
            &mut chunks,
        );

        assert!(
            chunks.is_empty(),
            "All covered lines should produce no top-level chunks"
        );
    }

    #[test]
    fn test_build_top_level_chunks_below_min_size() {
        // Lines too short (< MIN_CHUNK_CHARS) should be skipped
        let lines = vec!["x"];
        let covered_lines = vec![false];
        let mut chunks = Vec::new();

        build_top_level_chunks(
            &lines,
            &covered_lines,
            Language::Rust,
            1,
            "test.rs",
            "hash",
            None,
            &mut chunks,
        );

        assert!(
            chunks.is_empty(),
            "Text below MIN_CHUNK_CHARS should be skipped"
        );
    }

    #[test]
    fn test_gather_leading_context_python_triple_quote_multiline() {
        let lines = vec![
            "\"\"\"",
            "This is a multi-line",
            "docstring.",
            "\"\"\"",
            "def documented_func():",
            "    pass",
        ];
        let start = gather_leading_context(&lines, 4);
        assert!(
            start == 0,
            "Should walk back through entire triple-quote docstring, got start={}",
            start
        );
    }

    #[test]
    fn test_gather_leading_context_python_single_line_triple_quote() {
        let lines = vec![
            "\"\"\"Single-line docstring.\"\"\"",
            "def func():",
            "    pass",
        ];
        let start = gather_leading_context(&lines, 1);
        assert_eq!(
            start, 0,
            "Should walk back past single-line triple-quote docstring"
        );
    }

    #[test]
    fn test_gather_leading_context_python_triple_single_quote() {
        let lines = vec!["'''", "Docstring body", "'''", "def func():", "    pass"];
        let start = gather_leading_context(&lines, 3);
        assert_eq!(
            start, 0,
            "Should walk back through ''' multi-line docstring"
        );
    }

    #[test]
    fn test_gather_leading_context_rust_inner_attribute() {
        let lines = vec!["#![allow(unused)]", "fn main() {}"];
        let start = gather_leading_context(&lines, 1);
        assert_eq!(start, 0, "Should walk back past #![] inner attribute");
    }

    #[test]
    fn test_gather_leading_context_past_bounds() {
        let lines = vec!["line0"];
        // start_0idx beyond array length should return the same index
        let start = gather_leading_context(&lines, 5);
        assert_eq!(start, 5, "Out of bounds start should be returned as-is");
    }

    #[test]
    fn test_extract_imports_typescript() {
        let src = "import { Component } from '@angular/core';\nconst x = require('lodash');\n\nclass Foo {}";
        let imports = extract_imports(src, Language::TypeScript);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("import { Component }"));
        assert!(imports.contains("require('lodash')"));
    }

    #[test]
    fn test_extract_imports_csharp() {
        let src = "using System;\nusing System.Collections.Generic;\n\nclass Program {}";
        let imports = extract_imports(src, Language::CSharp);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("using System;"));
        assert!(imports.contains("using System.Collections.Generic;"));
    }

    #[test]
    fn test_extract_imports_ruby() {
        let src = "require 'json'\nrequire_relative 'utils'\n\ndef main; end";
        let imports = extract_imports(src, Language::Ruby);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("require 'json'"));
        assert!(imports.contains("require_relative 'utils'"));
    }

    #[test]
    fn test_extract_imports_php() {
        let src = "namespace App\\Controllers;\n\nuse App\\Models\\User;\nuse Illuminate\\Http\\Request;\n\nclass Controller {}";
        let imports = extract_imports(src, Language::Php);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("namespace App\\Controllers;"));
        assert!(imports.contains("use App\\Models\\User;"));
    }

    #[test]
    fn test_extract_imports_swift() {
        let src = "import UIKit\nimport Foundation\n\nclass ViewController {}";
        let imports = extract_imports(src, Language::Swift);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("import UIKit"));
        assert!(imports.contains("import Foundation"));
    }

    #[test]
    fn test_extract_imports_dart() {
        let src = "import 'dart:io';\npart 'generated.g.dart';\n\nvoid main() {}";
        let imports = extract_imports(src, Language::Dart);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("import 'dart:io';"));
        assert!(imports.contains("part 'generated.g.dart';"));
    }

    #[test]
    fn test_extract_imports_cpp() {
        let src = "#include <iostream>\n#include \"myheader.h\"\n\nint main() { return 0; }";
        let imports = extract_imports(src, Language::Cpp);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("#include <iostream>"));
    }

    #[test]
    fn test_extract_imports_kotlin() {
        let src = "package com.example.app\n\nimport kotlinx.coroutines.*\nimport java.util.List\n\nfun main() {}";
        let imports = extract_imports(src, Language::Kotlin);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("package com.example.app"));
        assert!(imports.contains("import kotlinx.coroutines.*"));
    }

    #[test]
    fn test_extract_imports_go_single() {
        let src = "package main\n\nimport \"fmt\"\n\nfunc main() {}";
        let imports = extract_imports(src, Language::Go);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("import \"fmt\""));
    }

    #[test]
    fn test_extract_imports_lua_no_imports() {
        // Lua doesn't match any import patterns (not in the language branches)
        let src = "local x = require('foo')\n\nfunction main() end";
        let imports = extract_imports(src, Language::Lua);
        assert!(imports.is_none(), "Lua has no import detection patterns");
    }

    #[test]
    fn test_extract_imports_rust_pub_use() {
        let src = "pub use claudear_core::types::*;\nuse std::io;\n\nfn main() {}";
        let imports = extract_imports(src, Language::Rust);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("pub use claudear_core::types::*;"));
        assert!(imports.contains("use std::io;"));
    }

    #[test]
    fn test_is_likely_generated_auto_pattern() {
        assert!(is_likely_generated("", "types.auto.dart"));
    }

    #[test]
    fn test_is_likely_generated_generated_dir() {
        assert!(is_likely_generated("", "src/generated/api.rs"));
    }

    #[test]
    fn test_is_likely_generated_freezed_dart() {
        assert!(is_likely_generated("", "model.freezed.dart"));
    }

    #[test]
    fn test_is_likely_generated_g_dart() {
        assert!(is_likely_generated("", "model.g.dart"));
    }

    #[test]
    fn test_is_likely_generated_bundle_js() {
        assert!(is_likely_generated("", "vendor.bundle.js"));
    }

    #[test]
    fn test_is_likely_generated_dist_dir() {
        assert!(is_likely_generated("", "project/dist/app.js"));
    }

    #[test]
    fn test_is_likely_generated_auto_generated_header() {
        let src = "# Auto-generated file\nclass Foo: pass";
        assert!(is_likely_generated(src, "model.py"));
    }

    #[test]
    fn test_is_likely_generated_automatically_generated_header() {
        let src = "// Automatically generated by codegen\npackage main";
        assert!(is_likely_generated(src, "main.go"));
    }

    #[test]
    fn test_is_likely_generated_pb_cc() {
        assert!(is_likely_generated("", "service.pb.cc"));
    }

    #[test]
    fn test_is_likely_generated_underscore_generated() {
        assert!(is_likely_generated("", "schema_generated.rs"));
    }

    #[test]
    fn test_chunk_file_standalone_constant() {
        let src = r#"
const MAX_RETRIES: u32 = 3;
const TIMEOUT: u64 = 5000;

fn process() {
    println!("processing");
}
"#;
        let (symbols, chunks) = chunk_file(src, Language::Rust, 1, "config.rs", "hash").unwrap();
        assert!(!symbols.is_empty());

        // Check that constants and function are both chunked
        let fn_chunk = chunks
            .iter()
            .find(|c| c.symbol_name.as_deref() == Some("process"));
        assert!(fn_chunk.is_some(), "Should have a process function chunk");
    }

    #[test]
    fn test_chunk_file_multiple_standalone_functions() {
        let src = r#"
fn alpha() -> i32 {
    1
}

fn beta() -> i32 {
    2
}

fn gamma() -> i32 {
    3
}
"#;
        let (_, chunks) = chunk_file(src, Language::Rust, 1, "funcs.rs", "hash").unwrap();
        let fn_names: Vec<&str> = chunks
            .iter()
            .filter_map(|c| c.symbol_name.as_deref())
            .collect();
        assert!(
            fn_names.contains(&"alpha"),
            "Missing alpha, got: {:?}",
            fn_names
        );
        assert!(
            fn_names.contains(&"beta"),
            "Missing beta, got: {:?}",
            fn_names
        );
        assert!(
            fn_names.contains(&"gamma"),
            "Missing gamma, got: {:?}",
            fn_names
        );
    }

    #[test]
    fn test_split_oversized_with_overlap() {
        // Create a chunk that will split into multiple parts and verify overlap
        let long_line = "y".repeat(250);
        let lines: Vec<String> = (0..40)
            .map(|i| format!("// ln{}: {}", i, long_line))
            .collect();
        let big_text = lines.join("\n");
        assert!(big_text.len() > MAX_CHUNK_CHARS);

        let chunk = CodeChunk {
            id: None,
            repo_id: 1,
            file_path: "overlap.rs".to_string(),
            chunk_type: "function".to_string(),
            symbol_name: Some("big_fn".to_string()),
            language: Language::Rust,
            start_line: 1,
            end_line: 40,
            chunk_text: big_text,
            context_text: "File: overlap.rs".to_string(),
            file_hash: "hash".to_string(),
            content_hash: None,
        };
        let result = split_oversized(vec![chunk]);
        assert!(result.len() > 1, "Should produce multiple chunks");

        // Verify that the second chunk's start overlaps with the first chunk's end
        if result.len() >= 2 {
            let first_lines: Vec<&str> = result[0].chunk_text.lines().collect();
            let second_lines: Vec<&str> = result[1].chunk_text.lines().collect();
            // The overlap should mean the first few lines of chunk 2 match the last few of chunk 1
            let overlap_end = first_lines.len().min(OVERLAP_LINES);
            let first_tail = &first_lines[first_lines.len().saturating_sub(overlap_end)..];
            let second_head = &second_lines[..overlap_end.min(second_lines.len())];
            // At least one line should overlap
            assert!(
                first_tail.iter().any(|line| second_head.contains(line)),
                "Split chunks should have overlapping context lines"
            );
        }
    }

    #[test]
    fn test_build_context_prefix_empty_imports() {
        let ctx = build_context_prefix(
            "test.rs",
            Language::Rust,
            "function",
            Some("f"),
            None,
            None,
            Some(""),
        );
        // Empty imports should not add "Imports:" section
        assert!(!ctx.contains("Imports:"));
    }

    #[test]
    fn test_chunk_file_javascript_with_require() {
        let src = "const fs = require('fs');\nconst path = require('path');\n\nfunction readFile(name) {\n    return fs.readFileSync(name);\n}\n";
        let (_, chunks) = chunk_file(src, Language::JavaScript, 1, "app.js", "hash").unwrap();

        // Should find a function chunk
        let fn_chunk = chunks
            .iter()
            .find(|c| c.symbol_name.as_deref() == Some("readFile"));
        assert!(fn_chunk.is_some(), "Should have readFile function chunk");

        // Context should include require imports
        if let Some(fc) = fn_chunk {
            assert!(
                fc.context_text.contains("Imports:"),
                "JS chunk should have import context with require()"
            );
        }
    }

    #[test]
    fn test_chunk_file_go_with_imports() {
        let src = "package main\n\nimport (\n\t\"fmt\"\n\t\"os\"\n)\n\nfunc main() {\n\tfmt.Println(\"hello\")\n}\n";
        let (_, chunks) = chunk_file(src, Language::Go, 1, "main.go", "hash").unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_extract_imports_go_empty_lines_in_block() {
        // Go import block with blank lines between imports should skip blanks
        let src = "package main\n\nimport (\n\t\"fmt\"\n\n\t\"os\"\n)\n\nfunc main() {}";
        let imports = extract_imports(src, Language::Go);
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert!(imports.contains("\"fmt\""));
        assert!(imports.contains("\"os\""));
    }

    #[test]
    fn test_build_top_level_chunks_single_blank_does_not_split() {
        // A single blank line should NOT split blocks (only 2+ consecutive blanks split)
        let lines = vec!["// First line of block", "", "// Third line of block"];
        let covered_lines = vec![false, false, false];
        let mut chunks = Vec::new();

        build_top_level_chunks(
            &lines,
            &covered_lines,
            Language::Rust,
            1,
            "test.rs",
            "hash",
            None,
            &mut chunks,
        );

        assert_eq!(
            chunks.len(),
            1,
            "Single blank line should not split blocks, got {} chunks",
            chunks.len()
        );
    }
}
