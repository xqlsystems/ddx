// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Source-to-source SQL rewriting: find every `grad`/`jvp` marker, replace it
//! with derivative SQL, leave everything else byte-identical.
//!
//! This is Path A (design.md §3.3), the universal path every target relies on.
//! It is a real subsystem, not a one-liner (design.md §3.2):
//!
//! * a **parse-free pre-gate** so a marker-free statement is never parsed, and
//!   so a `sqlparser` coverage gap can only ever bound a statement that
//!   *actually contains* a marker (F5);
//! * **splice by source span**, so everything outside a marker stays
//!   byte-identical — which requires a UTF-8-aware character-column→byte-offset
//!   conversion, because `sqlparser` spans are 1-based *characters*, not bytes
//!   (G3);
//! * **multiple and nested markers** — spliced in reverse source order, nested
//!   ones differentiated bottom-up (`grad(grad(f,x),x)` just works);
//! * a safe **fallback** to whole-statement reprinting on the empty spans the
//!   API documents as possible.
//!
//! Two guards run here, both catching a *silently-wrong* derivative and turning
//! it into a typed error: the ambiguity guard lives in the engine (F2), and the
//! CTE-computed-alias guard (F3/G4) lives in [`projection_guard`].

use std::collections::HashSet;
use std::fmt;
use std::ops::ControlFlow;

use sqlparser::ast::Spanned;
use sqlparser::ast::{
    Expr, Function, ObjectNamePart, Query, Select, SelectItem, SetExpr, Statement, TableFactor,
    Visit, VisitMut, Visitor, VisitorMut,
};
use sqlparser::dialect::Dialect;
use sqlparser::parser::Parser;
use sqlparser::tokenizer::{Location, Span, Token, TokenWithSpan, Tokenizer};

use crate::colref::{ColRef, IdentCasing};
use crate::engine::{differentiate, jvp, positional_args, RuleRegistry};
use crate::error::{DiffError, Result};

/// Which marker a function call is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkerKind {
    Grad,
    Jvp,
}

/// Classify a function call as a marker — but only an **unqualified**,
/// case-folded `grad`/`jvp` (design.md §3.2, F8). `myschema.grad(...)` and a
/// user's own multi-part function are left alone.
fn marker_kind(f: &Function) -> Option<MarkerKind> {
    if f.name.0.len() != 1 {
        return None;
    }
    let ObjectNamePart::Identifier(id) = &f.name.0[0] else {
        return None;
    };
    match id.value.to_ascii_lowercase().as_str() {
        "grad" => Some(MarkerKind::Grad),
        "jvp" => Some(MarkerKind::Jvp),
        _ => None,
    }
}

impl MarkerKind {
    /// The marker's SQL function name, for display.
    fn name(self) -> &'static str {
        match self {
            MarkerKind::Grad => "grad",
            MarkerKind::Jvp => "jvp",
        }
    }
}

fn is_marker_expr(e: &Expr) -> bool {
    matches!(e, Expr::Function(f) if marker_kind(f).is_some())
}

/// The [`MarkerKind`] of an expression that is a marker call, else `None`.
fn marker_expr_kind(e: &Expr) -> Option<MarkerKind> {
    match e {
        Expr::Function(f) => marker_kind(f),
        _ => None,
    }
}

/// A human-inspectable account of what [`crate::Ddx::rewrite_sql`] would do to a
/// statement, produced by [`crate::Ddx::explain`] — so a user can see the
/// derivative SQL *before* running anything. Inspect the fields directly, or
/// print the whole thing (`Display`) for a readable summary.
#[derive(Debug, Clone)]
pub struct Explanation {
    /// The original statement, unchanged.
    pub original: String,
    /// The statement after every `grad`/`jvp` marker is rewritten to derivative
    /// SQL — exactly what [`crate::Ddx::rewrite_sql`] returns.
    pub rewritten: String,
    /// One entry per top-level marker, in source order. Empty when the statement
    /// has no marker, or in the rare empty-span reprint fallback (where the
    /// rewrite still appears in [`Explanation::rewritten`]).
    pub steps: Vec<ExplainStep>,
}

/// One marker and the derivative SQL it rewrites to (part of an [`Explanation`]).
#[derive(Debug, Clone)]
pub struct ExplainStep {
    /// The marker function: `"grad"` or `"jvp"`.
    pub function: &'static str,
    /// The original marker call, exactly as written (e.g. `grad(sin(x), x)`).
    pub marker: String,
    /// The derivative SQL it becomes (e.g. `(cos(x))`).
    pub derivative: String,
}

impl fmt::Display for Explanation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.steps.is_empty() {
            return write!(
                f,
                "No grad/jvp markers to rewrite; the statement is unchanged:\n  {}",
                self.original
            );
        }
        let n = self.steps.len();
        writeln!(f, "ddx rewrites {n} marker{}:", if n == 1 { "" } else { "s" })?;
        for step in &self.steps {
            writeln!(f, "  • {} → {}", step.marker, step.derivative)?;
        }
        writeln!(f)?;
        writeln!(f, "  from: {}", self.original)?;
        write!(f, "  into: {}", self.rewritten)
    }
}

/// The parse-free pre-gate: a case-insensitive scan for an *unqualified*
/// `grad(`/`jvp(` — the equivalent of `(?i)(?:^|[^A-Za-z0-9_.])(grad|jvp)\s*\(`,
/// hand-rolled so the core depends on `sqlparser` only (design.md §3.2/§6). A
/// statement that doesn't hit is returned verbatim, never parsed (F5). It is a
/// best-effort filter: a false positive (e.g. `grad(` inside a string literal)
/// only costs a parse that then finds no marker, never a wrong rewrite.
fn pre_gate_hit(sql: &str) -> bool {
    // ASCII-lowercasing preserves byte length and offsets, so indices found in
    // `lower` are valid char boundaries in `sql`.
    let lower = sql.to_ascii_lowercase();
    for kw in ["grad", "jvp"] {
        let mut from = 0;
        while let Some(rel) = lower[from..].find(kw) {
            let idx = from + rel;
            from = idx + 1;

            // Preceding character must not be part of a longer identifier or a
            // qualifier (`.`), so `mygrad(` and `schema.grad(` don't match.
            let ok_prev = idx == 0
                || sql[..idx].chars().next_back().is_some_and(|prev| {
                    !(prev.is_ascii_alphanumeric() || prev == '_' || prev == '.')
                });
            if !ok_prev {
                continue;
            }

            // The next significant character must be `(`. sqlparser treats a SQL
            // comment as lexical whitespace, so `grad /* c */ (x, x)` and
            // `grad-- c\n(x, x)` are genuine marker calls — the scan skips
            // comments as well as whitespace, or the gate would miss them and
            // let a real marker reach execution un-rewritten (#52).
            let after = &sql[idx + kw.len()..];
            if after[skip_trivia(after)..].starts_with('(') {
                return true;
            }
        }
    }
    false
}

/// Byte offset of the first significant character in `s`, skipping leading
/// whitespace and SQL comments (`-- … end-of-line`, and `/* … */` block
/// comments, which nest in Postgres/DuckDB) — the trivia sqlparser's tokenizer
/// discards. Returns `s.len()` if the rest is all trivia.
///
/// Delimiters (`-`, `/`, `*`, whitespace, `\n`) are all ASCII, and a UTF-8
/// continuation byte is never equal to an ASCII byte, so scanning by bytes is
/// safe even with multibyte text inside a comment.
fn skip_trivia(s: &str) -> usize {
    let b = s.as_bytes();
    let n = b.len();
    let mut i = 0;
    loop {
        while i < n && b[i].is_ascii_whitespace() {
            i += 1;
        }
        // Line comment: `--` to end of line (or input).
        if i + 1 < n && b[i] == b'-' && b[i + 1] == b'-' {
            i += 2;
            while i < n && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Block comment: `/* … */`, nesting-aware.
        if i + 1 < n && b[i] == b'/' && b[i + 1] == b'*' {
            i += 2;
            let mut depth = 1usize;
            while i < n && depth > 0 {
                if i + 1 < n && b[i] == b'/' && b[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if i + 1 < n && b[i] == b'*' && b[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        break;
    }
    i
}

/// The public entry point behind [`crate::Ddx::rewrite_sql`].
pub(crate) fn rewrite_sql(
    sql: &str,
    dialect: &dyn Dialect,
    casing: IdentCasing,
    reg: &RuleRegistry,
) -> Result<String> {
    match resolve_markers(sql, dialect, casing, reg)? {
        Resolution::Verbatim => Ok(sql.to_string()),
        Resolution::Reprinted(out) => Ok(out),
        Resolution::Spliced(repls) => Ok(apply_splice(sql, repls)),
    }
}

/// The public entry point behind [`crate::Ddx::explain`]: the same marker
/// resolution as [`rewrite_sql`], but returned as inspectable structure (each
/// marker and the derivative SQL it becomes) plus the final rewritten
/// statement — so a user can see what will happen before running anything.
pub(crate) fn explain_sql(
    sql: &str,
    dialect: &dyn Dialect,
    casing: IdentCasing,
    reg: &RuleRegistry,
) -> Result<Explanation> {
    let (rewritten, steps) = match resolve_markers(sql, dialect, casing, reg)? {
        Resolution::Verbatim => (sql.to_string(), Vec::new()),
        // The empty-span fallback reprints the whole statement, so per-marker
        // byte ranges aren't available — report the rewrite without steps.
        Resolution::Reprinted(out) => (out, Vec::new()),
        Resolution::Spliced(repls) => {
            let steps = repls
                .iter()
                .map(|r| ExplainStep {
                    function: r.function.name(),
                    marker: r.marker.clone(),
                    derivative: r.derivative.clone(),
                })
                .collect();
            (apply_splice(sql, repls), steps)
        }
    };
    Ok(Explanation {
        original: sql.to_string(),
        rewritten,
        steps,
    })
}

/// Splice each replacement's derivative into `sql` by byte range, in reverse
/// source order so earlier offsets stay valid.
fn apply_splice(sql: &str, mut repls: Vec<Replacement>) -> String {
    repls.sort_by(|a, b| b.start.cmp(&a.start));
    let mut out = sql.to_string();
    for r in repls {
        out.replace_range(r.start..r.end, &r.derivative);
    }
    out
}

/// One marker's resolution: the byte range it occupies, its original call text,
/// and the derivative SQL it becomes.
struct Replacement {
    start: usize,
    end: usize,
    function: MarkerKind,
    marker: String,
    derivative: String,
}

/// How a statement resolves against its markers.
enum Resolution {
    /// No real marker — the input is returned unchanged.
    Verbatim,
    /// The empty-span fallback: only the fully-rewritten text is available (no
    /// per-marker byte ranges).
    Reprinted(String),
    /// The normal path: one [`Replacement`] per outermost marker.
    Spliced(Vec<Replacement>),
}

/// Run the marker pipeline (pre-gate → parse → collect → per-marker derivative)
/// *without* splicing, so both [`rewrite_sql`] and [`explain_sql`] share it.
fn resolve_markers(
    sql: &str,
    dialect: &dyn Dialect,
    casing: IdentCasing,
    reg: &RuleRegistry,
) -> Result<Resolution> {
    // 1. Parse-free pre-gate: no marker syntax, no parse, byte-identical out.
    if !pre_gate_hit(sql) {
        return Ok(Resolution::Verbatim);
    }

    // 2. The statement (or one of them) looks like it carries a marker; parse.
    let statements = Parser::parse_sql(dialect, sql)
        .map_err(|e| DiffError::Parse(format!("failed to parse SQL: {e}")))?;

    // 3. Statement-level context for the projection-boundary guard (F3/G4):
    //    the names of every *computed* select-list alias of a CTE/derived table.
    let mut aliases = ComputedAliases::default();
    for stmt in &statements {
        collect_computed_aliases(stmt, &mut aliases);
    }

    // 4. Locate the outermost markers (with their source spans). Nested markers
    //    are handled when their enclosing outermost marker is differentiated.
    let mut collector = MarkerCollector::default();
    for stmt in &statements {
        let _ = Visit::visit(stmt, &mut collector);
    }
    // Pre-gate false positive (e.g. only qualified markers, or `grad(` inside a
    // string literal): nothing to rewrite.
    if collector.found.is_empty() {
        return Ok(Resolution::Verbatim);
    }

    // 5. Empty spans are documented as possible; fall back to a correct (if not
    //    byte-identical) whole-statement reprint if any marker lacks a span.
    if collector.found.iter().any(|(span, _)| is_empty_span(span)) {
        return Ok(Resolution::Reprinted(reprint_fallback(
            statements, casing, reg, &aliases,
        )?));
    }

    // 6. Compute each replacement's byte range and derivative.
    //
    //    The marker name position (`span.start`) is reliable, but the Function's
    //    `span.end` is NOT: sqlparser under-reports it when the call's last
    //    argument ends in a `Cast` (excludes ` AS <type>`) or `Nested` (excludes
    //    the closing `)`), so trusting it under-splices and leaves corrupt SQL
    //    behind (#57). Instead, find the call's matching close paren over the
    //    token stream — re-tokenizing with the same dialect, which lexes strings
    //    and comments as single tokens so their parens don't miscount.
    let tokens = Tokenizer::new(dialect, sql)
        .tokenize_with_location()
        .map_err(|e| DiffError::Parse(format!("failed to tokenize SQL: {e}")))?;

    let mut repls = Vec::with_capacity(collector.found.len());
    for (span, marker_expr) in &collector.found {
        let derivative = differentiate_marker_tree(marker_expr, casing, reg, &aliases)?;
        let function = marker_expr_kind(marker_expr)
            .ok_or_else(|| DiffError::Internal("outermost marker lost its kind".into()))?;
        let start = locate(sql, span.start, false)
            .ok_or_else(|| DiffError::Internal("marker span start out of range".into()))?;
        let close = marker_call_close(&tokens, span.start).ok_or_else(|| {
            DiffError::Internal("could not locate the marker call's closing parenthesis".into())
        })?;
        // `close` is the location of the `)`; the exclusive byte end is one
        // character past it.
        let end = locate(sql, close, true)
            .ok_or_else(|| DiffError::Internal("marker span end out of range".into()))?;
        let marker = sql[start..end].to_string();
        repls.push(Replacement {
            start,
            end,
            function,
            marker,
            derivative,
        });
    }
    Ok(Resolution::Spliced(repls))
}

/// Differentiate one (possibly nested) marker subtree, returning the derivative
/// rendered to SQL text, parenthesized so it keeps the call's precedence.
fn differentiate_marker_tree(
    marker_expr: &Expr,
    casing: IdentCasing,
    reg: &RuleRegistry,
    aliases: &ComputedAliases,
) -> Result<String> {
    let mut clone = marker_expr.clone();
    let mut rw = MarkerRewriter {
        casing,
        reg,
        aliases,
    };
    if let ControlFlow::Break(err) = VisitMut::visit(&mut clone, &mut rw) {
        return Err(err);
    }
    Ok(clone.to_string())
}

/// The whole-statement reprint fallback (empty-span case).
fn reprint_fallback(
    mut statements: Vec<Statement>,
    casing: IdentCasing,
    reg: &RuleRegistry,
    aliases: &ComputedAliases,
) -> Result<String> {
    for stmt in &mut statements {
        let mut rw = MarkerRewriter {
            casing,
            reg,
            aliases,
        };
        if let ControlFlow::Break(err) = VisitMut::visit(stmt, &mut rw) {
            return Err(err);
        }
    }
    Ok(statements
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; "))
}

// ---------------------------------------------------------------------------
// Differentiating a single marker (args assumed already marker-free)
// ---------------------------------------------------------------------------

/// Differentiate a single marker call whose arguments are already free of
/// nested markers (guaranteed by the bottom-up post-order walk).
fn differentiate_marker(f: &Function, casing: IdentCasing, reg: &RuleRegistry) -> Result<Expr> {
    let kind = marker_kind(f).ok_or_else(|| DiffError::Internal("not a marker".into()))?;
    let args = positional_args(f).ok_or_else(|| {
        DiffError::InvalidMarker("marker call has non-positional arguments".into())
    })?;
    match kind {
        MarkerKind::Grad => {
            if args.len() != 2 {
                return Err(DiffError::InvalidMarker(format!(
                    "grad(expr, column) expects 2 arguments, got {}",
                    args.len()
                )));
            }
            let wrt = ColRef::from_wrt_arg("grad", args[1])?;
            differentiate(args[0], &wrt, casing, reg)
        }
        MarkerKind::Jvp => {
            if args.len() != 3 {
                return Err(DiffError::InvalidMarker(format!(
                    "jvp(expr, column, tangent) expects 3 arguments, got {}",
                    args.len()
                )));
            }
            let wrt = ColRef::from_wrt_arg("jvp", args[1])?;
            let seeds = vec![(wrt, args[2].clone())];
            jvp(args[0], &seeds, casing, reg)
        }
    }
}

/// The projection-boundary guard (design.md §3.5, F3/G4).
///
/// Errors if a marker argument references an identifier that is a *computed*
/// select-list alias of a CTE/derived table in the same statement, used as a
/// *non-`wrt`* term — differentiating it would silently treat an upstream
/// expression as a constant and drop gradient terms. The carve-out (G4): when
/// the alias *is* the `wrt` itself, every occurrence is the differentiation
/// leaf, so no term can be dropped and the guard stays quiet.
fn projection_guard(f: &Function, aliases: &ComputedAliases) -> Result<()> {
    if aliases.is_empty() {
        return Ok(());
    }
    let Some(args) = positional_args(f) else {
        return Ok(());
    };
    let Some(expr_arg) = args.first() else {
        return Ok(());
    };
    let wrt_name = args
        .get(1)
        .and_then(|a| ColRef::from_expr(a))
        .map(|c| c.name.value.to_ascii_lowercase());

    let mut cols = ColumnCollector::default();
    let _ = Visit::visit(*expr_arg, &mut cols);
    for c in cols.cols {
        let lname = c.name.value.to_ascii_lowercase();
        // Carve-out (G4): the wrt itself is always a leaf; never an error.
        if Some(&lname) == wrt_name.as_ref() {
            continue;
        }
        let is_boundary = match &c.qualifier {
            // A bare occurrence could bind to any computed alias in scope.
            None => aliases.bare.contains(&lname),
            // A qualified occurrence crosses a projection boundary only if the
            // qualifier names the relation that actually owns the alias. A base
            // column qualified to an unrelated table (e.g. `w.s` when the alias
            // `s` belongs to a different CTE) is NOT the alias — preserving the
            // qualifier-awareness the F2 ambiguity guard is built on.
            Some(q) => aliases
                .qualified
                .contains(&(q.value.to_ascii_lowercase(), lname.clone())),
        };
        if is_boundary {
            return Err(DiffError::ProjectionBoundary(format!(
                "`{}` is a computed select-list alias of a CTE/derived table used \
                 as a non-differentiation term; grad does not see through the \
                 projection boundary — differentiate inside that CTE instead",
                c.display()
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Visitors
// ---------------------------------------------------------------------------

/// Collects the outermost marker expressions (with their spans), skipping
/// markers nested inside another marker's arguments (handled bottom-up later).
#[derive(Default)]
struct MarkerCollector {
    depth: usize,
    found: Vec<(Span, Expr)>,
}

impl Visitor for MarkerCollector {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        if is_marker_expr(expr) {
            if self.depth == 0 {
                self.found.push((expr.span(), expr.clone()));
            }
            self.depth += 1;
        }
        ControlFlow::Continue(())
    }

    fn post_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        if is_marker_expr(expr) {
            self.depth -= 1;
        }
        ControlFlow::Continue(())
    }
}

/// Replaces each marker with `Nested(derivative)`, bottom-up (post-order), so a
/// nested marker's own arguments are already marker-free when it is reached.
struct MarkerRewriter<'a> {
    casing: IdentCasing,
    reg: &'a RuleRegistry,
    aliases: &'a ComputedAliases,
}

impl VisitorMut for MarkerRewriter<'_> {
    type Break = DiffError;

    fn post_visit_expr(&mut self, expr: &mut Expr) -> ControlFlow<DiffError> {
        let replacement = match expr {
            Expr::Function(f) if marker_kind(f).is_some() => {
                if let Err(err) = projection_guard(f, self.aliases) {
                    return ControlFlow::Break(err);
                }
                match differentiate_marker(f, self.casing, self.reg) {
                    Ok(d) => Some(d),
                    Err(err) => return ControlFlow::Break(err),
                }
            }
            _ => None,
        };
        if let Some(d) = replacement {
            *expr = Expr::Nested(Box::new(d));
        }
        ControlFlow::Continue(())
    }
}

/// Collects the column references directly appearing in an expression tree.
#[derive(Default)]
struct ColumnCollector {
    cols: Vec<ColRef>,
}

impl Visitor for ColumnCollector {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        match expr {
            Expr::Identifier(_) | Expr::CompoundIdentifier(_) => {
                if let Some(cr) = ColRef::from_expr(expr) {
                    self.cols.push(cr);
                }
            }
            _ => {}
        }
        ControlFlow::Continue(())
    }
}

// ---------------------------------------------------------------------------
// Computed-alias collection for the projection-boundary guard
// ---------------------------------------------------------------------------

/// The *computed* select-list aliases of the CTEs/derived tables in a
/// statement, recorded so the guard can distinguish a reference that crosses a
/// projection boundary from a same-named base column that does not.
#[derive(Default)]
struct ComputedAliases {
    /// Alias names referenceable by a *bare* (unqualified) occurrence.
    bare: HashSet<String>,
    /// `(owning relation name, alias name)` for CTE/derived-table computed
    /// aliases — so a *qualified* occurrence `rel.alias` is matched only against
    /// the relation that actually owns it. This is what keeps a base column
    /// like `w.s` from colliding with an unrelated CTE alias `s` (F2's
    /// qualifier-awareness, applied to the F3/G4 guard).
    qualified: HashSet<(String, String)>,
}

impl ComputedAliases {
    fn is_empty(&self) -> bool {
        self.bare.is_empty() && self.qualified.is_empty()
    }
}

fn collect_computed_aliases(stmt: &Statement, out: &mut ComputedAliases) {
    match stmt {
        Statement::Query(q) => walk_query(q, None, out),
        Statement::Insert(insert) => {
            if let Some(source) = &insert.source {
                walk_query(source, None, out);
            }
        }
        _ => {}
    }
}

/// `owner` is the name of the relation whose *own* projection aliases we are
/// collecting (a CTE name, or a derived-table alias) — `None` for the outer
/// query's own select list, whose aliases can only be referenced bare.
fn walk_query(q: &Query, owner: Option<&str>, out: &mut ComputedAliases) {
    if let Some(with) = &q.with {
        for cte in &with.cte_tables {
            let name = cte.alias.name.value.to_ascii_lowercase();
            walk_query(&cte.query, Some(&name), out);
        }
    }
    walk_set_expr(&q.body, owner, out);
}

fn walk_set_expr(body: &SetExpr, owner: Option<&str>, out: &mut ComputedAliases) {
    match body {
        SetExpr::Select(select) => walk_select(select, owner, out),
        SetExpr::Query(q) => walk_query(q, owner, out),
        SetExpr::SetOperation { left, right, .. } => {
            walk_set_expr(left, owner, out);
            walk_set_expr(right, owner, out);
        }
        _ => {}
    }
}

fn walk_select(select: &Select, owner: Option<&str>, out: &mut ComputedAliases) {
    for item in &select.projection {
        if let SelectItem::ExprWithAlias { expr, alias } = item {
            // A *computed* alias is one whose projected expression is not a
            // plain column reference. `ColRef::from_expr` is the single place
            // that recognizes a column reference (seeing through `Nested`).
            if ColRef::from_expr(expr).is_none() {
                let name = alias.value.to_ascii_lowercase();
                if let Some(o) = owner {
                    out.qualified.insert((o.to_string(), name.clone()));
                }
                out.bare.insert(name);
            }
        }
    }
    // Recurse into derived tables (FROM subqueries), which are projection
    // boundaries too — a derived table's aliases are owned by its own alias.
    for twj in &select.from {
        walk_table_factor(&twj.relation, out);
        for join in &twj.joins {
            walk_table_factor(&join.relation, out);
        }
    }
}

fn walk_table_factor(tf: &TableFactor, out: &mut ComputedAliases) {
    if let TableFactor::Derived {
        subquery, alias, ..
    } = tf
    {
        let owner = alias.as_ref().map(|a| a.name.value.to_ascii_lowercase());
        walk_query(subquery, owner.as_deref(), out);
    }
}

/// The [`Location`] of the `)` that closes the marker call whose name token
/// starts at `name_start`, found by matching parentheses over the token stream.
///
/// This is authoritative where `Expr::span()` is not: sqlparser under-reports a
/// `Function`'s span end when its last argument ends in a `Cast` (the ` AS
/// <type>` tail is excluded) or a `Nested` (the closing `)` is excluded), so a
/// span-based splice leaves trailing bytes behind and corrupts the output
/// (#57). Matching over tokens is exact because the tokenizer lexes string
/// literals and comments into single tokens, so parentheses inside them are
/// never counted. The marker name position (`name_start`) is itself reliable;
/// between it and the call's `(` there is only trivia (skipped here).
fn marker_call_close(tokens: &[TokenWithSpan], name_start: Location) -> Option<Location> {
    let mut depth = 0usize;
    let mut opened = false;
    for t in tokens.iter().filter(|t| t.span.start >= name_start) {
        match t.token {
            Token::LParen => {
                depth += 1;
                opened = true;
            }
            Token::RParen => {
                depth = depth.checked_sub(1)?;
                if opened && depth == 0 {
                    return Some(t.span.start);
                }
            }
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Span (1-based character line/column) → byte offset conversion (G3)
// ---------------------------------------------------------------------------

/// `sqlparser` uses `line: 0`/`column: 0` for an empty/unknown location.
fn is_empty_span(span: &Span) -> bool {
    span.start.line == 0 || span.start.column == 0 || span.end.line == 0 || span.end.column == 0
}

/// Convert a 1-based (line, character-column) [`Location`] to a byte offset in
/// `sql`. Character-column, not byte-column: a multibyte character before the
/// target shifts the byte offset past the column number (G3).
///
/// With `past = false` the returned offset is the *start* byte of the character
/// at `loc`; with `past = true` it is the byte *one past* that character — used
/// for an inclusive span end, so the whole marker (its closing `)` included) is
/// covered by `start..end`.
fn locate(sql: &str, loc: Location, past: bool) -> Option<usize> {
    let mut line: u64 = 1;
    let mut col: u64 = 1;
    for (byte_idx, ch) in sql.char_indices() {
        if line == loc.line && col == loc.column {
            return Some(if past {
                byte_idx + ch.len_utf8()
            } else {
                byte_idx
            });
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    // A location just past the final character maps to the end of the string.
    if line == loc.line && col == loc.column {
        return Some(sql.len());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_gate_matches_unqualified_markers() {
        assert!(pre_gate_hit("SELECT grad(x, x) FROM t"));
        assert!(pre_gate_hit("SELECT jvp(x, x, dx) FROM t"));
        assert!(pre_gate_hit("grad(x,x)")); // at start of input
        assert!(pre_gate_hit("SELECT GRAD (x, x) FROM t")); // case + whitespace
        assert!(pre_gate_hit("SELECT AVG(grad(x, x)) FROM t")); // after `(`
    }

    #[test]
    fn pre_gate_rejects_non_markers() {
        assert!(!pre_gate_hit("SELECT a + b FROM t")); // no marker
        assert!(!pre_gate_hit("SELECT mygrad(x) FROM t")); // longer identifier
        assert!(!pre_gate_hit("SELECT schema.grad(x, x) FROM t")); // qualified
        assert!(!pre_gate_hit("SELECT grad AS g FROM t")); // no open paren
        assert!(!pre_gate_hit("SELECT upgrade(x) FROM t")); // 'grad' inside a word
    }
}
