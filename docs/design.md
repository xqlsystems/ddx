# `ddx` — portable autograd for composable databases

_author_: Alex Merose
_co-author_: Claude (Opus 4.8, Fable 5, Opus 5.5), via Claude Code
_status_: Living design — built on DataFusion; DuckDB-Python in progress (M5)
_last updated_: 2026-10-18

---

## 1. What `ddx` is

`ddx` is autograd for composable databases: you write a model's forward pass
and loss as SQL, and take its gradient in SQL,

```sql
WITH loss AS (SELECT SUM(val * val) AS l FROM w)
SELECT i, val - 0.1 * grad(loss, val) AS val FROM w
```

and the gradient comes back as a column beside each row's value, computed by
the engine as a sequence of plans. The destination is training ML models in
SQL — a [differentiable](https://www.youtube.com/watch?v=LNNU33TmBFk) [database](https://www.youtube.com/watch?v=jUe3rvTmv7Q). It ships as engine-neutral Rust crates with
thin per-engine adapters. Its Python package, `ddxdb`, brings it to
[xarray-sql](https://github.com/xqlsystems/xarray-sql) on DataFusion today, and
on DuckDB-Python next (M5, §4.2). A DuckDB community extension, for
[duckdb-zarr](https://github.com/xqlsystems/duckdb-zarr) and other DuckDB users
outside Python, comes after (§8).

**Data model.** `ddx` assumes the XQL data model: an N-dimensional array is a
long/tidy relational table — one row per coordinate tuple, dimensions and
variables as columns (`temp(time, lat, lon)` becomes rows of `(time, lat, lon,
temp)`). A derivative is just another column aligned to the same coordinates,
which is what makes `grad` compose with ordinary SQL. See
[xql.systems](https://xql.systems) and
[xarray-sql](https://github.com/xqlsystems/xarray-sql) for the model in depth;
this doc takes it as given.

**Thesis — the vmap insight.** Because each row of a table is an independent
evaluation point, differentiating a column expression and letting the engine
evaluate it per row is the relational equivalent of `jax.vmap(jax.grad(f))` —
the rows are the batch dimension. This turns SQL into a place you can express
gradients, directional derivatives, and (bounded — §4) training loops. `ddx` is
named for exactly this: trained ML models as differentiable databases, `d/dx`
of a table.

**Grounding.** The design starts from a working prototype,
[xarray-sql#192](https://github.com/xqlsystems/xarray-sql/pull/192) [3], which
implements `grad`/`jvp`/`vjp` for DataFusion, and a follow-on demo,
[xarray-sql#196](https://github.com/xqlsystems/xarray-sql/pull/196) [4], which
trains a real MLP with every gradient computed in SQL. §3 and §4 explain what
each taught the design and generalize it into a reusable component. Every
foundational claim below that could be checked with a small program was
checked with one — the programs live in [`docs/spikes/`](spikes/README.md) and
are cited by tag throughout.

**One engine, built in two stages.** ddx was designed as two layers, and the
first became part of the second:

- **v1 — calculus as columns** (§3, kept as history). `ddx-core`
  differentiated SQL scalar expressions and rewrote `grad()`/`jvp()` calls to
  derivative SQL before planning. Its ceiling: an N-parameter gradient
  computed as N independent scalar derivations does not scale, which is the
  reason ML left symbolic differentiation for reverse-mode AD in the first
  place (Baydin et al. [1]).
- **v2 — query-level AD** (§4), the product. `ddx-ad` differentiates whole
  queries, not expressions, the way JAX differentiates functions: one rule per
  relational primitive (map, select, broadcast, reduce), composed, in reverse
  mode (`grad`, `vjp`) and forward mode (`jvp`), which compose with each other.
  The sharing that scalar mode lacks comes from materialized intermediate
  relations (the tape), rather than from shared work inside expressions. v1's scalar engine is
  v2's map rule. Verified machine-exact against `jax.grad` on an MLP and on
  attention.

Both follow the same shape: a minimal-dependency, engine-neutral core, with
thin adapters per engine. `ddx-core` depends on `sqlparser` only; `ddx-ad` on
`substrait` and `ddx-core`. Neither depends on `datafusion` or `duckdb`.

**v1's SQL surface is removed (2026-10-13, `S15`).** v2 covers what v1 did: a
row's derivative is the gradient of a sum over rows, or a `jvp` with a tangent
of 1. Two meanings of `grad` in one SQL statement were a correctness hazard.
v1's scalar differentiator stays, as `ddx-core`: it is v2's map rule. §3 is
kept as the record of how that differentiator was designed.

### 1.1 Non-goals

- Not a runtime tensor library or a GPU kernel engine. `ddx` differentiates
  SQL queries, never imperative code, and its tape is materialized relations,
  not a record of Python or Rust operations.
- Not a plan-interchange format of its own. ddx reads Substrait plans and
  writes Substrait, or SQL where an engine's Substrait consumer falls short
  (§4.2).
- Not a derivative where none is defined. An unsupported construct (a rank
  used as a value, a running `MAX`, a `UNION ALL` in reverse mode for now) is
  a typed error, never a silently wrong gradient (§4.6).
- Not every engine at once. DataFusion first, then DuckDB-Python; a DuckDB
  extension, and Postgres (which needs array/XQL support first, via `pgrx`),
  later.

### 1.2 Success criteria

- A single engine-independent core implements differentiation once; every
  engine integration is a thin adapter over it. The run loop and the SQL
  rewrite live in the core too, so an adapter implements only an engine's
  primitives (`S14`, `S17`).
- It ships in real, actively-used projects with no regressions, in order:
  xarray-sql on DataFusion (done, through `ddxdb`), xarray-sql on
  DuckDB-Python (M5), the other Arrow-based engines xarray-sql registers
  into, then duckdb-zarr through a DuckDB extension.
- The `grad`/`vjp`/`jvp` surface is portable — the same SQL-level calls, one
  core defining what they mean, on every target engine. Portability lives at
  the SQL surface and at the Substrait plan surface — never in a
  project-specific plan-interchange format.

---

## 2. Design principles

1. **Differentiate once, on a real IR, not a bespoke one.** v1 operates
   directly on `sqlparser::ast::Expr` — the same parser DataFusion uses. v2
   operates directly on `substrait::proto` types. Neither layer invents its
   own representation; each reuses the closest real one for its scope.
2. **Rewrite, don't execute.** `grad`, `vjp` and `jvp` in SQL are rewritten
   before the engine sees the statement, into reads of what ddx's programs
   computed — never functions that run per row (§4.8). Every integration is
   fundamentally "find the call, differentiate what it names, put the result
   in its place, hand plain SQL onward."
3. **Never guess; derive, or be told.** A derivative follows from what an
   operator computes: a `SUM` is linear whatever it sums, a join broadcasts
   whatever it pairs. ddx reads that off the operator, never off a guess about
   the query's intent, because a wrong guess is a silently wrong gradient.
   Where intent genuinely cannot be read off the plan, the user says it with a
   function ddx has claimed the name of: `grad`/`jvp` in v1, and
   `ddx_stop_gradient` in v2. (This principle first read "tag explicitly;
   never infer", and asked for a tag on every contraction and reduction.
   Building v2 showed those tags classified nothing the rules needed; see
   decision `S10`.)
4. **SQL (or its plan) is the portable surface.** `grad`, `vjp` and `jvp`
   are ordinary SQL calls that take a query. ddx reads each query's plan from
   the engine's own SQL-to-plan machinery, and hands the engine back plans, or
   SQL where its Substrait consumer falls short (§4.2). No project-specific
   interchange format carries meaning between engines.
5. **Fail loud, never silently wrong.** An unsupported construct is a typed
   error, not an approximate or silently-zero derivative. This is a
   numerical-correctness product, and every decision is weighed against this
   first.
6. **Prove it in real projects.** xarray-sql and duckdb-zarr are acceptance
   tests, not demos.

---

## 3. v1: calculus as columns

> **Removed (2026-10-13, `S15`).** The marker rewrite (`rewrite_sql`), both
> DataFusion paths (§3.3) and the per-engine integration (§3.4) are gone. The
> scalar engine of §3.2 and the differentiation surface of §3.6 remain, as
> `ddx-core`, the map rule of v2 (§4.3). The rest of this section is history,
> including §3.4's DuckDB plan, which §4.2 and decision `S22` replace.

### 3.1 The core insight: markers, not UDFs

A scalar UDF only ever sees *values* at runtime — never the *symbolic
expression* of its argument. But differentiation is a function of the
symbolic form, so `grad(...)` cannot be a real row function. In the
prototype, `grad`/`jvp`/`vjp` are markers: no-op functions whose only job is
to parse and carry the differentiation request through the pipeline. They are
always rewritten away before execution, and deliberately error if one ever
reaches execution (`[F-proto-3.1]`).

**Consequence:** "install a UDF in each database" is the wrong mental model.
What each engine needs is a *rewrite hook* at or before planning time. UDF
registration exists only to make the marker call parse.

Other decisions inherited from the prototype and kept: the data model stays
scalar-only (a gradient/Jacobian is several scalar columns, never a nested
array — a nested-array cell breaks the one-value-per-coordinate model);
higher-order differentiation falls out for free from bottom-up rewriting
(`grad(grad(f,x),x)` just works); differentiating through an aggregate is
linearity, so the marker goes *inside* the aggregate (`AVG(grad(loss,
theta))`) — this is what makes a gradient-descent step expressible in SQL;
and a "calculus compiler" is exported alongside the marker path —
`differentiate_sql(expr, wrt)` returns the derivative as SQL text, for
embedding an update rule where a marker can't reach (e.g. inside a recursive
term).

### 3.2 `ddx-core`: the engine

`ddx-core` differentiates `sqlparser::ast::Expr` directly — the
[`sqlparser`](https://docs.rs/sqlparser/) crate (Apache's
`datafusion-sqlparser-rs`, the same parser DataFusion uses), which ships
`DuckDbDialect`, `PostgreSqlDialect`, `GenericDialect`, and more. There is no
bespoke intermediate representation and no adapter layer — the AST is the IR.
Public surface:

```rust
// The entry point is an object, not free functions, so the user rule
// registry and dialect/identifier config have a home.
pub struct Ddx { /* rules: RuleRegistry, ident/dialect policy, … */ }

impl Ddx {
    pub fn new() -> Self;                                    // built-in rules
    pub fn register(&mut self, name: &str, rule: Rule);      // user-extensible

    // The whole path: parse the statement, find every grad/jvp call,
    // differentiate its argument, splice the derivative back by source
    // span, return SQL text. A statement with no marker returns
    // byte-identical.
    pub fn rewrite_sql(&self, sql: &str, dialect: &dyn Dialect) -> Result<String, DiffError>;

    // Lower-level, on the AST directly (used by the DataFusion bridge, §3.3).
    pub fn differentiate(&self, e: &ast::Expr, wrt: &ColRef) -> Result<ast::Expr, DiffError>;
    pub fn jvp(&self, e: &ast::Expr, seeds: &HashMap<ColRef, ast::Expr>) -> Result<ast::Expr, DiffError>;
    // No scalar `vjp` — the name is reserved for query-level reverse-mode
    // AD (§4), where it does actual reverse accumulation. `[Q7]`
}

// Column identity read off the AST. Stores sqlparser `Ident`s (which keep
// quote-style) and compares with per-dialect identifier folding, not
// raw-string equality.
pub struct ColRef { pub qualifier: Option<Ident>, pub name: Ident }
```

The rules match the `ast::Expr` variants v1 supports —
`Expr::BinaryOp{left,op,right}` (`+ - * /`), `Expr::Function`
(name-dispatched: `sin`, `power`, …), `Expr::UnaryOp` (minus), `Expr::Cast`,
`Expr::Nested`, `Expr::Identifier`/`CompoundIdentifier` (leaves), `Expr::Value`
(literals) — and return `NotImplemented` for everything else.

**Dependency: `sqlparser` only.** No DataFusion, no `protoc`, no engine
crate. `ddx-core` re-exports `sqlparser`, and pins the exact version
DataFusion requires — a `sqlparser` bump is a breaking release of
`ddx-core` `[G2]`.

**Design decisions inside the engine** (each one closes a way an unsupported
construct could otherwise produce a wrong number instead of an error):

- **Extensible rule registry, keyed by function name.** Built-ins populate a
  registry users can extend: `registry.register("myfn", rule)`. For a unary
  `f(u)`, a user rule supplies just `f'(u)`; the engine applies the chain
  rule automatically. Dispatch case-folds the function name, and a minimal
  canonicalization folds `pow` to `power` before dispatch. A fuller dialect
  name-normalization *table* is deferred to the §3.6 roadmap; note that some
  dialect spellings deliberately cannot be folded — `log` is natural log on
  some engines but base-10 on DuckDB, so folding `ln`/`log` together would be
  a silently-wrong derivative, exactly what this project refuses.
- **The smart constructors — `add`/`sub`/`mul`/`div`/`neg` — own three
  correctness properties, not just algebraic simplification:**
  - *0/1-folding*, the JAX-`Zero`-tangent equivalent: drops structurally-zero
    terms and short-circuits dead branches, keeping output compact. This is a
    stated NULL-semantics convention, not an accident: folding `0 *
    (NULL-valued expr)` to `0` where unfolded SQL would give `NULL` matches
    JAX's `Zero`-tangent treatment, but the two disagree on NULL-bearing rows
    — documented and tested, not silent `[F11]`.
  - *Numeric-type policy*: `div()` (and anything that can hit integer
    operands) wraps in `CAST(… AS DOUBLE)`, and literals are emitted
    `DOUBLE`-typed. Differentiation runs pre-binding, so operand types are
    always unknown, and SQL integer division truncates on some engines but
    not others — `grad(x/y,y)` on a `BIGINT` column silently gives `0`
    instead of the right fraction on one engine and the correct float on
    another without this `[F4]`/`[R1b]`.
  - *Precedence-safe construction*: composite operands are wrapped in
    `Expr::Nested` before rendering. `sqlparser`'s `Display` for a binary op
    has no precedence parentheses, so a *constructed* tree like `mul(add(a,b),
    c)` — exactly what the product rule builds — displays as `a + b * c`,
    which reparses as the wrong expression. This is a wrong number in valid
    SQL with nothing failing downstream: confirmed by spike, and fixed by
    `Nested`-wrapping `[G1]`.
- **Identifier folding, not raw-string equality, and the fold is
  per-dialect.** SQL unquoted identifiers are case-insensitive, so
  `grad(Temp*Temp, temp)` must match — otherwise it silently differentiates
  to `0`. The exact rule differs by engine: DataFusion/Postgres-style
  unquoted-lowercase-folds but quoted stays case-sensitive; DuckDB folds
  *quoted* identifiers too. `ColRef` equality takes the dialect and applies
  its rule to each part; output preserves original spelling `[F1]`.
- **Qualifier-aware, with an ambiguity guard on uncertain occurrences.**
  `ColRef` carries the qualifier straight off `CompoundIdentifier`, so
  `grad(a.x + b.x, a.x)` differentiates the right column with no catalog.
  The guard fires only when an occurrence of the `wrt` base name can't be
  pinned syntactically — a bare occurrence when `wrt` is qualified (or vice
  versa) — and hard-errors, demanding full qualification. A fully-qualified,
  unambiguous `wrt` like `grad(a.x*b.x, a.x)` is accepted `[F2]`.
- **Marker names are reserved precisely.** `grad`/`jvp` are claimed only as
  unqualified calls (`myschema.grad(…)` is left alone) and matched
  case-folded, so `GRAD(x,x)` is caught too `[F8]`/`[G7]`.
- **Splice by source span, never reprint the statement.** `rewrite_sql`
  first runs a parse-free, case-insensitive pre-gate — a scan for an
  unqualified `grad`/`jvp\(` substring — and returns the input verbatim if it
  doesn't hit, so a statement whose text contains no such substring is *never
  parsed* and can't be failed or reformatted by parser coverage gaps. The gate
  is a substring filter, so it also hits on a `grad(` that appears only inside
  a string literal or comment: such a statement *is* parsed, but with no real
  marker it still comes back verbatim (the collector finds nothing) — a
  false-positive costs a parse, never a wrong rewrite. The one residual is a
  marker-free statement that both mentions the substring *and* uses syntax the
  dialect can't parse: it would hard-error where a stricter gate wouldn't
  (documentation-only, `[F5]`). When the gate hits a real marker, only the
  marker call's byte range is replaced, everything else stays byte-identical. This is a real
  subsystem, not a one-liner: `sqlparser`'s `Spanned` gives line/column in
  1-based *characters*, not byte offsets, so the splice needs a
  UTF-8-aware conversion, must handle multiple and nested markers (spliced
  in reverse source order, nested ones rewritten bottom-up), and must fall
  back safely on the empty spans the API documents as possible `[F5]`/`[G3]`.
- **Port the prototype's 15 rule unit tests** — they pin the math unchanged.

### 3.3 The rewrite mechanism: two paths

**Path A — SQL source-to-source rewrite (universal, every target).**
Intercept the SQL string before it reaches the engine, rewrite every
`grad`/`jvp` call to derivative SQL, pass plain SQL onward. It runs before
planning, so it works for every query shape the parser accepts — recursive
CTEs, DML, subqueries — which is what lets a whole training loop live in one
query. Both xarray-sql and the DuckDB extension rely on it.

Applicability is capped by `sqlparser`'s per-dialect coverage on
marker-bearing queries — not by what `grad` touches, since the whole
statement must parse to find the marker. This is a real, permanent
version-treadmill (DuckDB moves faster than `sqlparser`'s `DuckDbDialect`
follows) but a narrower one than first assumed: spiked against `DuckDbDialect`
@ `sqlparser` 0.62.0, `SELECT * EXCLUDE`, `FROM`-first queries, bare `FROM t`,
lambdas, and `t.* REPLACE (…)` all parse; the real misses are `PIVOT` and `#1`
positional columns `[G9]`. The parse-free pre-gate and source-span splicing
above (§3.2) mean this coverage gap only ever bounds a query whose text
*contains a `grad(`/`jvp(` substring* — almost always a real marker, but also
the rare false positive where the substring sits inside a string literal or
comment (§3.2) — and reprint fidelity is never a separate risk.

**Path B — in-engine plan rewrite, native Rust DataFusion.** A marker UDF plus
an `AnalyzerRule` so `grad()` works bare, with no wrapper, across both the SQL
and DataFrame APIs. This exists in v1 for exactly one engine, native Rust
DataFusion, as the cheapest possible proof that `ddx-core` can drive an
in-engine plan-time rewrite and not merely a text preprocess — neither
acceptance target actually needs it, since xarray-sql is Python (Path A only)
and duckdb-zarr is DuckDB (no plan hooks at all, §3.4). It does not de-risk
DuckDB's harder C++ path; the two share only the shallow "walk plan, find
marker, substitute" pattern `[G6]`.

Implementation is a bridge, not a second rule engine. The rule walks the
bound `LogicalPlan`, and for each `grad()` call: unparses its argument with
DataFusion's `expr_to_sql` (which emits exactly `ddx-core`'s
`sqlparser::ast::Expr` input type, provided the two crates' `sqlparser`
versions are pinned identical — if they ever diverge, the bridge degrades to
a string round-trip, still one rule engine, less elegant `[G2]`);
differentiates via `ddx-core`; re-plans the result back to a DataFusion
`Expr` against the node's schema. Because the input is already bound, its
columns unparse qualified, so this path is binding-aware for free — the
ambiguity guard (§3.5) never fires here. Two practical details: DataFusion's
`add_analyzer_rule` runs after `TypeCoercion`, so the marker's argument may
already carry injected casts by the time the rule sees it (handled — `Cast`
has a rule — but the marker UDF must be coercion-tolerant, which
`Signature::any` achieves); and the re-plan step needs a function registry.

**Correction, from building it (M2).** This section previously named
`SessionState::create_logical_expr` as the re-plan seam `[G7]`. That seam is
not reachable from where the rewrite happens: `AnalyzerRule::analyze` receives
only `(LogicalPlan, &ConfigOptions)` — no `SessionState` — and a rule cannot
hold the state it is installed into without a reference cycle. What the bridge
does instead is plan the derivative itself with `SqlToRel` over a minimal
`ContextProvider` supplying just a function registry. This is *more*
self-contained than the seam it replaces, not a workaround: the only thing a
scalar expression needs from a context is function resolution, and a
differentiated expression has no table references to resolve, because
differentiation maps column references to column references. The optimistic
half of `[G2]` also held in practice — `expr_to_sql` emits and
`sql_to_expr` accepts the identical `sqlparser::ast::Expr`, so the bridge is
type-level end to end with no SQL string in between, and the documented
degrade-to-string fallback is unused. A test asserts the single-version
resolution so a future `datafusion` bump fails at the pin rather than
confusingly at the bridge.

### 3.4 Per-engine integration

| Integration | Dialect | How the rewritten SQL reaches the engine |
| --- | --- | --- |
| **Rust DataFusion** (`ddx-datafusion` helper) | DataFusion's | `ctx.sql(ddx.rewrite_sql(sql, dialect)?)` — one line |
| `ddxdb` (Python → DataFusion) | DataFusion-compatible | `Context.sql()` shim calls `rewrite_sql`, stock context plans it |
| `ddxdb` for DuckDB-python | `DuckDbDialect` | preprocess the string before `duckdb.sql(...)` |
| `ddx` (DuckDB community ext) | `DuckDbDialect` | `ddx('<sql>')` table function calls `rewrite_sql`, runs on an inner connection |

**DataFusion / xarray-sql.** `datafusion-python` doesn't expose injecting an
`AnalyzerRule` into its `SessionContext` `[R2]` — which is why the SQL rewrite
is the path here, not a limitation being worked around. The gap is structural,
not a missing convenience method: re-verified at M1 against `datafusion` 54.0.0
(`docs/spikes/datafusion_python_analyzer_rule_r2.py`), the *FFI capsule vocabulary*
the bindings are built on has no analyzer-rule capsule at all, so a compiled
Rust extension can't inject one either. `ddxdb` wraps a `Ddx`
and exposes `rewrite_sql` plus a `Context.sql()` shim; xarray-sql pulls it in
as an optional extra, `pip install "xarray-sql[ddx]"`, so autograd is opt-in
and costs nothing for users who don't ask for it `[Q4]`. Native Rust
DataFusion additionally gets `ddx-datafusion` (deps: `ddx-core` +
`datafusion`), exposing both the one-line `ddx_sql` helper (Path A) and the
marker-UDF + `AnalyzerRule` bridge (Path B, §3.3).

**DuckDB / duckdb-zarr.** DuckDB's actual C extension header
(`duckdb_extension.h`, what the `duckdb` crate's `loadable-extension`
feature binds) exposes registration for only scalar, aggregate, table, and
cast functions plus replacement scans — zero optimizer, parser, operator, or
bound-expression hooks, corroborated by duckdb-zarr itself, a mature
extension using exactly table functions and nothing deeper `[R1]`. So a
native bare-`grad()` rewrite is impossible in a Rust community extension: a
scalar UDF only ever receives executed values, never a symbolic argument
tree. The design instead:

- Ships a `ddx('<sql>')` **table function** as the primary, but explicitly
  *transitional*, form. The same C API exposes reading a literal SQL string
  at bind time and executing a query on an inner connection to the same
  database, so `ddx('<sql>')` reads the literal, rewrites markers via
  `ddx-core::rewrite_sql` with `DuckDbDialect`, runs the plain SQL on an
  inner connection, and streams the result back:
  ```sql
  INSTALL ddx FROM community;
  SELECT * FROM ddx('SELECT grad(sin(x), x) AS d FROM t');
  ```
  Re-entrancy is validated: an inner query on the same DB, run mid-execution
  of an outer table scan, is safe — no deadlock, reads of committed data
  work, DML works — with one real consequence: the inner connection runs in
  its own transaction and cannot see the *outer* connection's uncommitted
  writes `[R1b]`. So `ddx('…')` is the right tool for self-contained queries
  (including a whole recursive-CTE training loop passed as one string); a
  training loop that mutates parameters across statements inside an open
  transaction needs client-side Path A instead, which rewrites on the
  caller's own connection and preserves session/transaction visibility.
  `ddx('…')` defaults to read/SELECT-only (DuckDB's own precedent,
  `query('sql')`, is *not* actually SELECT-only, so this is ddx's own
  conservative choice, not inherited) and routes DML training loops through
  client-side Path A. Document dollar-quoting (`ddx($$ … $$)`) as the house
  style, since SQL-in-a-string quoting is unpleasant for the flagship
  recursive-CTE examples `[F7]`.
- Ships `ddxdb`'s client-side Path A for DuckDB-Python as a zero-hook
  fallback, available day one.
- Keeps a C++/Rust hybrid — a DuckDB `OptimizerExtension` walking the
  *bound* plan, bridged to `ddx-core` via [cxx.rs](https://cxx.rs/) — as the
  documented, correctness-superior route to bare `grad()` anywhere in a
  normal `SELECT`, deferred rather than built up front. Its advantage is
  structural: running after binding, it is immune to every silent-wrong
  class the syntactic path must guard against (identifier case, qualification
  ambiguity, parser coverage) because columns arrive already resolved. It
  stays deferred because its hard part — rebuilding a *bound* derivative
  expression with correct `ColumnBinding` indices and catalog entries on the
  way back — is orthogonal to what v1 needs to prove and version-coupled
  forever, and DataFusion's Path B already buys the in-engine validation far
  more cheaply. A miniature spike (round-trip one bound expression through
  `ddx-core` and back) is scheduled alongside DuckDB integration to keep this
  a known quantity rather than a standing unknown `[Q6]`.
- Rejects `CREATE MACRO` outright — macros are fixed expansions and cannot
  perform differentiation.

**Postgres / `ddx-pg`.** Later — needs array/XQL support first (via `pgrx`);
its native path would use a planner hook.

### 3.5 Column identity and the projection boundary

A pre-binding syntactic rewrite has two things to get right about columns.
The first — telling `a.x` from `b.x` — mostly dissolves with the ambiguity
guard already described (§3.2). The second does not dissolve, and is the more
important half of this section.

**Columns are leaves — `grad` does not see through CTEs or views.**
Differentiation stops at column references, so a column computed
*upstream* — a CTE, subquery, or view select-list expression — is an opaque
constant to it; every projection boundary is an implicit `stop_gradient`.
This is defensible relational semantics, but a real trap for the pitched use
case: factoring a loss through a CTE silently drops terms.

```sql
WITH v AS (SELECT x, sin(x) AS s FROM t)
SELECT grad(s * x, x) FROM v       -- ds/dx treated as 0 → result = s = sin(x)
SELECT grad(sin(x) * x, x) FROM t  -- inlined by hand → cos(x)*x + sin(x)
```

**The contract:** `grad` differentiates the expression as written, against
the relation it directly queries, never through view/CTE definitions.
`rewrite_sql` sees the whole statement, so a best-effort guard catches the
worst subcase: if a marker argument references an identifier that is a
computed select-list alias of a CTE/derived table *in the same statement*, it
errors with "differentiate inside the CTE instead" rather than silently
dropping the term. It cannot see catalog views — that residual is
documentation-only `[F3]`.

One carve-out is essential: when the computed alias *is* the `wrt` itself
(`grad(s*s, s)`), every occurrence of it is the differentiation leaf, so no
term can be silently dropped, and `d/ds(s*s) = 2s` is exactly right. The
guard fires only when a computed alias appears as a *non-`wrt`* term —
never when it is the `wrt` `[G4]`.

### 3.6 The differentiation surface

- `grad(expr, column)` → `d(expr)/d(column)`.
- `jvp(expr, column, tangent)` → forward-mode `d(expr)/d(column) · tangent`;
  a multi-input directional derivative is a sum of `jvp` terms.
- `differentiate_sql(expr, wrt)` → the derivative as SQL text — the "calculus
  compiler" escape hatch, for embedding an update rule where a marker can't
  reach.
- No scalar `vjp` — reserved for the query-level operation (§4) `[Q7]`.
- Rules: `+ - * /`; the unary chain rule for the trig/inverse-trig/exp/log/
  hyperbolic set plus `abs`; `power` with a constant base or exponent.
  Higher-order via nesting; through-aggregate via linearity.

**What you can write** (a `grad(...)` call rewrites *in place*, so anywhere a
scalar expression is legal, `grad` is legal):

| You write | Rewrites to | Works? |
| --- | --- | --- |
| `SELECT grad(sin(x)*y, x) FROM g` | `SELECT (cos(x)*y) FROM g` | ✅ |
| `SELECT grad(x*y,x) AS dfdx, grad(x*y,y) AS dfdy FROM g` | `SELECT y AS dfdx, x AS dfdy FROM g` | ✅ full gradient as tidy columns |
| `SELECT grad(grad(power(x,3),x),x) FROM g` | `… (6*power(x,1)) …` | ✅ higher-order (nesting) |
| `SELECT grad(a.v * b.w, a.v) FROM t a JOIN u b …` | `… (b.w) …` | ✅ qualified across joins |
| `SELECT jvp(sin(x),x,dx) FROM g` | `(cos(x)*dx)` | ✅ forward-mode directional derivative |
| `SELECT AVG(grad(loss, theta)) FROM batch` | `AVG( d(loss)/d(theta) )` | ✅ one gradient-descent step (linearity) |
| `SELECT a+b AS s, grad(s*s, s) FROM t` | `…, (s + s)` | ✅ differentiate w.r.t. a computed alias |
| `WITH RECURSIVE n AS (… x-(x*x-2)/grad(x*x-2,x) …) …` | `… /(x+x) …` | ✅ training loop in one query (but see the DataFusion caveat below) |
| `INSERT INTO p SELECT theta-lr*grad(loss,theta) FROM …` | rewritten SELECT | ✅ DML update rule |
| `SELECT grad(sin(x),x) FROM t` in **DuckDB** | needs `SELECT * FROM ddx('…')` — bare works only in native DataFusion | ⚠️ wrapper |

**What it refuses** (a clear error, never a wrong number):

| You write | Result |
| --- | --- |
| `grad(atan2(x,y), x)` | ❌ `NotImplemented` — `atan2` has no rule yet |
| `grad(power(x,x), x)` | ❌ `NotImplemented` — general `u^v` not yet |
| `grad(CASE WHEN x>0 THEN x END, x)` | ❌ `NotImplemented` — conditionals not yet |
| `grad(x > 0, x)` / string / date exprs | ❌ `NotImplemented` — not differentiable, permanently |
| `grad(a.x * b.x, x)` in a self-join | ❌ ambiguous unqualified `wrt`; write `a.x` |
| `grad(x * a.x, a.x)` where bare `x` also binds `a.x` | ❌ bare `x` may be the `wrt` column; qualify it |
| `WITH v AS (SELECT sin(x) AS s …) SELECT grad(s*x, x) FROM v` | ❌ `s` is a computed CTE alias used as a non-`wrt` term; differentiate inside the CTE |
| `grad(x*y, x+y)` | ❌ `wrt` must be a bare column, not an expression |
| `grad(SUM(f), x)` | ❌ rejected by SQL scoping; write `SUM(grad(f,x))` |

The mental model: if every function has a rule and the `wrt` is an
unambiguous column, it works in any query shape; otherwise a typed error at
rewrite time, before the query runs.

**A DataFusion caveat on the recursive-CTE row, found at M2 and worth knowing
before M4 leans on it.** DataFusion 54 mis-plans a recursive CTE whose *outer*
query references only a non-leading subset of the CTE's columns: projection
pushdown prunes the leading column and the recursive term's projection indices
are not remapped, giving `project index N out of bounds`. It is unrelated to
ddx — it reproduces with `cos()` in place of the marker on a context with no ddx
installed — and it is narrower than "recursive CTEs are broken": referencing
every column of the loop state in the outer query (`SELECT *` suffices, and a
`WHERE` on a column counts as referencing it) avoids it entirely. The Newton
fixture in `ddx-datafusion/tests/path_a.rs` runs a two-column loop with `grad`
in the recursive term for exactly this reason. Same coverage-gatekeeper shape as
`[F5]`/`[G9]`/`[S5]`: the bound is the engine's, ddx's job is to know where it
sits and route around it.

**Roadmap:** general `u^v` via `exp(v·ln u)`; `CASE`/`min`/`max` subgradients
with a documented kink convention (mirroring how `abs` pins its kink at `0`
via a portable `CASE`-based sign); `atan2`, `log(base,x)`, `cbrt`,
`expm1`/`log1p`; a dialect name-normalization table; and a clear, permanent
taxonomy of "not differentiable" (comparisons, string/temporal ops, window
functions). v2 has since given window aggregates rules: `SUM` and `AVG` over
any frame, `MAX` and `MIN` over each whole partition (§4.3).

### 3.7 Known limitation: symbolic expression swell

Product/quotient rules duplicate their operands, so an n-factor product
yields an O(n²) derivative, repeated differentiation compounds
multiplicatively, and an N-parameter gradient is N columns each re-deriving
the whole loss. 0/1-folding trims easy zeros but shares no subexpressions —
a term appearing k times is recomputed k times unless the engine's own CSE
catches it. With no reverse-mode accumulation at the scalar layer, an
N-parameter SGD step is N independent full derivations of the loss per row
per iteration — precisely why ML left symbolic differentiation for
reverse-mode AD (Baydin et al. [1]) `[F6]`/`[F10]`/`[G5]`. v1 accepts
this and positions around it (§1) rather than trying to out-engineer it: the
size/latency cliff gets measured, not guessed, by an explicit benchmark
(§8); the eventual remedy for very heavy scalar use is a let-binding pass
factoring shared subexpressions into projected columns. The real fix for
anything past a handful of parameters is v2.

### 3.8 Known limitation: ddx's arithmetic is real, the engine's may not be

`ddx` differentiates the expression as **real-valued arithmetic**. A database
does not always evaluate it that way, and where the two disagree the emitted
derivative is a correct derivative *of a different function* than the one the
engine computes. Found by the property suite
(`ddx-datafusion/tests/simulation.rs`, "a derivative does not depend on the
column storage type"); it affects **both** paths equally, because it is a
property of the model rather than of any bridge.

The sharp case is integer division:

```sql
-- x is BIGINT, value 3
SELECT ln(2 / x)            FROM t   -- -inf   : 2/3 truncates to 0, ln(0) = -inf
SELECT grad(ln(2 / x), x)   FROM t   -- -inf
-- the same column as DOUBLE
SELECT grad(ln(2 / x), x)   FROM t   -- -0.333 : the real-arithmetic answer
```

Under integer semantics `2 / x` is a step function: its true derivative is `0`
almost everywhere and undefined at the steps. `ddx` emits `-2 / (x * x)`, which
describes the real function. `DECIMAL` has the same shape and a milder
magnitude — the derivative is right about the decimal-truncated function, which
is not quite the function the user believes they wrote.

**This is not the same thing as `[F4]`/`[R1b]`,** and the distinction is the
whole point. That policy makes every derivative `ddx` *emits* DOUBLE-typed, so
the derivative's own arithmetic never truncates. It says nothing about the
arithmetic in the **primal the user wrote**, which the engine evaluates under
its own rules before differentiation is ever involved.

It is a silent divergence, which principle 5 exists to prevent, so it is stated
rather than left as folklore. The v1 position: **the primal's arithmetic is the
user's to declare.** Write `2.0 / x`, or cast the column, and the model and the
engine agree. Two candidate hardenings are recorded rather than built, because
each has a real cost:

- *Detect and refuse on Path B.* Post-binding, operand types are known, so a
  `/` with two integral operands inside a marker could be a typed error naming
  the fix. It is unavailable to Path A by construction (differentiation there is
  pre-binding, which is exactly why `[F4]` casts blindly), so building it would
  introduce a new, deliberate divergence between the paths — the thing §3.3 has
  otherwise worked to avoid.
- *Cast the primal.* Rewriting the user's `2 / x` to `2.0 / x` would make the
  engine compute the function `ddx` differentiates. It also silently changes the
  meaning of the user's own query, which is a larger liberty than this project
  should take without asking.

---

## 4. v2: query-level reverse-mode AD

### 4.1 Why this exists

v1's ceiling — an N-parameter gradient costs N independent scalar
derivations, with no sharing across them (§3.7) — is only a property of the
*scalar* surface, not of the underlying idea. The fix is not to abandon the
ML pitch; it's to lift differentiation from scalar expressions to whole
queries, where the sharing scalar mode lacks happens through *materialized
intermediate relations* — a tape — instead of inside expressions.

This is de-risked, not speculative. The prototype's [MNIST-MLP
demo](https://github.com/xqlsystems/xarray-sql/pull/196) (`nn.py`) already
trains a 196→32→10 network — about 160,000 parameters — where every gradient
is computed in SQL, with `grad` appearing only as the elementwise leaf
`grad(tanh(z), z)`. Read correctly, that backward pass *is* reverse-mode AD,
written by hand: it is nothing more than the mechanical application of one
**transpose rule per relational primitive** (§4.3), and all six parameter
gradients it computes match `jax.grad` to machine precision
(`docs/spikes/relational_ad_spike.py`, max error ~1e-18). The one gradient the
demo derived "by hand" — the softmax delta — is the *first* thing the rules
recover mechanically; it was never fundamental.

**The right axis is IR scope, not "symbolic vs. tape."** JAX is also
symbolic — it traces and rewrites a jaxpr; its architecture is JVP rules for
primitives plus transpose rules for the linear ones, with `vjp =
transpose(linearize(f))`. v2 is the same recipe, one scope up: primitives
are relational operators instead of tensor ops, and the tape is materialized
relations instead of a Python list. `ddx-core`'s scalar `grad` is not
superseded by any of this — it becomes the elementwise-primitive rule
(§4.3), unchanged. Everything built for v1 is the foundation v2 stands on.

**Generality past the MLP is confirmed.** `docs/spikes/attention_ad_spike.py`
builds a full single-head attention block (Q/K/V projections → `QKᵀ/√d` →
softmax over the key axis → `A@V`) from the *same* rules and matches
`jax.grad` on every weight and the input to ~1e-16; the causal mask is just
elementwise and also passes. LayerNorm (mean/variance = group-reduce +
elementwise), residual connections (elementwise add), and GELU/ReLU
(elementwise) all reduce to the same primitive set. (ReLU as it is usually
written, `CASE WHEN x > 0 …` or `greatest(x, 0)`, is differentiated by the
branch taken; §4.3.)

**Published precedent.** Tang et al. [2], *Auto-Differentiation of Relational
Computations for Very Large Scale Machine Learning*, do exactly this — a
functional relational algebra with a
gradient operator and per-operator relation-Jacobian products for reverse
mode — and show it performance-competitive at billion-node scale. What
`ddx` adds: they target a bespoke tensor-relational engine, not portable
SQL, and don't factor out a reusable scalar differentiator; `ddx`
contributes the engine-portable, SQL-surface, community-installable form,
with `ddx-core`'s scalar engine as the reusable leaf. On performance, `ddx`
deliberately does not adopt their trick of making relation values chunked
tensors — that would break the portable, one-value-per-cell surface the
whole project rests on. Instead `ddx` keeps the model pure-logical and pushes
BLAS-class speed into the physical plan: a fused-contraction "einsum"
operator (an aggregate-`HashJoin` that computes a grouped contraction
without materializing the full join, dispatching to a matmul kernel on the
dense path) as a `DataFusion` `ExecutionPlan`. Logical portability stays at
the top; engine-specific performance lives underneath, unchanged by it. This
operator is still to be spiked — the one open piece of the performance
story.

### 4.2 The mechanism: Substrait

v2 needs a real relational plan representation in a way v1 never did. v1
differentiates a scalar expression and splices *text* back into the source
query — it never needs a plan IR at all. v2 has to *synthesize new joins and
group-bys* (the backward contraction, the broadcast join) that exist nowhere
in the forward query's text; that is not a text-splice problem. So v2's core
operates directly on `substrait::proto` types — the real, generated Rust
types from the [Substrait](https://substrait.io/) crate — the same way v1's
core operates on real `sqlparser` types rather than a bespoke enum.

**Two things ruled this in, and one thing ruled two alternatives out.**
A bespoke Rust builder API (an early draft of this design) was rejected: it's
a new embedded DSL, not SQL, repeating exactly the mistake this project
already paid down once when it deleted a bespoke expression IR in favor of
`sqlparser`'s real type. DataFusion's own `LogicalPlan` was rejected too, and
not on taste: DuckDB's stable extension surface has zero plan hooks (§3.4),
so an IR keyed to a DataFusion Rust type is DataFusion-only by construction —
it breaks the "engine-independent core" success criterion outright `[S1]`.
Substrait is genuinely engine-neutral: both DataFusion and DuckDB produce and
consume it.

**Custom functions survive the trip.** Substrait's extension mechanism
(`extension_uris` + `simple_extension_declaration`) gives every function a
plan-local anchor, so a function ddx has claimed the name of, like
`ddx_stop_gradient`, is found in the plan wherever it sits.
`docs/spikes/substrait_ad_marker_spike.py` checks this with a custom identity
function (it tried `ddx_contract_mark`, from an earlier draft that tagged every
contraction). The function survives a same-engine round-trip *and* a genuine
cross-engine hop: a plan **produced by DataFusion** is **consumed and executed
correctly by DuckDB**, matching DataFusion's own result exactly. The reverse
direction (DuckDB produces, DataFusion consumes) deserializes cleanly;
execution in that direction wasn't exercised — an honest scope boundary, not a
claim. Neither engine's producer emits a fully spec-conformant extension URI
for a custom function (both use a bare anchor + name), which didn't break
either tested engine but is unverified for a third `[S2]`, and is why ddx
matches functions by name (`S9`).

**A recurring cost of this choice, found once already and worth budgeting
for.** v2 is now bounded by whatever relation vocabulary Substrait itself,
and each engine's producer/consumer, actually implement — the same
coverage-gatekeeper pattern v1 lives with for `sqlparser`'s dialect coverage,
recurring one layer up. §4.3's rank-select found a concrete instance: DuckDB's
own optimizer silently mangles a specific idiom before Substrait export. The
resolution there (spike each rule's forward idiom against both engines
before trusting it, verify a workaround rather than wait for an upstream
fix) is the template for handling this class of risk as more rules get
built `[S4]`/`[S5]`.

**What ddx needs from a Substrait producer, and from a consumer.** A
second-engine spike (DuckDB 1.5.6 producing the plan and running the steps,
alongside Ibis's producer) learned each of these the hard way:

| Producer or consumer behavior | Seen | What ddx does |
|---|---|---|
| Constants folded, or not | DataFusion's optimized plans fold; DuckDB only with its optimizer on; Ibis does not | folds a cast of a literal and literal arithmetic itself, so `power(x, cast(2 AS DOUBLE))` is `power(x, 2)` |
| Shared subtrees | DuckDB: `ReferenceRel` into an entry of `Plan.relations`; DataFusion: the tree written out | follows `ReferenceRel` by copying the subtree it names, and saves each distinct aggregate once either way |
| Function names | DataFusion: bare (`sum`); DuckDB: compound (`sum:fp64`), bare for its own | compares normalized names (`S9`) |
| Extension declarations | DataFusion 54 and 55: no URN (`u32::MAX`); DuckDB: a real `extension_urns` table | an emitted plan carries the input's URNs and declarations; a function ddx adds has no URN, as DataFusion writes it. A consumer that resolves only by URN (Acero) is not yet a target |
| Table names | some producers drop the schema (`s.t` arrives as `t`) | matches a `wrt` table name as `ColumnRef::table` states, refuses a bare name two schemas share, and hints at the bare name when the schema is missing |
| Producer coverage | DuckDB's producer refuses an Arrow-registered scan, which is how xarray-sql registers a Dataset | plans on a connection that mirrors the caller's catalog with schemas only (below) |
| What a consumer's reads see | DuckDB's consumer reads as if on a connection of its own: committed persistent tables and views over them, not registered objects, temp tables, or the caller's open transaction | DuckDB runs ddx's steps as SQL on the caller's connection (below) |
| Consumer coverage | DuckDB's consumer rejects every `window_function` expression | the same: ddx's reduce rules, window transposes and `jvp`'s `MAX` tangent all write windows |
| `emit` on a relation | DataFusion honours it on any relation; DuckDB 1.5.6's consumer ignores it on a join, filter, sort, fetch, cross join or set and returns the leading columns, with no error | writes an emit only on a projection, and narrows anything else with a projection over it; the DataFusion tests check every step for it |

**Substrait in, SQL out, where a consumer falls short (M5, `S22`).**
`docs/spikes/duckdb_substrait_visibility.py` shows DuckDB's consumer cannot
run ddx's programs over xarray-sql's data: its reads see only committed
persistent tables, and it rejects window functions. So on DuckDB-Python ddx
keeps Substrait for reading a query and writes SQL for running one. DuckDB's
own producer plans each query, on a planning connection that mirrors the
caller's catalog with schemas only (an empty table per table or registered
Dataset, and the caller's views), so the plan has DuckDB's own typing,
division and functions. `ddx-ad`'s `to_sql` writes each step and each check
as SQL in the engine's dialect, which the backend runs on the caller's own
connection (`CREATE TEMP TABLE step AS …` for a step, and for a check a query
that must return no rows), where both see what the caller's queries see:
registered data, temp tables and the open transaction. A window is then
ordinary SQL. The `Backend` protocol does not change: its `materialize` and
`returns_rows` receive each plan and run the SQL `to_sql` writes for it.

**One backend for SQL engines (`S23`).** Running needs only what every
Arrow-based engine xarray-sql registers into already has: run SQL, register
an Arrow table (a program's cotangent or tangent), drop a table, report a
schema, and a dialect for `to_sql`. So `ddxdb`'s backend is generic over
those primitives, with DuckDB its first instance and xarray-sql's per-engine
register adapters the natural place for each engine's glue. Planning is the
half that isn't universal: it needs a Substrait producer, which DataFusion
and DuckDB have and Polars and most databases behind ADBC or Flight SQL do
not. How to plan for those is deferred until one is needed (§9). The writer is a correctness surface of its
own; its oracle is every program the DataFusion tests build, run both as
Substrait and as the SQL it writes. This is planned, not built: §8's M5.

A host that holds plans as protobuf bytes, as most of the Substrait ecosystem
does, needs no `substrait` crate of its own: `decode_plan` takes the bytes and
each step and check gives its plan back as bytes. Running a program is
written once, in `ddx-ad`'s `run`; an engine implements `Backend`, four
primitives: whether plans return rows, materialize a plan as a table, drop a
table, and a table's schema in its own producer's terms. Each returns a
future, so an asynchronous engine (DataFusion) implements it directly and a
synchronous one (a Python object) returns futures already done. A step's plan
is `Unbound` until `run` binds its reads to the engine's tables, and the
engine is only ever handed a `Bound` one (`S17`).

`ddx-ad`'s dependencies stay symmetric with v1's: `substrait`, plus `ddx-core`
for the elementwise rule, and no `datafusion` or `duckdb`. `substrait` is pinned
exactly to the version `datafusion-substrait` uses, for the reason §6 gives for
`sqlparser`.

### 4.3 One transpose rule per relational primitive

JAX differentiates a function by giving each primitive a JVP or transpose
rule and composing them. A query is a composition of relational primitives,
and v2 does the same:

| Primitive | SQL | Transpose |
|---|---|---|
| **map** | a projected expression `y = f(x₁, …)` | `x̄ᵢ += ȳ · ∂f/∂xᵢ`, row by row, the partials from `ddx-core`; for `CASE`, `greatest` and `least`, the derivative of the branch the row takes |
| **select** | `WHERE`, a join condition, a semi-join, a filter on a rank | the cotangent stays on the rows that were kept |
| **broadcast** | a join | sum the cotangent back over the rows each input row was copied to |
| **reduce** | grouped `SUM` | broadcast the group's cotangent to every row summed |
| | `AVG` | the same, divided by the group's count |
| | `MAX`, `MIN` | the group's cotangent to the rows attaining it, shared evenly at a tie |
| | `COUNT` | nothing: it does not change when its argument does |
| | a window `SUM` or `AVG` | each row's cotangent (over its frame's count, for `AVG`), summed over the frame mirrored (`k PRECEDING` ↔ `k FOLLOWING`) |
| | a window `MAX` or `MIN` over each whole partition | the grouped rule, from the sum of the partition's cotangents |

**One linear form per reduction.** Every reduction's tangent is linear in its
argument's: `Σ over the group's rows of coef · ṫ`, with `coef` 1 for `SUM`,
`1/count` for `AVG`, and `1/(number attaining)` on the rows attaining a
`MAX` or `MIN` (0 elsewhere). ddx-ad's `reduce.rs` defines that form once, as
JAX's `deflinear` defines a linear primitive once. Forward mode renders it
as a tangent (`SUM` of the counted tangents, or `AVG` when they share it),
and reverse mode as its transpose (the group's cotangent times `coef`), so
the two modes have the same rules by construction. A window function is the
same reduction over each row's frame. Over a whole partition it is the
grouped reduction broadcast back to the rows. Over a running frame, a sum or
mean's transpose sums over the frame seen from the other side. A running
extreme moves with a different row in each frame, and is refused in both
modes. A rank or row number has no derivative. A window's transpose runs over
the rows it read, recomputed, each once: the cotangents of the rows above it
are first summed back to them by the dims of the inputs beneath it, so a later
filter, a join that copies rows, or a group with no cotangent cannot change
its counts, its ties or its frames. Those sums are materialized as a step, and
a window read along several paths is transposed once, from all of their sums:
a stack of windows, each layer reading the last both directly and through a
window, then costs the square of its depth rather than a power of it.

**Map is the seam between v1 and v2.** A projected column is a scalar function
of other columns of the same row, and its local derivatives are one call into
`ddx-core`'s v1 `differentiate` per input column. Nothing new is built here.

**A contraction is not a primitive.** `SUM(a.val * b.val) … GROUP BY` over a
join is broadcast, map and reduce, and composing their transposes gives the
familiar rule, `Ā = Σ_out C̄·B` and `B̄ = Σ_batch A·C̄`, both contractions
themselves. Verified against `nn.py`'s `g2`/`g1`/`g0` and
`relational_ad_spike.py`. The physical fused-contraction operator (§4.1) is
where a contraction becomes a unit again, for speed, not for the math.
Composed, the rules rebuild the join for every backward step and join it to
the cotangent. When a saved `SUM`'s region is a tree of inner joins and its
argument a product whose factors split the joined inputs into two sides,
ddx instead contracts the cotangent with the *other* side and hands the
result to each side's own rules as the cotangent of its factor, so the join
is never rebuilt (`S19`). A side may be several inputs: attention's output,
`SUM(w / z * v)` with the softmax's normalizer `z` joined on a grouping key,
splits into the scores with their normalizer, and the values, so the
division is transposed once per score, as `jax.grad` transposes `p = w / z`
and then `p @ v`. A join condition that reads only grouping keys (a causal
mask, `q.t >= k.t`) selects whole groups, so the transpose applies it to
the cotangent.

**Mean is a reduce rule, not a separate division.** `AVG(x)` is `SUM(x)` over
the group's count, and its transpose is too; nn.py's `-AVG(ln p)` loss
differentiates as written.

**Argmax, two ways.** `MAX`/`MIN` as an aggregate routes the cotangent to the
rows that attain the extreme and shares it evenly at an exact tie, which is
`jax.grad(jnp.max)`'s own convention, so it agrees with JAX everywhere. The
extreme and the count of rows attaining it are windows over the recomputed
rows themselves (`MAX(x) OVER (PARTITION BY` the group's keys`)`), never a
comparison with the saved value: a recomputation need not match the forward
pass to the last bit, and a saved maximum no recomputed row equals would send
no gradient at all (`S12`). For the same reason, where the argument reads the
output of an aggregate that rounds (a saved aggregate, or constant data or a
scalar subquery computed with a sum or average, not a maximum or count), a
row within 8 ulps of the extreme attains it: two groups that tie in exact
arithmetic can round apart differently from run to run. Any other argument,
a table's values included whether or not the table is differentiated,
compares exactly, so MAX(1, 1 + 2 ulps) has gradient
(0, 1), as `jax.grad` gives (`S13`). The
other idiom, nn.py's, ranks and filters:
```sql
WITH ranked AS (
  SELECT {group_dims}, {route_dim}, val,
         ROW_NUMBER() OVER (PARTITION BY {group_dims} ORDER BY val DESC, {route_dim}) AS rk
  FROM {input}
)
SELECT {group_dims}, {route_dim}, val FROM ranked WHERE rk = 1
```
That is select, and needs no rule of its own: only the kept row carries the
cotangent back. The rank itself has no derivative, and gradient reaching it is
refused. At a tie the query has picked one winner, so the whole cotangent goes
to it (the convention `docs/spikes/route_ad_spike.py` pins, which differs from JAX's
split). The ranking is recomputed in the backward pass, so its `ORDER BY`
should break ties deterministically. Both idioms are checked, math against
`jax.grad` away from ties in `docs/spikes/route_ad_spike.py` and the Substrait side
in `docs/spikes/duckdb_substrait_window_bug.py`: a plain window column round-trips
through DuckDB, in the sense that DuckDB produces a plan for it (by rewriting
the window into an aggregate joined back to the rows); DuckDB's consumer
rejects a `window_function` expression, so the other direction does not hold,
and neither do ddx's own windows, one reason DuckDB runs ddx's steps as SQL
(§4.2).
The full top-1-per-group idiom round-trips **silently
wrong** there — `from_substrait` returns every row instead of the top-1 rows —
because DuckDB's optimizer rewrites the idiom into an `arg_max`-join before
export. This reproduces with no ddx function involved; DataFusion round-trips
the identical idiom correctly. A verified two-step workaround (round-trip only
the window-column computation, then apply the `rk = 1` filter as plain
engine-native SQL) produces the correct result on DuckDB `[S3]`/`[S4]`.

**Stop-gradient** is the one thing a query tells ddx. `ddx_stop_gradient(x)`
is `x` at runtime and a constant to differentiation, as `lax.stop_gradient` is
in JAX. It is for when a stop is actually meant. nn.py's softmax subtracts a
per-row max before `exp`; with the `MAX` rule that shift's contribution
cancels exactly, so the loss differentiates correctly without it, and
stopping it just skips work.

Each backward step ddx emits is **plain** Substrait, with no ddx functions in
it, so a program can be differentiated again (§4.9).

### 4.4 `grad`, `vjp` and the tape

```rust
pub fn grad<D: Differentiable + ?Sized>(of: &D, wrt: &[ColumnRef]) -> Result<BackwardProgram, AdError>;
pub fn vjp<D: Differentiable + ?Sized>(of: &D, wrt: &[ColumnRef]) -> Result<BackwardProgram, AdError>;
// D: a query's Plan, or a program of either kind (§4.9)

pub struct ColumnRef { pub table: String, pub column: String }

pub struct BackwardProgram {
    pub inputs: Vec<InputTable>,      // vjp: the output's cotangent; grad: none
    pub checks: Vec<Check>,           // plans that must return no rows, run first
    pub forward_steps: Vec<Step>,     // the saved aggregates, then the value
    pub backward_steps: Vec<Step>,    // cotangents, then one gradient per table
    pub value: OutputTable,           // the query's own result
    pub gradients: Vec<OutputTable>,  // one per wrt table
}
pub struct Step { pub name: String, pub plan: Plan }

// What a program shares with its caller, in every mode (§4.8 too):
pub enum Of { Output, Table(Vec<String>) }   // the query's output, or a wrt table
pub struct InputTable { pub name: String, pub of: Of, pub columns: Vec<String>, pub keys: usize }
pub struct OutputTable { pub step: String, pub of: Of, pub columns: Vec<String>, pub tangents: Vec<Tangent> }
```

Every program has the same anatomy: input tables the caller registers
before it runs, checks, steps, and output tables it leaves. A cotangent and a
tangent are both input tables, shaped like what they are of (its keys, then
a number per value under the value's name); the value, a gradient, a `jvp`'s
output and a Hessian-vector product are all output tables.

As in JAX, `vjp` pulls a cotangent of the output back to the inputs, and
`grad` is `vjp` of a loss seeded with 1. `grad` requires one row and one
column, as `jax.grad` requires a scalar; anything else is an error that points
at `vjp`. `vjp` reads the cotangent from a table the caller supplies, the
program's one input table, keyed like the output.

**Dims and values.** A gradient has the shape of what it is taken with
respect to, and ddx's version of shape is the XQL model (§1). A relation's
columns are **dims**, coordinates that identify a row and are never
differentiated, or **values**, the numbers at a coordinate. A tangent or
cotangent has its primal's dims and values, so a gradient comes back shaped
like its table, as `jax.grad` returns a pytree shaped like its argument. For a
table the query reads, the `wrt` columns are its values and the others its
dims. For a relation the query computes, the plan says: a `GROUP BY` key is a
dim, an aggregate a value, and a join's dims are both sides'. A table the loss
reads but no `wrt` names is constant data.

**Saved and recomputed.** Like any AD system, ddx chooses which forward values
to save and which to recompute (JAX exposes the same choice as `jax.checkpoint`
policies). It saves the output of every aggregate that depends on a `wrt`
column, once, as `{prefix}saved_{n}` (the program's prefix, below): no saved relation is bigger than a layer's
output. The row-local work between two saved aggregates is a **region**, and
is recomputed inside each backward step, never written out, so a contraction's
`N × D × H` join never is. A region is rebuilt with the same relations in the
same order, so it produces the same rows, but with no projection dropping a
column. "The same rows" is an assumption about the engine, not a property of
the plan, and it is enforced where ddx knows it can fail. A window function
recomputed in a region must rank totally (its `PARTITION BY` and `ORDER BY`
include every dim of the rows it ranks), because SQL leaves ties unordered and
a parallel engine may break them differently on each run; so must a `LIMIT`'s
`ORDER BY`, and a ranking on a semi-join's right side, which decides which
rows are kept. An aggregate ddx has a rule for, over each whole partition
(`SUM(x) OVER (PARTITION BY g)`), is exempt, since no row order changes its
value; a recomputed `SUM` or `AVG` may still round differently, so a `MAX`
over a value read from one compares with §4.3's tolerance. A `ROW_NUMBER` or
`FIRST_VALUE` over a whole partition is not exempt: its value is the engine's
row order. A ranking or `LIMIT` inside constant data joined into a region,
where ddx has no dims to check, and a volatile function (`random()`, `now()`)
anywhere a region recomputes, are refused. Each refusal is made only if
gradient reaches the region. §4.6 lists the rest. Every intermediate value is then a column, defined as an expression
over columns to its left, and reverse column order is a reverse topological
order for the chain rule. This index is an implementation detail over the
real Substrait plan, the same relationship v1's `ColRef` has to
`sqlparser::ast::Expr`, not a competing IR.

Because Substrait plans are trees, a CTE read twice arrives as two identical
subtrees; attention's plan holds eighteen aggregates for the eight the query
writes. Identical aggregates are saved once and share one cotangent. They
are found by a structural id per relation, interned bottom-up from its own
fields and its inputs' ids, so finding them is linear in the plan's size
however many copies inlining made (#155): encoding each aggregate's whole
subtree, as ddx first did, re-encoded every aggregate beneath it, and took
most of the build of a GPT's loss (1,050 aggregates, about 25 distinct).

A plan may also say what it shares: Substrait writes a relation read more
than once in `plan.relations`, once, and a `ReferenceRel` wherever it is
read (DuckDB's producer does this for a CTE read twice). Reverse mode lowers
each such relation once and every reference reads its region, so the
program is the one its written-out form gives, byte for byte, at the cost
of the DAG rather than of its copies: forty layers that each read the last
twice, 2^40 copies written out, are differentiated in milliseconds (#19).
A relation read through a reference that reads no `wrt` table is written
out in each step that reads it, since an engine runs the steps and
DataFusion's consumer reads no `ReferenceRel`.

**One saved aggregate's backward step.** Saved aggregates are processed parents
first, starting from the output. For each: join the region beneath it to its
cotangent on the grouping keys (the reduce rule's broadcast); walk the region's
columns right to left applying the map rule, each column's cotangent appended
as a new column rather than inlined; and at each input, sum the cotangent by
the input's dims (the broadcast rule's transpose). An aggregate skips a row
whose argument is NULL, so that row's seed is NULL rather than the group's
cotangent, and nothing in it gets gradient. The result is the input's
contribution: a saved aggregate's cotangent, `{prefix}cotangent_{n}`, or a part
of a table's gradient.

**Fan-in accumulation is real, not hypothetical.** When a relation feeds more
than one consumer — attention's `X` feeding `Wq`, `Wk`, and `Wv` — each
consumer's contribution is summed, verified in `attention_ad_spike.py` (`Xbar =
Xq + Xk + Xv`, matching `jax.grad` to 1e-16). Relationally the sum is a `UNION
ALL` of the contributions, then `GROUP BY` the dims and `SUM`, never a join:
cotangents are sparse, and an inner join of nn.py's three per-layer
contributions to `weight` matches no rows at all, silently returning an empty
gradient.

**Gradients** are one step per table: every row of the table, its dims, and
each `wrt` value's gradient under the column's own name and in its type. Two
conventions are pinned, whatever the loss: `0` where no gradient reached a
row, and NULL where the row's value is NULL, as the scalar rules pin it. A gradient
shaped like its table makes an SGD step a plain join.

**Dims are checked, not assumed.** A gradient keyed by dims is only right if
the dims identify rows, which a plan cannot show. So each program carries a
check per `wrt` table, a plan that must return no rows (the dim tuples that
repeat), and an adapter runs the checks before the steps and refuses the
program if one returns a row. Without it, rows sharing dims got their summed
gradient, and an SGD join then multiplied rows.

A check is a grouped scan of its table, run on every run: on a large `wrt`
table the checks are most of a `jvp`'s run, and the adapter runs them at once.
ddx cannot see whether a table changed between runs, so it never skips one
(`S16`).

**Every table a program materializes is named under a prefix fresh to that
program** (`__ddx_{id}_`), so two programs on one engine never read or
replace each other's tables, and a user's table is never replaced. A run
returns what the program computed as values, as a JAX function does: the
adapter's `run(ctx, &program, &inputs)` takes the inputs as relations, one
per `InputTable` in order, and returns the value and the gradients as
relations that hold their rows, tangents named `{column}_tangent` as in SQL
(`OutputTable::query`); none of the program's tables stays on the engine
(`S11`). The step-by-step run underneath (`ddx_ad::run`, and the adapter's
own engine module) is what `run` and SQL are built on, not the API.

**Each step is a plain Substrait `Plan`,** handed to the engine's own consumer
(`from_substrait`, `datafusion-substrait`) rather than converted to SQL text by
ddx, and materialized under its name before the next step runs. For DuckDB,
for example:
```sql
CREATE TEMP TABLE __ddx_3f2a_cotangent_7 AS SELECT * FROM from_substrait($1)
```
A step that reads an earlier one is emitted **unbound**: its read names the
columns and leaves their types out. ddx does not know them without
re-implementing each engine's typing rules, and by the time the engine runs
the step, the table it reads exists. The adapter fills the types in from it
(`ddx_ad::bind_reads`), so they are the engine's own by construction
(`S7`).

### 4.5 Worked example

nn.py writes its forward pass as a chain of queries, then its backward pass
by hand: `delta2`, then `g2`, `gb2`, `delta1`, and so on, a query per error
and per gradient. With v2 the forward pass is one query, unchanged, and the
backward pass is gone:

```sql
WITH c0 AS (
  SELECT a.sample, w.out, SUM(a.val * w.val) AS z
  FROM (SELECT sample, height * 28 + width AS inp, images AS val FROM pixels) a
  JOIN weight w ON a.inp = w.inp AND w.layer = 0
  GROUP BY a.sample, w.out),
     fwd0 AS (SELECT c0.sample, c0.out, tanh(c0.z + b.val) AS val
              FROM c0 JOIN bias b ON c0.out = b.out AND b.layer = 0),
     ...                                          -- layers 1 and 2 alike
     m AS (SELECT sample, MAX(z) AS m FROM logits GROUP BY sample),
     e AS (SELECT logits.sample, logits.out, exp(logits.z - m.m) AS e
           FROM logits JOIN m ON logits.sample = m.sample),
     s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample)
SELECT -AVG(ln(e.e / s.s)) AS loss
FROM e JOIN s ON e.sample = s.sample JOIN labels y ON y.sample = e.sample
WHERE e.out = y.labels
```

`grad` of that loss with respect to `weight.val` and `bias.val` equals nn.py's
hand-written `g*` and `gb*` to 1e-12 (M4). There is no migration cost: the
query is the one nn.py already runs to report its loss.

### 4.6 What's verified, what's still open

**Verified, machine-exact against `jax.grad`:** the MLP's all six parameter
gradients; attention's four (including the causal mask); Route's math away
from ties. **Verified cross-engine:** a ddx-claimed function surviving the
Substrait round-trip.

**Built (M3):** `grad`, `vjp` and the rules, run on DataFusion, with every
gradient entry checked against a finite difference of the query computed by
DataFusion: a matrix product, nn.py's two-layer network in one query (both
layers' weights in one table), the same network with respect to its input,
nn.py's softmax cross-entropy as nn.py writes it (which also equals
`(softmax - onehot)/N` to 1e-12), single-head attention with respect to all
four inputs, `MAX`/`MIN` pooling and a rank filter, with both tie conventions
pinned. No query carries a label.
**Found and closed, not left as a risk:** the DuckDB Substrait window-idiom
bug (workaround verified, no upstream-fix dependency).
**Built (M4.6):** both modes read one linear form per reduction (§4.3), and
reverse mode differentiates window values (`SUM` and `AVG` over any frame,
`MAX` and `MIN` over each whole partition), checked against finite
differences, against each window's `GROUP BY` form, and against forward mode
on 13 aggregates and 12 windows; a softmax written with windows equals
`softmax − onehot`.

**Genuinely open:**
- Recomputation's other engine assumptions. A region must produce the same
  rows forward and backward. Rankings and `LIMIT`s are checked for a total
  order and volatile functions are refused (§4.4), but an engine whose plain
  scans are not repeatable would break the assumption too, and so can
  floating-point jitter at a near-tie (a filter or ranking on a value that a
  multi-partition `SUM` computed, recomputed a bit differently). The
  alternative to checking is saving: materialize a region instead of
  recomputing it, which the save-or-recompute policy (`S6`) could allow per
  region, and would let ddx accept the rankings over constant data it now
  refuses.
- The physical fused-contraction operator for BLAS-class performance on
  dense data (§4.1) — not yet spiked.
- Reverse over reverse (`vjp` of a `grad` program, §4.9): refused. A
  gradient step joins the table's rows to the summed contributions on the
  table's dims (a `LEFT JOIN`, so a row no gradient reached gets 0) and keeps
  the table's copy of them; `vjp` needs its output to keep every input's
  dims, and cannot yet see that the two copies are equal. Behind that, the
  gradient into the join's NULL-extended side needs a rule. The other orders
  work.
- The DuckDB Substrait extension is community-maintained, not core, as of
  1.5.4 — an ongoing-maintenance signal to watch, separate from the
  correctness bug already found and worked around. ddx will use only its
  producer (§4.2).
- Whether a third engine (beyond DataFusion and DuckDB) would tolerate the
  non-spec-conformant extension-URI form DataFusion emits, and ddx keeps for
  the functions it adds, is untested and only matters if/when a third engine
  is targeted.
- `UNION ALL` in reverse mode is refused (forward mode has its rule), which
  blocks the gradient of a GraphCast-shaped model in xarray-sql's ML demos.
  A `GROUP BY` key computed from a `wrt` value is refused too, even when the
  other keys determine it.
- Windows on an engine whose Substrait consumer rejects them. ddx's rules
  write windows: the `AVG`, `MAX` and `MIN` rules' counts and extremes over
  the recomputed rows (`S12`), the window transposes, and `jvp`'s `MAX`
  tangent. DuckDB will run steps as SQL (§4.2), where windows are ordinary,
  so no portable form is planned. Should an engine that runs only Substrait
  need one, a grouped aggregate joined back on the group keys
  (`IS NOT DISTINCT FROM`, for a NULL key) is the portable form of a
  whole-partition window, at the cost of recomputing the region, and a
  running frame's is a self-join on its range. Since a program is built
  before anything runs it, it would be an `Options` switch the adapter sets
  when it builds the program, covering `jvp`'s windows too.

### 4.7 What a backward program costs

**Against the bound.** Reverse mode's cheap-gradient principle bounds a
gradient's work by a small multiple of the forward pass's, about 2-3x when
contractions dominate: each forward contraction `Y = X·W` has two backward
ones, `X̄ = Ȳ·Wᵀ` and `W̄ = Xᵀ·Ȳ`. ddx's work per primitive, for `R` rows:

| primitive | forward | ddx's backward | optimal | note |
|---|---|---|---|---|
| map, a chain of `ℓ` expressions | `R·ℓ` | `R·ℓ` | `R·ℓ` | plan size `ℓ²`: every intermediate is a column (the tape) |
| contraction `X(N×D)·W(D×H)` as join + `SUM` | `NDH` | `2·NDH` | `2·NDH` | the join is recomputed inside each contribution, never written out (`S6`) |
| grouped `SUM`/`AVG` | `R` | `R` (a broadcast join) | `R` | |
| `MAX`/`MIN` | `R` | `R` (windows over the recomputed rows) | `R` | |
| a relation read `k` times | `k·R` | `k·R` | `k·R` | with `k ≥ 3` reads, the region's cotangents are materialized once |
| rank-select | `R log R` | `R log R` | `R log R` | the ranking is recomputed |
| a window aggregate | `R log R` | `R log R` | `R log R` | the rows beneath it are recomputed, and its cotangents summed back to them once per place that reads it, as a step; a stack of `k` windows costs `k²` in plan size |

So the work is within a constant of the bound wherever contractions
dominate. Measured (`tests/ad_perf.rs`, backward over forward):
single-head attention 2.5x at L = 16 rising to 3.0x at L = 256; nn.py's MLP
1.9-2.4x; residual blocks 4x at one block falling to 1x at six (DataFusion
recomputes the forward pass's reused CTEs, ddx's steps read saved ones). The
memory is the saved aggregates, one per aggregate that depends on a `wrt`
table: a layer's outputs, as a framework's saved activations.

**What is not proportional to the data: a fixed cost per step.** Each step
is consumed, optimized and planned by the engine before it runs: about 3 ms
on DataFusion for a typical step, independent of the rows, more for a long
row-local chain (the plan grows as `ℓ²`). A program has about `2A + T`
steps, for `A` saved aggregates and `T` `wrt` tables. For a transformer of
`D` blocks with the heads as a dim column (so `H` adds rows, not steps), a
block has about 14 aggregates (q, k and v projections, scores, the softmax's
max and sum, the weighted values, the output projection, two layer norms'
means and variances, two feed-forward contractions), so about `28·D` steps:
roughly `84·D` ms of fixed cost per training step.

**Depth.** `ddx-ad`'s `opt` module optimizes every plan and program ddx
writes, without changing what they compute: `prune` drops the columns
nothing reads (above); `fuse` folds a projection into its neighbour when one
of the two only picks columns, nothing is computed twice, no expression
grows and no window function moves (two levels of computed columns keep
their boundary, since DataFusion names a computed column by its expression
and the boundary keeps names short: without it, a tangent through stacked
aggregates grows its names exponentially); and `split` makes every subtree
32 relations deep a step of its own, run first and read by name in its
place, so no step is deeper. Depth matters beyond planning time:
datafusion-python decodes a plan with protobuf's default limit of 100 nested
messages, two per relation, and a consumer recurses once per relation.
Fusion takes a three-layer MLP's `jvp` step from 61 relations to 43;
splitting brings nn.py's, and a `jvp` through forty stacked aggregates,
within 32. A part costs writing it, on deep steps only. On nn.py and a
contraction, programs run 1–9% faster for fusion, with plans 2–4% smaller
and builds about 5% slower. Reverse mode's lowering walks chains of
projections, filters, sorts, limits and aggregates with a loop, so a deep
query does not overflow a 2 MB worker stack.

**How the overhead scales with a model.** In the data sizes (batch `B`,
sequence length `L`, width `d`, heads `H`) the backward-to-forward ratio is a
constant: every backward contraction mirrors a forward one, and the
recomputed row-local work (`O(B·L·d)` per block, `O(B·H·L²)` for the
softmax) is a fraction of the contractions' (`O(B·L·d²)`, `O(B·H·L²·d/H)`).
The attention benchmark shows the ratio flat as `L` grows 16-fold. In depth
`D` the fixed cost grows linearly, with the steps. So on a CPU engine, where
a contraction is a hash join emitting `N·D·H` rows, the data cost dominates
and the overhead is the inherent 2-3x. With an accelerated contraction
operator, the data cost per step falls to milliseconds and the fixed cost
would dominate, unless the engine's plans are kept across training steps: a
program is built once and its plans do not change between steps, so they
need planning once, not on every run.

**What would make it marginal**, in order of payoff for that setting:
caching each step's physical plan across training steps; a fused dense
contraction operator (`§4.1`), which ddx's backward contractions use as the
forward ones do; a contraction path for multi-way products (Blacher et al.,
*Efficient and Portable Einstein Summation in SQL*: engines do not push a
`SUM` below joins, so a product of three or more factors materializes the
full join); forward-mode tangents within a long row-local region (plans
linear in `ℓ`); and saving a subtree read many times once rather than
recomputing it.

---

### 4.8 `jvp`: forward mode

```rust
pub fn jvp<D: Differentiable + ?Sized>(of: &D, wrt: &[ColumnRef]) -> Result<ForwardProgram, AdError>;
// D: a query's Plan, or a program of either kind (§4.9)

pub struct ForwardProgram {
    pub inputs: Vec<InputTable>,      // a program's own inputs, then one tangent per wrt table
    pub checks: Vec<Check>,
    pub steps: Vec<Step>,
    pub value: OutputTable,           // the output, with its tangents
    pub gradients: Vec<OutputTable>,  // jvp of a grad program: each gradient with H·v
}
```

As in JAX, `jvp` pushes a tangent of the `wrt` columns forward to the
query's output, `J·t`. A tangent is shaped like its primal: for each `wrt`
table, a relation with the table's dims and a tangent per `wrt` column,
one input table of the program per `wrt` table, which the caller registers
before it runs. A row the tangent table lacks has tangent
0. The program's output holds the query's own columns and, after them, the
tangent of each column that depends on a `wrt` column, as `jax.jvp` returns
both.

Forward mode has no transposes and no tape. Tangents travel beside values
through the same operators, so a query's `jvp` is one plan: the query
rewritten so each relation carries, after its own columns, a tangent for
each of them that depends on `wrt` (its **dual**). A relation the plan
shares through a `ReferenceRel` (§4.4) is rewritten once, as a step of its
own that every reader reads with its tangents, the way a program's steps
read each other: it is computed once, not at every read, and an engine,
which reads no `ReferenceRel`, can run every step. Nothing is recomputed and
nothing saved, so the rules need none of §4.4's machinery for keeping
recomputations repeatable.

| Primitive | Tangent rule |
|---|---|
| map `y = f(x₁, …)` | `ẏ = Σ ∂f/∂xᵢ · ẋᵢ`, the partials from `ddx-core`, as the map transpose uses them |
| select (filter, join condition, semi-join, `LIMIT`, sort) | the kept rows keep their tangents |
| join, cross join | each side's tangents come along with its values; an outer join's NULL-extended side has NULL tangents |
| `SUM`, `AVG` (aggregate or window) | the same function over the tangents |
| `MAX`, `MIN` (aggregate, or window over a whole partition) | the mean tangent of the rows attaining it, found by a window over the group (`jax.jvp` of `jnp.max`) |
| `COUNT` | none |
| `UNION ALL` | each input's tangents, 0 where an input has none |
| `ddx_stop_gradient(x)` | none |
| a `wrt` table | its rows joined to the tangent table on its dims |

The reduce rows are each the reduction's linear form, `Σ coef · ṫ` over its
group, rendered as a tangent.

A column that depends on `wrt` only through conditions (a `CASE` test, a
comparison, a `COUNT`) has tangent 0, as `jax.jvp` gives and as `grad` gives
it gradient 0.

A `MAX`'s attaining rows are the ones equal to a window extreme computed
over the same rows in the same query, so they are compared exactly, with no
tolerance: values and tangents are computed together, once.

**NULL.** A tangent is NULL where its value is, and a NULL tangent beside a
value that is not NULL means zero (the value does not move with the NULL it
was computed past, as in `coalesce(x, 0)`). Aggregates and outputs read
tangents through that rule, so a `SUM` or `AVG` of tangents skips exactly
the rows the value's aggregate skips.

**Refusals** follow reverse mode's: a column that depends on `wrt` but has
no tangent ddx can give (a rank, a `GROUP BY` key computed from a `wrt`
value, an aggregate with no rule, a running `MAX` over an ordered frame)
carries its refusal, raised only if something reads its tangent.

**Forward over reverse.** `jvp` of a program rewrites its steps in
order, each as `jvp` rewrites a query, reading the duals of the steps before
it. A step's read of an earlier step names the columns it takes, since a
program's steps are pruned to what they read (§4.7), so a dual's columns are
matched by name. Of a `grad` program along `v`, the tangent of each gradient
is the Hessian-vector product `H·v`, shaped like the gradient; the value's
tangent is the loss's directional derivative. There is no `hvp`: it is
this composition, as in JAX it is `jax.jvp(jax.grad(f), …)`, and it also
gives mixed second derivatives (`grad` with respect to one table, a
tangent through another) and one `grad` program reused across directions.

**Verified** (`crates/ddx-datafusion/tests/jvp.rs`): each rule, row by row,
against a central finite difference of the same query along the same
direction; `jvp` of a loss against `⟨∇L, t⟩` from `grad`, and of a relation
against `vjp` by the dot-product test `⟨J t, c⟩ = ⟨t, Jᵀ c⟩`, to rounding;
and `H·v` against a finite difference of `grad`'s gradient.

**In SQL.** `grad`, `vjp` and `jvp` are each a column of a `SELECT`, with
the rows its derivative has, as JAX's results have the shapes of what they
differentiate. Each is rewritten before the engine sees the statement
(§4.4):

| call | in a `SELECT` over | is, row by row |
|---|---|---|
| `grad(f, c)` | the table whose column `c` is | the gradient at that row's `c` of the number the CTE `f` computes |
| `vjp(f, c, cotangent)` | the table whose column `c` is | the same for any `f`, pulled back along `cotangent` (a CTE or table shaped like `f`'s output: its keys, then its values) |
| `jvp(e, t.c, …, tangent)` | any rows: `e` is an expression of the `SELECT` | the tangent of `e` along `tangent` (shaped like `t`: its dims, then each column's tangent) |

A gradient has its table's rows, so `grad` and `vjp` sit in a `SELECT` over
the table, anywhere an expression can (its select list, `WHERE`, `GROUP BY`,
`HAVING`, a join's `ON`, or the `ORDER BY` of the query whose `SELECT` it
is; a call written twice reads one gradient column, so a `SELECT` and its
`GROUP BY` still agree), and an SGD step is one expression,
`SELECT i, val - lr * grad(loss, val) AS val FROM w`, where the relational
form needed a join on every dim. A call that is a column of its own, with no
alias, is named `{c}_grad`. A tangent has `e`'s rows, so `jvp` is a
column of its own of the `SELECT` computing `e` (the statement's or a
CTE's), named by its alias or, for a column `e`, `{e}_tangent`. One
`SELECT`'s `jvp` columns share a tangent; tangents along two tables are two
`SELECT`s, and add:

```sql
WITH h AS (SELECT x.n, w.o, tanh(SUM(x.v * w.val) + MAX(b.val)) AS y
           FROM x JOIN w ON x.i = w.i JOIN b ON w.o = b.o GROUP BY x.n, w.o),
     dw AS (SELECT i, o, 0.01 AS val FROM w),
     j AS (SELECT n, o, y, jvp(y, w.val, dw) AS t FROM h),
     k AS (SELECT n, o, jvp(y, b.val, db) AS t FROM h)          -- db: a table
SELECT j.n, j.o, j.y, j.t + k.t AS dy FROM j JOIN k ON j.n = k.n AND j.o = k.o
```

A `jvp`'s tangent may be a number, the same at every row:
`SELECT x, jvp(sin(x), t.x, 1) AS dx FROM t` is `cos(x)`, the derivative of
an expression row by row, ddx writing the tangent table itself (the table's
dims, the number under the column's name).

`ddx_ad::rewrite(&mut backend, &programs, statements, dialect)` is the whole
of it, on any `Backend` that can also plan SQL (`plan_sql`). It finds the calls of
several statements, each a `Derivative`: `Grad`, `Vjp` with its cotangent,
or `Jvp` with its tangent, a query or a number. It plans one program per
distinct derivative: per objective for `grad`, per objective and cotangent
for `vjp` (so statements updating different tables pay for one backward
pass), per `SELECT`, columns and tangent for `jvp`, whose `f` is the
`SELECT` with each `jvp(e, …)` written `e`. For each in turn it plans the
query on the engine, builds the program, registers its cotangent or tangent
as the program's input table and runs it. A query whose CTEs read one
another more than once is planned in pieces: an engine inlines a CTE at
every read, so a model whose layers each read the last twice would reach
the engine's optimizer and ddx at a size exponential in its depth (a GPT's
loss held 1,050 copies of about 25 aggregates; GraphCast's plan grew 2.7×
per round of message passing, of 16). Each CTE the engine would copy is
planned alone, in order, reading the ones before it as placeholders (empty
tables of their columns, made from the engine's plan of the CTE with
`LIMIT 0`), and the plans are stitched into one in which each such CTE is a
relation read through `ReferenceRel`s, which reverse and forward mode read
once (§4.4). A CTE whose split could change what a read means is left to
the engine: one a nested `WITH` defines again, one read inside an
expression's subquery (where ddx refuses a reference), one read under
another spelling, and every CTE of a `WITH RECURSIVE`. If planning the
pieces fails, the query is planned whole. A model is often a view, whose CTEs the
objective's text does not show, so first each view the objective reads,
whose SQL the engine keeps (`Backend::view_sql`: DataFusion's
`ViewTable::definition`, or its `information_schema.views` from Python), is
written into the objective as CTEs: its own CTEs under fresh names, then
its query, all before the objective's own, so that a view's reads, which
the engine resolves in its catalog, cannot reach one of them. A view with
no SQL (one registered from a DataFrame), under a name a CTE also has, or
whose SQL does not parse is left to the engine. So is a view whose SQL no
longer means what the engine runs: DataFusion keeps the plan a view was
made with, so after a view it reads is redefined, its SQL reads the new
one and the engine the old. Its backend gives the SQL only if the view's
query, read in its place (`ddx_ad::view_probe`), plans as the view's kept
plan. What the split costs is the
optimizations across the pieces: a reader's filter is not pushed into a
shared CTE, nor a column only some readers use pruned from it, as when
DuckDB computes a CTE read twice once; ddx prunes the steps it writes, and
a filter on a gradient's dims still restricts it, being planned against
the table. Then it rewrites each statement's
tree to read what the calls need, and writes it out as SQL again (`S18`),
returning it with the tables it reads, for the caller to drop once it has
planned it. A program built by an earlier call and kept in `programs` is
run again instead of built, so a training loop that rewrites the same
statements each step builds its programs once (`S20`). A table `grad` or
`vjp` reads becomes a relation over its rows, under its alias, that also
holds each call's gradient, left-joined on the table's dims; each call
reads its own. A `SELECT` with `jvp` columns becomes a read of its
program's value, each tangent in its call's place; its `LIMIT` stays, and
so does its `ORDER BY`, which may name only the `SELECT`'s output, by name
or position, since the tables it read are gone.

When a statement's `WHERE` compares a column of a table `grad` reads with a
literal (`layer = 1`), the program computes only those rows' gradients:
the comparison rejects NULL, so no other row reaches the result whatever
the joins. A comparison of a column the program differentiates, because
another call on the table names it, is not pushed. And the engine plans the
restriction itself, so its optimizer may give a plan the program cannot
take (a filter folded away, or moved into the scan); then the program
computes every row, which is always right.

A query or seed that reads, through the CTEs it reads, a call of itself
(`loss` reading a CTE that holds `grad(loss, …)`) is refused: no order runs
either first.

**Calls nest**, as JAX's transformations do. A call's `f`, `e`'s `SELECT`,
or its cotangent or tangent may read another call, whose job comes first:

```sql
WITH loss AS (SELECT SUM(val * val * val) AS l FROM w),
     g AS (SELECT i, grad(loss, val) AS val FROM w),
     v AS (SELECT i, 1.0 AS val FROM w)
SELECT i, jvp(val, w.val, v) AS hv FROM g                      -- H·v
```

The outer job's query is written over the inner programs' tables, and the
engine plans it as any query. Its program is built from that plan *after*
the inner programs (`compose::After`): their steps, renamed into one
namespace, then the plan as the step that writes the value. The engine's
reads of their tables become reads of those steps, and every plan is
declared from one set of functions, since the engine numbers them its own
way. `grad`, `vjp` and `jvp` of that program then differentiate through the
inner ones as of any program (§4.9): forward over reverse, reverse over
forward, forward over forward. A cotangent or tangent that reads a call is
only computed, never differentiated. Every job also runs on its own, so a
nested call costs its inner program once more than a composed program from
Rust does. Reverse over reverse is refused as it is elsewhere.

*Verified* (`tests/sql_grad.rs`, `tests/sql_jvp.rs`, `tests/sql_vjp.rs`,
`tests/sql_nesting.rs`): against the closed forms, against the programs row
by row, two `jvp` `SELECT`s against one program along both tangents, `vjp` along 1 against `grad`, a
`vjp`'s `WHERE` restricting its rows, a filter on another call's column and
one the engine plans away, a quoted table's name kept as written, `H·v` in
SQL against `jvp` of the
`grad` program, `grad` of a `jvp` and a query over a call, a tangent that is
another call, and nothing left behind.

### 4.9 Composition

`grad`, `vjp` and `jvp` each take a query's plan or a program of either
kind (`Differentiable`, sealed), so they compose as JAX's do:

| | is | gives |
|---|---|---|
| `jvp(&grad(&q, w)?, w)` | forward over reverse | `H·v` beside each gradient |
| `vjp(&jvp(&q, w)?, w)`, cotangent `(0, 1)` | reverse over forward | `H·v` as the gradient |
| `jvp(&jvp(&q, w)?, w)` | forward over forward | `uᵀ H v`, the tangent of the tangent |
| `vjp(&grad(&q, w)?, w)` | reverse over reverse | refused for now (§4.6) |

A program's **output** is its gradients if it has any, and otherwise its
value: what the function it computes returns.

- **`jvp` of a program** rewrites its steps in order (§4.8), so every
  intermediate table carries its tangent, and so does every output.
- **`grad` and `vjp` of a program** differentiate its one output table (a
  program with several gradients is refused: build it with respect to one
  table). Reverse mode needs the whole computation in one plan, to save what
  the backward pass reads and to send each cotangent back the way it came, so
  the output step's plan is written out with each read of an earlier step a
  read of that step's plan. Each step is written once, among the plan's
  relations, and read through a `ReferenceRel` wherever a step reads it
  (§4.4), so the plan is the program's size however often its steps read one
  another. A step the output does not reach is not read (another gradient's
  steps), and the program the plan gives is pruned, as every program is
  (§4.7).

Either way the new program reads the old one's input tables (its tangent,
its cotangent) as constants, which the caller registers as before: they come
first in the new program's `inputs`, and the old program's checks first in
its `checks`. Every plan of a program declares its functions from one set of
declarations that only grows, so a function anchor means the same function
in every step, and steps can be read together. A `jvp` program's tangent of
a column named `__ddx_tangent_{c}` is `__ddx_tangent_{c}_{n}` when that name
is taken; `OutputTable::tangents` says where each tangent is.

Reverse over forward runs within 0.96–1.13× of forward over reverse
(nn.py's loss and a contraction, at two sizes each), with plans up to 1.5×
larger. Composing needed two changes elsewhere: reverse mode no longer
refuses an exact aggregate over each whole partition (`MAX(x) OVER
(PARTITION BY g)`, which `jvp` writes for a `MAX`'s tangent) as a ranking
whose ties could break differently, since no order changes its value; and
reverse mode lowers a chain of single-input relations with a loop, since a
`jvp` step is about twice as deep as its query.

**Verified** (`crates/ddx-datafusion/tests/compose.rs`): reverse over forward
against forward over reverse, row by row, and forward over forward against
`uᵀ (H v)`, on an elementwise loss, a layer of three tables, and a `MAX` and
a mean; refusals for reverse over reverse, for a program with two outputs,
and for `grad` of a value with its tangent; and `vjp` of a `jvp` program on
a 2 MB stack.

## 5. Testing & verification

Differentiation is a numerical-correctness feature; the test strategy is
layered:

- **Unit (rule) tests in `ddx-core`** — port the prototype's 15 tests; every
  rule pinned symbolically.
- **Round-trip property tests, semantic, not just "parseable."**
  `construct → Display → reparse` must equal the constructed AST *modulo
  `Nested`* (normalize parentheses on both sides before comparing) — a test
  that only checks the output parses sails right past the precedence bug
  §3.2 found (`(a+b)*c` reparses fine, just wrong). Fuzz small random trees
  per dialect.

  The invariant is *exact*, and holding it to that standard is what makes it
  worth having. It was briefly weakened to "modulo `Nested` and associativity,"
  on the reasoning that a right-associated product reprinting left-associated is
  harmless because multiplication associates. That reasoning does not survive
  the operator sharing a precedence level with division: `a * (b / c)`
  reprinting as `a * b / c` reparses as `(a * b) / c`, which agrees in exact
  arithmetic and diverges without bound as `c` approaches zero. The continuous
  fuzz found it as a value change of twenty-four orders of magnitude. The fix
  belonged in the *renderer* — parenthesize a right operand that binds as
  tightly as its parent — not in a laxer comparison, and with it the exact
  invariant holds.
- **Numeric agreement against JAX** — the natural oracle, since the whole
  design mirrors JAX's forward/reverse structure for the same seed/cotangent
  semantics. Keep finite-difference as a cheap independent cross-check where
  a JAX equivalent is awkward.
- **Cross-engine equivalence** — the same expression, rewritten per-dialect,
  must evaluate to numerically equal columns in DuckDB and DataFusion.
- **Convention-pinning tests, not blind oracle comparison,** at every point
  where a convention genuinely differs rather than one side being wrong:
  - *Kinks* — `abs` at 0 gives `0`, pinned by emitting a portable
    `CASE WHEN u > 0 THEN 1 WHEN u < 0 THEN -1 ELSE 0 END` rather than an
    engine `signum`/`sign` builtin (DuckDB has only `sign`, DataFusion only
    `signum`, and `signum(0) = 1` — so a bare builtin would be both
    non-portable and *not* pin `0` at the kink). JAX's own convention at the
    kink differs (verify the exact value). Pin the convention explicitly; the
    same treatment Route's tie-break needs (§4.3).
  - *Domain-widening* — a derivative can fail where the primal doesn't
    (`sqrt(x)` is fine at 0; `1/(2*sqrt(x))` divides by zero), and engines
    disagree on the result (`inf` vs. `NULL` vs. error). Cross-engine
    equivalence needs a stated domain-edge policy: sample away from edges,
    or pin per-engine expected behavior.
  - *NULL/folding* — confirm folded and unfolded derivatives agree
    everywhere except the documented NULL-row cases (§3.2).
- **Real-integration acceptance** — end-to-end gradient descent and a
  recursive-CTE training loop converging to closed-form solutions, inside
  xarray-sql and duckdb-zarr.
- **v2: generated queries against a finite difference of the query.** The
  loss is a query the engine can run with no ddx involved, so the independent
  oracle for a gradient is the query itself, perturbed:
  `⟨∇L, d⟩ ≈ (L(θ + hd) − L(θ − hd))/2h` along random directions and single
  entries, screened at kinks (the second difference must shrink like `h²`) and
  Richardson-extrapolated. `ddx-datafusion/tests/ad_simulation.rs` generates
  loss queries from the §4.3 primitives and adds metamorphic relations that
  need no oracle: `∇(cL) = c∇L`, `∇(L·sg(L)) = L∇L`, `grad` = `vjp` seeded
  with 1, `vjp(R, c)` = `grad(Σ R·c)`, and invariance to CTE inlining, the
  unoptimized plan, row order, partition count, dim storage type, the `wrt`
  list, reuse of a program on new values, and concurrent programs. It also
  checks that `grad(loss, col)` in SQL matches the program. The plan ddx
  reads is varied too, because §4.2's coverage risk is about plan shapes. The
  query is re-planned with optimizer rules removed or reordered, and the
  Substrait plan is rewritten into equivalent shapes another producer might
  emit (identity and permuting projections, `WHERE true`, swapped join sides,
  an added sort, a computed-then-dropped column); DataFusion confirms each
  rewrite computes the same loss before ddx sees it. The data varies as well:
  exact ties, where only directions that keep the ties are checked (at a tie
  the loss need not be differentiable, and any subgradient is a convention);
  dense NULLs and NULL keys; huge, tiny, `-0.0`, NaN and infinite values; and
  thousand-row tables in several partitions. Near a kink, ⟨∇L, d⟩ must lie
  between the one-sided derivatives. `grad(loss, col)` in SQL is held to
  the program's gradient under schema-qualified and quoted names with a decoy
  of the old name, a shadowing CTE (refused: a `SELECT` over it reads the
  CTE's rows), the call in a subquery or a later CTE, and
  two losses in one statement; three SGD steps written in SQL must each be
  θ − lr·∇L and lower the loss. It runs nightly beside v1's soak, and a
  weekly mutation test seeds bugs into each rule to measure how fast the soak
  catches them.
- **v2-specific: spike each rule's forward idiom against both engines'
  actual Substrait implementations before trusting it** — the coverage
  discipline §4.2 commits to, now a standing test-plan item, not a one-time
  check.

---

## 6. Architecture: monorepo layout & dependency policy

```
ddx/                               (repo; crates published under the ddx-* names)
├── crates/
│   ├── ddx-core/                   # the scalar rules — differentiate sqlparser::ast::Expr
│   │                               #   dep: sqlparser only
│   ├── ddx-ad/                     # query-level AD over substrait::proto
│   │                               #   deps: substrait, ddx-core
│   └── ddx-datafusion/             # DataFusion adapter: ad
│                                   #   deps: ddx-ad, datafusion
├── python/
│   └── ddxdb/                      # PyO3/maturin wheel: ad on DataFusion; DuckDB-Python next (M5)
├── docs/
│   ├── design.md                   # this file
│   └── spikes/                     # runnable evidence for every load-bearing claim (README.md indexes them)
└── tests/                          # cross-engine numeric-agreement suites (vs JAX)
```

A DuckDB community extension and a Postgres integration, once laid out here as
`ddx-duckdb` and `future/`, are not in the repo; §8 says when they come.

`ddx-core` and `ddx-ad` each publish independently, with a single
minimal dependency (`sqlparser`, `substrait`) and no engine crate, so either
can be driven from a new engine without pulling in DataFusion or DuckDB. The
heavy per-engine dependencies are quarantined in the adapter crates.

**`sqlparser` version policy — a real cost of "one IR," paid once.**
`ddx-core`'s public API takes and returns `sqlparser::ast::Expr`. (Until v1's
SQL surface was removed, `S15`, the Path B bridge (§3.3) required `ddx-core` and DataFusion to resolve the
identical `sqlparser` version — a mismatch makes them two unrelated Rust
types and the bridge won't compile. `sqlparser` ships roughly one breaking
release every 1–3 months, and DataFusion adopts each with a lag, while the
DuckDB dialect coverage argument (§3.3) wants the newest release — the two
pulls are in tension. Policy: pin to DataFusion's requirement (the bridge is
a v1 deliverable and a broken bridge is a compile failure; the DuckDB-coverage
cost is bounded, §3.3); re-export it (`pub use sqlparser`) so downstream
consumers can't accidentally link a mismatch; treat a `sqlparser` bump as a
breaking release of `ddx-core`; and if the pins ever must diverge, degrade
the bridge to a string round-trip rather than break it `[G2]`.) The pin now
only keeps the build to one `sqlparser`.

**`datafusion` is pinned exactly too, always** (`=54.1.0`, with
`datafusion-substrait`; `==54.0.0` for the Python package, the PyPI release of the
same line). `ddx-datafusion`'s public API is DataFusion's types, a user's
`SessionContext` goes in and a `DataFrame` comes out, and v2 reads and writes
Substrait plans of the version DataFusion's producer writes. A caller therefore
links the DataFusion ddx was tested against; a range would promise a combination
no test ran. A bump moves the workspace, both Python manifests and their locks
together.

---

## 7. Naming & distribution

The project is named `ddx`, not `autograd`: "autograd" connotes a runtime,
tape-based system (PyTorch/HIPS), and this is symbolic differentiation as a
plan-time rewrite — literally `d/dx` of an expression — so the name sets the
right expectation instead of sending users looking for a tape and kernels.
It fits the XQL family (`xql.systems`, `xarray-sql`, `duckdb-zarr`), and the
thesis is in the name: "ML models as differentiable databases" → `d/dx` of a
table. Practically: `autograd` is taken on PyPI; the bare `ddx` crate on
crates.io is a dead project, so there's no umbrella crate (none needed);
`ddx-core`, `ddx-ad`, `ddx-datafusion`, `ddx-duckdb`, and `ddxdb` are all
free on crates.io, `ddxdb` is free on PyPI, and `ddx` is free on the DuckDB
community registry.

**Distribution:** Rust crates on crates.io as above. Python: `pip install
ddxdb` standalone, and `pip install "xarray-sql[ddx]"` as a coordinated
optional extra. DuckDB: from Python, `ddxdb` on a DuckDB connection (M5); a dedicated
extension, `INSTALL ddx FROM community`, later (§8). Repo: renamed
`substrait-autograd` → `ddx` on GitHub (old name redirects); tagline
"SQL-portable autograd," not "Substrait" — v1 doesn't use it, and v2's use is
an implementation detail, not the pitch.

---

## 8. Milestones

The plan builds the scalar core first, then puts the true-AD track
immediately after it — before broadening to a second engine — because
proving the ML headline on one engine de-risks the goal; a second engine is
breadth, not de-risking.

- **M0 — Extract the core.** Workspace setup; lift the prototype's
  `src/autograd.rs` into `ddx-core`, re-pointed onto `sqlparser::ast::Expr`;
  implement `rewrite_sql`; port the 15 rule tests. Also lands, before
  publish: the `Ddx` object API; per-dialect identifier folding + case
  tests; the ambiguity guard; the numeric-type policy + integer-column
  tests; precedence-safe construction + the semantic round-trip test;
  span→byte splicing + a multibyte/multi-marker test; pin and re-export
  `sqlparser`. *Exit:* `ddx-core` reproduces every prototype rule, rewrites
  SQL end-to-end, passes all of the above, depends only on `sqlparser`.
- **M1 — Confirm the DataFusion-Python constraint.** Verify
  `datafusion-python` still can't inject an `AnalyzerRule` (keeps v1 on the
  rewrite path). *Exit:* v1 path confirmed; any seam noted as future-only.
- **M2 — DataFusion, Python and native.** `ddxdb` wheel (`rewrite_sql` +
  `Context.sql()` shim), re-integrated into xarray-sql in place of its
  vendored `autograd.rs`. `ddx-datafusion`: marker UDFs + the `AnalyzerRule`
  bridge, plus the `ddx_sql` helper — mind the `TypeCoercion` ordering and
  the `create_logical_expr` seam. Needs a minimal JAX-oracle numeric-agreement
  harness pulled forward from M6, since the exit gate depends on it. *Exit:*
  xarray-sql green on `ddx-core` (vs. JAX, no regressions), and bare `grad()`
  runs end-to-end through the `AnalyzerRule` in a native DataFusion test.
- **M3 — Relational reverse-mode AD, phase 1: the rules.** (Planned as "the
  rules + Substrait markers": four extension-function markers,
  `ddx_contract_mark`, `ddx_reduce_mark`, `ddx_route_mark` and
  `ddx_stop_gradient`, and five transpose rules. Built instead as one rule per
  relational primitive, with only `ddx_stop_gradient` kept; see `S10`.)
  **Built:** `ddx-ad`'s `grad`/`vjp` and the rules, checked on DataFusion
  against finite differences for the MLP, attention and max-pool fixtures
  (§4.6). Clean up
  `nn.py` into the canonical relational-backprop example and regression
  fixture; `docs/spikes/` are the acceptance tests. *Exit:* the rules reproduce
  `jax.grad` on the MLP, attention, and Route fixtures (already machine-exact
  by hand), and a ddx-claimed function round-trips through both engines'
  Substrait implementations (already verified).
- **M4 — `grad` over queries, phase 2: the ML headline.** `grad(plan, wrt)`
  and `vjp(plan, wrt)` take a Substrait `Plan` and emit the backward
  program — a sequence of named, materializable `Plan`s (the tape) — exposed
  in the engines' own terms, down to `grad(loss, column)` in SQL. Stays
  pure-logical; performance is a separate, physical concern (the
  fused-contraction operator, still to spike). Runs first on DataFusion.
  *Exit:* train the `nn.py` MLP with gradients *emitted by* `ddx`'s `grad`, not
  hand-written, matching the demo and JAX. **Built:** `grad(loss,
  column)` in SQL, from Rust (`ddx_datafusion::ad::sql`) and Python
  (`ddxdb.Context.sql`, `ddxdb.ad`), over `grad`/`vjp` programs run on
  DataFusion. `ddx-datafusion/examples/nn` trains nn.py's MLP with one SQL
  statement per parameter table; its gradients equal nn.py's hand-written
  backward queries to 1e-12, and the spikes' MLP, attention and max-pool
  gradients, taken in SQL, equal `jax.grad` to 1e-12 (`tests/test_query_jax.py`),
  as do the gradients of nn.py's own network and SQL. Not built: a loss
  defined in a `WITH RECURSIVE` clause is refused by the SQL surface, so a
  whole training loop is not yet one recursive statement (§5). That refusal
  is ddx's, independent of DataFusion 54's recursive-CTE bug (§3.6); lifting it
  means differentiating a loss inside the recursive term. Query-level `jvp` is
  the next milestone, M4.5.
- **M4.5 — `jvp` over queries: the forward-mode half.** Completes the
  SQL surface as `grad`, `vjp` and `jvp`. Forward mode needs no transposes and
  no tape: tangents travel beside values through the same operators (map via
  `ddx-core`'s scalar `jvp`; sum, mean and join linear in the tangent; `MAX`
  and `MIN` taking the attaining row's tangent, averaged over ties), so a `jvp`
  is one rewritten query rather than a program. Surfaces as `jvp(e,
  table.column, tangent)`, a column of a `SELECT`, from Rust and Python. *Exit:*
  matches `jax.jvp` on the spikes' fixtures, passes the dot-product test
  ⟨J t, c⟩ = ⟨t, Jᵀ c⟩ against `vjp`, and gives a Hessian-vector product on
  the MLP (forward-over-reverse) matching `jax.jvp(jax.grad(f))`. **Built,
  released as ddxdb 0.3.0:** `ddx_ad::jvp` of a query and of a program (forward over reverse) and their
  DataFusion adapter (§4.8), checked against finite differences, `grad` and
  `vjp`; `grad`, `vjp` and `jvp` composed with each other (§4.9); `vjp(…)`
  and `jvp(…)` in SQL, nesting (§4.8); the Python API (`ddxdb` 0.3: `jvp`,
  `jvp_plan`, `ForwardProgram`, composition, `vjp(…)` and `jvp(…)` in SQL),
  checked against `jax.jvp` and, both ways, `jax.jvp(jax.grad(f))` on the
  MLP fixture; `run` returning values. The soak checks `jvp` on its random
  queries against reverse mode, exactly (`tests/ad_simulation/jvp.rs`:
  ⟨∇L, t⟩, linearity, a missing tangent row as 0, the dot-product test
  against `vjp`, `H·t`'s symmetry and its value, a repeated tangent key
  refused); 332 cases and 3,350 comparisons in a five-minute run, none
  failing.
- **M4.6 — One linear form per reduction; windows in reverse mode.** Both
  modes read each reduction's linear form from one table (§4.3, `S21`), and
  reverse mode differentiates window values, which a softmax written with
  windows needs. **Built,** released as ddx-ad 0.2.1 and ddxdb 0.3.1.
- **M5 — DuckDB-Python.** `grad`, `vjp` and `jvp` on a DuckDB-Python
  connection, which is how xarray-sql reaches DuckDB (§4.2, `S22`): `ddx-ad`
  writes a program's steps as SQL (`to_sql`), checked by running every
  DataFusion test program both ways; `ddxdb` gets a DuckDB backend that plans
  with DuckDB's producer on a schema-only mirror of the caller's catalog and
  runs each step and check on the caller's own connection, built generic
  over SQL engines with DuckDB its first instance (`S23`); and what DuckDB's producer
  writes (decimal literals, its optimizer's rewrites, compound function
  names) is read correctly or refused. The two engines are checked against
  each other and against JAX. *Exit:* xarray-sql's ML demos differentiate on
  their DuckDB host, checked against JAX as on DataFusion. (The first plan, a
  community extension whose `ddx('<sql>')` table function ran v1's rewrite,
  went with v1; §3.4 is its record.)
- **M6 — Math roadmap & hardening.** Extend the scalar rules (§3.6's
  roadmap), a dialect name table (which M5's `to_sql` starts), docs. Plus:
  the expression-swell size/latency benchmark (§3.7), and the
  convention-pinning tests for NULL-folding, kinks, and domain edges (§5).
- **Future, demand-driven:** the physical fused-contraction operator; more
  v2 architectures (conv, more normalization variants); a DuckDB community
  extension, once performance work settles (einfold, a query optimizer that
  rewrites SQL plans into array algorithms, is one route), with the C++/cxx.rs
  hybrid for a bare `grad()` in DuckDB and the spike that sizes it; `ddx-pg`.

---

## 9. Open questions

- **DuckDB outside Python.** M5 serves DuckDB-Python. A dedicated extension's
  surface (a table function taking SQL, since DuckDB's extension API has no
  planner hooks, or a bare `grad()` through a C++ `OptimizerExtension`) waits
  until performance work settles, and wants the duckdb-zarr maintainers' view.
- **Planning for an engine with no Substrait producer** (Polars, most
  databases behind ADBC or Flight SQL). Running is solved for any SQL engine
  whose tables persist between statements (`S23`: a step's table must be
  visible to the next statement); planning is not. The candidate is DataFusion's planner over the
  engine's schemas, with `to_sql` writing every step in the engine's dialect.
  ddx already vendors DataFusion's SQL parser, but a parser is not a planner:
  planning needs names, types and functions resolved. einfold's SQL front end
  is the same planner (it parses SQL into DataFusion's unoptimized plan), so
  einfold could host this front end, not replace it. The risk is a foreign
  planner reading a query differently from the engine (integer division,
  `log`'s base, decimal typing), which would give the derivative of a
  different function. `to_sql` writing the plan's meaning explicitly keeps
  ddx's value and gradient consistent with each other; a value check (the
  query run natively, compared with ddx's value step) would catch a
  divergence from the engine, at the cost of a forward query, and whether it
  always runs is undecided. An engine with no session state between
  statements would also need a program written as one statement, each step a
  CTE, which is unresolved: a CTE an engine inlines per reference recomputes
  the work a materialized step saves. Deferred until an engine needs it.
- **One SQL writer for ddx and einfold.** einfold also writes SQL for named
  dialects from relational plans, and found that unparsing DataFusion's
  optimized plans can give SQL DuckDB rejects or, once, a silently wrong
  answer (its spike S7). `to_sql` is built in `ddx-ad` for M5, with a
  standalone API (a Substrait plan and a dialect in, SQL out), so the two
  projects can share it later rather than keep two.
- **Decimals.** DuckDB types `0.1` as `DECIMAL`, so a DuckDB query's
  arithmetic may be decimal where ddx's rules assume real arithmetic, the
  shape of §3.8's integer-division case. Whether to differentiate decimal
  operands, cast them, or refuse them is M5's to decide.
- **Custom rule richness** — unary custom differentiation rules only, or
  binary/n-ary too? Unary is easy today (the engine already dispatches on
  function name); richer rules are a bigger trait.
- **Higher-order AD over an emitted backward plan** — decided (§4.9): a
  read of a materialized step is differentiated through by writing the step's
  plan in its place, and `jvp` reads each step's dual. Reverse over reverse
  waits on reverse mode learning that join keys are equal (§4.6).
- **Recomputation vs. saving inside a region.** A region's backward step
  rebuilds the region's joins rather than reading them from a saved relation
  (`S6`). For nn.py's first layer that is one extra join per backward step,
  the same size as the forward one. Whether some regions should be saved
  instead is a performance question for the fused-contraction work, not a
  correctness one. A contraction's region is no longer rebuilt at all
  (`S19`); other regions still are.

---

## Decision log

This is the audit trail behind the design above: findings from two rounds of
adversarial review on v1 (`F1`–`F12`, `G1`–`G9`), the spikes that resolved
open research questions (`R1`, `R1b`, `R2`), the answered decision points
(`Q1`–`Q7`), and the v2 pivot from a bespoke IR to Substrait (`S1`–`S5`). Each
entry is referenced from the main text where it applies; nothing here changes
the design as stated above — it's the evidence for why it's stated that way.

### v1, round 1 — silent-wrong findings (principle-5 violations)

**F1 — Column identity was raw-string equality; SQL identifiers aren't.**
`grad(Temp*Temp, temp)` would silently differentiate to `0` in DuckDB
(case-insensitive throughout) if matched by raw string. Also a regression
risk against the prototype, which got folding for free via DataFusion's own
parser. Fixed: per-dialect identifier folding in `ColRef`, casefold-unquoted
and dialect-specific on quoted (DuckDB folds quoted too; DataFusion/Postgres
don't) — corrected on a second pass after the first fix wrongly assumed
"exact-match quoted" was universal. → §3.2.

**F2 — The ambiguity guard was one-sided.** It fired only for an unqualified
`wrt`; the mirror case — a qualified `wrt` with a bare occurrence of the same
name elsewhere — was silently wrong (`grad(x * a.x, a.x)` in a join where `x`
binds to `a.x` should be `2x`, not `x`). Fixed: fire on any *uncertain*
occurrence of the `wrt` base name, regardless of which side is qualified —
corrected on a second pass after an initial "symmetric ≥2-qualifiers" version
over-fired and would have rejected `grad(a.x*b.x, a.x)`, a case the design
explicitly wants to accept. → §3.2, §3.5.

**F3 — Derivatives don't commute with query composition, and the doc never
said so.** `grad` treats a CTE-computed column as an opaque constant, so
factoring a loss through a CTE silently drops gradient terms — defensible
relational semantics, undocumented convention, real trap for the pitched use
case. Fixed: state the contract loudly (§3.5) and add a best-effort
same-statement CTE-alias guard, refined with a carve-out (`G4`) so it doesn't
reject differentiating w.r.t. the alias itself.

**F4 — The DOUBLE-literal fix didn't cover literal-free derivatives.** The
quotient rule routinely emits literal-free SQL (`grad(x/y,y)` →
`(-x)/(y*y)`), and integer division silently truncates on `BIGINT` columns in
one engine but not another. Fixed: the numeric-type policy moved into the
smart constructors themselves (`div()` wraps in `CAST(… AS DOUBLE)`
whenever operand types are unknown, which is always, pre-binding), not just
into literal emission. → §3.2.

### v1, round 1 — systems risks

**F5 — `sqlparser` is a whole-query gatekeeper, and reprinting amplifies
it.** Any DuckDB syntax `sqlparser`'s dialect lags on fails the *whole*
query inside `ddx('…')`, even when the `grad` itself is trivial — a
permanent version-treadmill. Fixed, two mitigations: no-marker queries pass
through byte-identical without ever being parsed (a parse-free pre-gate),
and when a marker is present, the derivative is spliced by source span
rather than by reprinting the statement, so reprint fidelity stops being a
separate risk on top of the coverage bound. → §3.2, §3.3.

**F6 — Symbolic expression swell, with nothing shared.** An n-factor product
yields an O(n²) derivative; a subexpression appearing k times is recomputed
k times; an N-parameter gradient multiplies this by N. Accepted for v1 with
two mitigations: a measured (not guessed) size/latency benchmark, and a
future let-binding remedy — the real fix is v2. → §3.7.

**F7 — `ddx('…')` had unvalidated mechanics.** Bind-time schema (a DuckDB
table function must declare result columns at bind, requiring a
`DESCRIBE` of the rewritten query on the inner connection — feasible-looking
but unspiked), lost connection-scoped state (temp tables, session `SET`s,
prepared statements are invisible inside the inner connection — broader than
the transaction-visibility finding `R1b` covered), and DML policy (DuckDB's
own `query('sql')` precedent is *not* actually SELECT-only, correcting an
earlier claim, so ddx's SELECT-only default is its own conservative choice,
not inherited). All three named as explicit M5 tasks, not discoveries made
under deadline. → §3.4.

**F8 — Markers hijacked every function spelled `grad`/`jvp`.** Including a
user's own UDF or a qualified `myschema.grad(…)`. Fixed: reserve only
unqualified spellings, documented. → §3.2.

### v1, round 1 — API and semantic debts

**F9 — The rule registry had no seam in the public API.** Every signature was
a free function; nowhere for a user's registry to live without forcing
global mutable state or a later API break. Fixed: the entry point is an
object, `Ddx`, from the start. → §3.2.

**F10 — `vjp` wasn't reverse mode, and the doc didn't say so.** As specified,
scalar `vjp` was a cotangent-scaled *forward* pass — no accumulation, no
amortization across N parameters — and the "reverse-mode" framing oversold
it to exactly the JAX audience the project courts. Resolved later, more
fully than a wording fix: see `Q7`.

**F11 — 0/1-folding changes NULL semantics; needed to be a stated
convention.** Folding `d/dx(x+y)` to `1` gives a non-`NULL` derivative even
where the primal would be `NULL`. Matches JAX's `Zero`-tangent treatment,
but folded and unfolded derivatives then disagree on NULL-bearing rows —
documented and tested, not left as a quirk. → §3.2, §5.

**F12 — Kinks and domain edges will make the oracle and the engines
disagree.** `abs` at 0 (JAX's own kink convention differs from ddx's pinned
`0`); derivatives that fail where the primal doesn't (`sqrt`'s derivative
divides by zero at 0, and engines disagree on the result); `tan` near π/2,
`ln` near 0. Fixed: pin conventions explicitly and state a domain-edge
policy, rather than comparing blindly. The `abs` kink in particular is pinned
by emitting a portable `CASE`-based sign, not an engine `signum`/`sign`
builtin — an early cut emitted `signum(u)`, which is non-portable (DuckDB has
no `signum`) and evaluates to `1` at 0 on DataFusion, pinning nothing; caught
in a later adversarial review. → §5.

### v1, round 2 (`G1`–`G9`) — a fresh pass with four independent evidence spikes

**G1 — `sqlparser`'s `Display` drops precedence parens on constructed trees
(confirmed by spike).** A constructed `(a+b)*c` Displays as `a + b * c`,
which reparses as the wrong expression — hits the product/quotient rules
immediately, a wrong number in valid SQL. Fixed: smart constructors wrap
composite operands in `Expr::Nested`. → §3.2, §5.

**G2 — `sqlparser` version lockstep undermines "one IR."** `ddx-core`'s
public API and the Path B bridge both need the *identical* `sqlparser`
version as DataFusion's, but `sqlparser` ships breaking releases every 1–3
months and DataFusion adopts each with a lag, while DuckDB-dialect coverage
wants the newest. Resolved with an explicit pin-and-degrade policy. → §3.3,
§6.

**G3 — Spans are line/column characters, not byte offsets (confirmed by
spike).** `grad` in a string containing a multibyte character before it
lands at a different byte offset than its column number suggests — naive
column-as-byte splicing corrupts the output. Fixed: a proper span→byte
conversion subsystem, not a one-liner. → §3.2.

**G4 — The CTE-alias guard (`F3`) forbade the design's own endorsed case.**
`grad(s*s, s)` — differentiating w.r.t. the alias itself — was rejected by
the naive version of the guard. Fixed: the guard fires only when a computed
alias appears as a *non-`wrt`* term. → §3.5.

**G5 — The ML pitch is where the design's own limits bite hardest.** F6+F10
compound: an N-parameter SGD step is N independent full derivations, exactly
why ML abandoned symbolic diff historically. This did not become "downplay
ML" — see the v2 pivot below (`S1`–`S5`) — it became "stage it, and prove the
staging is real."

**G6 — A stated claim was actually a contradiction.** DataFusion's Path B was
claimed to "de-risk" DuckDB's C++ path; it doesn't — the C++ path's hard part
(rebuilding a *bound* expression) is exactly what the DataFusion bridge
avoids by leaning on DataFusion's own unparse/re-plan utilities. Fixed:
removed the claim; the honest de-risker is the dedicated M5-adjacent spike.
→ §3.4.

**G7 — Day-one implementation details, not yet named.** Pre-gate/marker
matching needed to be case- and whitespace-tolerant; `TypeCoercion` runs
before the Path B `AnalyzerRule` sees the marker's argument; the numeric
oracle needed to be pulled forward to M2, since M2's exit gate depends on
it. All folded into §3.2/§3.3/§8.

**G8 — Should scalar `vjp` be cut?** Given `F10`, a two-token macro
(`mul(cotangent, grad(e,x))`) burns a name the JAX audience expects to mean
something real. Left open pending the v2 investigation; resolved by `Q7`
below once that investigation concluded.

**G9 — The `sqlparser`-gap examples were stale.** Spiked against
`DuckDbDialect` @ 0.62.0: three of four claimed gaps (`SELECT * EXCLUDE`,
`FROM`-first queries, lambdas) actually parse fine; only `PIVOT` and `#1`
positional columns genuinely fail. Refreshed the claim with a version-pinned
spike result instead of an assumption. → §3.3.

### Open research questions closed by spike (`R1`, `R1b`, `R2`)

**R1 — DuckDB's stable C extension API has no optimizer, parser, or
bound-expression hook.** Read directly off `duckdb_extension.h` (what the
`duckdb` crate's `loadable-extension` feature binds): registration exists for
scalar, aggregate, table, and cast functions plus replacement scans, and
nothing deeper. Corroborated by duckdb-zarr, a mature extension using exactly
table functions. This is what makes a native bare-`grad()` rewrite impossible
in a *Rust* community extension and puts `ddx('<sql>')` — plus the deferred
C++/cxx.rs hybrid — in the design instead. → §3.4.

**R1b — An inner query on the same DuckDB database, run mid-execution of an
outer one, is safe (`docs/spikes/duckdb_reentrancy_r1b.py`).** No deadlock; reads of
committed data work; DML works. One real consequence, and the reason the design
routes DML training loops elsewhere: the inner connection runs in its own
transaction, so it cannot see the *outer* connection's uncommitted writes. →
§3.4.

**R2 — `datafusion-python` cannot inject an `AnalyzerRule`
(`docs/spikes/datafusion_python_analyzer_rule_r2.py`).** Re-verified at M1 against
`datafusion` 54.0.0, checked three ways from the outside in: the public
`SessionContext` has no `add_analyzer_rule`; neither does the `_internal` PyO3
class beneath it, which also exposes no `SessionState`/`SessionStateBuilder`
handle (where the method lives in native Rust DataFusion); and — the check that
makes the finding durable rather than a snapshot — the compiled FFI capsule
vocabulary contains no `__datafusion_analyzer_rule__`, so a *compiled Rust*
extension has no way in either. The logical optimizer is closed the same way:
`remove_optimizer_rule` exists with no matching `add`.

The single rule-injection capsule that does exist,
`__datafusion_physical_optimizer_rule__`, is post-planning and therefore useless
here — by the time it could see a marker, the argument is a compiled physical
expression, not the symbolic form differentiation needs. The same spike pins
that underlying fact independently (§3.1): a *scalar* UDF named `grad`
registered on a live context receives `[1.0, 4.0, 9.0]` for `grad(x*x, x)` over
`x = [1,2,3]` — the *evaluated* argument, never `x*x`.

Three seams were found and recorded per M1's exit criterion; none displaces
Path A for bare `grad()`, and one is worth adopting on its own merits.

*(i) Plan serialization.* `LogicalPlan ↔ proto bytes` plus
`execute_logical_plan` does permit an out-of-engine plan rewrite — but it can't
intercept `ctx.sql()` transparently, it would make ddx manipulate DataFusion
protobuf plan messages (a DataFusion-specific plan-interchange format, which
§1.2 rules out), and it fails outright on in-memory tables, xarray-sql's common
case. Not adopted.

*(ii) `add_physical_optimizer_rule`* via a compiled extension — post-planning,
rejected on the merits above. Not adopted.

*(iii) Table functions (`register_udtf`).* Unlike a scalar UDF, a table function
receives an **unevaluated `RawExpr`**. This is the DataFusion-native analogue of
the `ddx('<sql>')` table function §3.4 already ships for DuckDB, and it needs no
compiled extension — a plain Python callable works — so it is worth taking up in
M2 for surface symmetry, letting the same `ddx('<sql>')` spelling work on both
engines. It is emphatically *not* a route to bare `grad()`, and does not soften
the scalar-UDF finding. Two limits close it, and the second is worth stating
precisely rather than in the obvious-but-wrong shorthand. First, a composite
argument arrives already **constant-folded** (`sin(2.0)*3.0` →
`Float64(2.7278…)`), so no structure survives. Second, a table function's
arguments are resolved against an **empty schema** — which is narrower than
"columns are rejected": a *bare* column reference does pass through, as an
unresolved and meaningless `Expr(x)`. What dies at planning is anything
requiring type resolution, so both `ddx(x + 1)` and `ddx(grad(x*x, x))` fail
with *"No field named x"*. Only opaque leaves survive, which means what the seam
can actually carry is a SQL **string** — handled by `rewrite_sql` exactly as
under Path A. It relocates Path A into the engine; it does not replace it.

So the structural claim is specifically about *bare* `grad()`: no seam, at any
layer, delivers a marker's symbolic argument over real table columns. Should an
`__datafusion_analyzer_rule__` capsule ever appear upstream, `ddx-datafusion`'s
Path B bridge is what would plug into it — a second reason to build Path B in M2
as designed. → §3.3, §3.4, §8 M1.

### Resolved decision points (`Q1`–`Q7`)

1. **DuckDB ergonomics** — `SELECT * FROM ddx('…')` accepted for v1; still
   pending a second opinion from the other duckdb-zarr maintainer (§9).
2. **Column binding** — qualifier-aware syntactic differentiation with a
   hard error on ambiguous unqualified `wrt`, settled by `F2`/`G4`. → §3.5.
3. **User-registrable rules** — yes, adopted; unary-only-vs-richer is still
   open (§9).
4. **xarray-sql integration** — ship as an optional extra
   (`xarray-sql[ddx]`), not a hard dependency. → §3.4.
5. **Naming** — resolved; no umbrella `ddx` crate needed since the individual
   crate names are all free. → §7.
6. **C++ hybrid timing** — keep `ddx('…')` first, brand it transitional,
   pull the *risk* forward via a miniature spike rather than building the
   full extension up front. The case for going C++ first is genuinely
   stronger than early drafts credited (the bound path is structurally
   immune to the whole F1/F2/F3/F5 class), but DataFusion's Path B already
   buys most of the in-engine validation far more cheaply, and `ddx('…')`'s
   only real risk is social (hardening into a permanent contract), which
   branding fixes without an architecture change. A concrete tripwire is
   named in §9. → §3.4.
7. **Scalar `vjp` — cut.** Resolved once the v2 investigation (`S1`–`S5`)
   confirmed the *real* `vjp` — query-level, actual reverse accumulation —
   is buildable and roadmapped. Shipping a scalar `vjp` that's definitionally
   a two-token macro would burn the name on something that doesn't earn it
   and mislead the exact audience the project courts. v1 ships `grad` +
   `jvp` only; `vjp` is reserved. → §3.6, §4.

### The v2 pivot: from a bespoke IR to Substrait (`S1`–`S5`)

**S1 — Rejected a bespoke Rust builder IR for v2; adopted Substrait
instead.** An early draft of the v2 design answered "how does the system
know a join is a contraction?" by inventing a `RelGraph`/`RelNode` Rust
builder API the user would construct by hand. Alex's review pushed back,
correctly: that's a new embedded DSL, not SQL, repeating the exact mistake
already paid down once when a bespoke `DExpr` was deleted in favor of
`sqlparser`'s real type. The "tag, don't infer" requirement itself was kept
— it's not negotiable, since a misclassified plan shape is a silently wrong
gradient — but realized instead as Substrait extension-function markers,
the same mechanism v1's `grad()` already uses, one layer down. DataFusion's
`LogicalPlan` was considered and rejected as the alternative carrier: DuckDB
has zero plan hooks (`R1`), so that type is DataFusion-only by construction.
→ §4.2.

**S2 — The marker mechanism verified cross-engine
(`docs/spikes/substrait_ad_marker_spike.py`).** Before this spike, "Substrait +
markers" was an architectural argument; after it, a checked claim. A
DataFusion-produced, marker-tagged plan is consumed and executed correctly
by DuckDB, matching DataFusion's own result exactly — the actual portability
claim, not merely "an engine doesn't corrupt its own plan." One honest gap:
the reverse direction (DuckDB produces, DataFusion consumes) deserializes
but wasn't executed. → §4.2.

**S3 — Route's math verified against `jax.grad`
(`docs/spikes/route_ad_spike.py`).** Machine-exact away from ties; a genuine,
now-pinned convention divergence at exact ties (first-index routing vs.
JAX's tie-splitting) — not a bug, but not something to claim agreement on
without saying so. → §4.3.

**S4 — A real DuckDB Substrait bug found, isolated, and worked around
(`docs/spikes/duckdb_substrait_window_bug.py`).** The Route rule's forward idiom
— `ROW_NUMBER()`-based top-1-per-group filtering — silently returns wrong
(unfiltered) results after a DuckDB Substrait round-trip, because DuckDB's
own optimizer rewrites the idiom into an `arg_max`-join before export and
that rewritten form doesn't survive the trip. Confirmed independent of any
ddx marker (pure DuckDB bug) and isolated from DataFusion, which round-trips
the identical query correctly. A two-step workaround (round-trip only the
window-column computation, then filter with plain engine-native SQL) is
verified correct — Route ships on both engines without waiting for an
upstream fix, though filing the bug against the DuckDB Substrait extension
project [5] is still worth doing on principle. → §4.3.

**S5 — Named the recurring risk pattern this reveals.** v2 is bounded by
whatever relation vocabulary Substrait and each engine's producer/consumer
actually implement — the same coverage-gatekeeper shape as `sqlparser`'s
dialect coverage for v1 (`F5`/`G9`), recurring one layer up. `S4` is the
first concrete instance; more rules (LayerNorm, conv) will likely surface
more. The resolution pattern each time: spike the forward idiom against
both engines before trusting a rule, and prefer a verified workaround over
waiting on an upstream fix when one exists. → §4.2, §4.6, §5.

---

### Building v2 (`S6`–`S23`)

**S6 — The tape is cut at aggregates, and nothing between them is
materialized.** Materializing every relation's output, or every relation's
cotangent, would write a contraction's whole join, `N × D × H` rows for
nn.py's first layer. Saving only aggregate outputs keeps each saved relation
no larger than a layer's output. The row-local work between two saved
aggregates is rebuilt inside each backward step, with every intermediate value
kept as a column so the chain rule can read it. → §4.4.

**S7 — Reads of earlier steps are bound late rather than typed by ddx.** A
Substrait read must state its column types. DataFusion's producer does not
fill in function output types, so ddx would have to derive them, which means
re-implementing each engine's typing rules and turning every mismatch into a
consumer error. The engine runs steps in order, so each table a step reads
exists before the step is consumed; the adapter binds the read's types from
it. → §4.4.

**S8 — Relations are dims and values; `wrt` names values.** The rules need to
know which columns identify a row. The XQL model already says: dims do, and a
variable is a value at a coordinate. Naming the differentiated columns
therefore names the dims too, and a cotangent or gradient has its primal's
dims and values, the relational version of "a gradient has its argument's
shape". Because a plan cannot show that dims identify rows, each program
checks it at run time before it runs, rather than trust it. → §4.4.

**S9 — Functions are recognized by name.** DataFusion 54 declares every
function, its own built-ins included, with a bare name and
`extension_urn_reference = u32::MAX`, so there is no URN to match. Names are
case-folded and a `:signature` suffix is ignored, as the spec's compound names
would have it. This extends `S2`'s observation to the producer side. → §4.2.

**S10 — The markers were dropped; the rules are per primitive.** The design
had users tag every contraction (`ddx_contract_mark`), reduction
(`ddx_reduce_mark`) and argmax (`ddx_route_mark`), on the reasoning of
principle 3: a misclassified operation is a silently wrong gradient. Building
the rules showed nothing was being classified. A `SUM` has the same transpose
whether one calls it a contraction or a reduction; an argmax by rank filter is
the select rule; and `MAX` has a rule of its own. The first implementation
kept the tags as required checks, which made users write annotations that
changed no number, and made the SQL surface a place no JAX user would
recognize. They were removed after review. What stays is the one annotation
that does change the gradient, `ddx_stop_gradient`, which JAX has too.
Principle 3 is restated accordingly. → §2, §4.3.

**S11 — Each program's tables live under a prefix of their own.** The first
adapters materialized every step under fixed names (`__ddx_value`,
`__ddx_saved_0`) in the caller's catalog. An adversarial review showed what
that costs: a user's table of the same name was replaced, activations stayed
pinned in memory, and two programs on one context read each other's tables,
giving a silently zero gradient, or failing with a missing table under
`tokio::spawn`. Every name now starts with a prefix fresh to its program, and
an adapter drops the intermediate tables once a run finishes. Running one
program twice at once still shares its names, so a caller that runs
concurrently builds a program per task. `ad::sql` keeps its programs from
call to call, and takes one out of its cache until the caller has dropped
the tables it wrote, so an overlapping call builds its own (`S20`). → §4.4.

**S12 — What the v2 soak found.** A generated-query soak checked
gradients against finite differences of the query and against ddx itself,
and every failure was in how rows are identified, not in a derivative. An
aggregate that skipped a NULL argument still sent gradient through the rest
of that row; MAX and MIN compared recomputed values with the saved extreme,
and a multi-partition `SUM` beneath them made that comparison fail about
half the time, silently zeroing the gradient; stacked rank filters gave the
engine two columns of one name; a gradient step's name had capitals the
engine folded away; and an unanalyzed plan's step declared a type its rows
did not have. Each is fixed where it arose (the reduce rules' NULL mask and
windows, window columns renamed in place, lower-case step names, a step's
table typed by the plan that ran). The soak's mutation test then measures how
fast it catches each rule broken on purpose. → §4.3, §4.4, §5.

**S13 — What the second round found: size, not calculus.** A forward-mode
oracle (every relation's tangent computed beside its value, independent of
the transposes) agreed with reverse mode on all eleven thousand comparisons
it made. What broke was the size of the plans an engine builds from ddx's.
DataFusion's Substrait consumer names a computed column by its whole
expression, and ddx computes each column from earlier ones, so a map layer
that reads its input twice doubled every name after it (twenty layers
exhausted 13 GB), and a NULL-skipping fold that nested its running sum did
the same per reader. One projection per cotangent column also made steps
hundreds of relations deep, and cloning one overflowed a worker thread's
stack. The fixes: the DataFusion adapter consumes a step with short column
names (the names mean nothing to ddx, whose plans refer to columns by
position); a column's cotangent terms are one flat sum; and cotangents are
projected in batches, as deep as the chain of columns reading each other.
datafusion-python's consumer cannot be given short names, so a very deep
chain is still large from Python. The same round found `grad(…)` in SQL
counting parentheses inside comments (now found by tokens), a vjp cotangent
whose keys repeat (now checked, like a wrt table's dims), and that the
MAX/MIN tie tolerance must apply only to values that can jitter: those
read from an aggregate that rounds, not a table's, differentiated or not. → §4.4.

**S14 — ddx composes: what a second engine needs is in `ddx-ad`.** A design
review aimed at ddx as a library for other engines, and a spike that ran it
on DuckDB, found that what is neither calculus nor engine had been written
once per adapter: the run protocol (checks first, steps in order, drop the
intermediates on success and failure) and `sql_all`'s planning (one program
per loss query, each call's dims and value columns). Both now live in
`ddx-ad` (`run` and `Backend`; `rewrite`), and the DataFusion and
Python adapters drive them. The spike also found two producer gaps, now
closed (a `ReferenceRel` was not followed; a cast of a literal was opaque, so
`x ** 2` from an unoptimized plan was refused), and that emitted plans
dropped the input's extension URNs. The API got the promises a published
crate needs: `AdError` and the program types are `non_exhaustive`, the
plan-reading modules are behind an `internals` feature, `Options` gives a
fixed namespace (the same plan then gives the same program, for golden tests
and caches), plans can cross as bytes, and how a `wrt` table name matches is
stated, with an ambiguous one refused. Two suggestions wait for a second
in-repo adapter: emitting the portable form of the windows in the reduce
rules (§4.6), and putting the simulation harness behind an engine trait so
it doubles as a conformance suite. → §4.2.

**S15 — v1's SQL surface is removed.** Once `grad`, `vjp` and `jvp` were
query-level calls in SQL, `grad` meant two things in one statement: v1's
derivative of an expression, row by row, and v2's gradient of a loss. Telling
them apart took a guess from the arguments, and a guess about meaning is the
kind of silent ambiguity principle 5 forbids. v2 covers v1: a row's derivative
of `f` is the gradient of `SUM(f)`, or a `jvp` with a tangent of 1. So the
marker rewrite, the DataFusion analyzer and text paths, and `ddxdb.rewrite_sql`
were removed. The scalar differentiator stays as `ddx-core`, v2's map rule.
The JAX oracle suite now checks each scalar rule through v2, as the gradient
of a sum, on DataFusion. → §1, §3.

**S16 — Every run checks its tables.** A program's checks (that each `wrt`
table's dims, and each input's keys, identify their rows) cost a grouped scan
each, every run. `Verified` let a caller keep the facts a run proved and skip
those checks on later runs over the same tables: about a quarter off a `jvp`
run in conjugate gradient (712 ms against 1,007 on 8M rows). But ddx cannot
see a table change, so the fact was the caller's promise, and a broken one
gave rows that share keys their summed gradient with no error: the one place
the API traded correctness for speed. It was removed, from Rust and Python. A
faster check, or a cache keyed on something an engine can vouch for, can come
back as an internal detail. → §4.4.

**S17 — One run, written once; engines implement `Backend`.** Running a
program took two shapes: `Runner`, a state machine an adapter drove step by
step (so an asynchronous engine could run it), and a synchronous `Backend`.
Each adapter wrote a loop over `Runner`'s actions, and the SQL layer grew a
second state machine of its own. The standard shape, as MADlib's driver
functions and every database client have it, is a trait of engine
operations with the library owning the driver: `Backend`'s methods return
futures, so DataFusion implements it directly and a Python engine returns
futures already done, and `ddx_ad::run` is the only loop. `Runner` and its
actions left the API. Two invariants moved into types: a program's plans are
`Unbound` (their reads of tables it writes name no column types) and an
engine receives only `Bound` plans, which only `Unbound::bind` makes; and
`Program` is `Sync`, so a run can be spawned. → §4.2.

**S18 — Calls in SQL are rewritten as a tree, not spliced as text.** v1
spliced each derivative into the statement by source span and kept every
other byte (§3.2), so a statement could go to any engine exactly as written.
Once calls became columns of a `SELECT`, splicing meant byte offsets for
calls, tables, `*`s and `SELECT`s, a parenthesis count in tokens because an
expression's span can stop short of its `)`, and rules for edits inside other
edits; two bugs came from those. The statement must parse for its calls to be
found anyway, so the rewrite now replaces nodes of the parsed tree,
each named by where its first token starts (`sqlparser`'s `VisitorMut`), and
writes the tree out. Each replacement is built as nodes (the relation
that replaces a table wraps the statement's own table node) and swapped in
only after the visitor has passed it, so the visitor never walks into
what ddx built and never mistakes it for the statement's. The query a
derivative differentiates (a CTE's query after the CTEs it reads, or a
`jvp`'s `SELECT`) is built from the statement's nodes the same way, and
its own calls are found in those nodes rather than in a reparse of its
text. The cost: comments and spacing are not kept, and the engine sees
`sqlparser`'s rendering of the statement; `tests/ad_sql_text.rs` and every
SQL test run through it. → §4.8.

**S19 — A contraction's transpose does not rebuild its join.** einfold's
spikes S22 and S25 (#125) found each gradient step of `matmul` rebuilding
the forward join (6.4 M rows at n = 50,000) only to test whether the
forward product was NULL, then hash-joining those rows to the cotangent:
1.1 s of a 1.4 s run. For `SUM(e_X · e_Y)` over one inner join of `X` and
`Y` (each side one input, the condition equalities between them, the
groupings columns of a side), `X`'s contribution is `∂e_X/∂x ·
Σ_y C̄·e_Y`, so it is emitted as the cotangent joined with `Y`, summed by the
join keys and `X`'s grouping keys, then joined to `X`. The NULL test moves
from every joined row to one per row of `X` (a NULL `e_Y` already makes its
term NULL, which `SUM` skips). Anything else (a filter across the sides, a
sum that is not a product, `AVG`, `MAX`, a side differentiated while the
other holds no grouping key, as in a matrix-vector product) keeps the
composed rules. Keys
joined with `=` drop their NULLs before the last join so that all its keys
are null-safe, which DataFusion hashes; a mixed join fell back to a nested
loop and took 40× longer. `contraction_transpose.rs` checks it against the
composed rules on random tables with NULLs, NaNs, infinities, missing rows,
NULL join keys under `=` and `IS NOT DISTINCT FROM`, two-key joins, and
repeated keys in constant data. The soak does not yet write this shape
(#127). `matmul`'s gradient at n = 50,000 went
from 1.77 s to 0.83 s; `attn` at L = 256 from 0.74 s to 0.52 s. → §4.3.

*Extended for causal attention (#153).* The rule above missed both of
attention's aggregates: the scores' join carries the causal mask, a filter
across the sides, and the output's region joins three inputs. Two
extensions, each exact for inner joins. A conjunct that reads only grouping
keys passes every row of a group or none, so it is applied to the
cotangent's rows instead. And a side may hold several inputs: inner joins
can be regrouped, so the inputs are split by which factors of the product
read them (an input no factor reads goes with the side its conditions tie it
to), each side is rebuilt as its own join (falling back to the composed
rules if the side's own conditions leave it in pieces, which would rebuild
it as a cross join the query never formed), and each is transposed by the
ordinary region rules (its projections, windows and inputs) from its
factor's cotangent, the contraction `t` of the cotangent with the other
side. This replaced inlining each side's factor down to one input's
columns, which could not pass through a window. `t` is materialized when a
side reads it more than once (several inputs, or a window). The output's
sides are the scores joined to the normalizer, and the values; nothing at
(token, token, feature) granularity is stored or rebuilt.
`contraction_transpose.rs` adds masked products, NULL grouping keys under a
mask, a divisor joined on a grouping key in three join orders and with a
table no factor reads, and the full attention query, each against the
composed rules. At 256 tokens, attention's steps went from 1.12 s to
0.28 s.

**S20 — The SQL surface keeps its programs, keyed without optimizing.**
`ad::sql` planned and built every program on every call, so a training loop
paid for differentiating its loss each step. In the GPT demo of xarray-sql's
ML benchmarks that was about 1.5 s of a 2.4 s step: DataFusion optimizing the
loss (800 ms), then `ddx_ad::grad`. A program depends on the tables' names
and schemas and the views' definitions, not their values (which is why
the program API is built once and run each step), so
`ddx_ad::rewrite` now keeps each program it builds in a `Programs` passed by
the caller, at most 16, the least recently used dropped first. The key must
be found without optimizing, or a hit would save only the build. It is the
program's transform, `wrt` columns, query text and row restrictions, plus a
new optional `Backend::plan_key` of each query: on DataFusion, the plan
before optimization printed with its schemas. DataFusion expands every view
(transitively) while it turns SQL into a plan, so this text holds each
view's definition and each table's columns, types and nullability, and ddx
never resolves a name itself. Keying on the statement text plus schemas ddx
looks up was rejected for that reason (views over views, default schemas).
An explicit prepared statement was rejected because every caller, the demos
included, would have to change. A backend without `plan_key` keeps nothing.
A program that reads another's result reads it by that program's table
names, so it is kept only while the program it reads is. A program is taken
out of the cache while it runs and comes back only when the caller releases
the tables it wrote (`Rewritten::release`), since its runs write the same
tables (S11): put back as `rewrite` returned, a call made before the first
caller had planned its statements would rewrite those tables under it, and
drop them when it released its own (found by CodeRabbit).
One semantic change: DataFusion's optimizer folds `now()` into a literal, so
a kept program holds the time of the call that built it. Measured on
`nn.rs`'s model, a SQL training step went from 107 ms to 91 ms, within 3% of
running a program built once (88 ms). The SQL training test runs 25 steps
again. → §4.8.

**S21 — One linear form per reduction, read by both modes; windows in reverse
mode.** Forward and reverse mode each held a rule per aggregate, and reverse
mode refused a window value, which blocked a softmax written with windows
(`x - MAX(x) OVER (PARTITION BY i)`). A JAX-style tape that transposes forward
mode's dual was considered and not built: reverse mode already transposes the
linear part over recomputed regions, so a tape would only have moved forward
mode's machinery. Instead each reduction's tangent is one linear form,
`Σ coef · ṫ` over its group (`reduce.rs`), which forward mode renders as a
tangent and reverse mode as its transpose, as JAX's `deflinear` defines a
linear primitive once. A window is the same reduction over each row's frame:
a frame's transpose is its mirror, and a whole partition's extreme is the
grouped rule. Review found three things. A window's transpose must run over
the rows it read, not the rows above it, or a later filter, semi-join or
duplicating join changes its counts, ties and frames; finite differences
caught the wrong gradient. A window read along several paths must be
transposed once, from the sum of its cotangents, with each window's summed
cotangent a step, or a stack of windows grows as a power of its depth, since
a plan is a tree. And two refusals changed: a rounding `SUM` or `AVG` over a
whole partition is accepted rather than refused, and a `ROW_NUMBER` or
`FIRST_VALUE` with no `ORDER BY` now needs its ties broken. → §4.3, §4.4.

**S22 — DuckDB-Python first: Substrait in, SQL out.** The first DuckDB plan
(§3.4) was a community extension whose `ddx('<sql>')` table function ran
v1's rewrite; it went with v1 (`S15`). xarray-sql reaches DuckDB through
Python, registering each Dataset as an Arrow dataset on the connection, so
DuckDB-Python is the target. The obvious backend, DuckDB's own Substrait
producer and consumer, fails there, as
`docs/spikes/duckdb_substrait_visibility.py` shows on DuckDB 1.5.5: the
producer refuses an Arrow-registered scan; the consumer reads as if on a
connection of its own, seeing committed persistent tables but not registered
objects, temp tables or the caller's open transaction; and it rejects every
window function, which ddx's rules write. Copying every table a query reads
into DuckDB, and writing windows as joins back, would make the consumer work,
at the cost of xarray-sql's lazy, pushed-down scans, of transaction
visibility, and of more reliance on a community-maintained extension.
Instead ddx plans with DuckDB's producer on a schema-only mirror of the
caller's catalog, so the plan keeps DuckDB's semantics, and writes each step
and check as SQL that runs on the caller's own connection. The SQL writer is new code
with an existing oracle: every DataFusion test program, run both ways. It
also makes any SQL engine with a Substrait producer a target, whatever its
consumer. A dedicated DuckDB extension waits until performance work
settles. → §4.2, §8.

**S23 — One backend for SQL engines; planning without a producer deferred.**
The question that followed `S22`: how would ddx work on any Arrow-based
engine xarray-sql registers into (DuckDB, Polars, any database behind ADBC
or Flight SQL)? Every one can run SQL, take an Arrow table, drop a table and
report a schema, and with `to_sql` that is all running a program needs. So
the DuckDB backend is built generic over those primitives, DuckDB its first
instance. Few can produce Substrait, which planning needs. Planning with
DataFusion over the engine's schemas, directly or through einfold's SQL
front end (which is DataFusion's planner too), was considered and deferred:
no engine needs it yet, einfold is still in design, and its central risk, a
foreign planner reading a query differently from the engine, deserves its
own decision when an engine needs it (§9). → §4.2, §9.

## References

Numbered where cited inline in the design above (e.g. `[1]`); everything
else the design links to (project pages, tool docs) stays as an ordinary
inline hyperlink at the point it's used and isn't repeated here.

**Papers**

1. Baydin, A. G., Pearlmutter, B. A., Radul, A., & Siskind, J. M. (2018).
   *Automatic Differentiation in Machine Learning: a Survey.* Journal of
   Machine Learning Research, 18(153), 1–43.
   [arXiv:1502.05767](https://arxiv.org/abs/1502.05767). Cited for the
   argument that reverse-mode AD displaced symbolic differentiation in ML for
   exactly the expression-swell reason v1's scalar engine hits (§3.7,
   Decision Log `F6`/`F10`/`G5`).
2. Tang et al. (2023). *Auto-Differentiation of Relational Computations for
   Very Large Scale Machine Learning.* Proceedings of the 40th International
   Conference on Machine Learning (ICML), PMLR 202:33581.
   [proceedings.mlr.press/v202/tang23a](https://proceedings.mlr.press/v202/tang23a/tang23a.pdf).
   The published precedent for v2's per-operator transpose-rule approach —
   independently arrived at, reviewed after the fact (§4.1).

**Prior work this design is grounded in**

3. [xarray-sql#192](https://github.com/xqlsystems/xarray-sql/pull/192) — the
   prototype implementing `grad`/`jvp`/`vjp` for DataFusion that §3
   generalizes into `ddx-core`. All 13 commits, and what was added and then
   removed, are load-bearing (§3.1).
4. [xarray-sql#196](https://github.com/xqlsystems/xarray-sql/pull/196) — the
   MLP-trained-in-SQL demo (`nn.py`) that motivates v2 and that §4's
   transpose rules are checked against, query-shape for query-shape (§4.5).
5. [substrait-io/duckdb-substrait-extension](https://github.com/substrait-io/duckdb-substrait-extension) —
   DuckDB's community Substrait extension. The round-trip bug found in
   Decision Log `S4` (`docs/spikes/duckdb_substrait_window_bug.py`) should be
   filed against this project.
