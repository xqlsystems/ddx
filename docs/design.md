# `ddx` — portable autograd for composable databases

_author_: Alex Merose
_co-author_: Claude (Opus 4.8, Fable 5), via Claude Code
_status_: Design — iterating toward implementation
_last updated_: 2026-07-20

---

## 1. What `ddx` is

`ddx` is autograd for composable databases: you write calculus directly in SQL,

```sql
SELECT i, grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g
```

and derivatives come back as ordinary columns, evaluated row by row by the
engine alongside everything else. The destination is training ML models in
SQL — a [differentiable](https://www.youtube.com/watch?v=LNNU33TmBFk) [database](https://www.youtube.com/watch?v=jUe3rvTmv7Q). It ships as one engine-neutral Rust core with
thin per-engine adapters, into DataFusion (via a Python package, `ddxdb`, into
[xarray-sql](https://github.com/xqlsystems/xarray-sql)) and DuckDB (via a
community extension, into
[duckdb-zarr](https://github.com/xqlsystems/duckdb-zarr)).

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
checked with one — the programs live in [`spikes/`](../spikes/README.md) and
are cited by tag throughout.

**Two layers, one engine.** The design has two committed layers, built on one
differentiation engine:

- **v1 — calculus as columns** (§3). `ddx-core` differentiates SQL scalar
  expressions and rewrites `grad()`/`jvp()` calls to derivative SQL before
  planning. This is a real product on its own: sensitivity columns, small
  Jacobians, Newton steps, curve fitting, physical derivatives on gridded
  data — the sweet spot no other SQL-native tool covers. It has a real
  ceiling: an N-parameter gradient computed as N independent scalar
  derivations does not scale, which is the reason ML left symbolic
  differentiation for reverse-mode AD in the first place (Baydin et al. [1]).
- **v2 — query-level reverse-mode AD** (§4), the ML headline. The scalar
  engine from v1 becomes the *elementwise leaf* of a system that
  differentiates whole queries — not expressions — by applying transpose
  rules to relational operators (contraction, elementwise, reduce, route,
  stop-gradient). Sharing that scalar mode lacks happens through materialized
  intermediate relations (the tape) instead of inside expressions. Verified
  machine-exact against `jax.grad` on an MLP and on attention.

Both layers follow the same shape: a minimal-dependency, engine-neutral core,
with thin adapters per engine. v1's core depends on `sqlparser` only; v2's
depends on `substrait` only. Neither depends on `datafusion` or `duckdb`.

### 1.1 Non-goals

- Not a runtime tensor library or a GPU kernel engine. `ddx` differentiates SQL
  — scalar expressions in v1, whole queries in v2 — never arbitrary imperative
  code, and there is no tape of Python/Rust operations to record.
- Not general `u^v` power, `CASE`/conditional subgradients, or other
  non-smooth ops in v1 (§3.6 roadmap). An unsupported node is a typed error,
  never a silently wrong derivative.
- Not using Substrait in v1 (§3.3) — v1 needs no plan IR at all. v2 does, and
  uses Substrait; the two are consistent, not a reversal (§4.2, `[S1]`).
- Not two injection paths everywhere in v1. The universal path is a SQL
  source-to-source rewrite; DataFusion additionally gets an in-engine plan
  rewrite as the reference proof the core can drive one. Other engines' plan
  rewrites are deferred.
- Postgres is later — it needs array/XQL support first (via `pgrx`), and its
  planner-hook story differs from the two first targets.

### 1.2 Success criteria

- A single engine-independent core (`ddx-core`) implements each layer's
  differentiation algorithm once; every database integration is a thin
  adapter over it.
- It ships in two real, actively-used projects with no regressions:
  xarray-sql (DataFusion/Python) and duckdb-zarr (Rust community extension).
  These are sequenced, not simultaneous — xarray-sql is the v1 acceptance
  target; duckdb-zarr comes after the v2 track (§8).
- The `grad`/`jvp` surface is portable — the same SQL-level functions, one
  shared core defining what they mean, adopted by every target engine.
  Portability lives at the SQL surface for v1 and at the Substrait plan
  surface for v2 — never in a project-specific plan-interchange format.

---

## 2. Design principles

1. **Differentiate once, on a real IR, not a bespoke one.** v1 operates
   directly on `sqlparser::ast::Expr` — the same parser DataFusion uses. v2
   operates directly on `substrait::proto` types. Neither layer invents its
   own representation; each reuses the closest real one for its scope.
2. **Rewrite, don't execute.** `grad`/`jvp` are compile-time markers, always
   rewritten away before execution — never functions that run per row. Every
   integration is fundamentally "find the marker, differentiate what it
   wraps, splice the result back, hand plain output onward."
3. **Tag explicitly; never infer.** Recognizing what a piece of SQL means (a
   join is a contraction, an aggregate is a reduction) is never done by
   pattern-matching plan shape — the same shape means different things in
   different queries, and a misclassification is a silently wrong gradient.
   Every semantically special operation is marked, by the user, with a
   function ddx has claimed the name of.
4. **SQL (or its plan) is the portable surface.** v1's `grad`/`jvp` are
   ordinary SQL function calls; v2's markers are the same, one layer down in
   the plan. No project-specific interchange format carries meaning between
   engines — each engine's own SQL-to-plan machinery does.
5. **Fail loud, never silently wrong.** An unsupported construct is a typed
   error, not an approximate or silently-zero derivative. This is a
   numerical-correctness product, and every decision is weighed against this
   first.
6. **Prove it in real projects.** xarray-sql and duckdb-zarr are acceptance
   tests, not demos.

---

## 3. v1: calculus as columns

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
has a rule — but the marker UDF must be coercion-tolerant); and the re-plan
step needs a function registry, the seam is `SessionState::create_logical_expr`
`[G7]`.

### 3.4 Per-engine integration

| Integration | Dialect | How the rewritten SQL reaches the engine |
| --- | --- | --- |
| **Rust DataFusion** (`ddx-datafusion` helper) | DataFusion's | `ctx.sql(ddx.rewrite_sql(sql, dialect)?)` — one line |
| `ddxdb` (Python → DataFusion) | DataFusion-compatible | `Context.sql()` shim calls `rewrite_sql`, stock context plans it |
| `ddxdb` for DuckDB-python | `DuckDbDialect` | preprocess the string before `duckdb.sql(...)` |
| `ddx` (DuckDB community ext) | `DuckDbDialect` | `ddx('<sql>')` table function calls `rewrite_sql`, runs on an inner connection |

**DataFusion / xarray-sql.** `datafusion-python` doesn't expose injecting an
`AnalyzerRule` into its `SessionContext` `[R2]` — which is why the SQL rewrite
is the path here, not a limitation being worked around. `ddxdb` wraps a `Ddx`
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
| `WITH RECURSIVE n AS (… x-(x*x-2)/grad(x*x-2,x) …) …` | `… /(x+x) …` | ✅ training loop in one query |
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

**Roadmap:** general `u^v` via `exp(v·ln u)`; `CASE`/`min`/`max` subgradients
with a documented kink convention (mirroring how `abs` pins its kink at `0`
via a portable `CASE`-based sign); `atan2`, `log(base,x)`, `cbrt`,
`expm1`/`log1p`; a dialect name-normalization table; and a clear, permanent
taxonomy of "not differentiable" (comparisons, string/temporal ops, window
functions).

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
(`spikes/relational_ad_spike.py`, max error ~1e-18). The one gradient the
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

**Generality past the MLP is confirmed.** `spikes/attention_ad_spike.py`
builds a full single-head attention block (Q/K/V projections → `QKᵀ/√d` →
softmax over the key axis → `A@V`) from the *same* rules and matches
`jax.grad` on every weight and the input to ~1e-16; the causal mask is just
elementwise and also passes. LayerNorm (mean/variance = group-reduce +
elementwise), residual connections (elementwise add), and GELU/ReLU
(elementwise) all reduce to the same primitive set.

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

### 4.2 The mechanism: Substrait + extension-function markers

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

The remaining requirement — tag explicitly, never infer (§2, principle 3) —
is realized the same way v1's `grad()` already works, one layer down:
**extension-function markers**. Instead of writing `SUM(a.val * b.val)`, a
user (or a model-definition helper built on top) writes
`SUM(ddx_contract_mark(a.val * b.val))` — an identity scalar function whose
only job is to survive planning and mark that specific aggregate as a
contraction. Substrait's extension mechanism (`extension_uris` +
`simple_extension_declaration`) gives every custom function a plan-local
anchor, independent of whatever base relational shape it sits inside — this
is mature, standard Substrait usage for custom *functions* specifically
(custom *relation* types are a much less mature corner of Substrait, and
`ddx` never needs one — only tagged functions inside ordinary relations).

**This is verified, not just architecturally plausible.**
`spikes/substrait_ad_marker_spike.py` wraps a contraction's operand in
`ddx_contract_mark(...)` and confirms it survives as a distinguishable
extension-function anchor through a same-engine round-trip *and* a genuine
cross-engine hop: a plan **produced by DataFusion**, containing the marker,
is **consumed and executed correctly by DuckDB**, matching DataFusion's own
result exactly. The reverse direction (DuckDB produces, DataFusion consumes)
deserializes cleanly; execution in that direction wasn't exercised — an
honest scope boundary, not a claim. One nuance: neither engine's producer
emits a fully spec-conformant extension URI for the custom marker (both use
a bare anchor + name), which didn't break either tested engine but is
unverified for a third `[S2]`.

**A recurring cost of this choice, found once already and worth budgeting
for.** v2 is now bounded by whatever relation vocabulary Substrait itself,
and each engine's producer/consumer, actually implement — the same
coverage-gatekeeper pattern v1 lives with for `sqlparser`'s dialect coverage,
recurring one layer up. §4.3's Route rule found a concrete instance: DuckDB's
own optimizer silently mangles a specific idiom before Substrait export. The
resolution there (spike each rule's forward idiom against both engines
before trusting it, verify a workaround rather than wait for an upstream
fix) is the template for handling this class of risk as more rules get
built `[S4]`/`[S5]`.

`ddx-core` v2's dependency stays symmetric with v1's: `substrait` only, no
`datafusion`, no `duckdb`.

### 4.3 The five transpose rules

Each backward step ddx-core emits is **plain, unmarked** Substrait — ddx only
needs markers to recognize the *user's* input, never to tag its own output
(unless a second round of differentiation is wanted over an already-emitted
backward query — open, §4.6).

**Contraction.** Forward, as the user writes it:
```sql
SELECT a.{batch...}, b.{out...}, SUM(ddx_contract_mark(a.val * b.val)) AS val
FROM {a} a JOIN {b} b ON a.j = b.j
GROUP BY a.{batch...}, b.{out...}
```
ddx recognizes the marker inside an `AggregateRel`'s measure and reads the
contracted dim and surviving dims directly off the enclosing `JoinRel`
condition and `GROUP BY` — no separate shape argument, since the plan the
user wrote already contains it. Transpose: given cotangent `C̄`, `Ā = Σ_out
C̄·B` and `B̄ = Σ_batch A·C̄` — both themselves contractions, so the rule's own
backward pass never needs a sixth primitive. Verified against `nn.py`'s
`g2`/`g1`/`g0` and `relational_ad_spike.py`.

**Elementwise.** No marker at all — any projection expression not wrapped in
one of the other markers is, by default, an elementwise map, and its local
derivative is one call into `ddx-core`'s existing v1 `differentiate` on the
same underlying expression type. Transpose: `X̄ = Ȳ · f'(X)`. This is the
literal seam between v1 and v2 — nothing new is built here.

**Reduce, with broadcast as its transpose.** Forward:
`SUM(ddx_reduce_mark(val)) GROUP BY {surviving dims}` — distinguished from
Contraction by the absence of an enclosing join feeding the aggregate.
Transpose: a broadcast join, fanning the cotangent back out to every row
that was summed. Mean is deliberately *not* a separate variant: `nn.py`
always divides by `N` as a separate elementwise step after a plain `SUM`,
and the design follows that pattern rather than threading a global `N`
through the transpose rule.

**Route** (argmax/argmin — needed for max-pooling; not yet used for a
gradient in the prototype, only for reporting accuracy). Forward:
```sql
WITH ranked AS (
  SELECT {group_dims}, {route_dim}, ddx_route_mark(val) AS val,
         ROW_NUMBER() OVER (PARTITION BY {group_dims} ORDER BY val DESC) AS rk
  FROM {input}
)
SELECT {group_dims}, {route_dim}, val FROM ranked WHERE rk = 1
```
Transpose: a scatter — cotangent flows only to the winning row, zero
elsewhere. Verified two ways, both closed by spike:
- *Math*, `spikes/route_ad_spike.py`: machine-exact (0.00e+00) against
  `jax.grad` for a max-pool-style layer, away from ties. At an exact tie the
  conventions genuinely differ — this rule's deterministic first-index
  tiebreak (matching SQL's own `ROW_NUMBER()` ordering) routes the whole
  cotangent to the first winner, while `jax.grad(jnp.max)` splits it evenly
  across every tied entry. Both are standard, defensible conventions; they
  are not the same one, and the rule pins its own rather than claiming JAX
  agreement at ties.
- *Substrait feasibility*, `spikes/duckdb_substrait_window_bug.py`: a plain
  window function as an output column round-trips correctly through DuckDB.
  The *full* top-1-per-group idiom above round-trips **silently wrong** — no
  exception, but `from_substrait` returns every row instead of the filtered
  top-1 rows — because DuckDB's own optimizer rewrites the idiom into an
  `arg_max`-join before Substrait export, and that rewritten form doesn't
  survive the round-trip. This reproduces with no ddx marker involved at
  all — a pre-existing DuckDB bug. DataFusion round-trips the identical
  idiom correctly, isolating this as DuckDB-specific, not a general gap in
  Substrait's window-relation support. A verified two-step workaround —
  round-trip only the window-column computation through Substrait, then
  apply the `rk = 1` filter as a second, plain, engine-native SQL statement
  ddx authors directly — produces the correct result on DuckDB with no need
  to wait on an upstream fix `[S3]`/`[S4]`.

**StopGradient.** An explicit marker, `ddx_stop_gradient(x)`, cutting
cotangent flow past a subexpression. Needed for one specific, easy-to-miss
case: `nn.py`'s softmax subtracts a per-row max purely for numerical
stability before `exp`, which is mathematically a no-op for the gradient
(the standard log-sum-exp identity), but a naive rule walk would — correctly,
per Route's own rule — try to route gradient into that max, which is wrong.
The stop-gradient marker prevents it explicitly rather than relying on the
walker to special-case numerical-stability shifts.

### 4.4 The backward-program emitter and the tape

```rust
pub fn vjp_query(
    &self,
    plan: &Plan,              // a substrait::proto::Plan, already containing
                               // ddx's marker functions — recognized, not built
    wrt: &[RelRef],            // which Source relations to differentiate w.r.t.
) -> Result<BackwardProgram, DiffError>;
```

`vjp_query` first walks the plan once to locate every ddx marker anchor and
build a lightweight annotated-node index (which aggregates are contractions
vs. reduces, where the stop-gradient edges are) — this index is purely an
implementation detail of the walker over the real Substrait plan, the same
relationship v1's `ColRef` has to `sqlparser::ast::Expr`, not a competing IR.
It then walks in reverse topological order, applying each node's transpose
rule and accumulating cotangents.

**Fan-in accumulation is real, not hypothetical.** When a relation feeds more
than one consumer — attention's `X` feeding `Wq`, `Wk`, and `Wv` — each
consumer's contribution is summed via an ordinary elementwise-add step,
verified in `attention_ad_spike.py` (`Xbar = Xq + Xk + Xv`, matching
`jax.grad` to 1e-16). This is why the walk must process a node only once
every one of its consumers has contributed — the standard reverse-mode-AD
scheduling discipline, applied to a relational-node graph instead of a
tensor-op graph.

**The tape** is a sequence of named, materialized relations —
`__ddx_fwd_{node_id}` / `__ddx_bwd_{node_id}` — exactly matching the
projection-boundary contract in §3.5, which already forces pre-activations
to exist as real columns, and `nn.py`'s own `.cache()`/`register_table`
pattern, just auto-named instead of hand-named. Each backward step is a
plain, unmarked Substrait `Plan`, handed to *the engine's own* Substrait
consumer (`from_substrait`, `datafusion-substrait`) rather than converted to
SQL text by `ddx-core` itself — one less thing for ddx to get right per
engine, reusing machinery both target engines already ship. Materialization
wraps that consumer call in ordinary SQL/DataFrame code, e.g. for DuckDB:
```sql
CREATE TEMP TABLE __ddx_bwd_7 AS SELECT * FROM from_substrait($1)
```

```rust
pub struct BackwardProgram {
    pub forward_steps: Vec<(Ident, Plan)>,
    pub backward_steps: Vec<(Ident, Plan)>,
    pub gradients: HashMap<RelRef, Ident>,
}
```

A loss is not a special type — it's simply the plan's terminal
(no-further-consumer) relation, ordinarily a `ddx_reduce_mark`-tagged `SUM`
over an elementwise residual or cross-entropy term, seeded with cotangent
`1.0`.

### 4.5 Worked example

The literal SQL a user writes for `nn.py`'s layer-2 weight gradient, changed
by exactly one thing from the hand-written original — the marker:

```sql
SELECT a.out AS inp, d.out AS out, SUM(ddx_contract_mark(a.val * d.val)) AS val
FROM (SELECT sample, out, val FROM fwd1) a
JOIN delta2 d ON a.sample = d.sample
GROUP BY a.out, d.out
```

This shape is spike-confirmed to round-trip through both engines' Substrait
producers/consumers and execute to the numerically correct contraction. The
concrete, honest migration cost: existing hand-written SQL like `nn.py`'s
needs exactly one function-name change per contraction (`SUM(x)` →
`SUM(ddx_contract_mark(x))`) to opt into v2's automatic differentiation — a
small, mechanical, per-query cost worth stating plainly rather than implying
v2 is a drop-in replacement for hand-written backprop.

### 4.6 What's verified, what's still open

**Verified, machine-exact against `jax.grad`:** the MLP's all six parameter
gradients; attention's four (including the causal mask); Route's math away
from ties. **Verified cross-engine:** the marker-tagging mechanism itself.
**Found and closed, not left as a risk:** the DuckDB Substrait window-idiom
bug (workaround verified, no upstream-fix dependency).

**Genuinely open:**
- The physical fused-contraction operator for BLAS-class performance on
  dense data (§4.1) — not yet spiked.
- Higher-order AD over an already-emitted backward query (differentiating
  ddx's own generated plan a second time) — the backward output is currently
  unmarked by design (§4.3); whether it needs markers for this case is
  undecided.
- The DuckDB Substrait extension is community-maintained, not core, as of
  1.5.4 — an ongoing-maintenance signal to watch, separate from the
  correctness bug already found and worked around.
- Whether a third engine (beyond DataFusion and DuckDB) would tolerate the
  non-spec-conformant extension-URI form both current producers emit is
  untested and only matters if/when a third engine is targeted.

---

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
- **v2-specific: spike each rule's forward idiom against both engines'
  actual Substrait implementations before trusting it** — the coverage
  discipline §4.2 commits to, now a standing test-plan item, not a one-time
  check.

---

## 6. Architecture: monorepo layout & dependency policy

```
ddx/                               (repo; crates published under the ddx-* names)
├── crates/
│   ├── ddx-core/                   # v1 engine — differentiate sqlparser::ast::Expr
│   │                               #   + rewrite_sql; dep: sqlparser only
│   ├── ddx-ad/                      # v2 engine — vjp_query over substrait::proto
│   │                               #   dep: substrait only
│   ├── ddx-datafusion/             # markers + AnalyzerRule (Path B) + ddx_sql helper
│   │                               #   deps: ddx-core, ddx-ad, datafusion
│   └── ddx-duckdb/                 # DuckDB community extension: `ddx('<sql>')` + v2 table fn
├── python/
│   └── ddxdb/                      # PyO3/maturin wheel: rewrite_sql + Context.sql() shim
├── docs/
│   └── design.md                   # this file
├── tests/                          # cross-engine numeric-agreement suites (vs JAX)
├── spikes/                         # runnable evidence for every load-bearing claim (README.md indexes them)
└── future/                         # deferred, not on the critical path
    ├── ddx-duckdb-cpp/             #   C++/cxx.rs hybrid for bare grad() in DuckDB
    └── ddx-pg/                     #   Postgres via pgrx (needs array/XQL support first)
```

`ddx-core` and `ddx-ad` each publish independently, with a single
minimal dependency (`sqlparser`, `substrait`) and no engine crate, so either
can be driven from a new engine without pulling in DataFusion or DuckDB. The
heavy per-engine dependencies are quarantined in the adapter crates.

**`sqlparser` version policy — a real cost of "one IR," paid once.**
`ddx-core`'s public API takes and returns `sqlparser::ast::Expr`, and the
Path B bridge (§3.3) requires `ddx-core` and DataFusion to resolve the
identical `sqlparser` version — a mismatch makes them two unrelated Rust
types and the bridge won't compile. `sqlparser` ships roughly one breaking
release every 1–3 months, and DataFusion adopts each with a lag, while the
DuckDB dialect coverage argument (§3.3) wants the newest release — the two
pulls are in tension. Policy: pin to DataFusion's requirement (the bridge is
a v1 deliverable and a broken bridge is a compile failure; the DuckDB-coverage
cost is bounded, §3.3); re-export it (`pub use sqlparser`) so downstream
consumers can't accidentally link a mismatch; treat a `sqlparser` bump as a
breaking release of `ddx-core`; and if the pins ever must diverge, degrade
the bridge to a string round-trip rather than break it `[G2]`.

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
optional extra. DuckDB: `INSTALL ddx FROM community;` → the `ddx('<sql>')`
table function (v1) and its v2 counterpart. Repo: renamed
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
- **M3 — Relational reverse-mode AD, phase 1: the rules + Substrait
  markers.** Register the extension-function markers
  (`ddx_contract_mark`, `ddx_reduce_mark`, `ddx_route_mark`,
  `ddx_stop_gradient`) and implement the five transpose rules. Clean up
  `nn.py` into the canonical relational-backprop example and regression
  fixture; `spikes/` are the acceptance tests. *Exit:* the rules reproduce
  `jax.grad` on the MLP, attention, and Route fixtures (already machine-exact
  by hand), and markers round-trip through both engines' Substrait
  implementations (already verified).
- **M4 — `vjp` over queries, phase 2: the ML headline.** `vjp_query(plan,
  wrt)` takes a marker-tagged Substrait `Plan` and emits the backward
  program — a sequence of named, materializable `Plan`s (the tape). Stays
  pure-logical; performance is a separate, physical concern (the
  fused-contraction operator, still to spike). Runs first on DataFusion.
  *Exit:* train the `nn.py` MLP with gradients *emitted by* `ddx.vjp`, not
  hand-written, matching the demo and JAX.
- **M5 — DuckDB.** `ddx-duckdb` = the `ddx('<sql>')` table function (v1) plus
  its v2 counterpart, and the `ddxdb` client-side path for DuckDB-python.
  Integrate with duckdb-zarr; run the re-entrancy smoke test. Named tasks,
  not discoveries: the bind-time-schema spike (declare result columns by
  `DESCRIBE`-ing the rewritten query on the inner connection), the DML
  policy decision (SELECT-only by default), and branding the extension
  transitional in its own docs. *Exit:* `grad` works end-to-end via `SELECT
  * FROM ddx('…')` on a real duckdb-zarr dataset, schema/DML behavior
  documented.
- **M5-adjacent spike** (scheduled with M5, not after): the miniature of the
  full C++ hybrid — from an `OptimizerExtension`, serialize one `grad`
  bound expression out to `ddx-core`, differentiate, and rebuild one bound
  derivative expression back in with correct `ColumnBinding` indices and
  catalog entries. *Exit:* a yes/no on tractability in days, so the full
  extension is schedulable on demand rather than a multi-week unknown,
  without sitting on duckdb-zarr's critical path.
- **M6 — Math roadmap & hardening.** Extend v1's rule set (§3.6), cross-engine
  equivalence vs. JAX, dialect canonicalization table, docs. Plus: the
  expression-swell size/latency benchmark (§3.7), and the convention-pinning
  tests for NULL-folding, kinks, and domain edges (§5).
- **Future, demand-driven:** the physical fused-contraction operator; more
  v2 architectures (conv, more normalization variants); the C++/cxx.rs
  hybrid for bare `grad()` in DuckDB; `ddx-pg`.

---

## 9. Open questions

- **DuckDB ergonomics** — `SELECT * FROM ddx('…')` is the accepted v1 shape,
  pending a second opinion from the other duckdb-zarr maintainer before the
  extension's surface locks.
- **Custom rule richness** — ship unary custom differentiation rules only in
  v1, or binary/n-ary too? Unary is easy today (the engine already
  dispatches on function name); richer rules are a bigger trait, likely
  post-v1 regardless.
- **The C++ hybrid's trigger condition** — deferred by default (§3.4), but if
  the syntactic ambiguity guards prove messier than expected during M0, that
  shifts the balance toward accelerating it, since the bound path makes
  those guards unnecessary by construction. Decide if that tripwire fires,
  not preemptively.
- **Higher-order AD over an emitted v2 backward plan** — genuinely
  undecided (§4.6); revisit once M4 has a working single-order emitter to
  reason about concretely.

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
(`spikes/substrait_ad_marker_spike.py`).** Before this spike, "Substrait +
markers" was an architectural argument; after it, a checked claim. A
DataFusion-produced, marker-tagged plan is consumed and executed correctly
by DuckDB, matching DataFusion's own result exactly — the actual portability
claim, not merely "an engine doesn't corrupt its own plan." One honest gap:
the reverse direction (DuckDB produces, DataFusion consumes) deserializes
but wasn't executed. → §4.2.

**S3 — Route's math verified against `jax.grad`
(`spikes/route_ad_spike.py`).** Machine-exact away from ties; a genuine,
now-pinned convention divergence at exact ties (first-index routing vs.
JAX's tie-splitting) — not a bug, but not something to claim agreement on
without saying so. → §4.3.

**S4 — A real DuckDB Substrait bug found, isolated, and worked around
(`spikes/duckdb_substrait_window_bug.py`).** The Route rule's forward idiom
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
   Decision Log `S4` (`spikes/duckdb_substrait_window_bug.py`) should be
   filed against this project.
