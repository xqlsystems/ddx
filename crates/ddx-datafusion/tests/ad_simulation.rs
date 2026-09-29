// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Simulation / property tests for query-level AD (v2): `grad`, `vjp`, `run`
//! and `grad(loss, table.column)` in SQL, on generated loss queries.
//!
//! The hand-written v2 tests check a gradient on a fixture somebody thought
//! of. This file generates the queries: random compositions of the relational
//! primitives design.md §4.3 gives a rule each (map, select, broadcast,
//! reduce), over small random tables, written the way a user writes SQL — CTEs,
//! joins on shared dims, grouped SUM/AVG/MAX/MIN, filters, semi-joins, a rank
//! filter, a softmax with and without its shift stopped. Each is then held to
//! two kinds of property.
//!
//! # The oracle: a finite difference the engine computes
//!
//! A gradient is a claim about the loss, and the loss is a query DataFusion can
//! run with no ddx involved. So the independent check is a finite difference of
//! that query, the parameters perturbed in memory and the tables re-registered:
//! `⟨∇L, d⟩ ≈ (L(θ + h·d) − L(θ − h·d)) / 2h` for random directions `d`, and for
//! a few single entries. It shares nothing with ddx but the engine's
//! arithmetic, so it cannot share a misconception with the rules.
//!
//! A finite difference is only a derivative where the loss is smooth, and the
//! generator makes kinks on purpose (ReLU, `greatest`, `MAX`, a rank filter, a
//! filter on a value). Points are screened rather than compared blindly: the
//! second difference must shrink like `h²` when `h` halves (at a kink it only
//! halves), and the central differences at `h` and `h/2` must agree before
//! their Richardson extrapolation is compared. A screened point is counted as
//! skipped, and the retention rate is asserted, so a generator that drifted
//! into kinks everywhere would fail loudly instead of passing vacuously.
//!
//! # Metamorphic relations: ddx against itself
//!
//! The rest need no oracle for the derivative, only that two programs agree:
//!
//! - **calculus**: `∇(cL) = c∇L`, `∇(L + c) = ∇L`, `∇(L²) = 2L∇L`,
//!   `∇sin(L) = cos(L)∇L`, `∇(L·L) = 2L∇L` with the loss CTE read twice,
//!   `∇(L·sg(L)) = L∇L`, `∇(L + sg(L)) = ∇L`.
//! - **JAX's identities**: `grad` is `vjp` seeded with 1; `vjp` of a relation
//!   `R` with cotangent `c` is `grad` of `Σ R·c`; `vjp` is linear in `c`.
//! - **invariance**: the gradient does not depend on how the query is spelled
//!   (CTEs or inline subqueries, optimized or unoptimized plan), on row order,
//!   on how many partitions the engine uses, on the order or case of the `wrt`
//!   list, or on which other tables are differentiated alongside.
//! - **the program contract**: a program built once runs on new values of the
//!   same tables (a training loop) and gives the fresh program's gradient;
//!   repeated dims are refused by the checks, never answered; after `run` only
//!   the value and the gradients remain, after `release` nothing does, and a
//!   user's tables are never touched.
//! - **the SQL surface**: `grad(loss, t.val)` in SQL is the program's gradient,
//!   and an SGD step written as a join is `θ − lr·∇L`.
//! - **shape**: one gradient row per table row, keyed by the table's dims,
//!   DOUBLE, NULL exactly where the value is NULL.
//!
//! A refusal (`NotImplemented`, `NotScalar`, …) is always allowed — the
//! generator reaches past what ddx supports on purpose — and is tallied by
//! kind. What is never allowed: a panic, an `Internal` error, a program ddx
//! accepted that the engine that produced the plan cannot run, and above all a
//! number that is wrong.
//!
//! # Soak mode
//!
//! [`soak_v2_query_ad`] is an `#[ignore]`d variant that runs every property on
//! fresh seeds for a wall-clock budget, with the same knobs and log format as
//! ddx-core's soak, so `.github/scripts/report_fuzz_findings.sh` triages both:
//!
//! ```text
//! DDX_SOAK_SECS=300  DDX_SOAK_BASE=0  DDX_SOAK_LOG=/path/to/soak.log \
//!   cargo test -p ddx-datafusion --test ad_simulation --release \
//!   -- --ignored --nocapture soak_v2_query_ad
//! ```
//!
//! Every failure is reported with its seed; `DDX_SOAK_BASE=<seed>` replays it
//! as iteration 0, and `DDX_V2_SEED=<seed>` with `replay_one_seed` prints the
//! generated SQL and runs only that case.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, AsArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use datafusion::arrow::datatypes::{
    DataType, Field, Float32Type, Float64Type, Int32Type, Int64Type, Schema, UInt64Type,
};
use datafusion::datasource::MemTable;
use datafusion::error::DataFusionError;
use datafusion::execution::SessionStateBuilder;
use datafusion::optimizer::{Optimizer, OptimizerRule};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_substrait::logical_plan::consumer::from_substrait_plan;
use datafusion_substrait::logical_plan::producer::to_substrait_plan;
use ddx_core::test_utils::{gen_expr, seeded, Failures, Rng};
use ddx_datafusion::ad::{self, AdError, BackwardProgram, ColumnRef};

#[path = "ad_simulation/mutate.rs"]
mod mutate;

// ---------------------------------------------------------------------------
// Tables.
// ---------------------------------------------------------------------------

/// The dims a generated relation can have, and how many values each takes.
/// Every table and every generated relation is keyed by a subset of these, and
/// two relations join on the dims they share, so the generator never has to
/// invent a join condition.
const DIMS: &[&str] = &["s", "i", "j"];

/// Parameter tables (the `wrt` candidates) and constant data tables.
const PARAMS: &[(&str, &[&str])] = &[("w", &["i", "j"]), ("b", &["j"]), ("u", &["i"])];
const DATA: &[(&str, &[&str])] = &[("x", &["s", "i"]), ("y", &["s", "j"]), ("m", &["i", "j"])];

/// How every dim of a case is stored. A dim is only ever compared for
/// equality, so its type must not change a gradient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyType {
    Int64,
    Int32,
    Utf8,
}

impl KeyType {
    fn data_type(self) -> DataType {
        match self {
            KeyType::Int64 => DataType::Int64,
            KeyType::Int32 => DataType::Int32,
            KeyType::Utf8 => DataType::Utf8,
        }
    }

    /// Keys as an array; [`NULL_KEY`] is a NULL.
    fn array(self, keys: Vec<i64>) -> ArrayRef {
        let keys = keys.into_iter().map(|k| (k != NULL_KEY).then_some(k));
        match self {
            KeyType::Int64 => Arc::new(Int64Array::from(keys.collect::<Vec<_>>())),
            KeyType::Int32 => Arc::new(Int32Array::from(
                keys.map(|k| k.map(|k| k as i32)).collect::<Vec<_>>(),
            )),
            KeyType::Utf8 => Arc::new(StringArray::from(
                keys.map(|k| k.map(|k| k.to_string())).collect::<Vec<_>>(),
            )),
        }
    }

    /// A key as a SQL literal.
    fn lit(self, k: u64) -> String {
        match self {
            KeyType::Utf8 => format!("'{k}'"),
            _ => k.to_string(),
        }
    }
}

/// Set by the soak, which alone runs `big` mode (see [`Modes::draw`]).
static SOAKING: AtomicBool = AtomicBool::new(false);

/// A NULL key, in a data table only: a parameter's dims identify its rows.
const NULL_KEY: i64 = i64::MIN;

/// One table, held in memory so it can be perturbed and re-registered.
#[derive(Clone, Debug)]
struct Table {
    name: String,
    dims: Vec<String>,
    keys: Vec<Vec<i64>>,
    vals: Vec<Option<f64>>,
    param: bool,
    key_type: KeyType,
    /// How its rows are split into partitions of batches when registered:
    /// the engine must not care, and more than one batch or partition is
    /// what a real table has.
    chunks: Vec<Vec<usize>>,
}

impl Table {
    fn unique(&self) -> bool {
        self.keys.iter().collect::<BTreeSet<_>>().len() == self.keys.len()
    }

    fn batch(&self) -> RecordBatch {
        let mut fields: Vec<Field> = self
            .dims
            .iter()
            .map(|d| Field::new(d, self.key_type.data_type(), true))
            .collect();
        fields.push(Field::new("val", DataType::Float64, true));
        let mut cols: Vec<ArrayRef> = (0..self.dims.len())
            .map(|k| {
                self.key_type
                    .array(self.keys.iter().map(|r| r[k]).collect())
            })
            .collect();
        cols.push(Arc::new(Float64Array::from(self.vals.clone())));
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).expect("a valid batch")
    }

    fn register(&self, ctx: &SessionContext) -> Result<(), DataFusionError> {
        let batch = self.batch();
        let mut partitions = Vec::new();
        let mut at = 0;
        for sizes in &self.chunks {
            let mut part = Vec::new();
            for &n in sizes {
                let n = n.min(batch.num_rows() - at);
                part.push(batch.slice(at, n));
                at += n;
            }
            partitions.push(part);
        }
        if at < batch.num_rows() || partitions.is_empty() {
            partitions.push(vec![batch.slice(at, batch.num_rows() - at)]);
        }
        let table = MemTable::try_new(batch.schema(), partitions)?;
        ctx.deregister_table(self.name.as_str())?;
        ctx.register_table(self.name.as_str(), Arc::new(table))?;
        Ok(())
    }
}

/// Domain sizes for one case's dims.
#[derive(Clone, Debug)]
struct Domains(BTreeMap<&'static str, i64>);

impl Domains {
    fn size(&self, d: &str) -> i64 {
        self.0[d]
    }
    fn rows(&self, dims: &[&'static str]) -> i64 {
        dims.iter().map(|d| self.size(d)).product()
    }
}

fn gen_table(
    rng: &mut Rng,
    dom: &Domains,
    name: &str,
    dims: &[&'static str],
    param: bool,
    key_type: KeyType,
) -> Table {
    // Every dim tuple, in a shuffled order: the gradient is keyed by dims and
    // must not care what order the rows arrived in.
    let mut keys: Vec<Vec<i64>> = vec![vec![]];
    for d in dims {
        keys = keys
            .into_iter()
            .flat_map(|k| {
                (0..dom.size(d)).map(move |v| {
                    let mut k = k.clone();
                    k.push(v);
                    k
                })
            })
            .collect();
    }
    // A parameter table sometimes has an orphan row, whose dims nothing else
    // has: an inner join drops it, and its gradient must be exactly 0.
    if param && !dims.is_empty() && rng.below(100) < 25 {
        let mut k: Vec<i64> = dims
            .iter()
            .map(|d| rng.below(dom.size(d) as u64) as i64)
            .collect();
        let at = rng.below(k.len() as u64) as usize;
        k[at] = 90 + at as i64;
        keys.push(k);
    }
    // Constant data may repeat a key: a join then multiplies rows, which is
    // data, not a promise ddx relies on (only a wrt table's dims are checked).
    if !param && !keys.is_empty() && rng.below(100) < 15 {
        let r = rng.below(keys.len() as u64) as usize;
        keys.push(keys[r].clone());
    }
    shuffle(rng, &mut keys);
    let vals = keys
        .iter()
        .map(|_| {
            // Values are occasionally NULL: the NULL-row convention for
            // parameters, three-valued logic for data.
            if rng.below(100) < null_pct() {
                None
            } else {
                Some(round6(rng.range(-1.0, 1.0)))
            }
        })
        .collect();
    Table {
        name: name.to_string(),
        dims: dims.iter().map(|d| d.to_string()).collect(),
        keys,
        vals,
        param,
        key_type,
        chunks: Vec::new(),
    }
}

/// Percent of parameter values generated NULL (`DDX_V2_NULL_PCT`, default
/// 4). Setting it to 0 replays a case minus its NULLs, which is how a
/// failure is attributed to them.
fn null_pct() -> u64 {
    env_u64("DDX_V2_NULL_PCT", 4)
}

/// Values with few significant digits, so a failure report is readable and
/// distinct parameters never tie by accident at full precision.
fn round6(v: f64) -> f64 {
    (v * 1e6).round() / 1e6 + 1e-9 * (v * 7919.0).sin()
}

fn shuffle<T>(rng: &mut Rng, xs: &mut [T]) {
    for i in (1..xs.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        xs.swap(i, j);
    }
}

// ---------------------------------------------------------------------------
// Modes: what a case's data looks like, beyond the ordinary.
// ---------------------------------------------------------------------------

/// Extreme values a case can hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Extreme {
    /// Parameters thirty times larger: exp and softmax near overflow.
    Huge,
    /// Parameters around 1e-160: products underflow to subnormals and zero.
    Tiny,
    /// Some values are `-0.0`.
    NegZero,
    /// A data value is NaN, which is not NULL: aggregates do not skip it.
    NanData,
    /// A data value is infinite.
    InfData,
}

/// A case's modes. Each is drawn from its own generator seeded by the case's
/// seed and applied after the case is generated, so a seed with no mode
/// replays as it did before modes existed. `DDX_V2_MODES=ties,nulls,
/// extreme,big` forces the listed modes on every case instead.
#[derive(Clone, Copy, Debug, Default)]
struct Modes {
    /// Parameters drawn from four values, so MAX, MIN and rankings tie
    /// exactly: the conventions the finite difference cannot see.
    ties: bool,
    /// About a third of all values NULL, and some NULL keys in data.
    nulls: bool,
    extreme: Option<Extreme>,
    /// Domains ten times larger: thousands of rows per relation.
    big: bool,
}

impl Modes {
    fn draw(seed: u64) -> (Modes, Rng) {
        let mut r = seeded(seed, 0xD47A_5EED);
        let extreme = |r: &mut Rng| {
            *r.pick(&[
                Extreme::Huge,
                Extreme::Tiny,
                Extreme::NegZero,
                Extreme::NanData,
                Extreme::InfData,
            ])
        };
        let modes = match std::env::var("DDX_V2_MODES") {
            Ok(names) => {
                let on = |n: &str| names.split(',').any(|x| x.trim() == n);
                Modes {
                    ties: on("ties"),
                    nulls: on("nulls"),
                    extreme: on("extreme").then(|| extreme(&mut r)),
                    big: on("big"),
                }
            }
            Err(_) => Modes {
                ties: r.below(100) < 15,
                nulls: null_pct() > 0 && r.below(100) < 10,
                extreme: (r.below(100) < 8).then(|| extreme(&mut r)),
                // Drawn either way, so seeds replay alike, but only a soak
                // uses it: a big case is slow, and a bug found only in many
                // partitions need not be deterministic, which a PR gate
                // must be.
                big: r.below(100) < 4 && SOAKING.load(Ordering::Relaxed),
            },
        };
        (modes, r)
    }

    fn names(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.ties {
            v.push("ties");
        }
        if self.nulls {
            v.push("nulls");
        }
        if let Some(e) = self.extreme {
            v.push(match e {
                Extreme::Huge => "huge",
                Extreme::Tiny => "tiny",
                Extreme::NegZero => "-0.0",
                Extreme::NanData => "NaN data",
                Extreme::InfData => "inf data",
            });
        }
        if self.big {
            v.push("big");
        }
        v
    }
}

/// The four values a parameter takes in `ties` mode.
const TIE_VALUES: &[f64] = &[-0.5, 0.25, 0.75, 1.0];

/// Apply `modes` to freshly generated tables, drawing from `r` only.
fn apply_modes(tables: &mut [Table], modes: &Modes, r: &mut Rng) {
    for t in tables.iter_mut() {
        // Batches and partitions, always.
        let n = t.keys.len();
        let parts = 1 + r.below(3) as usize;
        let mut left = n;
        t.chunks = (0..parts)
            .map(|p| {
                let share = if p + 1 == parts {
                    left
                } else {
                    r.below(left as u64 + 1) as usize
                };
                left -= share;
                let mut sizes = Vec::new();
                let mut rem = share;
                while rem > 0 {
                    let k = 1 + r.below(rem as u64) as usize;
                    sizes.push(k);
                    rem -= k;
                }
                sizes
            })
            .collect();
        if modes.ties && t.param {
            for v in t.vals.iter_mut().flatten() {
                *v = *r.pick(TIE_VALUES);
            }
        }
        if modes.nulls {
            for v in t.vals.iter_mut() {
                if r.below(100) < 30 {
                    *v = None;
                }
            }
            if !t.param {
                for k in t.keys.iter_mut().flat_map(|k| k.iter_mut()) {
                    if r.below(100) < 10 {
                        *k = NULL_KEY;
                    }
                }
            }
        }
        match modes.extreme {
            Some(Extreme::Huge) if t.param => t.vals.iter_mut().flatten().for_each(|v| *v *= 30.0),
            Some(Extreme::Tiny) if t.param => {
                t.vals.iter_mut().flatten().for_each(|v| *v *= 1e-160)
            }
            Some(Extreme::NegZero) => {
                for v in t.vals.iter_mut().flatten() {
                    if r.below(100) < 30 {
                        *v = -0.0;
                    }
                }
            }
            Some(Extreme::NanData | Extreme::InfData) if !t.param && !t.vals.is_empty() => {
                let k = r.below(t.vals.len() as u64) as usize;
                t.vals[k] = Some(if modes.extreme == Some(Extreme::NanData) {
                    f64::NAN
                } else {
                    f64::INFINITY
                });
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// The generator: a DAG of relations, each one CTE with dims and one value `v`.
// ---------------------------------------------------------------------------

/// One generated relation. Its body refers to earlier relations as `§n§`,
/// which [`Case::render`] turns into a CTE name or an inline subquery.
#[derive(Clone, Debug)]
struct Node {
    body: String,
    /// Whether its dims identify its rows. Not so after a union, or over
    /// data that repeats a key; `vjp` of such a relation has no cotangent
    /// keyed by dims, so the vjp relation checks skip it.
    unique: bool,
    dims: Vec<&'static str>,
    /// The parameter tables it reads, other than through a stop-gradient.
    reads: BTreeSet<String>,
    /// Kinds of primitive it (or anything beneath it) uses, for coverage.
    kinds: BTreeSet<&'static str>,
    /// Whether a later relation may read it. A softmax with its shift
    /// stopped keeps its exponentials and their sum to itself: only the
    /// ratio is shift-invariant, so a gradient through them is, by
    /// stop-gradient's definition, not the loss's derivative.
    shared: bool,
}

#[derive(Clone, Debug)]
struct Case {
    modes: Modes,
    tables: Vec<Table>,
    nodes: Vec<Node>,
    root: usize,
    /// The loss head over the root relation, an ungrouped aggregate of `v`.
    head: String,
    /// The parameter tables differentiated with respect to.
    wrt: Vec<String>,
}

/// Unary maps: smooth, bounded on the generator's ranges, and each on a rule
/// ddx-core has. The last few have kinks on purpose.
const UNARY: &[&str] = &[
    "tanh({v})",
    "sin({v})",
    "({v} * {v})",
    "(0.5 * {v} + 0.25)",
    "exp(0.5 * {v})",
    "sqrt({v} * {v} + 1.0)",
    "ln({v} * {v} + 1.0)",
    "atan({v})",
    "(1.0 / (1.0 + exp(-{v})))",
    "power({v}, 3)",
    "cos({v})",
    "CAST({v} AS DOUBLE)",
    "(-{v})",
    "({v} / 2.0)",
    "power(2.0, {v})",
    // A stop-gradient the value cannot see: its primal and its gradient are
    // both `v`'s, so the finite difference still applies.
    "({v} + 0.0 * ddx_stop_gradient({v}))",
    "(ddx_stop_gradient({v}) * 0.0 + {v})",
    "CASE WHEN {v} > 0 THEN {v} ELSE 0.1 * {v} END",
    "greatest({v}, -0.3)",
    "least({v}, 0.4)",
    "abs({v})",
];

/// Does a map expression have a kink: a branch, a clamp, or `abs`?
fn is_kinked(f: &str) -> bool {
    let f = f.to_ascii_lowercase();
    ["case ", "greatest(", "least(", "abs("]
        .iter()
        .any(|k| f.contains(k))
}

const BINARY: &[&str] = &[
    "({a} + {b})",
    "({a} * {b})",
    "({a} - {b})",
    "({a} * tanh({b}))",
    "({a} / (1.0 + {b} * {b}))",
    "greatest({a}, {b})",
];

/// Token-aware rename of `x` and `y` in ddx-core's generated scalar text.
fn rename_xy(text: &str, x: &str, y: &str) -> String {
    let b = text.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_alphabetic() || b[i] == b'_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            match &text[start..i] {
                "x" => out.push_str(x),
                "y" => out.push_str(y),
                w => out.push_str(w),
            }
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

fn list(dims: &[&str], prefix: &str) -> String {
    dims.iter()
        .map(|d| format!("{prefix}{d}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `d, ` for a select list with dims, or nothing.
fn lead(dims: &[&str], prefix: &str) -> String {
    if dims.is_empty() {
        String::new()
    } else {
        format!("{}, ", list(dims, prefix))
    }
}

fn group_by(dims: &[&str]) -> String {
    if dims.is_empty() {
        String::new()
    } else {
        format!(" GROUP BY {}", list(dims, ""))
    }
}

/// `a JOIN b ON shared` or a cross join when nothing is shared.
fn join_clause(a: usize, b: usize, shared: &[&str]) -> String {
    if shared.is_empty() {
        format!("§{a}§ a CROSS JOIN §{b}§ b")
    } else {
        let on: Vec<String> = shared.iter().map(|d| format!("a.{d} = b.{d}")).collect();
        format!("§{a}§ a JOIN §{b}§ b ON {}", on.join(" AND "))
    }
}

struct Gen<'r> {
    /// The most rows a join may produce.
    cap: i64,
    key_type: KeyType,
    rng: &'r mut Rng,
    dom: Domains,
    tables: Vec<Table>,
    nodes: Vec<Node>,
}

impl Gen<'_> {
    fn push(
        &mut self,
        body: String,
        dims: Vec<&'static str>,
        reads: BTreeSet<String>,
        kinds: BTreeSet<&'static str>,
    ) -> usize {
        self.push_u(body, dims, reads, kinds, true)
    }

    fn push_u(
        &mut self,
        body: String,
        dims: Vec<&'static str>,
        reads: BTreeSet<String>,
        kinds: BTreeSet<&'static str>,
        unique: bool,
    ) -> usize {
        self.nodes.push(Node {
            body,
            unique,
            dims,
            reads,
            kinds,
            shared: true,
        });
        self.nodes.len() - 1
    }

    /// Push a relation derived from `c` row by row: its dims, reads,
    /// uniqueness and kinds, plus `kind`.
    fn derive(&mut self, c: usize, body: String, kind: &'static str) -> usize {
        let n = &self.nodes[c];
        let (dims, reads, mut kinds, unique) =
            (n.dims.clone(), n.reads.clone(), n.kinds.clone(), n.unique);
        kinds.insert(kind);
        self.push_u(body, dims, reads, kinds, unique)
    }

    fn node(&mut self, depth: u32) -> usize {
        if depth == 0 || self.rng.below(100) < 18 {
            return self.leaf();
        }
        if !self.nodes.is_empty() && self.rng.below(100) < 10 {
            // Reuse: a relation read twice (attention's X feeding Q, K and V).
            let k = self.rng.below(self.nodes.len() as u64) as usize;
            if let Some(k) = (0..=k).rev().find(|&i| self.nodes[i].shared) {
                return k;
            }
            return self.leaf();
        }
        match self.rng.below(100) {
            0..=19 => self.map(depth),
            20..=39 => self.join(depth),
            40..=59 => self.reduce(depth),
            60..=66 => self.filter(depth),
            67..=73 => self.rank(depth),
            74..=80 => self.softmax(depth),
            81..=86 => self.semi(depth),
            87..=91 => self.top_k(depth),
            92..=96 => self.union(depth),
            _ => self.distinct(depth),
        }
    }

    fn leaf(&mut self) -> usize {
        let param = self.rng.below(100) < 65;
        let (name, dims) = if param {
            *self.rng.pick(PARAMS)
        } else {
            *self.rng.pick(DATA)
        };
        let dims: Vec<&'static str> = dims.to_vec();
        let mut reads = BTreeSet::new();
        if param {
            reads.insert(name.to_string());
        }
        let unique = self
            .tables
            .iter()
            .find(|t| t.name == name)
            .is_some_and(Table::unique);
        let value = if !param && self.rng.below(100) < 20 {
            // Integer-typed data: DataFusion then does mixed arithmetic.
            "CAST(val * 10 AS BIGINT)"
        } else {
            "val"
        };
        let body = format!("SELECT {}{value} AS v FROM {name}", lead(&dims, ""));
        self.push_u(body, dims, reads, BTreeSet::from(["read"]), unique)
    }

    fn map(&mut self, depth: u32) -> usize {
        let c = self.node(depth - 1);
        let f = match self.rng.below(100) {
            // Occasionally, ddx-core's own generator: the whole v1 grammar,
            // domain edges and all. A NaN loss is skipped, not compared.
            0..=9 => rename_xy(&gen_expr(self.rng, 2), "{v}", "0.7"),
            // A constant subquery over data, as a scalar.
            10..=14 => format!(
                "({{v}} * (SELECT {}(val) FROM {}))",
                self.rng.pick(&["AVG", "MAX", "SUM"]),
                self.rng.pick(DATA).0
            ),
            15..=18 => "COALESCE({v}, 0.25)".to_string(),
            _ => self.rng.pick(UNARY).to_string(),
        };
        let body = format!(
            "SELECT {}{} AS v FROM §{c}§ c",
            lead(&self.nodes[c].dims, ""),
            f.replace("{v}", "v")
        );
        let n = self.derive(c, body, "map");
        if is_kinked(&f) {
            self.nodes[n].kinds.insert("kink");
        }
        n
    }

    fn join(&mut self, depth: u32) -> usize {
        let a = self.node(depth - 1);
        let b = self.node(depth - 1);
        let (na, nb) = (self.nodes[a].clone(), self.nodes[b].clone());
        let shared: Vec<&'static str> = na
            .dims
            .iter()
            .filter(|d| nb.dims.contains(d))
            .copied()
            .collect();
        let only_b: Vec<&'static str> = nb
            .dims
            .iter()
            .filter(|d| !na.dims.contains(d))
            .copied()
            .collect();
        let mut dims = na.dims.clone();
        dims.extend(&only_b);
        dims.sort_by_key(|d| DIMS.iter().position(|x| x == d));
        if self.dom.rows(&dims) > self.cap {
            return a;
        }
        let select: Vec<String> = dims
            .iter()
            .map(|d| {
                if na.dims.contains(d) {
                    format!("a.{d}")
                } else {
                    format!("b.{d}")
                }
            })
            .collect();
        let mut kinds: BTreeSet<&'static str> = na.kinds.union(&nb.kinds).copied().collect();
        // A left join keeps every row of `a`; `b`'s side may be NULL, which
        // COALESCE turns back into a number. Only on shared dims, so the
        // result's dims are `a`'s.
        let left = !shared.is_empty() && only_b.is_empty() && self.rng.below(100) < 15;
        let op = if left {
            kinds.insert("left-join");
            "({a} + COALESCE({b}, 0.5))".to_string()
        } else if self.rng.below(100) < 8 {
            kinds.insert("case");
            "CASE WHEN {b} > 0 THEN {a} ELSE 0.5 * {a} END".to_string()
        } else {
            self.rng.pick(BINARY).to_string()
        };
        if is_kinked(&op) {
            kinds.insert("kink");
        }
        let op = op.replace("{a}", "a.v").replace("{b}", "b.v");
        let mut from = join_clause(a, b, &shared);
        if left {
            from = from.replacen(" JOIN ", " LEFT JOIN ", 1);
        }
        // A constant condition in the join: select, on the join itself.
        if !shared.is_empty() && self.rng.below(100) < 15 {
            let d = *self.rng.pick(&shared);
            let k = self.key_type.lit(self.rng.below(self.dom.size(d) as u64));
            write!(from, " AND a.{d} <> {k}").unwrap();
        }
        let body = format!(
            "SELECT {}{op} AS v FROM {from}",
            if select.is_empty() {
                String::new()
            } else {
                format!("{}, ", select.join(", "))
            }
        );
        let reads = na.reads.union(&nb.reads).cloned().collect();
        kinds.insert(if shared.is_empty() { "cross" } else { "join" });
        self.push_u(body, dims, reads, kinds, na.unique && nb.unique)
    }

    fn reduce(&mut self, depth: u32) -> usize {
        let c = self.node(depth - 1);
        let n = self.nodes[c].clone();
        if n.dims.is_empty() {
            return c;
        }
        let mut keep: Vec<&'static str> = n
            .dims
            .iter()
            .filter(|_| self.rng.below(2) == 0)
            .copied()
            .collect();
        if keep.len() == n.dims.len() {
            keep.pop();
        }
        let (agg, kind) = match self.rng.below(100) {
            0..=29 => ("SUM(v)", "sum"),
            30..=41 => ("AVG(v)", "avg"),
            42..=51 => ("MAX(v)", "max"),
            52..=58 => ("MIN(v)", "min"),
            59..=65 => ("SUM(v * v)", "sum"),
            66..=71 => ("SUM(v) / COUNT(*)", "count"),
            72..=77 => ("AVG(tanh(v))", "avg"),
            78..=83 => ("SUM(v) * MAX(v)", "max"),
            // The same aggregate twice: saved once, one shared cotangent.
            84..=88 => ("SUM(v) + 0.5 * SUM(v)", "dup-aggregate"),
            89..=93 => ("AVG(v) - MIN(v)", "min"),
            _ => ("SUM(v) / COUNT(v)", "count"),
        };
        let having = match self.rng.below(100) {
            0..=7 if !keep.is_empty() => " HAVING COUNT(*) >= 1".to_string(),
            8..=12 if !keep.is_empty() => {
                format!(" HAVING SUM(v) > {:.2}", self.rng.range(-1.5, 0.0))
            }
            _ => String::new(),
        };
        let mut kinds = n.kinds.clone();
        kinds.insert(kind);
        if !having.is_empty() {
            kinds.insert("having");
        }
        let body = format!(
            "SELECT {}{agg} AS v FROM §{c}§ c{}{having}",
            lead(&keep, ""),
            group_by(&keep)
        );
        self.push_u(body, keep, n.reads.clone(), kinds, true)
    }

    fn filter(&mut self, depth: u32) -> usize {
        let c = self.node(depth - 1);
        let dims = self.nodes[c].dims.clone();
        let (cond, kind) = if !dims.is_empty() && self.rng.below(100) < 60 {
            let d = *self.rng.pick(&dims);
            let k = self.key_type.lit(self.rng.below(self.dom.size(d) as u64));
            if self.rng.below(2) == 0 {
                (format!("{d} <> {k}"), "filter")
            } else {
                (format!("{d} <= {k}"), "filter")
            }
        } else {
            // A filter on a value: select at a point that moves with θ. The
            // finite difference is screened wherever a row sits on the edge.
            (
                format!("v > {:.3}", self.rng.range(-0.8, 0.2)),
                "value-filter",
            )
        };
        let body = format!("SELECT * FROM §{c}§ c WHERE {cond}");
        self.derive(c, body, kind)
    }

    /// ORDER BY … LIMIT: a top-k by value, broken by every dim.
    fn top_k(&mut self, depth: u32) -> usize {
        let c = self.node(depth - 1);
        let dims = self.nodes[c].dims.clone();
        let order = if dims.is_empty() {
            "v DESC".to_string()
        } else {
            format!("v DESC, {}", list(&dims, ""))
        };
        let body = format!(
            "SELECT * FROM §{c}§ c ORDER BY {order} LIMIT {}",
            1 + self.rng.below(3)
        );
        self.derive(c, body, "limit")
    }

    /// UNION ALL of two relations with the same dims: rows can then repeat
    /// their dims, so it is reduced right away.
    fn union(&mut self, depth: u32) -> usize {
        let a = self.node(depth - 1);
        let b = self.node(depth - 1);
        let (na, nb) = (self.nodes[a].clone(), self.nodes[b].clone());
        if na.dims != nb.dims {
            return a;
        }
        let dims = na.dims.clone();
        let mut kinds: BTreeSet<&'static str> = na.kinds.union(&nb.kinds).copied().collect();
        kinds.insert("union");
        let reads: BTreeSet<String> = na.reads.union(&nb.reads).cloned().collect();
        let u = self.push_u(
            format!(
                "SELECT {d}v FROM §{a}§ a UNION ALL SELECT {d}v FROM §{b}§ b",
                d = lead(&dims, "")
            ),
            dims.clone(),
            reads.clone(),
            kinds.clone(),
            false,
        );
        kinds.insert("sum");
        self.push_u(
            format!(
                "SELECT {}SUM(v) AS v FROM §{u}§ c{}",
                lead(&dims, ""),
                group_by(&dims)
            ),
            dims,
            reads,
            kinds,
            true,
        )
    }

    /// SELECT DISTINCT over a varied value: a GROUP BY on it.
    fn distinct(&mut self, depth: u32) -> usize {
        let c = self.node(depth - 1);
        let body = format!(
            "SELECT DISTINCT {}v FROM §{c}§ c",
            lead(&self.nodes[c].dims, "")
        );
        self.derive(c, body, "distinct")
    }

    fn semi(&mut self, depth: u32) -> usize {
        let c = self.node(depth - 1);
        let dims = self.nodes[c].dims.clone();
        let Some(&d) = dims
            .iter()
            .find(|d| DATA.iter().any(|(_, dd)| dd.contains(d)))
        else {
            return c;
        };
        let (data, _) = DATA.iter().find(|(_, dd)| dd.contains(&d)).unwrap();
        let body = if self.rng.below(3) == 0 {
            format!("SELECT * FROM §{c}§ c WHERE EXISTS (SELECT 1 FROM {data} q WHERE q.{d} = c.{d} AND q.val < 0)")
        } else {
            format!(
                "SELECT * FROM §{c}§ c WHERE {d} {} (SELECT {d} FROM {data} WHERE val > {:.2})",
                if self.rng.below(4) == 0 {
                    "NOT IN"
                } else {
                    "IN"
                },
                self.rng.range(-0.5, 0.5)
            )
        };
        self.derive(c, body, "semi")
    }

    fn rank(&mut self, depth: u32) -> usize {
        let c = self.node(depth - 1);
        let dims = self.nodes[c].dims.clone();
        if dims.is_empty() {
            return c;
        }
        let part: Vec<&'static str> = dims
            .iter()
            .filter(|_| self.rng.below(2) == 0)
            .copied()
            .collect();
        let rest: Vec<&'static str> = dims.iter().filter(|d| !part.contains(d)).copied().collect();
        if rest.is_empty() {
            return c;
        }
        // The ranking breaks ties by every dim it does not partition by, so
        // it is total, which ddx requires of a recomputed ranking.
        let over = format!(
            "{}ORDER BY v {}, {}",
            if part.is_empty() {
                String::new()
            } else {
                format!("PARTITION BY {} ", list(&part, ""))
            },
            if self.rng.below(3) == 0 {
                "ASC"
            } else {
                "DESC"
            },
            list(&rest, "")
        );
        let keep = if self.rng.below(4) == 0 {
            "rk <= 2"
        } else {
            "rk = 1"
        };
        let body = format!(
            "SELECT {}v FROM (SELECT {}v, ROW_NUMBER() OVER ({over}) AS rk FROM §{c}§ c) r WHERE {keep}",
            lead(&dims, ""),
            lead(&dims, ""),
        );
        self.derive(c, body, "rank")
    }

    /// nn.py's softmax over one dim, as four relations: the max, the shifted
    /// exponentials, their sum, the ratio. The shift cancels, so stopping it
    /// or not must give the same gradient (with the MAX rule it cancels too).
    fn softmax(&mut self, depth: u32) -> usize {
        let c = self.node(depth - 1);
        let n = self.nodes[c].clone();
        if n.dims.is_empty() {
            return c;
        }
        let axis = *self.rng.pick(&n.dims);
        let g: Vec<&'static str> = n.dims.iter().filter(|d| **d != axis).copied().collect();
        let mut kinds = n.kinds.clone();
        kinds.insert("softmax");
        kinds.insert("max");
        let mx = self.push(
            format!(
                "SELECT {}MAX(v) AS v FROM §{c}§ c{}",
                lead(&g, ""),
                group_by(&g)
            ),
            g.clone(),
            n.reads.clone(),
            kinds.clone(),
        );
        let shift = if self.rng.below(2) == 0 {
            "ddx_stop_gradient(b.v)"
        } else {
            "b.v"
        };
        if shift != "b.v" {
            kinds.insert("stop-gradient");
        }
        let e = self.push_u(
            format!(
                "SELECT {}exp(a.v - {shift}) AS v FROM {}",
                lead(&n.dims, "a."),
                join_clause(c, mx, &g)
            ),
            n.dims.clone(),
            n.reads.clone(),
            kinds.clone(),
            n.unique,
        );
        let s = self.push(
            format!(
                "SELECT {}SUM(v) AS v FROM §{e}§ c{}",
                lead(&g, ""),
                group_by(&g)
            ),
            g.clone(),
            n.reads.clone(),
            kinds.clone(),
        );
        if shift != "b.v" {
            self.nodes[e].shared = false;
            self.nodes[s].shared = false;
        }
        let ratio = if self.rng.below(3) == 0 {
            "ln(a.v / b.v)"
        } else {
            "a.v / b.v"
        };
        self.push_u(
            format!(
                "SELECT {}{ratio} AS v FROM {}",
                lead(&n.dims, "a."),
                join_clause(e, s, &g)
            ),
            n.dims.clone(),
            n.reads.clone(),
            kinds,
            n.unique,
        )
    }
}

const HEADS: &[&str] = &[
    "SUM(v)",
    "SUM(v)",
    "AVG(v)",
    "SUM(v * v)",
    "MAX(v)",
    "MIN(v)",
    "sin(SUM(v))",
    "SUM(v) / COUNT(*)",
    "AVG(v) + 0.5 * MAX(v)",
    "-AVG(ln(v * v + 0.5))",
    "SUM(v * v) * 0.1 + SUM(v)",
];

/// Seed `seed`'s case: generated from one stream, then its modes applied
/// from another. The first stream is returned for the properties to go on
/// drawing from, so a seed replays its checks exactly.
fn case_for(seed: u64) -> (Case, Rng) {
    let mut rng = seeded(seed, 0xAD_5EED_0002);
    let (modes, mut mrng) = Modes::draw(seed);
    let case = gen_case(&mut rng, modes, &mut mrng);
    (case, rng)
}

/// A random case whose loss reads at least one parameter table.
fn gen_case(rng: &mut Rng, modes: Modes, mrng: &mut Rng) -> Case {
    loop {
        let mut dom = Domains(BTreeMap::from([
            ("s", 1 + rng.below(3) as i64),
            ("i", 1 + rng.below(4) as i64),
            ("j", 1 + rng.below(3) as i64),
        ]));
        if modes.big {
            for (d, n) in dom.0.iter_mut() {
                *n *= if *d == "j" { 4 } else { 10 };
            }
        }
        let key_type = match rng.below(10) {
            0 => KeyType::Int32,
            1 => KeyType::Utf8,
            _ => KeyType::Int64,
        };
        let mut tables = Vec::new();
        for (name, dims) in PARAMS {
            tables.push(gen_table(rng, &dom, name, dims, true, key_type));
        }
        for (name, dims) in DATA {
            tables.push(gen_table(rng, &dom, name, dims, false, key_type));
        }
        apply_modes(&mut tables, &modes, mrng);
        let depth = 1 + rng.below(4) as u32;
        let mut g = Gen {
            cap: if modes.big { 6000 } else { 48 },
            key_type,
            rng,
            dom: dom.clone(),
            tables,
            nodes: Vec::new(),
        };
        let root = g.node(depth);
        let Gen {
            rng, tables, nodes, ..
        } = g;
        let reads: Vec<String> = nodes[root].reads.iter().cloned().collect();
        if reads.is_empty() {
            continue;
        }
        // Differentiate with respect to a random non-empty subset of the
        // parameters the loss reads, in a random order.
        let mut wrt: Vec<String> = reads
            .iter()
            .filter(|_| rng.below(3) != 0)
            .cloned()
            .collect();
        if wrt.is_empty() {
            wrt.push(rng.pick(&reads).clone());
        }
        shuffle(rng, &mut wrt);
        let head = rng.pick(HEADS).to_string();
        return Case {
            modes,
            tables,
            nodes,
            root,
            head,
            wrt,
        };
    }
}

impl Case {
    /// The CTEs `r0 … rN`, each relation once.
    fn ctes(&self) -> String {
        self.nodes
            .iter()
            .enumerate()
            .map(|(k, n)| {
                format!(
                    "r{k} AS ({})",
                    self.body(k, false).unwrap_or_else(|| n.body.clone())
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Node `k`'s body with its references resolved: to CTE names, or
    /// (`inline`) to the referenced bodies as subqueries.
    fn body(&self, k: usize, inline: bool) -> Option<String> {
        let mut out = String::new();
        let mut rest = self.nodes[k].body.as_str();
        while let Some(at) = rest.find('§') {
            out.push_str(&rest[..at]);
            let after = &rest[at + '§'.len_utf8()..];
            let end = after.find('§')?;
            let n: usize = after[..end].parse().ok()?;
            if inline {
                write!(out, "({})", self.body(n, true)?).ok()?;
            } else {
                write!(out, "r{n}").ok()?;
            }
            rest = &after[end + '§'.len_utf8()..];
        }
        out.push_str(rest);
        Some(out)
    }

    /// Whether the case holds exact ties by construction: its ties mode,
    /// or `-0.0` values.
    fn exact_ties(&self) -> bool {
        self.modes.ties || self.modes.extreme == Some(Extreme::NegZero)
    }

    /// `WITH r…, loss AS (SELECT head AS loss FROM root) SELECT outer AS loss
    /// FROM <from>`: the loss CTE, and an outer query over it that the
    /// metamorphic relations vary. `from` defaults to `loss`.
    fn with_loss(&self, outer: &str, from: &str) -> String {
        format!(
            "WITH {}, loss AS (SELECT {} AS loss FROM r{} c) SELECT {outer} AS loss FROM {from}",
            self.ctes(),
            self.head,
            self.root
        )
    }

    /// The same case with every read of table `t` reading `to` instead: a
    /// qualified or quoted name, or a copy registered elsewhere.
    fn reading(&self, t: &str, to: &str) -> Case {
        let mut c = self.clone();
        let suffix = format!(" FROM {t}");
        for n in &mut c.nodes {
            if n.body.ends_with(&suffix) {
                let keep = n.body.len() - suffix.len();
                n.body = format!("{} FROM {to}", &n.body[..keep]);
            }
        }
        c
    }

    /// The loss query as the user would write it.
    fn loss_sql(&self) -> String {
        format!(
            "WITH {} SELECT {} AS loss FROM r{} c",
            self.ctes(),
            self.head,
            self.root
        )
    }

    /// The same loss with no CTEs: every relation inlined as a subquery.
    fn inline_sql(&self) -> String {
        format!(
            "SELECT {} AS loss FROM ({}) c",
            self.head,
            self.body(self.root, true).expect("a well-formed body")
        )
    }

    fn wrt_refs(&self) -> Vec<ColumnRef> {
        self.wrt
            .iter()
            .map(|t| ColumnRef::new(t.as_str(), "val"))
            .collect()
    }

    fn table(&self, name: &str) -> &Table {
        self.tables
            .iter()
            .find(|t| t.name == name)
            .expect("a generated table")
    }

    fn kinds(&self) -> BTreeSet<&'static str> {
        self.nodes[self.root].kinds.clone()
    }

    fn describe(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "  loss  = {}", self.loss_sql());
        let _ = writeln!(s, "  wrt   = {:?}", self.wrt);
        if !self.modes.names().is_empty() {
            let _ = writeln!(s, "  modes = {:?}", self.modes.names());
        }
        for t in &self.tables {
            let rows: Vec<String> = t
                .keys
                .iter()
                .zip(&t.vals)
                .map(|(k, v)| format!("{k:?}={}", v.map_or("NULL".into(), |v| format!("{v}"))))
                .collect();
            let _ = writeln!(s, "  {}({}) = {}", t.name, t.dims.join(","), rows.join(" "));
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Running queries.
// ---------------------------------------------------------------------------

fn fresh_ctx(partitions: usize) -> SessionContext {
    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(partitions));
    ddx_datafusion::register_stop_gradient(&ctx);
    ctx
}

async fn setup(case: &Case, partitions: usize) -> Result<SessionContext, String> {
    let ctx = fresh_ctx(partitions);
    for t in &case.tables {
        t.register(&ctx).map_err(|e| e.to_string())?;
    }
    Ok(ctx)
}

fn cell(a: &ArrayRef, r: usize) -> Option<f64> {
    if a.is_null(r) {
        return None;
    }
    Some(match a.data_type() {
        DataType::Float64 => a.as_primitive::<Float64Type>().value(r),
        DataType::Float32 => a.as_primitive::<Float32Type>().value(r) as f64,
        DataType::Int64 => a.as_primitive::<Int64Type>().value(r) as f64,
        DataType::Int32 => a.as_primitive::<Int32Type>().value(r) as f64,
        DataType::UInt64 => a.as_primitive::<UInt64Type>().value(r) as f64,
        DataType::Utf8 => a.as_string::<i32>().value(r).parse().unwrap_or(f64::NAN),
        DataType::Utf8View => a.as_string_view().value(r).parse().unwrap_or(f64::NAN),
        _ => f64::NAN,
    })
}

/// A result set: its column names and types, and its rows as `f64`s.
struct Rows {
    names: Vec<String>,
    types: Vec<DataType>,
    rows: Vec<Vec<Option<f64>>>,
}

async fn query(ctx: &SessionContext, sql: &str) -> Result<Rows, String> {
    let df = ctx.sql(sql).await.map_err(|e| e.to_string())?;
    let schema = df.schema().inner().clone();
    let batches = df.collect().await.map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            rows.push((0..b.num_columns()).map(|c| cell(b.column(c), r)).collect());
        }
    }
    Ok(Rows {
        names: schema.fields().iter().map(|f| f.name().clone()).collect(),
        types: schema
            .fields()
            .iter()
            .map(|f| f.data_type().clone())
            .collect(),
        rows,
    })
}

/// The loss, or `None` when it is NULL, not finite, or not one row.
async fn loss(ctx: &SessionContext, sql: &str) -> Result<Option<f64>, String> {
    let r = query(ctx, sql).await?;
    Ok(match r.rows.as_slice() {
        [row] if row.len() == 1 => row[0].filter(|v| v.is_finite()),
        _ => None,
    })
}

/// A gradient: dim tuple → the gradient of `val` there.
type Grad = BTreeMap<Vec<i64>, Option<f64>>;

/// Why a `grad` did not produce a program.
enum Refusal {
    /// A refusal ddx is entitled to.
    Allowed(String),
    /// Something that is always a bug.
    Bug(String),
}

fn classify(e: DataFusionError) -> Refusal {
    if let DataFusionError::External(boxed) = &e {
        if let Some(ad) = boxed.downcast_ref::<AdError>() {
            return match ad {
                AdError::Internal(_) => Refusal::Bug(format!("[internal] {ad}")),
                AdError::InvalidPlan(_) => {
                    Refusal::Bug(format!("[invalid-plan] a plan DataFusion produced: {ad}"))
                }
                AdError::NotImplemented(m) => {
                    Refusal::Allowed(format!("NotImplemented: {}", short(m)))
                }
                AdError::NotScalar(m) => Refusal::Allowed(format!("NotScalar: {}", short(m))),
                AdError::UnknownWrt(m) => Refusal::Allowed(format!("UnknownWrt: {}", short(m))),
                AdError::InvalidWrt(m) => Refusal::Allowed(format!("InvalidWrt: {}", short(m))),
                AdError::Diff(d) => Refusal::Allowed(format!("Diff: {}", short(&d.to_string()))),
                // A kind of refusal the harness does not know yet (the soak
                // sets no options, so InvalidOptions too) is worth a look.
                _ => Refusal::Bug(format!("[unexpected refusal] {ad}")),
            };
        }
    }
    // Planning the loss query itself can fail on a generated query DataFusion
    // does not support; that is the generator's problem, not ddx's.
    Refusal::Allowed(format!("DataFusion: {}", short(&e.to_string())))
}

/// The first few words of a message, as a histogram key.
fn short(m: &str) -> String {
    m.split_whitespace().take(7).collect::<Vec<_>>().join(" ")
}

/// Did a step fail in DataFusion's *physical* planning, after its logical
/// plan was accepted and typed? Such a failure is an engine bug that ddx's
/// shape reaches, not a ddx bug: the optimizer variants find them in rule sets
/// DataFusion does not ship (it cannot execute a query's own COALESCE unless
/// SimplifyExpressions has rewritten it, and its ProjectionPushdown and
/// sanity check reject some plans it planned itself). ddx's own steps no
/// longer use COALESCE (#74). They are tallied by kind so a soak still shows
/// them.
fn engine_fault(msg: &str) -> Option<String> {
    const PHYSICAL: &[&str] = &[
        "should have been simplified",
        "SanityCheckPlan",
        "ProjectionPushdown",
        "EnforceDistribution",
        "EnforceSorting",
        "LimitPushdown",
    ];
    PHYSICAL
        .iter()
        .find(|p| msg.contains(**p))
        .map(|p| p.to_string())
}

/// Build and run `grad` of `sql`, and read every gradient.
async fn grad_of(
    ctx: &SessionContext,
    sql: &str,
    wrt: &[ColumnRef],
) -> Result<(BackwardProgram, BTreeMap<String, Grad>), Refusal> {
    let program = ad::grad(ctx, sql, wrt).await.map_err(classify)?;
    let grads = run_and_read(ctx, &program).await.map_err(Refusal::Bug)?;
    Ok((program, grads))
}

async fn run_and_read(
    ctx: &SessionContext,
    program: &BackwardProgram,
) -> Result<BTreeMap<String, Grad>, String> {
    ad::run(ctx, program)
        .await
        .map_err(|e| format!("[accepted-but-failed] ad::run of a program ddx accepted: {e}"))?;
    read_grads(ctx, program).await
}

async fn read_grads(
    ctx: &SessionContext,
    program: &BackwardProgram,
) -> Result<BTreeMap<String, Grad>, String> {
    let mut out = BTreeMap::new();
    for g in &program.gradients {
        let r = query(ctx, &format!("SELECT * FROM \"{}\"", g.step))
            .await
            .map_err(|e| format!("[read] cannot read gradient step {}: {e}", g.step))?;
        let table = g.table.last().cloned().unwrap_or_default();
        let ncols = r.names.len();
        if ncols == 0 || r.names[ncols - 1] != "val" {
            return Err(format!(
                "[shape] gradient of {table} has columns {:?}",
                r.names
            ));
        }
        if r.types[ncols - 1] != DataType::Float64 {
            return Err(format!(
                "[shape] gradient of {table}.val is {:?}, not the column's Float64",
                r.types[ncols - 1]
            ));
        }
        let mut grad = Grad::new();
        for row in &r.rows {
            let key: Vec<i64> = row[..ncols - 1]
                .iter()
                .map(|v| v.unwrap_or(f64::NAN) as i64)
                .collect();
            if grad.insert(key.clone(), row[ncols - 1]).is_some() {
                return Err(format!(
                    "[shape] gradient of {table} has two rows for dims {key:?}"
                ));
            }
        }
        out.insert(table, grad);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Comparing gradients.
// ---------------------------------------------------------------------------

/// Tolerance for two ddx programs computing the same gradient by different
/// plans: float noise at the scale of the gradient, not of each entry.
const META_RTOL: f64 = 1e-9;
const META_ATOL: f64 = 1e-11;

fn scale_of(g: &BTreeMap<String, Grad>) -> f64 {
    g.values()
        .flat_map(|t| t.values())
        .filter_map(|v| *v)
        .filter(|v| v.is_finite())
        .fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Compare `got` with `factor · want` entry by entry.
fn compare(
    label: &str,
    want: &BTreeMap<String, Grad>,
    got: &BTreeMap<String, Grad>,
    factor: f64,
    rtol: f64,
) -> Option<String> {
    let scale = (scale_of(want) * factor.abs())
        .max(scale_of(got))
        .max(1e-300);
    for (table, w) in want {
        let Some(g) = got.get(table) else {
            return Some(format!("[{label}] no gradient for {table}"));
        };
        if g.len() != w.len() {
            return Some(format!(
                "[{label}] {table}: {} rows vs {}",
                g.len(),
                w.len()
            ));
        }
        for (key, wv) in w {
            let Some(gv) = g.get(key) else {
                return Some(format!("[{label}] {table}: no row for dims {key:?}"));
            };
            let ok = match (wv, gv) {
                (None, None) => true,
                (Some(a), Some(b)) if a.is_nan() && b.is_nan() => true,
                // An expected value past f64 (an overflowed or NaN gradient
                // scaled by a chain factor) is not a comparison: huge values
                // reach it by different roundings on each side (seed 600912).
                (Some(a), Some(_)) if !(a * factor).is_finite() => true,
                // An infinite gradient equals itself; their difference is NaN.
                (Some(a), Some(b)) if a * factor == *b => true,
                // The absolute floor is for gradients that are zero in exact
                // arithmetic (a softmax's outputs sum to one, so a loss over
                // their sum has none), where two correct plans leave
                // different rounding residue of order ε times the
                // intermediates, not times the gradient.
                (Some(a), Some(b)) => {
                    (a * factor - b).abs() <= rtol * scale + META_ATOL * factor.abs().max(1.0)
                }
                _ => false,
            };
            if !ok {
                return Some(format!(
                    "[{label}] {table}{key:?}: expected {factor} × {wv:?} = {:?}, got {gv:?} \
                     (gradient scale {scale:.3e})",
                    wv.map(|v| v * factor)
                ));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// The finite-difference oracle.
// ---------------------------------------------------------------------------

/// A direction in parameter space: `(table, row, component)`.
type Direction = Vec<(String, usize, f64)>;

async fn loss_at(
    ctx: &SessionContext,
    case: &Case,
    sql: &str,
    dir: &Direction,
    t: f64,
) -> Result<Option<f64>, String> {
    let mut tables: BTreeMap<&str, Table> = BTreeMap::new();
    for (name, row, d) in dir {
        let tb = tables
            .entry(name.as_str())
            .or_insert_with(|| case.table(name).clone());
        if let Some(v) = tb.vals[*row].as_mut() {
            *v += t * d;
        }
    }
    for tb in tables.values() {
        tb.register(ctx).map_err(|e| e.to_string())?;
    }
    let out = loss(ctx, sql).await;
    for name in tables.keys() {
        case.table(name).register(ctx).map_err(|e| e.to_string())?;
    }
    out
}

enum Fd {
    Agree,
    /// Not smooth here, but ⟨∇L, d⟩ lies between the one-sided derivatives.
    Bracketed,
    /// Not comparable here: a non-finite loss or gradient.
    Screened,
    Disagree(String),
}

/// `⟨∇L, d⟩` against a screened, Richardson-extrapolated central difference.
async fn fd_check(
    ctx: &SessionContext,
    case: &Case,
    sql: &str,
    l0: f64,
    grads: &BTreeMap<String, Grad>,
    dir: &Direction,
) -> Result<Fd, String> {
    let mut ad_dot = 0.0;
    let mut ad_abs = 0.0;
    for (name, row, d) in dir {
        let key = &case.table(name).keys[*row];
        let g = grads
            .get(name)
            .and_then(|g| g.get(key))
            .copied()
            .flatten()
            .ok_or_else(|| {
                format!("[shape] no gradient for {name}{key:?}, whose value is not NULL")
            })?;
        ad_dot += g * d;
        ad_abs += (g * d).abs();
    }
    // A NaN or infinite gradient is compared too, not skipped: where the
    // loss is finite and smooth along d, the finite difference below is a
    // finite number, and no convention makes the gradient anything else.
    // (Where the derivative really is infinite, sqrt at 0, the gates screen
    // the point.)
    let finite_grad = ad_dot.is_finite();
    if !finite_grad {
        // NaN data makes whatever it reaches NaN, a gradient included, even
        // where the loss has filtered it away along d; skip. Infinite data
        // is compared: ddx-core writes a quotient's derivative term by term,
        // so a partial whose limit at ∞ is 0 is 0, not ∞/∞.
        if case.modes.extreme == Some(Extreme::NanData) {
            return Ok(Fd::Screened);
        }
        ad_abs = 0.0;
    }
    // The step follows the parameters' magnitude.
    let h = if case.modes.extreme == Some(Extreme::Tiny) {
        1e-163
    } else {
        1e-3
    };
    // A step the loss is not linear over tells nothing: sin of a sum near
    // 1e8 turns through many periods in one step.
    if finite_grad && ad_abs * h > 1e-2 * l0.abs().max(1.0) {
        return Ok(Fd::Screened);
    }
    let mut at = BTreeMap::new();
    for k in [-2i32, -1, 1, 2] {
        let t = h * k as f64 / 2.0;
        match loss_at(ctx, case, sql, dir, t).await? {
            Some(v) => {
                at.insert(k, v);
            }
            None => return Ok(Fd::Screened),
        }
    }
    let (m2, m1, p1, p2) = (at[&-2], at[&-1], at[&1], at[&2]);
    // Second differences at h and h/2: they shrink 4× on a smooth loss.
    let noise = 64.0 * f64::EPSILON * (l0.abs() + m2.abs() + p2.abs() + 1e-300);
    let a_h = p2 - 2.0 * l0 + m2;
    let a_h2 = p1 - 2.0 * l0 + m1;
    let d1 = (p2 - m2) / (2.0 * h);
    let d2 = (p1 - m1) / h;
    let scale = ad_abs.max(d2.abs()).max(1e-6);
    // On a smooth loss a(h) = f''h² + O(h⁴) and a(h/2) = f''h²/4 + O(h⁴),
    // so a(h) − 4·a(h/2) is O(h⁴); a kink within the step adds O(jump · h)
    // to it whatever the smooth curvature, which a plain ratio of the two
    // misses when a quadratic term dominates both.
    let kink = (a_h - 4.0 * a_h2).abs() > 16.0 * noise + 1e-7 * scale * h;
    if !finite_grad {
        // A loss that does not move along d beyond rounding, with a NaN
        // gradient, is a chain rule through an infinite partial that meets
        // a zero one: sqrt of ln(softmax) of a single row is sqrt(0) for
        // every value (seed 811), and tanh(b · ∞) is 1 whatever b is, so
        // tanh'(∞) · ∞ = 0 · ∞ (seed 500192). jax.grad gives NaN there too,
        // so it is screened, not failed.
        if ad_dot.is_nan() && (p2 - m2).abs() <= noise && (p1 - m1).abs() <= noise {
            return Ok(Fd::Screened);
        }
        // Infinite data makes 0 · ∞ part of reverse mode wherever a row that
        // does not move the loss (σ(x · w) with x = ∞ is 1 for every w)
        // carries an infinite partial: jax.grad gives NaN there too (seed
        // 600421). A finite gradient is still compared; a NaN one is not.
        if ad_dot.is_nan() && case.modes.extreme == Some(Extreme::InfData) {
            return Ok(Fd::Screened);
        }
        let smooth = !kink && (d1 - d2).abs() <= 1e-3 * scale && d2.is_finite();
        return Ok(if smooth {
            Fd::Disagree(format!(
                "[finite-diff] ⟨∇L, d⟩ = {ad_dot} but the loss is smooth along d and moves \
                 at {d2:.12e}: the gradient is not finite where the loss is"
            ))
        } else {
            Fd::Screened
        });
    }
    let exact_ties = case.exact_ties();
    if (kink || (d1 - d2).abs() > 1e-3 * scale) && exact_ties {
        // A tie between computed values (0.25 · 1 and -0.5 · -0.5) can still
        // break along a tie-preserving direction, and at a tie the bracket
        // below is not sound (several kinks meet, with any signs). `-0.0`
        // values tie too: a parameter at -0 equals data at -0.
        return Ok(Fd::Screened);
    }
    if kink || (d1 - d2).abs() > 1e-3 * scale {
        // Not smooth along d, so no central difference is the derivative.
        // Every convention ddx pins at a kink (MAX shares evenly at a tie, a
        // rank filter gives its winner everything, a CASE takes the branch
        // the row takes, abs gives 0) is still a derivative of one of the
        // pieces meeting here, so ⟨∇L, d⟩ must lie between the one-sided
        // derivatives. Their own truncation error is the slack. This holds
        // for one kink near the point, which is all random values produce;
        // at an exact tie several meet at once and it need not (ties mode
        // screens instead).
        let fwd = [(p2 - l0) / h, (p1 - l0) / (h / 2.0)];
        let bwd = [(l0 - m2) / h, (l0 - m1) / (h / 2.0)];
        let (lo, hi) = (fwd[1].min(bwd[1]), fwd[1].max(bwd[1]));
        let slack =
            (fwd[0] - fwd[1]).abs() + (bwd[0] - bwd[1]).abs() + 1e-6 * scale + 8.0 * noise / h;
        if ad_dot >= lo - slack && ad_dot <= hi + slack {
            return Ok(Fd::Bracketed);
        }
        let what: Vec<String> = dir
            .iter()
            .take(6)
            .map(|(n, r, d)| format!("{n}{:?}·{d:.3}", case.table(n).keys[*r]))
            .collect();
        return Ok(Fd::Disagree(format!(
            "[subgradient] at a kink along d, ⟨∇L, d⟩ = {ad_dot:.12e} is outside the \
             one-sided derivatives [{lo:.12e}, {hi:.12e}] (slack {slack:.3e}); no \
             convention at a kink gives that. d = {}{}",
            what.join(" + "),
            if dir.len() > 6 { " + …" } else { "" }
        )));
    }
    let fd = (4.0 * d2 - d1) / 3.0;
    // Richardson assumes the loss is smooth to fourth order. At a kink of a
    // C¹ piece (greatest(v, 0)² at v = 0) its error is O(h), and |d1 − d2|
    // estimates it; on a smooth loss that term is O(h²) and adds nothing.
    let tol = 1e-6 * scale + 8.0 * noise / h + 2.0 * (d1 - d2).abs();
    if (fd - ad_dot).abs() <= tol {
        Ok(Fd::Agree)
    } else {
        let what: Vec<String> = dir
            .iter()
            .take(6)
            .map(|(n, r, d)| format!("{n}{:?}·{d:.3}", case.table(n).keys[*r]))
            .collect();
        Ok(Fd::Disagree(format!(
            "[finite-diff] ⟨∇L, d⟩ = {ad_dot:.12e} but the loss moves at {fd:.12e} \
             (|Δ| {:.3e}, tol {tol:.3e}) along d = {}{}",
            (fd - ad_dot).abs(),
            what.join(" + "),
            if dir.len() > 6 { " + …" } else { "" }
        )))
    }
}

/// Random directions over every non-NULL `wrt` entry, then a few single
/// entries.
fn directions(rng: &mut Rng, case: &Case) -> Vec<Direction> {
    let entries: Vec<(String, usize)> = case
        .wrt
        .iter()
        .flat_map(|t| {
            let tb = case.table(t);
            (0..tb.vals.len())
                .filter(|r| tb.vals[*r].is_some())
                .map(|r| (t.clone(), r))
                .collect::<Vec<_>>()
        })
        .collect();
    if entries.is_empty() {
        return Vec::new();
    }
    let mut dirs = Vec::new();
    // `-0.0` values tie exactly too (a parameter at -0 equals data at -0,
    // and three parameters at -0 tie each other), so they get the same
    // tie-preserving directions (seeds 1300234, 1300617).
    if case.exact_ties() {
        // At an exact tie the loss need not be differentiable at all: the
        // median of three tied values moves at median(d) along any d, the
        // same on both sides, yet has no gradient, and ddx's convention is
        // one valid subgradient of many. A direction that breaks ties
        // therefore proves nothing either way. One that moves every
        // parameter by a function of its value keeps equal values equal, so
        // the ties persist, the loss is smooth along it, and the central
        // difference is exact: it checks that a tie's shared cotangent adds
        // up, whatever the split.
        for _ in 0..3 {
            let mut coef: BTreeMap<u64, f64> = BTreeMap::new();
            dirs.push(
                entries
                    .iter()
                    .map(|(t, r)| {
                        let v = case.table(t).vals[*r].unwrap_or(0.0);
                        // In -0.0 mode a zero parameter stays put: it ties
                        // with zero data and makes computed values tie
                        // (u + z against u when z is -0), which moving it
                        // would break (seed 1300617).
                        if v == 0.0 && case.modes.extreme == Some(Extreme::NegZero) {
                            return (t.clone(), *r, 0.0);
                        }
                        let c = *coef
                            .entry(v.to_bits())
                            .or_insert_with(|| rng.range(-1.0, 1.0));
                        (t.clone(), *r, c)
                    })
                    .collect(),
            );
        }
        return dirs;
    }
    for _ in 0..2 {
        dirs.push(
            entries
                .iter()
                .map(|(t, r)| (t.clone(), *r, rng.range(-1.0, 1.0)))
                .collect(),
        );
    }
    for _ in 0..2 {
        let (t, r) = rng.pick(&entries).clone();
        dirs.push(vec![(t, r, 1.0)]);
    }
    dirs
}

// ---------------------------------------------------------------------------
// One case, every property.
// ---------------------------------------------------------------------------

/// What happened to one case, for the run's tally.
#[derive(Default, Debug)]
struct Outcome {
    failures: Vec<String>,
    /// The base `grad` ran and was checked.
    accepted: bool,
    refusal: Option<String>,
    fd_compared: u32,
    fd_screened: u32,
    fd_bracketed: u32,
    meta_compared: u32,
    kinds: BTreeSet<&'static str>,
    /// Failures DataFusion's physical planning is responsible for (see
    /// [`engine_fault`]): tallied, not failed.
    engine: Vec<String>,
    /// The plan rewrites that were compared, for coverage.
    rewrites: Vec<String>,
}

impl Outcome {
    fn fail(&mut self, s: impl Into<String>) {
        self.failures.push(s.into());
    }
}

/// Which property groups to run. The soak runs them all.
#[derive(Clone, Copy)]
struct Props {
    fd: bool,
    calculus: bool,
    vjp: bool,
    invariance: bool,
    contract: bool,
    surface: bool,
    shapes: bool,
    names: bool,
    train: bool,
}

const ALL: Props = Props {
    fd: true,
    calculus: true,
    vjp: true,
    invariance: true,
    contract: true,
    surface: true,
    shapes: true,
    names: true,
    train: true,
};

async fn check_case(seed: u64, props: Props) -> Outcome {
    let (case, mut rng) = case_for(seed);
    let mut out = Outcome {
        kinds: case.kinds(),
        ..Outcome::default()
    };
    if let Err(e) = check_case_inner(&mut rng, &case, props, &mut out).await {
        out.fail(format!("[harness] {e}"));
    }
    if !out.failures.is_empty() {
        let details = case.describe();
        for f in &mut out.failures {
            f.push('\n');
            f.push_str(&details);
        }
    }
    out
}

async fn check_case_inner(
    rng: &mut Rng,
    case: &Case,
    props: Props,
    out: &mut Outcome,
) -> Result<(), String> {
    let ctx = setup(case, 4).await?;
    let sql = case.loss_sql();
    let wrt = case.wrt_refs();

    // A loss DataFusion cannot compute (a NaN cast to an integer) is the
    // generator's reach exceeding the engine's, not a finding.
    let l0 = match loss(&ctx, &sql).await {
        Ok(Some(l0)) => l0,
        Ok(None) => {
            out.refusal = Some("loss is NULL or not finite".into());
            return Ok(());
        }
        Err(e) => {
            out.refusal = Some(format!("the loss query fails: {}", short(&e)));
            return Ok(());
        }
    };

    let (program, grads) = match grad_of(&ctx, &sql, &wrt).await {
        Ok(ok) => ok,
        Err(Refusal::Allowed(why)) => {
            out.refusal = Some(why);
            return Ok(());
        }
        Err(Refusal::Bug(why)) => {
            out.fail(why);
            return Ok(());
        }
    };
    out.accepted = true;

    // The value step is the loss.
    match query(&ctx, &format!("SELECT * FROM \"{}\"", program.value)).await {
        Ok(r) => match r.rows.as_slice() {
            [row]
                if row.len() == 1
                    && row[0].is_some_and(|v| (v - l0).abs() <= 1e-9 * l0.abs().max(1.0)) => {}
            rows => out.fail(format!(
                "[value] the value step holds {rows:?}, the loss is {l0}"
            )),
        },
        Err(e) => out.fail(format!("[value] cannot read the value step: {e}")),
    }

    // Shape: one row per table row, keyed by its dims, NULL exactly where
    // the value is NULL.
    for t in &case.wrt {
        let tb = case.table(t);
        let Some(g) = grads.get(t) else {
            out.fail(format!("[shape] no gradient step for wrt table {t}"));
            continue;
        };
        if g.len() != tb.keys.len() {
            out.fail(format!(
                "[shape] {t} has {} rows, its gradient {}",
                tb.keys.len(),
                g.len()
            ));
        }
        for (key, val) in tb.keys.iter().zip(&tb.vals) {
            match (val, g.get(key)) {
                (_, None) => out.fail(format!("[shape] {t}: no gradient row for dims {key:?}")),
                (None, Some(Some(gv))) => out.fail(format!(
                    "[null] {t}{key:?}: the value is NULL but its gradient is {gv} (the convention is NULL)"
                )),
                (Some(v), Some(None)) => out.fail(format!(
                    "[null] {t}{key:?}: the value is {v} but its gradient is NULL \
                     (NULL is reserved for a NULL value; unreached is 0)"
                )),
                _ => {}
            }
        }
    }
    if !out.failures.is_empty() {
        return Ok(());
    }

    if props.fd {
        for dir in directions(rng, case) {
            match fd_check(&ctx, case, &sql, l0, &grads, &dir).await? {
                Fd::Agree => out.fd_compared += 1,
                Fd::Screened => out.fd_screened += 1,
                Fd::Bracketed => out.fd_bracketed += 1,
                Fd::Disagree(msg) => {
                    out.fail(msg);
                    break;
                }
            }
        }
    }

    // Everything below compares ddx with itself.
    let meta = |label: &str,
                r: Result<(BackwardProgram, BTreeMap<String, Grad>), Refusal>,
                factor: f64,
                rtol: f64,
                out: &mut Outcome,
                strict: bool| {
        match r {
            Ok((_, g)) => {
                out.meta_compared += 1;
                if let Some(f) = compare(label, &grads, &g, factor, rtol) {
                    out.fail(f);
                }
            }
            // A physical-planning fault is DataFusion's, tallied here as
            // everywhere (a single-partition context reaches one ddx's
            // fan-in union can trip, pinned in ad_findings.rs).
            Err(Refusal::Bug(b)) => match engine_fault(&b) {
                Some(kind) => out.engine.push(kind),
                None => out.fail(format!("[{label}] {b}")),
            },
            Err(Refusal::Allowed(why)) if strict => out.fail(format!(
                "[{label}] refused a query equivalent to one it accepted: {why}"
            )),
            Err(Refusal::Allowed(_)) => {}
        }
    };

    if props.calculus {
        let with = |outer: &str| case.with_loss(outer, "loss");
        let pairs = |outer: &str| case.with_loss(outer, "loss l CROSS JOIN loss k");
        let checks: Vec<(&str, String, f64)> = vec![
            ("scale", with("2.5 * loss"), 2.5),
            ("shift", with("loss + 3.0"), 1.0),
            ("square", with("loss * loss"), 2.0 * l0),
            ("sin", with("sin(loss)"), l0.cos()),
            ("read-twice", pairs("l.loss * k.loss"), 2.0 * l0),
            (
                "stop-product",
                pairs("l.loss * ddx_stop_gradient(k.loss)"),
                l0,
            ),
            ("stop-sum", pairs("l.loss + ddx_stop_gradient(k.loss)"), 1.0),
        ];
        for (label, q, factor) in checks {
            // Huge values take a transform past what f64 can compare: the
            // square of a loss near 1e270 overflows, and sin of a loss past
            // 1e15 is rounding (an ulp there exceeds 2π). Such a check says
            // nothing about ddx.
            if !(factor * scale_of(&grads)).is_finite() || (label == "sin" && l0.abs() > 1e15) {
                continue;
            }
            // A chain factor can make the gradient much larger or smaller
            // than the base's, so the tolerance follows the factor.
            let r = grad_of(&ctx, &q, &wrt).await;
            meta(label, r, factor, META_RTOL * 10.0, out, false);
        }
    }

    if props.vjp {
        vjp_checks(rng, &ctx, case, &sql, &wrt, &grads, l0, out).await?;
    }

    if props.invariance {
        // Spelled without CTEs.
        let r = grad_of(&ctx, &case.inline_sql(), &wrt).await;
        meta("inline", r, 1.0, META_RTOL, out, false);

        // The unoptimized plan, handed straight to ddx-ad.
        let lp = ctx
            .sql(&sql)
            .await
            .map_err(|e| e.to_string())?
            .into_unoptimized_plan();
        match ad::grad_plan(&ctx, &lp, &wrt) {
            Ok(p) => match run_and_read(&ctx, &p).await {
                Ok(g) => {
                    out.meta_compared += 1;
                    if let Some(f) = compare("unoptimized", &grads, &g, 1.0, META_RTOL) {
                        out.fail(f);
                    }
                }
                Err(e) => out.fail(format!("[unoptimized] {e}")),
            },
            Err(e) => {
                if let Refusal::Bug(b) = classify(e) {
                    out.fail(format!("[unoptimized] {b}"));
                }
            }
        }

        // The wrt list: reversed, case-folded, and one table at a time.
        let mut rev = wrt.clone();
        rev.reverse();
        let r = grad_of(&ctx, &sql, &rev).await;
        meta("wrt-order", r, 1.0, META_RTOL, out, true);
        let upper: Vec<ColumnRef> = case
            .wrt
            .iter()
            .map(|t| ColumnRef::new(t.to_uppercase(), "VAL"))
            .collect();
        let r = grad_of(&ctx, &sql, &upper).await;
        meta("wrt-case", r, 1.0, META_RTOL, out, true);
        if wrt.len() > 1 {
            for w in &wrt {
                match grad_of(&ctx, &sql, std::slice::from_ref(w)).await {
                    Ok((_, g)) => {
                        out.meta_compared += 1;
                        let want: BTreeMap<String, Grad> = grads
                            .iter()
                            .filter(|(k, _)| **k == w.table)
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        if let Some(f) = compare("wrt-alone", &want, &g, 1.0, META_RTOL) {
                            out.fail(f);
                        }
                    }
                    Err(Refusal::Bug(b)) => out.fail(format!("[wrt-alone] {b}")),
                    Err(Refusal::Allowed(_)) => {}
                }
            }
        }

        // Partitions and row order.
        // Each is compared only where the engine computes the same loss
        // there: over NaN data a MAX can depend on the order it meets rows
        // in, and then the query itself differs, before ddx is involved.
        let same_loss = |l: Option<f64>| match l {
            Some(v) => v == l0 || (v - l0).abs() <= 1e-12 * l0.abs().max(1.0),
            None => false,
        };
        for parts in [1usize, 7] {
            let other = setup(case, parts).await?;
            if !same_loss(loss(&other, &sql).await?) {
                continue;
            }
            let r = grad_of(&other, &sql, &wrt).await;
            meta(
                if parts == 1 {
                    "partitions-1"
                } else {
                    "partitions-7"
                },
                r,
                1.0,
                1e-8,
                out,
                true,
            );
        }
        let mut shuffled = case.clone();
        for t in &mut shuffled.tables {
            let mut idx: Vec<usize> = (0..t.keys.len()).collect();
            shuffle(rng, &mut idx);
            t.keys = idx.iter().map(|&k| t.keys[k].clone()).collect();
            t.vals = idx.iter().map(|&k| t.vals[k]).collect();
        }
        let other = setup(&shuffled, 4).await?;
        if same_loss(loss(&other, &sql).await?) {
            let r = grad_of(&other, &sql, &wrt).await;
            meta("row-order", r, 1.0, 1e-8, out, true);
        }
    }

    if props.contract {
        contract_checks(rng, case, &sql, &wrt, &program, &grads, out).await?;
    }

    if props.surface {
        surface_checks(rng, case, &grads, out).await?;
    }

    if props.shapes {
        optimizer_checks(rng, case, &grads, out).await?;
        mutation_checks(rng, case, &ctx, &sql, l0, &grads, out).await?;
    }

    if props.names {
        for _ in 0..2 {
            name_checks(rng, case, &grads, out).await?;
        }
    }

    if props.train {
        train_checks(case, &grads, out).await?;
    }
    Ok(())
}

/// A statement's rows as a gradient: every column but the last is a dim.
async fn grad_rows(ctx: &SessionContext, sql: &str) -> Result<Grad, String> {
    let frames = ad::sql(ctx, sql).await.map_err(|e| e.to_string())?;
    let batches = frames.collect().await.map_err(|e| e.to_string())?;
    let mut g = Grad::new();
    for b in &batches {
        let n = b.num_columns();
        for r in 0..b.num_rows() {
            let key: Vec<i64> = (0..n - 1)
                .map(|c| cell(b.column(c), r).map_or(NULL_KEY, |v| v as i64))
                .collect();
            g.insert(key, cell(b.column(n - 1), r));
        }
    }
    Ok(g)
}

/// `grad(loss, t.col)` in SQL against names and places a user might use:
/// the table under a schema or a quoted name while a decoy of its old name
/// holds other values, a CTE shadowing it, the call in a subquery or a later
/// CTE, two losses in one statement, and user tables whose names look like
/// ddx's own, which must come through untouched.
async fn name_checks(
    rng: &mut Rng,
    case: &Case,
    grads: &BTreeMap<String, Grad>,
    out: &mut Outcome,
) -> Result<(), String> {
    let t = rng.pick(&case.wrt).clone();
    let tb = case.table(&t).clone();
    let dims: Vec<&str> = tb.dims.iter().map(String::as_str).collect();
    let want = BTreeMap::from([(t.clone(), grads[&t].clone())]);
    let ctx = setup(case, 4).await?;
    // A decoy under the table's own name, with other values, so reading the
    // wrong one cannot pass by coincidence.
    let decoy = || {
        let mut d = tb.clone();
        for v in d.vals.iter_mut().flatten() {
            *v = round6(*v * 0.5 + 0.37);
        }
        d
    };
    let kind = *rng.pick(&[
        "schema",
        "quoted",
        "shadow",
        "subquery",
        "later-cte",
        "two-losses",
        "lookalikes",
    ]);
    let (sql, value) = match kind {
        "schema" | "quoted" => {
            let (reference, register) = if kind == "schema" {
                ctx.sql("CREATE SCHEMA IF NOT EXISTS s1")
                    .await
                    .map_err(|e| e.to_string())?
                    .collect()
                    .await
                    .map_err(|e| e.to_string())?;
                (format!("s1.{t}"), format!("s1.{t}"))
            } else {
                (format!("\"T {t}\""), format!("T {t}"))
            };
            let batch = tb.batch();
            let copy =
                MemTable::try_new(batch.schema(), vec![vec![batch]]).map_err(|e| e.to_string())?;
            let table_ref = if kind == "schema" {
                datafusion::sql::TableReference::parse_str(&register)
            } else {
                datafusion::sql::TableReference::bare(register.clone())
            };
            ctx.register_table(table_ref, Arc::new(copy))
                .map_err(|e| e.to_string())?;
            decoy().register(&ctx).map_err(|e| e.to_string())?;
            let moved = case.reading(&t, &reference);
            (
                format!(
                    "WITH {}, loss AS (SELECT {} AS loss FROM r{} c) SELECT {}val FROM grad(loss, {reference}.val)",
                    moved.ctes(),
                    case.head,
                    case.root,
                    lead(&dims, "")
                ),
                "val",
            )
        }
        "shadow" => (
            format!(
                "WITH {t} AS (SELECT * FROM {t}), {}, loss AS (SELECT {} AS loss FROM r{} c) \
                 SELECT {}val FROM grad(loss, {t}.val)",
                case.ctes(),
                case.head,
                case.root,
                lead(&dims, "")
            ),
            "val",
        ),
        "subquery" => (
            format!(
                "{} SELECT * FROM (SELECT {}val FROM grad(loss, {t}.val)) AS g",
                loss_prefix(case),
                lead(&dims, "")
            ),
            "val",
        ),
        "later-cte" => (
            format!(
                "WITH {}, loss AS (SELECT {} AS loss FROM r{} c), \
                 g AS (SELECT * FROM grad(loss, {t}.val)) SELECT {}val FROM g",
                case.ctes(),
                case.head,
                case.root,
                lead(&dims, "")
            ),
            "val",
        ),
        "two-losses" => {
            let on = if dims.is_empty() {
                "ON true".to_string()
            } else {
                format!(
                    "ON {}",
                    dims.iter()
                        .map(|d| format!("a.{d} = b.{d}"))
                        .collect::<Vec<_>>()
                        .join(" AND ")
                )
            };
            (
                format!(
                    "WITH {}, loss AS (SELECT {} AS loss FROM r{} c), \
                     loss2 AS (SELECT 2.0 * loss AS loss FROM loss) \
                     SELECT {}b.val - 2.0 * a.val AS val \
                     FROM grad(loss, {t}.val) a JOIN grad(loss2, {t}.val) b {on}",
                    case.ctes(),
                    case.head,
                    case.root,
                    lead(&dims, "a.")
                ),
                "zero",
            )
        }
        _ => {
            // Tables named like the steps ddx writes, minus the reserved
            // prefix: ddx must not read or replace them.
            for name in [
                "value",
                "saved_0",
                "cotangent_0",
                "grad_0_w",
                "__ddxvalue",
                "ddx_value",
            ] {
                let batch = RecordBatch::try_new(
                    Arc::new(Schema::new(vec![Field::new(
                        "sentinel",
                        DataType::Float64,
                        false,
                    )])),
                    vec![Arc::new(Float64Array::from(vec![42.0]))],
                )
                .map_err(|e| e.to_string())?;
                register_batch(&ctx, name, batch)?;
            }
            (
                format!(
                    "{} SELECT {}val FROM grad(loss, {t}.val)",
                    loss_prefix(case),
                    lead(&dims, "")
                ),
                "lookalikes",
            )
        }
    };
    let got = match grad_rows(&ctx, &sql).await {
        Ok(g) => g,
        Err(e) => {
            // The shadowing CTE is a DataFusion question first: skip it if
            // the loss alone does not plan that way.
            if kind == "shadow" && e.contains("DataFusion") {
                return Ok(());
            }
            if let Some(ad) = e.strip_prefix("External error: ") {
                if !ad.contains("internal error") && !ad.contains("invalid Substrait plan") {
                    // A refusal the program API did not make: plan-shape
                    // dependent coverage, tallied (see surface_checks).
                    out.engine.push(format!("sql refused: {}", short(ad)));
                    return Ok(());
                }
            }
            out.fail(format!("[names] {kind}: {e}\n  {sql}"));
            return Ok(());
        }
    };
    out.meta_compared += 1;
    match value {
        "zero" => {
            // grad(2L) − 2·grad(L) is zero wherever it is defined.
            let zeros: Grad = got
                .keys()
                .map(|k| (k.clone(), grads[&t].get(k).copied().flatten().map(|_| 0.0)))
                .collect();
            let wrap = |g: Grad| BTreeMap::from([(t.clone(), g)]);
            if let Some(f) = compare("names", &wrap(zeros), &wrap(got), 1.0, META_RTOL) {
                out.fail(format!("{f}\n  two losses in one statement: {sql}"));
            }
        }
        _ => {
            let key = if kind == "quoted" {
                format!("T {t}")
            } else {
                t.clone()
            };
            let _ = key;
            let wrap = |g: Grad| BTreeMap::from([(t.clone(), g)]);
            if let Some(f) = compare("names", &want, &wrap(got), 1.0, META_RTOL) {
                out.fail(format!("{f}\n  {kind}: {sql}"));
            }
        }
    }
    if kind == "lookalikes" {
        for name in [
            "value",
            "saved_0",
            "cotangent_0",
            "grad_0_w",
            "__ddxvalue",
            "ddx_value",
        ] {
            match query(&ctx, &format!("SELECT sentinel FROM \"{name}\"")).await {
                Ok(r) if r.rows == vec![vec![Some(42.0)]] => {}
                Ok(r) => out.fail(format!(
                    "[names] the user's table {name} now holds {:?}",
                    r.rows
                )),
                Err(e) => out.fail(format!("[names] the user's table {name} is gone: {e}")),
            }
        }
    }
    let left = ddx_tables(&ctx);
    if !left.is_empty() {
        out.fail(format!(
            "[catalog] ad::sql left {left:?} on the context ({kind})"
        ));
    }
    Ok(())
}

/// `WITH r…, loss AS (…)`: the start of a statement that calls grad.
fn loss_prefix(case: &Case) -> String {
    format!(
        "WITH {}, loss AS (SELECT {} AS loss FROM r{} c)",
        case.ctes(),
        case.head,
        case.root
    )
}

/// Three SGD steps written in SQL with `grad(loss, t.val)`, one statement per
/// table, sharing one program per step. Each must move θ to exactly
/// θ − lr·∇L(θ) as a fresh program computes it, a small enough step must not
/// raise the loss, and nothing may be left on the context.
async fn train_checks(
    case: &Case,
    grads: &BTreeMap<String, Grad>,
    out: &mut Outcome,
) -> Result<(), String> {
    let mut theta = case.clone();
    let mut g = grads.clone();
    let sql = case.loss_sql();
    let wrt = case.wrt_refs();
    // A step of at most 1e-3 in any coordinate. Not 1e-3 / |∇L| unbounded:
    // a gradient that is zero but for rounding (a softmax's, summed) would
    // make that 1e14 and the comparison a comparison of noise.
    let lr = 1e-3 / scale_of(&g).max(1.0);
    for step in 0..3 {
        let ctx = setup(&theta, 4).await?;
        let Some(before) = loss(&ctx, &sql).await.ok().flatten() else {
            return Ok(());
        };
        let mut statements = Vec::new();
        for t in &case.wrt {
            let tb = theta.table(t);
            let dims: Vec<&str> = tb.dims.iter().map(String::as_str).collect();
            let on = if dims.is_empty() {
                "ON true".to_string()
            } else {
                format!(
                    "ON {}",
                    dims.iter()
                        .map(|d| format!("p.{d} = g.{d}"))
                        .collect::<Vec<_>>()
                        .join(" AND ")
                )
            };
            statements.push(format!(
                "{} SELECT {}p.val - {lr:e} * g.val AS val FROM {t} p JOIN grad(loss, {t}.val) g {on}",
                loss_prefix(&theta),
                lead(&dims, "p.")
            ));
        }
        let refs: Vec<&str> = statements.iter().map(String::as_str).collect();
        let frames = match ad::sql_all(&ctx, &refs).await {
            Ok(f) => f,
            Err(e) => {
                if let Refusal::Bug(b) = classify(e) {
                    out.fail(format!("[train] step {step}: {b}"));
                }
                return Ok(());
            }
        };
        let mut next = theta.clone();
        for (k, df) in frames.into_iter().enumerate() {
            let t = &case.wrt[k];
            let batches = df
                .collect()
                .await
                .map_err(|e| format!("[train] step {step}: {e}"))?;
            let mut moved = Grad::new();
            for b in &batches {
                let n = b.num_columns();
                for r in 0..b.num_rows() {
                    let key: Vec<i64> = (0..n - 1)
                        .map(|c| cell(b.column(c), r).map_or(NULL_KEY, |v| v as i64))
                        .collect();
                    moved.insert(key, cell(b.column(n - 1), r));
                }
            }
            let tb = theta.table(t);
            let want: Grad = tb
                .keys
                .iter()
                .zip(&tb.vals)
                .map(|(key, v)| {
                    let gv = g.get(t).and_then(|g| g.get(key)).copied().flatten();
                    (key.clone(), v.zip(gv).map(|(v, gv)| v - lr * gv))
                })
                .collect();
            let wrap = |x: Grad| BTreeMap::from([(t.clone(), x)]);
            out.meta_compared += 1;
            if let Some(f) = compare("train", &wrap(want), &wrap(moved.clone()), 1.0, META_RTOL) {
                out.fail(format!("{f}\n  SGD step {step} in SQL is not θ − lr·∇L"));
                return Ok(());
            }
            let nt = next.tables.iter_mut().find(|x| &x.name == t).unwrap();
            for (key, v) in nt.keys.iter().zip(nt.vals.iter_mut()) {
                *v = moved.get(key).copied().flatten();
            }
        }
        let left = ddx_tables(&ctx);
        if !left.is_empty() {
            out.fail(format!(
                "[catalog] SGD step {step} left {left:?} on the context"
            ));
        }
        // Along −∇L some small enough step must lower the loss, unless the
        // gradient is zero: try steps shrinking by 10× from the one taken.
        // Curvature can make the SGD step itself overshoot, so that alone
        // proves nothing.
        let descent: Direction = case
            .wrt
            .iter()
            .flat_map(|t| {
                let tb = theta.table(t);
                (0..tb.keys.len())
                    .filter_map(|r| {
                        let gv = g
                            .get(t)
                            .and_then(|x| x.get(&tb.keys[r]))
                            .copied()
                            .flatten()?;
                        (tb.vals[r].is_some() && gv != 0.0).then(|| (t.clone(), r, -gv))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let decrease: f64 = descent.iter().map(|(_, _, d)| d * d).sum::<f64>() * lr;
        if !descent.is_empty()
            && !case.kinds().iter().any(|k| KINKY.contains(k))
            && decrease > 1e-9 * before.abs().max(1.0)
        {
            let mut lowered = false;
            let mut tried = Vec::new();
            for k in 0..4 {
                let eps = lr * 10f64.powi(-k);
                if let Some(v) = loss_at(&ctx, &theta, &sql, &descent, eps).await? {
                    tried.push((eps, v));
                    if v < before {
                        lowered = true;
                        break;
                    }
                }
            }
            if !lowered && !tried.is_empty() {
                out.fail(format!(
                    "[descent] at SGD step {step}, no step along −∇L lowers the loss \
                     {before:.15e}: {tried:?}"
                ));
            }
        }
        theta = next;
        let ctx = setup(&theta, 4).await?;
        if loss(&ctx, &sql).await.ok().flatten().is_none() {
            return Ok(());
        }
        g = match grad_of(&ctx, &sql, &wrt).await {
            Ok((_, g)) => g,
            Err(_) => return Ok(()),
        };
    }
    Ok(())
}

/// Primitives with kinks, where a step can cross one and the descent lemma
/// does not apply.
const KINKY: &[&str] = &[
    "max",
    "min",
    "rank",
    "limit",
    "value-filter",
    "kink",
    "case",
    "having",
];

/// A context with the given optimizer rules, in the given order.
fn ctx_with_rules(rules: Vec<Arc<dyn OptimizerRule + Send + Sync>>) -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_config(SessionConfig::new().with_target_partitions(4))
        .with_default_features()
        .with_optimizer_rules(rules)
        .build();
    let ctx = SessionContext::new_with_state(state);
    ddx_datafusion::register_stop_gradient(&ctx);
    ctx
}

/// The optimizer decides the plan ddx reads, and nothing about the gradient
/// may depend on which rules it ran or in what order. Each variant must give
/// the base gradient or refuse; the loss must first mean the same thing
/// there, or the variant is skipped.
async fn optimizer_checks(
    rng: &mut Rng,
    case: &Case,
    grads: &BTreeMap<String, Grad>,
    out: &mut Outcome,
) -> Result<(), String> {
    let sql = case.loss_sql();
    let wrt = case.wrt_refs();
    let defaults = Optimizer::new().rules;
    let mut variants: Vec<(String, Vec<Arc<dyn OptimizerRule + Send + Sync>>)> = Vec::new();
    for _ in 0..2 {
        let k = rng.below(defaults.len() as u64) as usize;
        let mut rules = defaults.clone();
        let gone = rules.remove(k);
        variants.push((format!("without {}", gone.name()), rules));
    }
    let kept: Vec<_> = defaults
        .iter()
        .filter(|_| rng.below(10) >= 3)
        .cloned()
        .collect();
    variants.push((
        format!(
            "only {:?}",
            kept.iter()
                .map(|r| r.name().to_string())
                .collect::<Vec<_>>()
        ),
        kept,
    ));
    let mut shuffled = defaults.clone();
    shuffle(rng, &mut shuffled);
    variants.push(("rules shuffled".into(), shuffled));
    variants.push(("no optimizer rules".into(), vec![]));

    for (label, rules) in variants {
        let fuses_limits = rules.iter().any(|r| r.name() == "push_down_limit");
        let ctx = ctx_with_rules(rules);
        for t in &case.tables {
            t.register(&ctx).map_err(|e| e.to_string())?;
        }
        // The loss itself must plan and agree, or this variant says nothing.
        let base = setup(case, 4).await?;
        let (Ok(Some(want)), Ok(Some(got))) = (loss(&base, &sql).await, loss(&ctx, &sql).await)
        else {
            continue;
        };
        if (want - got).abs() > 1e-9 * want.abs().max(1.0) {
            continue;
        }
        // DataFusion 54 drops a sort beneath a limit whose projection
        // removes the sort key, when the limit sits under a join and
        // push_down_limit did not fuse them (see ad_findings.rs, upstream).
        // Its wrong rows are the engine's, whatever ddx does with them.
        let limit_unsound = case.kinds().contains("limit") && !fuses_limits;
        match grad_of(&ctx, &sql, &wrt).await {
            Ok((_, g)) => {
                out.meta_compared += 1;
                if let Some(f) = compare("optimizer", grads, &g, 1.0, META_RTOL) {
                    if limit_unsound {
                        out.engine.push("sort dropped under a limit".into());
                    } else {
                        out.fail(format!("{f}\n  optimizer: {label}"));
                    }
                }
            }
            Err(Refusal::Bug(b)) => match engine_fault(&b) {
                Some(kind) => out.engine.push(kind),
                None => out.fail(format!("[optimizer] {b}\n  optimizer: {label}")),
            },
            Err(Refusal::Allowed(_)) => {}
        }
    }
    Ok(())
}

/// The Substrait plan rewritten into equivalent shapes (see `mutate`): each
/// must give the base gradient or refuse. DataFusion first consumes the
/// rewritten plan and computes the loss, so a rewrite that changed the
/// query is the harness's mistake and is skipped.
async fn mutation_checks(
    rng: &mut Rng,
    case: &Case,
    ctx: &SessionContext,
    sql: &str,
    l0: f64,
    grads: &BTreeMap<String, Grad>,
    out: &mut Outcome,
) -> Result<(), String> {
    let wrt = case.wrt_refs();
    let df = ctx.sql(sql).await.map_err(|e| e.to_string())?;
    let optimized = df
        .clone()
        .into_optimized_plan()
        .map_err(|e| e.to_string())?;
    let unoptimized = df.into_unoptimized_plan();
    for (k, lp) in [&optimized, &optimized, &optimized, &unoptimized]
        .into_iter()
        .enumerate()
    {
        let Ok(plan) = to_substrait_plan(lp, &ctx.state()) else {
            continue;
        };
        let Some((mutated, kinds)) = mutate::mutate(&plan, rng) else {
            continue;
        };
        let which = if k == 3 { "unoptimized" } else { "optimized" };
        // Is it still the same query?
        let same = match from_substrait_plan(&ctx.state(), &mutated).await {
            Ok(lp) => match ctx.execute_logical_plan(lp).await {
                Ok(df) => match df.collect().await {
                    Ok(b) if b.iter().map(|b| b.num_rows()).sum::<usize>() == 1 => {
                        let b = b.iter().find(|b| b.num_rows() == 1).unwrap();
                        cell(b.column(0), 0)
                            .is_some_and(|v| (v - l0).abs() <= 1e-9 * l0.abs().max(1.0))
                    }
                    _ => false,
                },
                Err(_) => false,
            },
            Err(_) => false,
        };
        if !same {
            out.rewrites.extend(
                kinds
                    .iter()
                    .map(|k| format!("{k} (not equivalent to DataFusion)")),
            );
            continue;
        }
        let label = format!("{which} plan, rewritten by {kinds:?}");
        match ddx_ad::grad(&mutated, &wrt) {
            Ok(program) => match run_and_read(ctx, &program).await {
                Ok(g) => {
                    out.meta_compared += 1;
                    out.rewrites.extend(kinds.iter().map(|k| k.to_string()));
                    if let Some(f) = compare("plan-rewrite", grads, &g, 1.0, META_RTOL) {
                        out.fail(format!("{f}\n  {label}"));
                    }
                    let _ = ad::release(ctx, &program);
                }
                Err(e) => match engine_fault(&e) {
                    Some(kind) => out.engine.push(kind),
                    None => out.fail(format!("[plan-rewrite] {e}\n  {label}")),
                },
            },
            Err(e) => {
                // A refusal is allowed, but an equivalent plan ddx refuses is
                // coverage it lacks, so the tally shows which rewrite did it.
                out.rewrites.extend(
                    kinds
                        .iter()
                        .map(|k| format!("{k} (refused: {})", short(&e.to_string()))),
                );
                if let Refusal::Bug(b) = classify(DataFusionError::External(Box::new(e))) {
                    out.fail(format!("[plan-rewrite] {b}\n  {label}"));
                }
            }
        }
    }
    Ok(())
}

/// `grad` is `vjp` seeded with 1; `vjp` of the root relation with a random
/// cotangent is `grad` of the cotangent-weighted sum; `vjp` is linear.
#[allow(clippy::too_many_arguments)]
async fn vjp_checks(
    rng: &mut Rng,
    ctx: &SessionContext,
    case: &Case,
    sql: &str,
    wrt: &[ColumnRef],
    grads: &BTreeMap<String, Grad>,
    _l0: f64,
    out: &mut Outcome,
) -> Result<(), String> {
    // Seeded with 1 (and 2.5) on the scalar loss.
    for seed in [1.0, 2.5] {
        let program = match ad::vjp(ctx, sql, wrt).await {
            Ok(p) => p,
            Err(e) => {
                match classify(e) {
                    Refusal::Bug(b) => out.fail(format!("[vjp-seed] {b}")),
                    Refusal::Allowed(why) => out.fail(format!(
                        "[vjp-seed] vjp refused a loss grad accepted: {why}"
                    )),
                }
                return Ok(());
            }
        };
        if program.cotangent.len() != 1 {
            out.fail(format!(
                "[vjp-seed] a scalar loss's cotangent should be one column, got {:?}",
                program.cotangent
            ));
            return Ok(());
        }
        let cot = Table {
            name: program.cotangent_table.clone(),
            dims: vec![],
            keys: vec![vec![]],
            vals: vec![Some(seed)],
            param: false,
            key_type: KeyType::Int64,
            chunks: Vec::new(),
        };
        let batch = cot.batch();
        let renamed = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                &program.cotangent[0],
                DataType::Float64,
                true,
            )])),
            vec![batch.column(0).clone()],
        )
        .map_err(|e| e.to_string())?;
        register_batch(ctx, &program.cotangent_table, renamed)?;
        match run_and_read(ctx, &program).await {
            Ok(g) => {
                out.meta_compared += 1;
                if let Some(f) = compare("vjp-seed", grads, &g, seed, META_RTOL) {
                    out.fail(f);
                }
            }
            Err(e) => out.fail(format!("[vjp-seed] {e}")),
        }
        ctx.deregister_table(program.cotangent_table.as_str())
            .map_err(|e| e.to_string())?;
    }

    // vjp of the root relation itself, when its dims identify its rows (a
    // cotangent keyed by dims means nothing otherwise).
    let kt = case.tables[0].key_type;
    let root = &case.nodes[case.root];
    if !root.unique {
        return Ok(());
    }
    let dims = root.dims.clone();
    let out_sql = format!(
        "WITH {} SELECT {}v FROM r{} c",
        case.ctes(),
        lead(&dims, ""),
        case.root
    );
    let program = match ad::vjp(ctx, &out_sql, wrt).await {
        Ok(p) => p,
        Err(e) => {
            if let Refusal::Bug(b) = classify(e) {
                out.fail(format!("[vjp-relation] {b}"));
            }
            return Ok(());
        }
    };
    let rows = query(ctx, &out_sql).await?;
    // A NULL dim identifies no row, so no cotangent can be keyed by it.
    if rows
        .rows
        .iter()
        .any(|r| r[..dims.len()].iter().any(Option::is_none))
    {
        return Ok(());
    }
    // The cotangent: the output's dims, then its value `v`, under the names
    // the program asks for.
    let mut cots: Vec<Vec<f64>> = Vec::new();
    for _ in 0..2 {
        cots.push(
            rows.rows
                .iter()
                .map(|_| round6(rng.range(-1.0, 1.0)))
                .collect(),
        );
    }
    let mut vjp_grads = Vec::new();
    for c in &cots {
        let mut fields = Vec::new();
        let mut cols: Vec<ArrayRef> = Vec::new();
        for name in &program.cotangent {
            if let Some(k) = dims.iter().position(|d| d == name) {
                fields.push(Field::new(name, kt.data_type(), false));
                cols.push(
                    kt.array(
                        rows.rows
                            .iter()
                            .map(|r| r[k].unwrap_or(0.0) as i64)
                            .collect(),
                    ),
                );
            } else if name == "v" {
                fields.push(Field::new(name, DataType::Float64, true));
                cols.push(Arc::new(Float64Array::from(c.clone())));
            } else {
                out.fail(format!(
                    "[vjp-relation] the cotangent asks for column `{name}`, which the output {:?} does not have",
                    rows.names
                ));
                return Ok(());
            }
        }
        let batch =
            RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).map_err(|e| e.to_string())?;
        register_batch(ctx, &program.cotangent_table, batch)?;
        match run_and_read(ctx, &program).await {
            Ok(g) => vjp_grads.push(g),
            Err(e) => {
                out.fail(format!("[vjp-relation] {e}"));
                return Ok(());
            }
        }
        ctx.deregister_table(program.cotangent_table.as_str())
            .map_err(|e| e.to_string())?;
    }

    // Against grad of Σ R·c, the cotangent as a user table.
    for (k, c) in cots.iter().enumerate() {
        let mut fields: Vec<Field> = dims
            .iter()
            .map(|d| Field::new(*d, kt.data_type(), false))
            .collect();
        fields.push(Field::new("cv", DataType::Float64, true));
        let mut cols: Vec<ArrayRef> = (0..dims.len())
            .map(|d| {
                kt.array(
                    rows.rows
                        .iter()
                        .map(|r| r[d].unwrap_or(0.0) as i64)
                        .collect(),
                )
            })
            .collect();
        cols.push(Arc::new(Float64Array::from(c.clone())));
        let batch =
            RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).map_err(|e| e.to_string())?;
        register_batch(ctx, "cot_user", batch)?;
        let on = if dims.is_empty() {
            "loss_r r CROSS JOIN cot_user k".to_string()
        } else {
            format!(
                "loss_r r JOIN cot_user k ON {}",
                dims.iter()
                    .map(|d| format!("r.{d} = k.{d}"))
                    .collect::<Vec<_>>()
                    .join(" AND ")
            )
        };
        let weighted = format!(
            "WITH {}, loss_r AS (SELECT * FROM r{} c) SELECT SUM(r.v * k.cv) AS loss FROM {on}",
            case.ctes(),
            case.root
        );
        match grad_of(ctx, &weighted, wrt).await {
            Ok((_, g)) => {
                out.meta_compared += 1;
                if let Some(f) = compare("vjp-vs-weighted-grad", &g, &vjp_grads[k], 1.0, 1e-8) {
                    out.fail(f);
                }
            }
            Err(Refusal::Bug(b)) => out.fail(format!("[vjp-vs-weighted-grad] {b}")),
            Err(Refusal::Allowed(_)) => {}
        }
        ctx.deregister_table("cot_user")
            .map_err(|e| e.to_string())?;
    }

    // Linearity: vjp(c0 + 2 c1) = vjp(c0) + 2 vjp(c1), checked by building
    // the combined cotangent's expected gradient from the two.
    let combined: Vec<f64> = cots[0]
        .iter()
        .zip(&cots[1])
        .map(|(a, b)| a + 2.0 * b)
        .collect();
    let mut fields = Vec::new();
    let mut cols: Vec<ArrayRef> = Vec::new();
    for name in &program.cotangent {
        if let Some(k) = dims.iter().position(|d| d == name) {
            fields.push(Field::new(name, kt.data_type(), false));
            cols.push(
                kt.array(
                    rows.rows
                        .iter()
                        .map(|r| r[k].unwrap_or(0.0) as i64)
                        .collect(),
                ),
            );
        } else {
            fields.push(Field::new(name, DataType::Float64, true));
            cols.push(Arc::new(Float64Array::from(combined.clone())));
        }
    }
    let batch =
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).map_err(|e| e.to_string())?;
    register_batch(ctx, &program.cotangent_table, batch)?;
    match run_and_read(ctx, &program).await {
        Ok(g) => {
            out.meta_compared += 1;
            let mut want = vjp_grads[0].clone();
            for (t, grad) in want.iter_mut() {
                for (key, v) in grad.iter_mut() {
                    let other = vjp_grads[1]
                        .get(t)
                        .and_then(|g| g.get(key))
                        .copied()
                        .flatten();
                    *v = match (*v, other) {
                        (Some(a), Some(b)) => Some(a + 2.0 * b),
                        _ => None,
                    };
                }
            }
            if let Some(f) = compare("vjp-linear", &want, &g, 1.0, 1e-8) {
                out.fail(f);
            }
        }
        Err(e) => out.fail(format!("[vjp-linear] {e}")),
    }
    ctx.deregister_table(program.cotangent_table.as_str())
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn register_batch(ctx: &SessionContext, name: &str, batch: RecordBatch) -> Result<(), String> {
    let table = MemTable::try_new(batch.schema(), vec![vec![batch]]).map_err(|e| e.to_string())?;
    ctx.deregister_table(name).map_err(|e| e.to_string())?;
    ctx.register_table(name, Arc::new(table))
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn ddx_tables(ctx: &SessionContext) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for c in ctx.catalog_names() {
        let Some(cat) = ctx.catalog(&c) else { continue };
        for s in cat.schema_names() {
            let Some(schema) = cat.schema(&s) else {
                continue;
            };
            for t in schema.table_names() {
                if t.starts_with("__ddx_") {
                    out.insert(t);
                }
            }
        }
    }
    out
}

/// The program's contract with an adapter: reuse across values, the dims
/// check, and the catalog it leaves behind.
async fn contract_checks(
    rng: &mut Rng,
    case: &Case,
    sql: &str,
    wrt: &[ColumnRef],
    program: &BackwardProgram,
    grads: &BTreeMap<String, Grad>,
    out: &mut Outcome,
) -> Result<(), String> {
    // After run: exactly the value and the gradients remain.
    let ctx = setup(case, 4).await?;
    let p = match ad::grad(&ctx, sql, wrt).await {
        Ok(p) => p,
        Err(e) => {
            out.fail(format!(
                "[determinism] a second grad of the same query failed: {e}"
            ));
            return Ok(());
        }
    };
    if let Err(e) = ad::run(&ctx, &p).await {
        out.fail(format!("[accepted-but-failed] {e}"));
        return Ok(());
    }
    let want: BTreeSet<String> = std::iter::once(p.value.clone())
        .chain(p.gradients.iter().map(|g| g.step.clone()))
        .collect();
    let have = ddx_tables(&ctx);
    if have != want {
        out.fail(format!(
            "[catalog] after run the context holds {have:?}, expected {want:?}"
        ));
    }
    // The user's tables are untouched.
    for t in &case.tables {
        let r = query(
            &ctx,
            &format!(
                "SELECT {}val FROM {}",
                lead(&t.dims.iter().map(String::as_str).collect::<Vec<_>>(), ""),
                t.name
            ),
        )
        .await?;
        let mut got: Vec<(Vec<i64>, Option<f64>)> = r
            .rows
            .iter()
            .map(|row| {
                (
                    row[..row.len() - 1]
                        .iter()
                        .map(|v| v.map_or(NULL_KEY, |v| v as i64))
                        .collect(),
                    row[row.len() - 1],
                )
            })
            .collect();
        let mut want: Vec<(Vec<i64>, Option<f64>)> =
            t.keys.iter().cloned().zip(t.vals.iter().copied()).collect();
        // By bits, so a NaN equals itself, and by value as well as key, since
        // data may repeat a key.
        let bits = |rows: &mut Vec<(Vec<i64>, Option<f64>)>| {
            let mut b: Vec<(Vec<i64>, Option<u64>)> = rows
                .iter()
                .map(|(k, v)| (k.clone(), v.map(f64::to_bits)))
                .collect();
            b.sort();
            b
        };
        if bits(&mut got) != bits(&mut want) {
            out.fail(format!(
                "[catalog] running the program changed the user's table {}",
                t.name
            ));
        }
    }
    if let Err(e) = ad::release(&ctx, &p) {
        out.fail(format!("[catalog] release failed: {e}"));
    }
    let left = ddx_tables(&ctx);
    if !left.is_empty() {
        out.fail(format!(
            "[catalog] after release the context still holds {left:?}"
        ));
    }

    // Two programs on one context at once, as `ad::sql` from two tasks would
    // run them: each has its own prefix, so neither may see the other's
    // tables (S11). Spawned, so they really do interleave on the runtime.
    {
        let ctx = Arc::new(setup(case, 4).await?);
        let spell = [
            sql.to_string(),
            case.inline_sql(),
            case.with_loss("2.0 * loss", "loss"),
        ];
        let mut tasks = Vec::new();
        for q in spell.clone() {
            let (ctx, wrt) = (ctx.clone(), wrt.to_vec());
            tasks.push(tokio::spawn(async move {
                match grad_of(&ctx, &q, &wrt).await {
                    Ok((_, g)) => Some(Ok(g)),
                    Err(Refusal::Bug(b)) => Some(Err(b)),
                    Err(Refusal::Allowed(_)) => None,
                }
            }));
        }
        for (k, t) in tasks.into_iter().enumerate() {
            match t.await {
                Ok(Some(Ok(g))) => {
                    out.meta_compared += 1;
                    let factor = if k == 2 { 2.0 } else { 1.0 };
                    if let Some(f) = compare("concurrent", grads, &g, factor, META_RTOL) {
                        out.fail(format!("{f}\n  three programs ran at once on one context"));
                    }
                }
                Ok(Some(Err(b))) => out.fail(format!("[concurrent] {b}")),
                Ok(None) => {}
                Err(e) => out.fail(format!("[concurrent] a task panicked: {e}")),
            }
        }
        let left = ddx_tables(&ctx);
        let kept = left
            .iter()
            .filter(|t| !t.contains("value") && !t.contains("grad_"))
            .count();
        if kept > 0 {
            out.fail(format!(
                "[catalog] concurrent runs left intermediates {left:?}"
            ));
        }
    }

    // A program is built once and run on every training step: new values in
    // the same tables must give the fresh program's gradient.
    let mut moved = case.clone();
    for t in moved.tables.iter_mut().filter(|t| t.param) {
        for v in t.vals.iter_mut().flatten() {
            *v = round6(*v + rng.range(-0.3, 0.3));
        }
    }
    // Sometimes a training step also sees a different number of rows.
    let grow = rng.below(3) == 0;
    if grow {
        for t in moved
            .tables
            .iter_mut()
            .filter(|t| t.param && !t.dims.is_empty())
        {
            let mut k: Vec<i64> = t.keys[0].clone();
            k[0] = 77;
            t.keys.push(k);
            t.vals.push(Some(0.125));
        }
    }
    let ctx = setup(&moved, 4).await?;
    if loss(&ctx, sql).await?.is_some() {
        let fresh = grad_of(&ctx, sql, wrt).await;
        let reused = run_and_read(&ctx, program).await;
        match (fresh, reused) {
            (Ok((_, f)), Ok(r)) => {
                out.meta_compared += 1;
                if let Some(msg) = compare(
                    if grow { "reuse-grown" } else { "reuse" },
                    &f,
                    &r,
                    1.0,
                    META_RTOL,
                ) {
                    out.fail(format!(
                        "{msg}\n  a program built at θ0 and run at θ1 disagrees with one built at θ1"
                    ));
                }
            }
            (Ok(_), Err(e)) => out.fail(format!(
                "[reuse] the reused program failed on new values: {e}"
            )),
            _ => {}
        }
    }

    // Repeated dims must be refused by the checks, never answered.
    let target = rng.pick(&case.wrt).clone();
    let mut dup = case.clone();
    {
        let t = dup.tables.iter_mut().find(|t| t.name == target).unwrap();
        if !t.dims.is_empty() {
            let r = rng.below(t.keys.len() as u64) as usize;
            t.keys.push(t.keys[r].clone());
            t.vals.push(Some(0.5));
        }
    }
    if !dup.table(&target).dims.is_empty() {
        let ctx = setup(&dup, 4).await?;
        if let Ok(p) = ad::grad(&ctx, sql, wrt).await {
            match ad::run(&ctx, &p).await {
                Ok(()) => out.fail(format!(
                    "[dims-check] {target} has two rows with the same dims, and the program ran \
                     instead of refusing"
                )),
                Err(e) => {
                    let msg = e.to_string();
                    if !msg.contains("invalid wrt") {
                        out.fail(format!("[dims-check] refused, but not by the check: {msg}"));
                    }
                }
            }
            let left = ddx_tables(&ctx);
            if !left.is_empty() {
                out.fail(format!(
                    "[catalog] a refused run left {left:?} on the context"
                ));
            }
        }
    }
    let _ = grads;
    Ok(())
}

/// Whether to spell `t`'s column upper-case: a fixed choice per table name,
/// so both statements for one table agree.
fn rng_upper(t: &str) -> bool {
    t.bytes().map(u32::from).sum::<u32>() % 2 == 0
}

/// `grad(loss, t.val)` in SQL, and an SGD step written as a join.
async fn surface_checks(
    rng: &mut Rng,
    case: &Case,
    grads: &BTreeMap<String, Grad>,
    out: &mut Outcome,
) -> Result<(), String> {
    let ctx = setup(case, 4).await?;
    // The statement is text ddx rewrites before the engine sees it, so it is
    // spelled the ways a user might: the call's case and spacing, a comment
    // that mentions grad( and holds multibyte characters, a quoted CTE name,
    // an upper-case column, and CTEs on either side of the loss.
    let name = *rng.pick(&["loss", "loss", "\"Loss Fn\"", "l_0"]);
    let call = *rng.pick(&["grad(", "GRAD(", "Grad (", "grad\n  ("]);
    let comment = *rng.pick(&["", "/* grad(loss, w.val) — café ☕ */ ", "-- grad(\n"]);
    let before = if rng.below(3) == 0 {
        "pre AS (SELECT 1 AS one), "
    } else {
        ""
    };
    let after = if rng.below(3) == 0 {
        ", post AS (SELECT 2 AS two)"
    } else {
        ""
    };
    let loss_cte = format!(
        "{comment}WITH {before}{}, {name} AS (SELECT {} AS loss FROM r{} c){after}",
        case.ctes(),
        case.head,
        case.root
    );
    let column = |t: &str| {
        if rng_upper(t) {
            format!("{}.VAL", t.to_uppercase())
        } else {
            format!("{t}.val")
        }
    };
    let mut statements = Vec::new();
    for t in &case.wrt {
        let tb = case.table(t);
        let dims: Vec<&str> = tb.dims.iter().map(String::as_str).collect();
        statements.push(format!(
            "{loss_cte} SELECT {}val FROM {call}{name}, {})",
            lead(&dims, ""),
            column(t)
        ));
        let on = if dims.is_empty() {
            "ON true".to_string()
        } else {
            format!(
                "ON {}",
                dims.iter()
                    .map(|d| format!("p.{d} = g.{d}"))
                    .collect::<Vec<_>>()
                    .join(" AND ")
            )
        };
        statements.push(format!(
            "{loss_cte} SELECT {}p.val - 0.1 * g.val AS val FROM {t} p JOIN {call}{name}, {}) AS g {on}",
            lead(&dims, "p."),
            column(t)
        ));
    }
    let refs: Vec<&str> = statements.iter().map(String::as_str).collect();
    let frames = match ad::sql_all(&ctx, &refs).await {
        Ok(f) => f,
        Err(e) => {
            match classify(e) {
                Refusal::Bug(b) => out.fail(format!("[sql] {b}")),
                // Allowed, but the same loss planned through the SQL surface
                // is refused where the program API accepted it: coverage
                // that depends on plan shape, tallied, not failed.
                Refusal::Allowed(why) => out.engine.push(format!("sql refused: {why}")),
            }
            return Ok(());
        }
    };
    let left = ddx_tables(&ctx);
    if !left.is_empty() {
        out.fail(format!(
            "[catalog] ad::sql_all left {left:?} on the context"
        ));
    }
    for (k, df) in frames.into_iter().enumerate() {
        let t = &case.wrt[k / 2];
        let tb = case.table(t);
        let batches = match df.collect().await {
            Ok(b) => b,
            Err(e) => {
                out.fail(format!(
                    "[sql] statement {k} failed: {e}\n  {}",
                    statements[k]
                ));
                continue;
            }
        };
        let mut got = Grad::new();
        for b in &batches {
            for r in 0..b.num_rows() {
                let n = b.num_columns();
                let key: Vec<i64> = (0..n - 1)
                    .map(|c| cell(b.column(c), r).unwrap_or(f64::NAN) as i64)
                    .collect();
                got.insert(key, cell(b.column(n - 1), r));
            }
        }
        let want: Grad = if k % 2 == 0 {
            grads[t].clone()
        } else {
            tb.keys
                .iter()
                .zip(&tb.vals)
                .map(|(key, v)| {
                    let g = grads[t].get(key).copied().flatten();
                    (key.clone(), v.zip(g).map(|(v, g)| v - 0.1 * g))
                })
                .collect()
        };
        let label = if k % 2 == 0 { "sql-grad" } else { "sql-sgd" };
        let wrap = |g: Grad| BTreeMap::from([(t.clone(), g)]);
        out.meta_compared += 1;
        if let Some(f) = compare(label, &wrap(want), &wrap(got), 1.0, META_RTOL) {
            out.fail(format!("{f}\n  statement: {}", statements[k]));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Driving it.
// ---------------------------------------------------------------------------

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("a runtime")
}

/// Run one case, turning a panic into a failure.
fn run_one(rt: &tokio::runtime::Runtime, seed: u64, props: Props) -> Outcome {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rt.block_on(check_case(seed, props))
    })) {
        Ok(o) => o,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic>".into());
            let (case, _) = case_for(seed);
            Outcome {
                failures: vec![format!("[panic] {msg}\n{}", case.describe())],
                ..Outcome::default()
            }
        }
    }
}

/// The tally a run reports.
#[derive(Default)]
struct Tally {
    cases: u64,
    accepted: u64,
    failures: u64,
    fd_compared: u64,
    fd_screened: u64,
    fd_bracketed: u64,
    meta_compared: u64,
    refusals: BTreeMap<String, u64>,
    kinds: BTreeMap<&'static str, u64>,
    engine: BTreeMap<String, u64>,
    rewrites: BTreeMap<String, u64>,
}

impl Tally {
    fn add(&mut self, o: &Outcome) {
        self.cases += 1;
        self.accepted += o.accepted as u64;
        self.failures += o.failures.len() as u64;
        self.fd_compared += o.fd_compared as u64;
        self.fd_screened += o.fd_screened as u64;
        self.fd_bracketed += o.fd_bracketed as u64;
        self.meta_compared += o.meta_compared as u64;
        if let Some(r) = &o.refusal {
            *self.refusals.entry(r.clone()).or_default() += 1;
        }
        for e in &o.engine {
            *self.engine.entry(e.clone()).or_default() += 1;
        }
        for r in &o.rewrites {
            *self.rewrites.entry(r.clone()).or_default() += 1;
        }
        if o.accepted {
            for k in &o.kinds {
                *self.kinds.entry(k).or_default() += 1;
            }
        }
    }

    fn summary(&self) -> String {
        let mut s = format!(
            "cases={} accepted={} failures={} fd_compared={} fd_bracketed={} fd_screened={} \
             meta_compared={}",
            self.cases,
            self.accepted,
            self.failures,
            self.fd_compared,
            self.fd_bracketed,
            self.fd_screened,
            self.meta_compared
        );
        let _ = write!(s, "\n  accepted cases by primitive: {:?}", self.kinds);
        if !self.rewrites.is_empty() {
            let _ = write!(s, "\n  plan rewrites compared: {:?}", self.rewrites);
        }
        if !self.engine.is_empty() {
            let _ = write!(
                s,
                "\n  engine faults and SQL-only refusals (not failures): {:?}",
                self.engine
            );
        }
        let mut refusals: Vec<(&String, &u64)> = self.refusals.iter().collect();
        refusals.sort_by(|a, b| b.1.cmp(a.1));
        for (r, n) in refusals.iter().take(12) {
            let _ = write!(s, "\n  refused {n}×: {r}");
        }
        s
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A bounded run over seeds `0..n`: every failure is collected, then the run
/// asserts it was clean and that enough cases were actually checked.
fn bounded(label: &str, salt: u64, n: u64, props: Props, min_accepted: u64) {
    let rt = runtime();
    let mut tally = Tally::default();
    // ddx-core's reporter: every failure tagged with its seed, and a floor on
    // the cases actually exercised, so a generator whose queries ddx refuses
    // wholesale cannot pass by checking nothing.
    let mut fail = Failures::new();
    for k in 0..n {
        let seed = salt.wrapping_add(k);
        let o = run_one(&rt, seed, props);
        tally.add(&o);
        if o.accepted {
            fail.tested();
        }
        for f in &o.failures {
            fail.push(seed, f.clone());
        }
    }
    eprintln!("{label}: {}", tally.summary());
    fail.assert_clean(label, min_accepted as u32);
}

/// Seeds per bounded property group. Each case runs a dozen or more programs,
/// so this stays small for `cargo test`; the soak is where the volume is.
fn seeds() -> u64 {
    env_u64("DDX_V2_SEEDS", 24)
}

const NONE: Props = Props {
    fd: false,
    calculus: false,
    vjp: false,
    invariance: false,
    contract: false,
    surface: false,
    shapes: false,
    names: false,
    train: false,
};

#[test]
fn gradients_agree_with_finite_differences_of_the_query() {
    bounded(
        "finite differences",
        0,
        seeds(),
        Props { fd: true, ..NONE },
        seeds() / 3,
    );
}

#[test]
fn gradients_obey_the_calculus() {
    bounded(
        "calculus",
        1_000,
        seeds(),
        Props {
            calculus: true,
            ..NONE
        },
        seeds() / 3,
    );
}

#[test]
fn vjp_is_grads_transpose() {
    bounded(
        "vjp",
        2_000,
        seeds(),
        Props { vjp: true, ..NONE },
        seeds() / 3,
    );
}

#[test]
fn gradients_do_not_depend_on_how_the_query_is_run() {
    bounded(
        "invariance",
        3_000,
        seeds(),
        Props {
            invariance: true,
            ..NONE
        },
        seeds() / 3,
    );
}

#[test]
fn programs_keep_their_contract() {
    bounded(
        "contract",
        4_000,
        seeds(),
        Props {
            contract: true,
            ..NONE
        },
        seeds() / 3,
    );
}

#[test]
fn gradients_do_not_depend_on_the_plans_shape() {
    bounded(
        "plan shapes",
        6_000,
        seeds(),
        Props {
            shapes: true,
            ..NONE
        },
        seeds() / 3,
    );
}

#[test]
fn grad_in_sql_survives_names_and_training_loops() {
    bounded(
        "names and training",
        7_000,
        seeds(),
        Props {
            names: true,
            train: true,
            ..NONE
        },
        seeds() / 3,
    );
}

#[test]
fn grad_in_sql_is_the_programs_gradient() {
    bounded(
        "sql surface",
        5_000,
        seeds(),
        Props {
            surface: true,
            ..NONE
        },
        seeds() / 3,
    );
}

/// Run `case`'s program one step at a time, printing what each step holds:
/// for finding the step where a gradient goes wrong.
async fn debug_steps(case: &Case) {
    let ctx = match setup(case, 4).await {
        Ok(c) => c,
        Err(e) => return eprintln!("setup: {e}"),
    };
    let program = match ad::grad(&ctx, &case.loss_sql(), &case.wrt_refs()).await {
        Ok(p) => p,
        Err(e) => return eprintln!("grad: {e}"),
    };
    // Recomputation assumes a relation comes out the same every time it is
    // computed; report any that does not, bit for bit.
    for (k, n) in case.nodes.iter().enumerate() {
        let order: Vec<String> = n
            .dims
            .iter()
            .map(|d| d.to_string())
            .chain(["v".into()])
            .collect();
        let sql = format!(
            "WITH {} SELECT * FROM r{k} ORDER BY {}",
            case.ctes(),
            order.join(", ")
        );
        let mut seen = BTreeSet::new();
        for _ in 0..30 {
            if let Ok(r) = query(&ctx, &sql).await {
                let bits: Vec<Option<u64>> = r
                    .rows
                    .iter()
                    .flat_map(|row| row.iter().map(|c| c.map(f64::to_bits)))
                    .collect();
                seen.insert(bits);
            }
        }
        if seen.len() > 1 {
            eprintln!(
                "r{k} is not bit-reproducible: {} distinct results in 30 runs",
                seen.len()
            );
        }
    }
    for step in program.steps() {
        if let Err(e) = ad::run_step(&ctx, step).await {
            return eprintln!("step {} failed: {e}", step.name);
        }
        match query(&ctx, &format!("SELECT * FROM \"{}\"", step.name)).await {
            Ok(r) => {
                eprintln!("== {} {:?}: {} rows", step.name, r.names, r.rows.len());
                for row in r.rows.iter().take(12) {
                    eprintln!("   {row:?}");
                }
            }
            Err(e) => eprintln!("== {}: {e}", step.name),
        }
    }
}

/// Print one seed's case and run every property on it:
/// `DDX_V2_SEED=<seed> cargo test … -- --ignored --nocapture replay_one_seed`.
#[test]
#[ignore]
fn replay_one_seed() {
    let seed = env_u64("DDX_V2_SEED", 0);
    let (case, _) = case_for(seed);
    eprintln!("seed {seed}:\n{}", case.describe());
    eprintln!("inline = {}", case.inline_sql());
    if std::env::var("DDX_V2_DEBUG").is_ok() {
        runtime().block_on(debug_steps(&case));
    }
    let o = run_one(&runtime(), seed, soak_props());
    let mut tally = Tally::default();
    tally.add(&o);
    eprintln!("{}", tally.summary());
    for f in &o.failures {
        eprintln!("\nFAILURE: {f}");
    }
    assert!(o.failures.is_empty());
}

/// The property groups a soak runs: all of them, or the comma-separated
/// names in `DDX_V2_PROPS` (`fd,calculus,vjp,invariance,contract,surface,
/// shapes,names,train`), to spend a soak's budget on one surface.
fn soak_props() -> Props {
    let Ok(names) = std::env::var("DDX_V2_PROPS") else {
        return ALL;
    };
    let on = |n: &str| names.split(',').any(|x| x.trim() == n);
    Props {
        fd: on("fd"),
        calculus: on("calculus"),
        vjp: on("vjp"),
        invariance: on("invariance"),
        contract: on("contract"),
        surface: on("surface"),
        shapes: on("shapes"),
        names: on("names"),
        train: on("train"),
    }
}

/// The long-running soak: every property on fresh seeds for a wall-clock
/// budget. Same knobs and log lines as ddx-core's soak.
#[test]
#[ignore]
fn soak_v2_query_ad() {
    use std::time::Instant;

    let budget = env_u64("DDX_SOAK_SECS", 30);
    let base = env_u64("DDX_SOAK_BASE", 0);
    let log_path = std::env::var("DDX_SOAK_LOG").ok();
    let mut log = log_path.as_ref().map(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .unwrap_or_else(|e| panic!("cannot open DDX_SOAK_LOG `{p}`: {e}"))
    });
    let mut logline = |s: &str| {
        eprintln!("{s}");
        if let Some(f) = log.as_mut() {
            let _ = writeln!(f, "{s}");
            let _ = f.flush();
        }
    };

    SOAKING.store(true, Ordering::Relaxed);
    let props = soak_props();
    let rt = runtime();
    let start = Instant::now();
    let mut tally = Tally::default();
    let mut iters = 0u64;
    let mut last_beat = 0u64;
    logline(&format!(
        "SOAK start: budget={budget}s base={base} log={log_path:?}"
    ));
    logline(
        "REPRO DDX_SOAK_SECS=15 DDX_SOAK_BASE={seed} cargo test -p ddx-datafusion \
         --test ad_simulation --release -- --ignored --nocapture soak_v2_query_ad",
    );
    while start.elapsed().as_secs() < budget {
        let seed = base.wrapping_add(iters);
        let o = run_one(&rt, seed, props);
        tally.add(&o);
        for f in &o.failures {
            logline(&format!("\nFAILURE (seed={seed}, base={base}):\n{f}"));
        }
        iters += 1;
        let elapsed = start.elapsed().as_secs();
        if elapsed >= last_beat + 10 {
            last_beat = elapsed;
            logline(&format!(
                "HEARTBEAT elapsed={elapsed}s iters={iters} failures={} accepted={} fd={} meta={}",
                tally.failures, tally.accepted, tally.fd_compared, tally.meta_compared
            ));
        }
    }
    logline(&format!(
        "SOAK done: elapsed={}s iters={iters} failures={} base={base} next_base={}\n  {}",
        start.elapsed().as_secs(),
        tally.failures,
        base.wrapping_add(iters),
        tally.summary()
    ));
    assert_eq!(
        tally.failures, 0,
        "the v2 soak found {} failure(s); see FAILURE lines",
        tally.failures
    );
}
