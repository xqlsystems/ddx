// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The SQL surface's fast paths, on the soak's cases: none may move a
//! gradient.
//!
//! Each makes ddx do less work by trusting something about a statement, and
//! each is wrong in a way no other check sees if that trust is misplaced:
//!
//! - **the split** (a CTE read more than once is planned on its own) trusts
//!   that it resolves CTE names as the engine does. A *decoy* table under
//!   each CTE's name, holding other values, makes a misresolved read a wrong
//!   number rather than a coincidence. The names are spelled at random as the
//!   engine resolves them alike (`r3`, `R3`, `"r3"`), and a CTE is read inside
//!   a subquery, which the split must leave to the engine along with what it
//!   reads.
//! - **views** (a view the objective reads is written out as CTEs) trust
//!   that the view's SQL is what the engine reads. Some of the case's CTEs
//!   become views, a chain of them; some are registered from a DataFrame,
//!   with no SQL; the whole loss may be a view; a decoy schema holds views of
//!   the same names.
//! - **the program cache** trusts that its key changes whenever the
//!   statement's meaning does. A statement runs, then something it reads
//!   changes without its text changing (a table's values or column type, a
//!   view's definition, a decoy appearing), and it runs again.
//! - **shared relations** (`ReferenceRel`, which ddx's split writes and
//!   DuckDB's producer will) trust that reverse mode, forward mode and their
//!   composition read a DAG as the tree it stands for. The engine's plan has
//!   repeated subtrees moved into relations of their own at random (some
//!   read once, some nested), and `grad`, `jvp` and `jvp` of `grad` must not
//!   move.
//!
//! The oracle for the first three is the programmatic API on the same loss
//! text, in the same context: DataFusion plans it whole, with every view and
//! CTE inlined by the engine's own rules, and ddx differentiates that tree.
//! For the last, it is the same program built from the tree.

use std::collections::{BTreeMap, HashMap};

use datafusion::arrow::compute::cast;
use ddx_ad::substrait::proto::plan_rel::RelType as PlanRelType;
use ddx_ad::substrait::proto::rel::RelType;
use ddx_ad::substrait::proto::{Plan, PlanRel, ReferenceRel, Rel};

use super::jvp::{run_jvp, Tan};
use super::*;

pub(super) async fn perf_checks(
    rng: &mut Rng,
    case: &Case,
    sql: &str,
    grads: &BTreeMap<String, Grad>,
    out: &mut Outcome,
) -> Result<(), String> {
    let t = rng.pick(&case.wrt).clone();
    match rng.below(4) {
        0 => decoy_checks(rng, case, &t, out).await,
        1 => view_checks(rng, case, &t, out).await,
        2 => cache_checks(rng, case, &t, out).await,
        _ => shared_checks(rng, case, sql, grads, out).await,
    }
}

/// How a statement spells the loss: its CTEs, and the loss over them.
struct Spelling {
    /// `WITH …` clause entries, joined, or empty.
    ctes: String,
    /// The loss's query over them: `SELECT head AS loss FROM …`.
    loss: String,
}

impl Spelling {
    /// The loss as one query, for the programmatic oracle.
    fn query(&self) -> String {
        if self.ctes.is_empty() {
            self.loss.clone()
        } else {
            format!("WITH {} {}", self.ctes, self.loss)
        }
    }

    /// `grad(loss, val)` of table `t` in SQL.
    fn statement(&self, case: &Case, t: &str) -> String {
        let dims: Vec<&str> = case.table(t).dims.iter().map(String::as_str).collect();
        let with = if self.ctes.is_empty() {
            String::new()
        } else {
            format!("{}, ", self.ctes)
        };
        format!(
            "WITH {with}loss AS ({}) SELECT {}grad(loss, val) AS val FROM {t}",
            self.loss,
            lead(&dims, "")
        )
    }
}

/// Node `k`'s body, each relation it reads named by `name`.
fn render(case: &Case, k: usize, name: &mut dyn FnMut(usize) -> String) -> String {
    let mut out = String::new();
    for (i, part) in case.nodes[k].body.split('§').enumerate() {
        match (i % 2, part.parse::<usize>()) {
            (1, Ok(n)) => out.push_str(&name(n)),
            _ => out.push_str(part),
        }
    }
    out
}

/// A name for CTE `k` as the engine resolves it alike: unquoted in either
/// case, or quoted in lower case.
fn spelled(rng: &mut Rng, k: usize) -> String {
    match rng.below(4) {
        0 => format!("R{k}"),
        1 => format!("\"r{k}\""),
        _ => format!("r{k}"),
    }
}

/// The relation `r{k}` holds, its value moved, as a table of that name: a
/// read that reaches it instead of the CTE gives another number.
async fn register_decoy(
    ctx: &SessionContext,
    case: &Case,
    k: usize,
    name: &str,
) -> Result<(), String> {
    let dims = case.nodes[k].dims.clone();
    let sql = format!(
        "WITH {} SELECT {}v * -0.5 + 0.37 AS v FROM r{k}",
        case.ctes(),
        lead(&dims, "")
    );
    let batches = ctx
        .sql(&sql)
        .await
        .map_err(|e| e.to_string())?
        .collect()
        .await
        .map_err(|e| e.to_string())?;
    let Some(first) = batches.first() else {
        return Ok(());
    };
    let table =
        MemTable::try_new(first.schema(), vec![batches.clone()]).map_err(|e| e.to_string())?;
    ctx.register_table(
        datafusion::common::TableReference::bare(name),
        Arc::new(table),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// The case's CTEs spelled at random, a decoy under each name (and under
/// its upper-case quoted name, which no read names), and perhaps a CTE read
/// inside a subquery of the loss.
async fn decoy_checks(
    rng: &mut Rng,
    case: &Case,
    t: &str,
    out: &mut Outcome,
) -> Result<(), String> {
    let ctx = setup(case, 4).await?;
    let n = case.nodes.len();
    for k in 0..n {
        // A relation without a `v` (none the generator makes) has no decoy.
        let _ = register_decoy(&ctx, case, k, &format!("r{k}")).await;
        if rng.below(2) == 0 {
            let _ = register_decoy(&ctx, case, k, &format!("R{k}")).await;
        }
    }
    let mut ctes: Vec<String> = (0..n)
        .map(|k| {
            let body = render(case, k, &mut |i| spelled(rng, i));
            format!("{} AS ({body})", spelled(rng, k))
        })
        .collect();
    // The loss reads the root twice, through `rt`, so the root is planned on
    // its own (as most of the soak's cases, which read each CTE once, never
    // are) unless something rules it out. The oracle reads the same text.
    let root = &case.nodes[case.root];
    let (a, b) = (spelled(rng, case.root), spelled(rng, case.root));
    ctes.push(if root.dims.is_empty() {
        format!("rt AS (SELECT a.* FROM {a} a CROSS JOIN {b} b)")
    } else {
        let on: Vec<String> = root.dims.iter().map(|d| format!("a.{d} = b.{d}")).collect();
        format!(
            "rt AS (SELECT a.* FROM {a} a JOIN {b} b ON {})",
            on.join(" AND ")
        )
    });
    let read_in_subquery = n > 1 && rng.below(2) == 0;
    let filter = if read_in_subquery {
        // Another CTE, on its values, which a decoy holds others of: a read
        // that reaches the decoy keeps other rows.
        let k = (case.root + 1 + rng.below(n as u64 - 1) as usize) % n;
        let from = spelled(rng, k);
        match root.dims.iter().find(|d| case.nodes[k].dims.contains(d)) {
            Some(d) => format!(" WHERE c.{d} IN (SELECT {d} FROM {from} WHERE v > 0)"),
            None => format!(" WHERE EXISTS (SELECT 1 FROM {from} WHERE v > 0)"),
        }
    } else {
        String::new()
    };
    let spelling = Spelling {
        ctes: ctes.join(", "),
        loss: format!("SELECT {} AS loss FROM rt c{filter}", case.head),
    };
    let label = if read_in_subquery {
        "perf-decoy-subquery"
    } else {
        "perf-decoy"
    };
    agree(label, &ctx, case, t, &spelling, out).await?;
    Ok(())
}

/// The case's first CTEs as views of the same names, the rest as CTEs.
async fn view_checks(rng: &mut Rng, case: &Case, t: &str, out: &mut Outcome) -> Result<(), String> {
    let ctx = setup(case, 4).await?;
    let Some((spelling, _)) = views_or_skip(rng, &ctx, case, out).await else {
        return Ok(());
    };
    agree("perf-view", &ctx, case, t, &spelling, out).await?;
    Ok(())
}

/// Make views of the first `c` CTEs, at random: from SQL, or (about one in
/// five) from a DataFrame, which has none. Returns the spelling of the loss
/// over them and the views made from SQL, by node, with their bodies.
async fn as_views(
    rng: &mut Rng,
    ctx: &SessionContext,
    case: &Case,
) -> Result<(Spelling, Vec<(usize, String)>), String> {
    let n = case.nodes.len();
    let c = 1 + rng.below(n as u64) as usize;
    let mut from_sql = Vec::new();
    for k in 0..c {
        let body = render(case, k, &mut |i| format!("r{i}"));
        if rng.below(5) == 0 {
            let df = ctx.sql(&body).await.map_err(|e| e.to_string())?;
            ctx.register_table(format!("r{k}").as_str(), df.into_view())
                .map_err(|e| e.to_string())?;
        } else {
            run(ctx, &format!("CREATE VIEW r{k} AS {body}")).await?;
            from_sql.push((k, body));
        }
    }
    // Views of the same names in another schema, which no read reaches.
    if rng.below(3) == 0 {
        run(ctx, "CREATE SCHEMA other").await?;
        for k in 0..c {
            let dims = case.nodes[k].dims.clone();
            run(
                ctx,
                &format!(
                    "CREATE VIEW other.r{k} AS SELECT {}v * -0.5 + 0.37 AS v FROM r{k}",
                    lead(&dims, "")
                ),
            )
            .await?;
        }
    }
    let rest: Vec<String> = (c..n)
        .map(|k| format!("r{k} AS ({})", render(case, k, &mut |i| format!("r{i}"))))
        .collect();
    let loss = format!("SELECT {} AS loss FROM r{} c", case.head, case.root);
    let spelling = if rng.below(3) == 0 {
        // The whole loss as a view, as xarray-sql's GPT defines its model.
        let with = if rest.is_empty() {
            String::new()
        } else {
            format!("WITH {} ", rest.join(", "))
        };
        run(ctx, &format!("CREATE VIEW lossv AS {with}{loss}")).await?;
        Spelling {
            ctes: String::new(),
            loss: "SELECT loss FROM lossv".into(),
        }
    } else {
        Spelling {
            ctes: rest.join(", "),
            loss,
        }
    };
    Ok((spelling, from_sql))
}

/// [`as_views`], or `None` (tallied) where DataFusion will not make them:
/// it plans a view's definition when it creates it.
async fn views_or_skip(
    rng: &mut Rng,
    ctx: &SessionContext,
    case: &Case,
    out: &mut Outcome,
) -> Option<(Spelling, Vec<(usize, String)>)> {
    match as_views(rng, ctx, case).await {
        Ok(v) => Some(v),
        Err(e) => {
            out.engine
                .push(format!("perf: DataFusion made no view: {}", short(&e)));
            None
        }
    }
}

async fn run(ctx: &SessionContext, sql: &str) -> Result<(), String> {
    ctx.sql(sql)
        .await
        .map_err(|e| format!("{e}\n  {sql}"))?
        .collect()
        .await
        .map_err(|e| format!("{e}\n  {sql}"))?;
    Ok(())
}

/// A statement run twice on one context, so its program is cached between,
/// with something it reads changed in between and its text not.
async fn cache_checks(
    rng: &mut Rng,
    case: &Case,
    t: &str,
    out: &mut Outcome,
) -> Result<(), String> {
    let ctx = setup(case, 4).await?;
    let views = rng.below(2) == 0;
    let (spelling, from_sql) = if views {
        let Some(v) = views_or_skip(rng, &ctx, case, out).await else {
            return Ok(());
        };
        v
    } else {
        (
            Spelling {
                ctes: case.ctes(),
                loss: format!("SELECT {} AS loss FROM r{} c", case.head, case.root),
            },
            Vec::new(),
        )
    };
    if !agree("perf-cache-first", &ctx, case, t, &spelling, out).await? {
        return Ok(());
    }
    // A DataFusion view keeps the tables it was created over (#172): after
    // one is registered again, the engine reads the old one and ddx, which
    // reads by name, the new, so no oracle agrees with both. Under views,
    // only a view's definition changes.
    let change = if views {
        if from_sql.is_empty() || rng.below(3) == 0 {
            "nothing"
        } else {
            // A view's definition: its value scaled.
            let (k, body) = &from_sql[rng.below(from_sql.len() as u64) as usize];
            let dims = case.nodes[*k].dims.clone();
            run(
                &ctx,
                &format!(
                    "CREATE OR REPLACE VIEW r{k} AS SELECT {}v * 1.5 AS v FROM ({body}) q",
                    lead(&dims, "")
                ),
            )
            .await?;
            "a view's definition"
        }
    } else {
        match rng.below(4) {
            0 => "nothing",
            1 => {
                for k in 0..case.nodes.len() {
                    let _ = register_decoy(&ctx, case, k, &format!("r{k}")).await;
                }
                "decoys under the CTEs' names"
            }
            2 if case.tables.iter().any(|tb| !case.wrt.contains(&tb.name)) => {
                // A data table's value column as FLOAT: the same values,
                // another type, which the plan casts. (A wrt column's
                // gradient has its column's type.)
                let data: Vec<&Table> = case
                    .tables
                    .iter()
                    .filter(|tb| !case.wrt.contains(&tb.name))
                    .collect();
                let tb = (*rng.pick(&data)).clone();
                let batch = tb.batch();
                let mut cols = batch.columns().to_vec();
                let last = cols.len() - 1;
                cols[last] = cast(&cols[last], &DataType::Float32).map_err(|e| e.to_string())?;
                let mut fields: Vec<Field> = batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.as_ref().clone())
                    .collect();
                fields[last] = Field::new("val", DataType::Float32, true);
                let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), cols)
                    .map_err(|e| e.to_string())?;
                register_batch(&ctx, &tb.name, batch)?;
                "a table's value type"
            }
            _ => {
                // A table's values.
                let mut tb = rng.pick(&case.tables).clone();
                for v in tb.vals.iter_mut().flatten() {
                    *v = round6(*v * 0.5 + 0.37);
                }
                tb.register(&ctx).map_err(|e| e.to_string())?;
                "a table's values"
            }
        }
    };
    let label = format!("perf-cache after {change}");
    agree(&label, &ctx, case, t, &spelling, out).await?;
    Ok(())
}

/// `spelling`'s gradient of `t` through the SQL surface against the
/// program built from its loss planned whole. Returns whether both ran.
async fn agree(
    label: &str,
    ctx: &SessionContext,
    case: &Case,
    t: &str,
    spelling: &Spelling,
    out: &mut Outcome,
) -> Result<bool, String> {
    let before = ddx_tables(ctx);
    let query = spelling.query();
    let wrt = [ColumnRef::new(t, "val")];
    let want = match ad::grad(ctx, query.as_str(), &wrt).await {
        Ok(program) => {
            let g = run_and_read(ctx, &program).await;
            let _ = ddx_datafusion::engine::release(ctx, &program);
            match g {
                Ok(g) => g,
                Err(e) => {
                    match engine_fault(&e) {
                        Some(kind) => out.engine.push(kind),
                        None => out.fail(format!("[{label}] the oracle: {e}\n  {query}")),
                    }
                    return Ok(false);
                }
            }
        }
        Err(e) => {
            // A spelling the engine or ddx may refuse is not a finding; an
            // internal error or invalid plan is.
            match classify(e) {
                Refusal::Bug(b) => out.fail(format!("[{label}] the oracle: {b}\n  {query}")),
                Refusal::Allowed(why) => out
                    .engine
                    .push(format!("{label}: the oracle refused: {why}")),
            }
            return Ok(false);
        }
    };
    let statement = spelling.statement(case, t);
    let got = match grad_rows(ctx, &statement).await {
        Ok(g) => BTreeMap::from([(t.to_string(), g)]),
        Err(e) => {
            sql::failed(label, &statement, &e, out);
            return Ok(false);
        }
    };
    out.meta_compared += 1;
    let rtol = if case.modes.extreme == Some(Extreme::Huge) {
        1e-6
    } else {
        META_RTOL
    };
    if let Some(f) = compare(label, &want, &got, 1.0, rtol) {
        if case.modes.ulps && case.windowed_extremes() {
            out.engine.push(TIE_EDGE.into());
        } else {
            out.fail(format!("{f}\n  {statement}"));
        }
    }
    let after = ddx_tables(ctx);
    let left: Vec<_> = after.difference(&before).collect();
    if !left.is_empty() {
        out.fail(format!(
            "[{label}] the statement left ddx tables {left:?}\n  {statement}"
        ));
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// Shared relations.
// ---------------------------------------------------------------------------

/// `grad`, `jvp`, and `jvp` of `grad`, of the engine's plan with subtrees
/// moved into relations of their own, against the same of the tree.
async fn shared_checks(
    rng: &mut Rng,
    case: &Case,
    sql: &str,
    grads: &BTreeMap<String, Grad>,
    out: &mut Outcome,
) -> Result<(), String> {
    let ctx = setup(case, 4).await?;
    let wrt = case.wrt_refs();
    let lp = ctx
        .sql(sql)
        .await
        .map_err(|e| e.to_string())?
        .into_optimized_plan()
        .map_err(|e| e.to_string())?;
    let Ok(tree) = to_substrait_plan(&lp, &ctx.state()) else {
        return Ok(());
    };
    let tree = *tree;
    let Some((dag, moved)) = share(&tree, rng) else {
        return Ok(());
    };
    let label = format!("perf-shared ({moved} relations)");
    out.rewrites.push(format!("shared ({moved})"));

    let refused = |what: &str, e: AdError, out: &mut Outcome| {
        if let Refusal::Bug(b) = classify(DataFusionError::External(Box::new(e))) {
            out.fail(format!("[{label}] {what}: {b}"));
        } else {
            out.rewrites.push(format!("shared, {what} refused"));
        }
    };

    // Reverse mode.
    let program = match ddx_ad::grad(&dag, &wrt) {
        Ok(p) => p,
        Err(e) => {
            refused("grad", e, out);
            return Ok(());
        }
    };
    match run_and_read(&ctx, &program).await {
        Ok(g) => {
            out.meta_compared += 1;
            if let Some(f) = compare("perf-shared-grad", grads, &g, 1.0, META_RTOL) {
                if case.modes.ulps && case.windowed_extremes() {
                    out.engine.push(TIE_EDGE.into());
                } else {
                    out.fail(format!("{f}\n  {label}"));
                }
            }
        }
        Err(e) => match engine_fault(&e) {
            Some(kind) => out.engine.push(kind),
            None => out.fail(format!("[{label}] grad: {e}")),
        },
    }
    let _ = ddx_datafusion::engine::release(&ctx, &program);

    // Forward mode, and forward over reverse, along one tangent.
    let sparse = rng.below(2) == 0;
    let tan = Tan::random(rng, case, sparse);
    let Ok(tree_program) = ddx_ad::grad(&tree, &wrt) else {
        return Ok(());
    };
    let pairs = [
        ("jvp", ddx_ad::jvp(&tree, &wrt), ddx_ad::jvp(&dag, &wrt)),
        (
            "jvp of grad",
            ddx_ad::jvp(&tree_program, &wrt),
            ddx_ad::jvp(&program, &wrt),
        ),
    ];
    for (what, want, got) in pairs {
        let (want, got) = match (want, got) {
            (Ok(w), Ok(g)) => (w, g),
            (Err(_), _) => continue,
            (Ok(_), Err(e)) => {
                refused(what, e, out);
                continue;
            }
        };
        let (want, got) = match (
            run_jvp(&ctx, case, &want, &tan, false).await,
            run_jvp(&ctx, case, &got, &tan, false).await,
        ) {
            (Ok(w), Ok(g)) => (w, g),
            (Err(_), _) => continue,
            (Ok(_), Err(e)) => {
                match engine_fault(&e) {
                    Some(kind) => out.engine.push(kind),
                    None => out.fail(format!("[{label}] {what}: {e}")),
                }
                continue;
            }
        };
        out.meta_compared += 1;
        if want.gradients.len() != got.gradients.len() {
            out.fail(format!(
                "[perf-shared-{what}] {} gradients, the tree's {}\n  {label}",
                got.gradients.len(),
                want.gradients.len()
            ));
            continue;
        }
        let tables = std::iter::once((&want.output, &got.output))
            .chain(want.gradients.iter().zip(&got.gradients));
        for (w, g) in tables {
            if let Some(f) = same_rows(w, g) {
                if case.modes.ulps && case.windowed_extremes() {
                    out.engine.push(TIE_EDGE.into());
                } else {
                    out.fail(format!("[perf-shared-{what}] {f}\n  {label}"));
                }
                break;
            }
        }
    }
    Ok(())
}

/// Whether two programs' rows agree, as sets, within the meta tolerance.
fn same_rows(want: &Rows, got: &Rows) -> Option<String> {
    if want.names != got.names {
        return Some(format!(
            "columns {:?}, the tree's {:?}",
            got.names, want.names
        ));
    }
    let sort = |rows: &[Vec<Option<f64>>]| {
        let mut rows = rows.to_vec();
        rows.sort_by(|a, b| {
            a.iter()
                .zip(b)
                .map(|(x, y)| {
                    let (x, y) = (
                        x.unwrap_or(f64::NEG_INFINITY),
                        y.unwrap_or(f64::NEG_INFINITY),
                    );
                    x.total_cmp(&y)
                })
                .find(|o| o.is_ne())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows
    };
    let (w, g) = (sort(&want.rows), sort(&got.rows));
    if w.len() != g.len() {
        return Some(format!("{} rows, the tree's {}", g.len(), w.len()));
    }
    let scale = w
        .iter()
        .flatten()
        .flatten()
        .filter(|v| v.is_finite())
        .fold(0.0f64, |m, v| m.max(v.abs()));
    for (a, b) in w.iter().zip(&g) {
        let ok = a.iter().zip(b).all(|(x, y)| match (x, y) {
            (None, None) => true,
            (Some(x), Some(y)) => {
                x == y
                    || (x.is_nan() && y.is_nan())
                    || (x - y).abs() <= META_RTOL * scale + META_ATOL
            }
            _ => false,
        });
        if !ok {
            return Some(format!("row {b:?}, the tree's {a:?}"));
        }
    }
    None
}

/// `plan` with subtrees moved into relations of their own and read through
/// `ReferenceRel`: most that repeat, a few read once, nested where one holds
/// another. `None` if nothing moved.
fn share(plan: &Plan, rng: &mut Rng) -> Option<(Plan, usize)> {
    let mut out = plan.clone();
    let PlanRelType::Root(root) = out.relations.first_mut()?.rel_type.as_mut()? else {
        return None;
    };
    let root = root.input.as_mut()?;
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    count(root, &mut counts, true);
    let chosen: BTreeSet<String> = counts
        .into_iter()
        .filter(|(_, n)| rng.below(100) < if *n > 1 { 70 } else { 8 })
        .map(|(k, _)| k)
        .collect();
    if chosen.is_empty() {
        return None;
    }
    let mut moved: Vec<Rel> = Vec::new();
    let mut ordinals: HashMap<String, i32> = HashMap::new();
    let new_root = {
        let mut ctx = Share {
            chosen: &chosen,
            moved: &mut moved,
            ordinals: &mut ordinals,
        };
        let mut r = root.clone();
        for input in inputs_mut(&mut r) {
            *input = ctx.rel(input);
        }
        r
    };
    if moved.is_empty() {
        return None;
    }
    *root = new_root;
    let n = moved.len();
    out.relations.extend(moved.into_iter().map(|r| PlanRel {
        rel_type: Some(PlanRelType::Rel(r)),
    }));
    Some((out, n))
}

/// Each subtree of `rel` but `rel` itself (the root stays put), by its
/// debug rendering, with how often it appears.
fn count(rel: &Rel, counts: &mut BTreeMap<String, usize>, top: bool) {
    if !top {
        *counts.entry(format!("{rel:?}")).or_default() += 1;
    }
    for input in inputs(rel) {
        count(input, counts, false);
    }
}

fn inputs(rel: &Rel) -> Vec<&Rel> {
    match rel.rel_type.as_ref() {
        Some(RelType::Filter(r)) => r.input.as_deref().into_iter().collect(),
        Some(RelType::Aggregate(r)) => r.input.as_deref().into_iter().collect(),
        Some(RelType::Project(r)) => r.input.as_deref().into_iter().collect(),
        Some(RelType::Sort(r)) => r.input.as_deref().into_iter().collect(),
        Some(RelType::Fetch(r)) => r.input.as_deref().into_iter().collect(),
        Some(RelType::Window(r)) => r.input.as_deref().into_iter().collect(),
        Some(RelType::Join(r)) => r
            .left
            .as_deref()
            .into_iter()
            .chain(r.right.as_deref())
            .collect(),
        Some(RelType::Cross(r)) => r
            .left
            .as_deref()
            .into_iter()
            .chain(r.right.as_deref())
            .collect(),
        Some(RelType::Set(r)) => r.inputs.iter().collect(),
        _ => Vec::new(),
    }
}

struct Share<'a> {
    chosen: &'a BTreeSet<String>,
    moved: &'a mut Vec<Rel>,
    ordinals: &'a mut HashMap<String, i32>,
}

impl Share<'_> {
    fn rel(&mut self, rel: &Rel) -> Rel {
        let key = format!("{rel:?}");
        if self.chosen.contains(&key) {
            if let Some(&o) = self.ordinals.get(&key) {
                return reference(o);
            }
        }
        let mut r = rel.clone();
        for input in inputs_mut(&mut r) {
            *input = self.rel(input);
        }
        if !self.chosen.contains(&key) {
            return r;
        }
        self.moved.push(r);
        // The root is relation 0; the moved ones follow it.
        let o = self.moved.len() as i32;
        self.ordinals.insert(key, o);
        reference(o)
    }
}

fn reference(ordinal: i32) -> Rel {
    Rel {
        rel_type: Some(RelType::Reference(ReferenceRel {
            subtree_ordinal: ordinal,
        })),
    }
}

/// A relation's inputs, every kind ddx reads.
fn inputs_mut(rel: &mut Rel) -> Vec<&mut Rel> {
    match rel.rel_type.as_mut() {
        Some(RelType::Filter(r)) => r.input.as_deref_mut().into_iter().collect(),
        Some(RelType::Aggregate(r)) => r.input.as_deref_mut().into_iter().collect(),
        Some(RelType::Project(r)) => r.input.as_deref_mut().into_iter().collect(),
        Some(RelType::Sort(r)) => r.input.as_deref_mut().into_iter().collect(),
        Some(RelType::Fetch(r)) => r.input.as_deref_mut().into_iter().collect(),
        Some(RelType::Window(r)) => r.input.as_deref_mut().into_iter().collect(),
        Some(RelType::Join(r)) => r
            .left
            .as_deref_mut()
            .into_iter()
            .chain(r.right.as_deref_mut())
            .collect(),
        Some(RelType::Cross(r)) => r
            .left
            .as_deref_mut()
            .into_iter()
            .chain(r.right.as_deref_mut())
            .collect(),
        Some(RelType::Set(r)) => r.inputs.iter_mut().collect(),
        _ => Vec::new(),
    }
}
