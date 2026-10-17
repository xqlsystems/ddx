// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad`, `vjp` and `jvp` columns where a user may put them and a random
//! generator rarely does, each against its meaning.
//!
//! A call is a column whose value at a row is what its program computed
//! there. So every statement here is written twice: with the call, and with
//! the program's result registered as a plain table and joined in where the
//! call stood. The two must return the same rows, whatever is around the
//! call: a self-join, `DISTINCT`, a window, an `ON` condition, `UNION ALL`,
//! a wildcard, an outer join and a `WHERE` ddx pushes into its program,
//! quoted and upper-case names, a `jvp` in a CTE read twice, an `ORDER BY`
//! by position, a cotangent that is another call.

use datafusion::arrow::array::{Array, AsArray};
use datafusion::arrow::datatypes::{DataType, Float64Type};
use datafusion::prelude::SessionContext;
use ddx_datafusion::ad::{self, ColumnRef};

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// Rows as sortable text: every column cast to DOUBLE, rounded to 9
/// significant digits, NULL as `∅`.
async fn rows(ctx: &SessionContext, df: datafusion::prelude::DataFrame) -> Vec<String> {
    let _ = ctx;
    let batches = df.collect().await.unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let cols: Vec<_> = (0..b.num_columns())
            .map(|c| datafusion::arrow::compute::cast(b.column(c), &DataType::Float64).unwrap())
            .collect();
        for r in 0..b.num_rows() {
            let row: Vec<String> = cols
                .iter()
                .map(|a| {
                    let a = a.as_primitive::<Float64Type>();
                    if a.is_null(r) {
                        "∅".to_string()
                    } else {
                        let v = a.value(r);
                        let v = if v == 0.0 { 0.0 } else { v };
                        format!("{v:.9e}")
                    }
                })
                .collect();
            out.push(row.join(" | "));
        }
    }
    out.sort();
    out
}

fn ctx() -> SessionContext {
    let ctx = SessionContext::new();
    ddx_datafusion::register_stop_gradient(&ctx);
    ctx
}

const LOSS: &str =
    "WITH loss AS (SELECT SUM(w.val * w.val * x.val) + SUM(sin(w.val)) AS l FROM w JOIN x ON w.i = x.i)";

async fn setup(ctx: &SessionContext) {
    exec(
        ctx,
        "CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 0.5), (1, -1.25), (2, 2.0), (3, NULL), (4, 0.75)",
    )
    .await;
    exec(
        ctx,
        "CREATE TABLE x (i BIGINT, val DOUBLE) AS VALUES (0, 3.0), (1, -0.5), (2, 1.5), (3, 2.0), (4, -2.0)",
    )
    .await;
    // The programs' results, as plain tables: gw(i, val) = ∂loss/∂w.val,
    // gx(i, val) = ∂loss/∂x.val.
    for (t, g) in [("w", "gw"), ("x", "gx")] {
        let p = ad::grad(
            ctx,
            &format!("{LOSS} SELECT l FROM loss"),
            &[ColumnRef::new(t, "val")],
        )
        .await
        .unwrap();
        let out = ad::run(ctx, &p, &[]).await.unwrap();
        let batches = out.gradients[0].clone().collect().await.unwrap();
        let table =
            datafusion::datasource::MemTable::try_new(batches[0].schema(), vec![batches]).unwrap();
        ctx.register_table(g, std::sync::Arc::new(table)).unwrap();
    }
}

/// `with_call` and `oracle` return the same rows.
async fn agree(label: &str, with_call: &str, oracle: &str) -> Result<(), String> {
    let ctx = ctx();
    setup(&ctx).await;
    let want = rows(
        &ctx,
        ctx.sql(oracle).await.map_err(|e| format!("oracle: {e}"))?,
    )
    .await;
    let got = match ad::sql(&ctx, with_call).await {
        Ok(df) => rows(&ctx, df).await,
        Err(e) => return Err(format!("{label}: refused or failed: {e}\n  {with_call}")),
    };
    if got != want {
        return Err(format!(
            "{label}: rows differ\n  with the call: {got:#?}\n  oracle:        {want:#?}\n  {with_call}"
        ));
    }
    let left: Vec<String> = ctx
        .catalog("datafusion")
        .unwrap()
        .schema("public")
        .unwrap()
        .table_names()
        .into_iter()
        .filter(|t| t.starts_with("__ddx_"))
        .collect();
    if !left.is_empty() {
        return Err(format!("{label}: left tables {left:?}"));
    }
    Ok(())
}

#[tokio::test]
async fn grad_where_a_user_puts_it() {
    let g = "grad(loss, val)";
    let cases: Vec<(&str, String, String)> = vec![
        (
            "star",
            format!("{LOSS} SELECT *, {g} AS g FROM w"),
            "SELECT w.*, gw.val AS g FROM w JOIN gw ON w.i = gw.i".into(),
        ),
        (
            "qualified star in a join",
            format!("{LOSS} SELECT w.*, x.val AS xv, grad(loss, w.val) AS g FROM w JOIN x ON w.i = x.i"),
            "SELECT w.*, x.val AS xv, gw.val AS g FROM w JOIN x ON w.i = x.i JOIN gw ON w.i = gw.i".into(),
        ),
        (
            "both stars in a join",
            format!("{LOSS} SELECT *, grad(loss, w.val) AS g FROM w JOIN x ON w.i = x.i"),
            "SELECT w.*, x.*, gw.val AS g FROM w JOIN x ON w.i = x.i JOIN gw ON w.i = gw.i".into(),
        ),
        (
            "self-join, one side differentiated",
            format!("{LOSS} SELECT a.i, b.i AS j, grad(loss, a.val) AS g FROM w a JOIN w b ON b.i = a.i + 1"),
            "SELECT a.i, b.i AS j, gw.val AS g FROM w a JOIN w b ON b.i = a.i + 1 JOIN gw ON a.i = gw.i".into(),
        ),
        (
            "self-join, both sides differentiated",
            format!("{LOSS} SELECT a.i, grad(loss, a.val) - grad(loss, b.val) AS g FROM w a JOIN w b ON b.i = a.i + 1"),
            "SELECT a.i, ga.val - gb.val AS g FROM w a JOIN w b ON b.i = a.i + 1 \
             JOIN gw ga ON a.i = ga.i JOIN gw gb ON b.i = gb.i".into(),
        ),
        (
            "DISTINCT",
            format!("{LOSS} SELECT DISTINCT {g} > 0 AS up FROM w"),
            "SELECT DISTINCT gw.val > 0 AS up FROM w JOIN gw ON w.i = gw.i".into(),
        ),
        (
            "a window over the gradient",
            format!("{LOSS} SELECT i, SUM({g}) OVER (ORDER BY i) AS run FROM w"),
            "SELECT w.i, SUM(gw.val) OVER (ORDER BY w.i) AS run FROM w JOIN gw ON w.i = gw.i".into(),
        ),
        (
            "in a join's ON",
            format!("{LOSS} SELECT w.i, x.i AS j FROM w JOIN x ON grad(loss, w.val) > x.val"),
            "SELECT w.i, x.i AS j FROM w JOIN gw ON w.i = gw.i JOIN x ON gw.val > x.val".into(),
        ),
        (
            "UNION ALL of two tables' gradients",
            format!(
                "{LOSS} SELECT i, grad(loss, val) AS g FROM w UNION ALL SELECT i, grad(loss, val) AS g FROM x"
            ),
            "SELECT i, val AS g FROM gw UNION ALL SELECT i, val AS g FROM gx".into(),
        ),
        (
            "the table in a CTE and in the main query",
            format!(
                "{LOSS}, a AS (SELECT i, {g} AS g FROM w WHERE i < 3) \
                 SELECT a.i, a.g, grad(loss, w.val) * 2 AS h FROM a JOIN w ON a.i = w.i"
            ),
            "SELECT w.i, gw.val AS g, gw.val * 2 AS h FROM w JOIN gw ON w.i = gw.i WHERE w.i < 3".into(),
        ),
        (
            "outer join, the table on the NULL side, restricted",
            format!(
                "{LOSS} SELECT x.i, grad(loss, w.val) AS g FROM x LEFT JOIN w ON x.i = w.i + 2 WHERE w.i <= 1"
            ),
            "SELECT x.i, gw.val AS g FROM x LEFT JOIN w ON x.i = w.i + 2 LEFT JOIN gw ON w.i = gw.i WHERE w.i <= 1".into(),
        ),
        (
            "outer join, the NULL side kept",
            format!("{LOSS} SELECT x.i, grad(loss, w.val) AS g FROM x LEFT JOIN w ON x.i = w.i + 2"),
            "SELECT x.i, gw.val AS g FROM x LEFT JOIN w ON x.i = w.i + 2 LEFT JOIN gw ON w.i = gw.i".into(),
        ),
        (
            "restricted by IN and by a dim compared on the right",
            format!("{LOSS} SELECT i, {g} AS g FROM w WHERE i IN (1, 4) AND 3 > i"),
            "SELECT w.i, gw.val AS g FROM w JOIN gw ON w.i = gw.i WHERE w.i IN (1, 4) AND 3 > w.i".into(),
        ),
        (
            "restricted under an OR, which must not be pushed",
            format!("{LOSS} SELECT i, {g} AS g FROM w WHERE i = 0 OR val > 1"),
            "SELECT w.i, gw.val AS g FROM w JOIN gw ON w.i = gw.i WHERE w.i = 0 OR w.val > 1".into(),
        ),
        (
            "restricted to no rows",
            format!("{LOSS} SELECT i, {g} AS g FROM w WHERE i = 99"),
            "SELECT w.i, gw.val AS g FROM w JOIN gw ON w.i = gw.i WHERE w.i = 99".into(),
        ),
        (
            "a scalar subquery over the gradients",
            format!("{LOSS} SELECT i, (SELECT MAX({g}) FROM w) AS m FROM x"),
            "SELECT x.i, (SELECT MAX(val) FROM gw) AS m FROM x".into(),
        ),
        (
            "a correlated subquery beside the call",
            format!(
                "{LOSS} SELECT w.i, {g} AS g, (SELECT MAX(x.val) FROM x WHERE x.i = w.i) AS xv FROM w"
            ),
            "SELECT w.i, gw.val AS g, x.val AS xv FROM w JOIN gw ON w.i = gw.i JOIN x ON x.i = w.i".into(),
        ),
        (
            "the call inside a CASE inside an aggregate with FILTER",
            format!("{LOSS} SELECT SUM(CASE WHEN {g} > 0 THEN {g} ELSE 0 END) FILTER (WHERE i > 0) AS s FROM w"),
            "SELECT SUM(CASE WHEN gw.val > 0 THEN gw.val ELSE 0 END) FILTER (WHERE w.i > 0) AS s \
             FROM w JOIN gw ON w.i = gw.i".into(),
        ),
        (
            "a NATURAL JOIN with the table",
            format!("{LOSS} SELECT i, grad(loss, w.val) AS g FROM w NATURAL JOIN (SELECT i FROM x) y"),
            "SELECT w.i, gw.val AS g FROM w JOIN gw ON w.i = gw.i".into(),
        ),
        (
            "JOIN USING",
            format!("{LOSS} SELECT i, grad(loss, w.val) AS g FROM w JOIN x USING (i)"),
            "SELECT w.i, gw.val AS g FROM w JOIN gw ON w.i = gw.i".into(),
        ),
    ];
    let mut failures = Vec::new();
    for (label, with_call, oracle) in &cases {
        if let Err(e) = agree(label, with_call, oracle).await {
            failures.push(e);
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} failed:\n\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n\n")
    );
}

#[tokio::test]
async fn jvp_and_vjp_where_a_user_puts_them() {
    // jvp of sin(val) along 1 is cos(val), row by row; vjp of a relation
    // along a cotangent that is another call's gradient.
    let cases: Vec<(&str, String, String)> = vec![
        (
            "jvp, number tangent, row by row",
            "SELECT i, sin(val) AS y, jvp(sin(val), w.val, 1) AS dy FROM w".into(),
            "SELECT i, sin(val) AS y, cos(val) AS dy FROM w".into(),
        ),
        (
            "jvp in a CTE read twice",
            "WITH d AS (SELECT i, jvp(val * val, w.val, 1) AS dy FROM w) \
             SELECT a.i, a.dy + b.dy AS s FROM d a JOIN d b ON a.i = b.i"
                .into(),
            "SELECT i, 4.0 * val AS s FROM w".into(),
        ),
        (
            "jvp, ORDER BY position, LIMIT",
            "SELECT i, jvp(val * val, w.val, 1) AS dy FROM w ORDER BY 2 DESC NULLS LAST LIMIT 2"
                .into(),
            "SELECT i, 2.0 * val AS dy FROM w ORDER BY 2 DESC NULLS LAST LIMIT 2".into(),
        ),
        (
            "jvp under GROUP BY",
            "SELECT i % 2 AS k, jvp(SUM(val * val), w.val, 1) AS d FROM w GROUP BY i % 2".into(),
            "SELECT i % 2 AS k, SUM(2.0 * val) AS d FROM w GROUP BY i % 2".into(),
        ),
        (
            "jvp with WHERE",
            "SELECT i % 2 AS k, jvp(SUM(val), w.val, 2.5) AS d FROM w WHERE i < 4 GROUP BY i % 2"
                .into(),
            "SELECT i % 2 AS k, CASE WHEN COUNT(val) > 0 THEN 2.5 * COUNT(val) END AS d \
             FROM w WHERE i < 4 GROUP BY i % 2"
                .into(),
        ),
        (
            "jvp of a join, both tables' rows",
            "SELECT w.i, jvp(w.val * x.val, x.val, 1) AS d FROM w JOIN x ON w.i = x.i".into(),
            "SELECT w.i, w.val AS d FROM w JOIN x ON w.i = x.i".into(),
        ),
        (
            "jvp, unnamed bare column",
            "SELECT i, jvp(val, w.val, 1) FROM w".into(),
            "SELECT i, CASE WHEN val IS NULL THEN NULL ELSE 1.0 END AS val_tangent FROM w".into(),
        ),
        (
            "vjp along another call's gradient",
            format!(
                "{LOSS}, out AS (SELECT i, val * val AS v FROM w), \
                 c AS (SELECT i, grad(loss, val) AS v FROM w) \
                 SELECT i, vjp(out, val, c) AS g FROM w"
            ),
            "SELECT w.i, 2.0 * w.val * gw.val AS g FROM w JOIN gw ON w.i = gw.i".into(),
        ),
        (
            "grad and jvp in one statement",
            format!(
                "{LOSS}, d AS (SELECT i, jvp(val * val, w.val, 1) AS dy FROM w) \
                 SELECT w.i, grad(loss, w.val) + d.dy AS s FROM w JOIN d ON w.i = d.i"
            ),
            "SELECT w.i, gw.val + 2.0 * w.val AS s FROM w JOIN gw ON w.i = gw.i".into(),
        ),
    ];
    let mut failures = Vec::new();
    for (label, with_call, oracle) in &cases {
        if let Err(e) = agree(label, with_call, oracle).await {
            failures.push(e);
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} failed:\n\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n\n")
    );
}

#[tokio::test]
async fn statements_of_one_batch_share_and_restrict() {
    // One loss, three statements, restricted three ways; each must see its
    // own rows' gradients.
    let ctx = ctx();
    setup(&ctx).await;
    let s = [
        format!("{LOSS} SELECT i, grad(loss, val) AS g FROM w WHERE i < 2"),
        format!("{LOSS} SELECT i, grad(loss, val) AS g FROM w WHERE i >= 3"),
        format!("{LOSS} SELECT i, grad(loss, val) AS g FROM w WHERE i = 2 OR i = 4"),
        format!("{LOSS} SELECT i, grad(loss, val) AS g FROM x WHERE i = 1"),
    ];
    let o = [
        "SELECT i, val AS g FROM gw WHERE i < 2",
        "SELECT i, val AS g FROM gw WHERE i >= 3",
        "SELECT i, val AS g FROM gw WHERE i = 2 OR i = 4",
        "SELECT i, val AS g FROM gx WHERE i = 1",
    ];
    let texts: Vec<&str> = s.iter().map(String::as_str).collect();
    let frames = ad::sql_all(&ctx, &texts).await.unwrap();
    let mut failures = Vec::new();
    for ((df, oracle), sql) in frames.into_iter().zip(o).zip(&s) {
        let got = rows(&ctx, df).await;
        let want = rows(&ctx, ctx.sql(oracle).await.unwrap()).await;
        if got != want {
            failures.push(format!("{got:?} vs {want:?}\n  {sql}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[tokio::test]
async fn grad_over_what_is_not_the_table_is_refused() {
    // A SELECT over a derived table or a CTE reads their rows, not the
    // table's: ddx must refuse, not answer for other rows.
    let ctx = ctx();
    setup(&ctx).await;
    for sql in [
        format!("{LOSS} SELECT i, grad(loss, val) AS g FROM (SELECT i, val FROM w WHERE i > 0) s"),
        format!("{LOSS}, w2 AS (SELECT i, val * 2 AS val FROM w) SELECT i, grad(loss, val) AS g FROM w2"),
    ] {
        match ad::sql(&ctx, &sql).await {
            Ok(df) => panic!("accepted, rows {:?}\n  {sql}", rows(&ctx, df).await),
            Err(e) => assert!(!e.to_string().contains("internal"), "{e}\n  {sql}"),
        }
    }
}

#[tokio::test]
async fn grad_in_group_by_order_by_and_under_a_schema() {
    // Each is valid SQL ddx documents as supported ("anywhere an expression
    // can"), and each failed when found by fuzzing: the same call in the
    // SELECT and its GROUP BY got two hidden columns, so the SELECT's no
    // longer matched its key; an ORDER BY call was refused as outside a
    // SELECT's columns; `public.w.val` over `FROM public.w` was refused as
    // not in the FROM.
    let g = "grad(loss, val)";
    // The loss names the table as the statement does: ddx matches tables by
    // name, and cannot know that the engine's default schema makes `w` and
    // `public.w` one table.
    let public = LOSS.replace("FROM w JOIN", "FROM public.w JOIN");
    let cases: Vec<(&str, String, String)> = vec![
        (
            "schema-qualified table",
            format!("{public} SELECT i, grad(loss, public.w.val) AS g FROM public.w"),
            "SELECT w.i, gw.val AS g FROM w JOIN gw ON w.i = gw.i".into(),
        ),
        (
            "the gradient as a GROUP BY key",
            format!("{LOSS} SELECT {g} > 0 AS up, COUNT(*) AS n FROM w GROUP BY {g} > 0"),
            "SELECT gw.val > 0 AS up, COUNT(*) AS n FROM w JOIN gw ON w.i = gw.i GROUP BY gw.val > 0".into(),
        ),
        (
            "ORDER BY the gradient, LIMIT",
            format!("{LOSS} SELECT i FROM w ORDER BY {g} DESC NULLS LAST LIMIT 2"),
            "SELECT w.i FROM w JOIN gw ON w.i = gw.i ORDER BY gw.val DESC NULLS LAST LIMIT 2".into(),
        ),
    ];
    let mut failures = Vec::new();
    for (label, with_call, oracle) in &cases {
        if let Err(e) = agree(label, with_call, oracle).await {
            failures.push(e);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    // Spelled two ways, it is refused, saying which name the loss reads.
    let ctx = ctx();
    setup(&ctx).await;
    let e = ad::sql(
        &ctx,
        &format!("{LOSS} SELECT i, grad(loss, public.w.val) AS g FROM public.w"),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(e.contains("name it `w`"), "{e}");
}
