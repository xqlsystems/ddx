// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx_ad::to_sql`, relation by relation and expression by expression.
//!
//! Each case is a query DataFusion plans and optimizes, so the plan has the
//! shapes optimization produces (pushed-down projections, folded constants,
//! rewritten joins and subqueries), not only hand-written ones. The plan is
//! written as DataFusion SQL by `to_sql`, run, and its rows compared with
//! the query's own. The whole suite also runs every program it builds this
//! way (`DDX_STEPS_AS_SQL=1`, see `ddx_datafusion::engine`).

mod common;

use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::prelude::SessionContext;
use ddx_ad::substrait::proto::extensions::simple_extension_declaration::MappingType;
use ddx_ad::{to_sql, AdError, Dialect};

async fn fixture() -> SessionContext {
    let ctx = SessionContext::new();
    for sql in [
        "CREATE TABLE t (k BIGINT, g VARCHAR, x DOUBLE, n INT) AS VALUES \
         (1, 'a', 1.5, 3), (2, 'a', -2.0, NULL), (3, 'b', 0.25, 7), (4, 'b', NULL, 2), \
         (5, 'c', 4.0, -5)",
        "CREATE TABLE u (k BIGINT, y DOUBLE) AS VALUES (1, 10.0), (3, 30.0), (3, 31.0), \
         (6, 60.0), (NULL, 0.5)",
    ] {
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
    }
    ctx
}

/// The rows of `sql`'s result, one line each, sorted unless `ordered`.
async fn rows(ctx: &SessionContext, sql: &str, ordered: bool) -> Vec<String> {
    let batches = ctx
        .sql(sql)
        .await
        .unwrap_or_else(|e| panic!("{e}\n{sql}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}\n{sql}"));
    let text = pretty_format_batches(&batches).unwrap().to_string();
    // Drop the header: the column names are the root's, compared below.
    let mut lines: Vec<String> = text.lines().skip(3).map(str::to_string).collect();
    if !ordered {
        lines.sort();
    }
    lines
}

/// `sql`, planned by DataFusion and written back as SQL by `to_sql`, gives
/// the rows `sql` does, in order if `ordered`.
async fn round_trips(ctx: &SessionContext, sql: &str, ordered: bool) {
    let plan = common::substrait_of(ctx, sql, true).await;
    let written = to_sql(&plan, Dialect::DataFusion).unwrap();
    assert_eq!(
        rows(ctx, &written, ordered).await,
        rows(ctx, sql, ordered).await,
        "\n{sql}\nwritten as\n{written}"
    );
    // And with every subtree read twice shared, as a producer that shares a
    // CTE writes it: one CTE read wherever it is shared.
    let shared = common::shared(&plan);
    let written = to_sql(&shared, Dialect::DataFusion).unwrap();
    assert_eq!(
        rows(ctx, &written, ordered).await,
        rows(ctx, sql, ordered).await,
        "\n{sql}\nshared, written as\n{written}"
    );
}

#[tokio::test]
async fn reads_projections_and_filters() {
    let ctx = fixture().await;
    for sql in [
        "SELECT * FROM t",
        "SELECT x, k FROM t",
        "SELECT k, x * 2 + 1 AS y, -x AS z FROM t WHERE x > 0 OR n IS NULL",
        "SELECT k FROM t WHERE g = 'a' AND x IS NOT NULL",
        "SELECT k, n / 2 AS half, n % 3 AS r, x / 4 AS q FROM t",
        "SELECT * FROM (VALUES (1, 'it''s'), (2, NULL)) AS v(a, b)",
        "SELECT a FROM (VALUES (1.5), (2.5)) AS v(a) WHERE a > 10",
    ] {
        round_trips(&ctx, sql, false).await;
    }
}

#[tokio::test]
async fn joins_of_every_kind() {
    let ctx = fixture().await;
    for sql in [
        "SELECT t.k, u.y FROM t JOIN u ON t.k = u.k",
        "SELECT t.k, u.y FROM t LEFT JOIN u ON t.k = u.k",
        "SELECT t.k, u.y FROM t RIGHT JOIN u ON t.k = u.k",
        "SELECT t.k, u.k AS uk, u.y FROM t FULL JOIN u ON t.k = u.k",
        "SELECT t.k, u.y FROM t JOIN u ON t.k = u.k AND u.y > t.x",
        "SELECT t.k, u.y FROM t JOIN u ON t.k IS NOT DISTINCT FROM u.k",
        "SELECT t.k, u.y FROM t CROSS JOIN u WHERE u.y < 20",
        "SELECT k FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k)",
        "SELECT k FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.k)",
        "SELECT k FROM t WHERE k IN (SELECT k FROM u)",
        "SELECT k FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.y > 50)",
        "SELECT k FROM t WHERE x > (SELECT avg(y) FROM u) - 30",
        "SELECT k, (SELECT max(y) FROM u WHERE u.k = t.k) AS m FROM t",
    ] {
        round_trips(&ctx, sql, false).await;
    }
}

#[tokio::test]
async fn aggregates() {
    let ctx = fixture().await;
    for sql in [
        "SELECT sum(x), count(*), count(x), avg(x), min(n), max(n) FROM t",
        "SELECT g, sum(x) AS s, count(DISTINCT n) AS d FROM t GROUP BY g",
        "SELECT g, sum(x) FILTER (WHERE n > 0) AS s FROM t GROUP BY g",
        "SELECT DISTINCT g FROM t",
        "SELECT g, k % 2 AS parity, sum(x) FROM t GROUP BY g, k % 2",
        "SELECT sum(x) FROM t WHERE k > 100",
    ] {
        round_trips(&ctx, sql, false).await;
    }
}

#[tokio::test]
async fn sorts_fetches_and_sets() {
    let ctx = fixture().await;
    for sql in [
        "SELECT k, x FROM t ORDER BY x DESC NULLS LAST, k",
        "SELECT k FROM t ORDER BY k LIMIT 2",
        "SELECT k FROM t ORDER BY k DESC LIMIT 2 OFFSET 1",
    ] {
        round_trips(&ctx, sql, true).await;
    }
    for sql in [
        "SELECT k FROM t LIMIT 0",
        "SELECT k FROM t UNION ALL SELECT k FROM u",
        "SELECT k FROM t UNION SELECT k FROM u",
        "SELECT k FROM t EXCEPT SELECT k FROM u",
        "SELECT k FROM u INTERSECT SELECT k FROM t",
        "SELECT k FROM u INTERSECT ALL SELECT k FROM u",
    ] {
        round_trips(&ctx, sql, false).await;
    }
}

#[tokio::test]
async fn windows_with_their_frames() {
    let ctx = fixture().await;
    for sql in [
        "SELECT k, sum(x) OVER (PARTITION BY g) AS s FROM t",
        "SELECT k, sum(x) OVER (PARTITION BY g ORDER BY k) AS s FROM t",
        "SELECT k, sum(x) OVER (ORDER BY k ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS s FROM t",
        "SELECT k, max(x) OVER (PARTITION BY g ORDER BY k ROWS BETWEEN UNBOUNDED PRECEDING \
         AND CURRENT ROW) AS m FROM t",
        "SELECT k, count(*) OVER (ORDER BY n RANGE BETWEEN 2 PRECEDING AND 2 FOLLOWING) AS c FROM t",
        "SELECT k, row_number() OVER (PARTITION BY g ORDER BY x DESC) AS r, \
         rank() OVER (ORDER BY g) AS rk FROM t",
        "SELECT k, avg(x) OVER () AS a FROM t",
    ] {
        round_trips(&ctx, sql, false).await;
    }
}

#[tokio::test]
async fn expressions_and_literals() {
    let ctx = fixture().await;
    for sql in [
        "SELECT k, CASE WHEN x > 1 THEN 'big' WHEN x > 0 THEN 'small' END AS c FROM t",
        "SELECT k, CASE g WHEN 'a' THEN 1 WHEN 'b' THEN 2 ELSE 0 END AS c FROM t",
        "SELECT k, CAST(n AS DOUBLE) AS d, CAST(x AS BIGINT) AS i, TRY_CAST(g AS INT) AS j FROM t",
        "SELECT k FROM t WHERE n IN (2, 3, 99)",
        "SELECT k FROM t WHERE g NOT IN ('a', 'z')",
        "SELECT k, coalesce(x, 0) AS c, abs(x) AS a, sqrt(abs(x)) AS r, exp(x) AS e, \
         ln(abs(x) + 1) AS l, power(x, 2) AS p, signum(x) AS s FROM t",
        "SELECT k, greatest(x, 0) AS relu, least(x, 1) AS m, isnan(x) AS nan FROM t",
        "SELECT k, x IS NOT DISTINCT FROM 1.5 AS same, NOT (x > 0) AS neg FROM t",
        "SELECT 0.1 AS a, 1e300 AS b, -0.30000000000000004 AS c, CAST(2.5 AS REAL) AS d",
        "SELECT CAST('NaN' AS DOUBLE) AS a, CAST('inf' AS DOUBLE) AS b, \
         CAST('-inf' AS DOUBLE) AS c",
        "SELECT CAST(12.345 AS DECIMAL(10, 3)) AS d, CAST(-0.5 AS DECIMAL(4, 2)) AS e",
        "SELECT DATE '2024-02-29' AS d, TIMESTAMP '1969-12-31 23:59:59.25' AS ts",
        "SELECT 'it''s' AS s, CAST(NULL AS DOUBLE) AS n, TRUE AS b",
    ] {
        round_trips(&ctx, sql, false).await;
    }
}

#[tokio::test]
async fn a_column_named_like_a_keyword_or_with_a_quote_is_quoted() {
    let ctx = SessionContext::new();
    ctx.sql(r#"CREATE TABLE "Odd ""T""" ("select" DOUBLE, "A b" INT) AS VALUES (1.0, 2)"#)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    round_trips(
        &ctx,
        r#"SELECT "select" * 2 AS "x""y", "A b" FROM "Odd ""T""""#,
        false,
    )
    .await;
}

#[tokio::test]
async fn a_function_with_no_spelling_is_refused_by_name() {
    let ctx = fixture().await;
    let mut plan = common::substrait_of(&ctx, "SELECT abs(x) FROM t", true).await;
    for decl in &mut plan.extensions {
        if let Some(MappingType::ExtensionFunction(f)) = &mut decl.mapping_type {
            f.name = "frobnicate:fp64".into();
        }
    }
    let err = to_sql(&plan, Dialect::DuckDb).unwrap_err();
    assert!(
        matches!(&err, AdError::NotImplemented(m) if m.contains("`frobnicate`")),
        "{err}"
    );
}

#[tokio::test]
async fn each_dialect_spells_sign_and_integer_division_its_own_way() {
    let ctx = fixture().await;
    let plan = common::substrait_of(&ctx, "SELECT signum(x), n / 2 FROM t", true).await;
    let datafusion = to_sql(&plan, Dialect::DataFusion).unwrap();
    let duckdb = to_sql(&plan, Dialect::DuckDb).unwrap();
    assert!(
        datafusion.contains("signum(") && datafusion.contains(" / "),
        "{datafusion}"
    );
    assert!(
        duckdb.contains("sign(") && duckdb.contains(" // "),
        "{duckdb}"
    );
}
