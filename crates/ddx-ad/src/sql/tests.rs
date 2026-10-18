// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Unit tests of the SQL layer: finding calls, planning derivatives,
//! rewriting statements.

use super::find::*;
use super::plan::*;
use super::rewrite::*;
use crate::error::AdError;
use crate::relation::ColumnRef;
use crate::tables::{InputTable, OutputTable};
use ddx_core::sqlparser::dialect::GenericDialect;

const SQL: &str = "WITH d AS (SELECT * FROM x), \
                   loss AS (SELECT SUM(w.val * d.v) AS l FROM w JOIN d ON w.i = d.i) \
                   SELECT w.i, w.val - 0.1 * GRAD(loss, w.val) AS val FROM w";

fn find(sql: &str) -> Calls {
    Calls::find(sql, &GenericDialect {}).unwrap().unwrap()
}

fn filter_of(sql: &str) -> Option<String> {
    let conjuncts = &find(sql).calls[0].pushdown;
    (!conjuncts.is_empty()).then(|| {
        conjuncts
            .iter()
            .map(|c| c.sql.as_str())
            .collect::<Vec<_>>()
            .join(" AND ")
    })
}

/// `calls` rewritten as if each gradient were `g_{table}` (`g_{table}_{k}`
/// for the `k`th other query differentiated), with dims `i`, and each value
/// `v`: `l` and two jvps' expressions, of which `l` and the first have
/// tangents.
fn rewritten(calls: &Calls) -> String {
    let value = OutputTable {
        step: "v".into(),
        of: crate::tables::Of::Output,
        columns: vec![
            "l".into(),
            "__ddx_jvp_0".into(),
            "__ddx_jvp_1".into(),
            "t0".into(),
            "t1".into(),
        ],
        tangents: vec![
            crate::tables::Tangent {
                column: "l".into(),
                tangent: "t0".into(),
            },
            crate::tables::Tangent {
                column: "__ddx_jvp_0".into(),
                tangent: "t1".into(),
            },
        ],
    };
    let mut queries: Vec<&ddx_core::sqlparser::ast::Query> = Vec::new();
    let steps: Vec<String> = calls
        .calls
        .iter()
        .map(|c| {
            let k = queries
                .iter()
                .position(|q| *q == c.derivative.of())
                .unwrap_or_else(|| {
                    queries.push(c.derivative.of());
                    queries.len() - 1
                });
            let table = &c.derivative.wrt()[0].table;
            match k {
                0 => format!("g_{table}"),
                k => format!("g_{table}_{k}"),
            }
        })
        .collect();
    calls
        .rewrite(&|i| {
            Ok(match calls.calls[i].derivative.transform() {
                Transform::Jvp => Answer::Value(Box::leak(Box::new(value.clone()))),
                _ => Answer::Gradient {
                    step: Box::leak(steps[i].clone().into_boxed_str()),
                    dims: vec!["i"],
                    column: Box::leak(
                        calls.calls[i].derivative.wrt()[0]
                            .column
                            .clone()
                            .into_boxed_str(),
                    ),
                },
            })
        })
        .unwrap()
}

#[test]
fn a_call_is_found_with_its_objective_query() {
    let found = find(SQL);
    let call = &found.calls[0];
    let Derivative::Grad { of, wrt } = &call.derivative else {
        panic!("{:?}", call.derivative)
    };
    let of = of.to_string();
    assert!(
        of.starts_with("WITH d AS (SELECT * FROM x), loss AS ("),
        "{of}"
    );
    assert!(of.ends_with("SELECT * FROM loss"), "{of}");
    assert_eq!(wrt, &vec![ColumnRef::new("w", "val")]);
}

#[test]
fn a_gradient_is_a_column_of_its_table_s_rows() {
    let out = rewritten(&find(SQL));
    assert!(
        out.ends_with(
            "SELECT w.i, w.val - 0.1 * w.\"__ddx_d0\" AS val FROM (SELECT __ddx_t.*, \
             __ddx_g0.\"val\" AS \"__ddx_d0\" FROM w AS __ddx_t LEFT JOIN \"g_w\" AS \
             __ddx_g0 ON (__ddx_t.\"i\" IS NOT DISTINCT FROM __ddx_g0.\"i\")) AS w"
        ),
        "{out}"
    );
    // Nothing reads the objective once its call reads `g_w`.
    assert!(out.starts_with("SELECT w.i"), "{out}");
}

#[test]
fn a_rewritten_statement_keeps_only_the_ctes_it_still_reads() {
    // `loss` and `d`, which only `loss` reads, go; `kept`, which the body
    // reads (spelled in another case), stays, and so does `base`, which
    // only `kept` reads.
    let sql = "WITH base AS (SELECT i, 2 AS k FROM x), kept AS (SELECT i, k FROM base), \
               d AS (SELECT * FROM x), \
               loss AS (SELECT SUM(w.val * d.v) AS l FROM w JOIN d ON w.i = d.i) \
               SELECT w.i, w.val - 0.1 * grad(loss, w.val) AS val FROM w JOIN KEPT ON w.i = kept.i";
    let out = rewritten(&find(sql));
    assert!(
        out.starts_with(
            "WITH base AS (SELECT i, 2 AS k FROM x), kept AS (SELECT i, k FROM base) SELECT"
        ),
        "{out}"
    );
    // A CTE read only inside a subquery is read.
    let sql = "WITH t AS (SELECT 1 AS i), loss AS (SELECT SUM(val) AS l FROM w) \
               SELECT i, grad(loss, val) AS g FROM w WHERE i IN (SELECT i FROM t)";
    let out = rewritten(&find(sql));
    assert!(out.starts_with("WITH t AS (SELECT 1 AS i) SELECT"), "{out}");
}

#[test]
fn a_star_over_a_table_a_call_reads_returns_none_of_its_gradient_columns() {
    let sql = "WITH loss AS (SELECT SUM(a * b) AS l FROM w) \
               SELECT *, p.*, x.*, grad(loss, p.a) AS da FROM w p JOIN x ON p.i = x.i";
    let out = rewritten(&find(sql));
    assert!(
        out.contains(
            "SELECT * EXCLUDE (\"__ddx_d0\"), p.* EXCLUDE (\"__ddx_d0\"), x.*, p.\"__ddx_d0\" AS da"
        ),
        "{out}"
    );
    let e = Calls::find(
        &sql.replace("SELECT *,", "SELECT * EXCLUDE (b),"),
        &GenericDialect {},
    )
    .unwrap_err();
    assert!(matches!(e, AdError::NotImplemented(_)), "{e}");
}

#[test]
fn calls_on_one_table_share_its_relation_and_name_it_by_its_alias() {
    let sql = "WITH loss AS (SELECT SUM(a * b) AS l FROM w), \
               l2 AS (SELECT 2 * l AS l FROM loss) \
               SELECT p.i, grad(loss, p.a) AS da, grad(loss, b) AS db, grad(l2, a) AS d2 \
               FROM w AS p";
    let found = find(sql);
    assert_eq!(found.reads.len(), 1);
    let out = rewritten(&found);
    assert!(
        out.contains(
            "SELECT p.i, p.\"__ddx_d0\" AS da, p.\"__ddx_d1\" AS db, p.\"__ddx_d2\" AS d2"
        ),
        "{out}"
    );
    assert!(out.ends_with(") AS p"), "{out}");
    let planned = Derivatives::plan(&[sql], &GenericDialect {}).unwrap();
    assert_eq!(planned.derivatives().len(), 2);
    assert_eq!(
        planned.derivatives()[0].wrt(),
        vec![ColumnRef::new("w", "a"), ColumnRef::new("w", "b")]
    );
}

#[test]
fn two_tables_of_one_objective_share_its_job() {
    let sql = "WITH loss AS (SELECT SUM(w.val * b.val) AS l FROM w JOIN b ON w.o = b.o) \
               SELECT w.i, grad(loss, w.val) AS gw, grad(loss, b.val) AS gb \
               FROM w JOIN b ON w.o = b.o";
    let planned = Derivatives::plan(&[sql], &GenericDialect {}).unwrap();
    assert_eq!(planned.derivatives().len(), 1);
    assert_eq!(
        planned.derivatives()[0].wrt(),
        vec![ColumnRef::new("w", "val"), ColumnRef::new("b", "val")]
    );
    assert_eq!(find(sql).reads.len(), 2);
}

#[test]
fn case_variants_of_a_column_are_one_wrt_column() {
    let sql = "WITH loss AS (SELECT SUM(val) AS l FROM w) \
               SELECT grad(loss, w.val) AS a, grad(loss, W.VAL) AS b FROM w";
    let planned = Derivatives::plan(&[sql], &GenericDialect {}).unwrap();
    assert_eq!(
        planned.derivatives()[0].wrt(),
        vec![ColumnRef::new("w", "val")]
    );
}

#[test]
fn a_comparison_of_a_dim_with_a_literal_is_pushed_and_nothing_else() {
    let with = |w: &str| {
        format!(
            "WITH loss AS (SELECT SUM(val) AS l FROM w) \
             SELECT g.i, grad(loss, g.val) FROM w g JOIN x ON g.i = x.i WHERE {w}"
        )
    };
    assert_eq!(filter_of(&with("g.layer = 0")), Some("layer = 0".into()));
    assert_eq!(
        filter_of(&with(
            "g.layer IN (0, 2) AND g.i BETWEEN 1 AND 4 AND 3 > g.k"
        )),
        Some("layer IN (0, 2) AND i BETWEEN 1 AND 4 AND 3 > k".into())
    );
    // The column differentiated, another table's, a bare column of a
    // SELECT over two tables, an OR, a negation, or a comparison with a
    // column is not pushed.
    for w in [
        "g.val > 0",
        "x.layer = 0",
        "layer = 0",
        "g.layer = 0 OR g.i = 1",
        "g.layer NOT IN (0)",
        "g.layer = x.layer",
    ] {
        assert_eq!(filter_of(&with(w)), None, "{w}");
    }
    assert_eq!(
        filter_of(&with("g.layer = 0 AND g.val > 0")),
        Some("layer = 0".into())
    );
    // Over one table, a bare column is its.
    let one = "WITH loss AS (SELECT SUM(val) AS l FROM w) \
               SELECT grad(loss, val) FROM w WHERE layer = 0";
    assert_eq!(filter_of(one), Some("layer = 0".into()));
}

#[test]
fn a_job_restricts_a_table_only_when_every_call_on_it_does() {
    let stmt = |w: &str| {
        format!(
            "WITH loss AS (SELECT SUM(val) AS l FROM w) \
             SELECT i, grad(loss, val) FROM w{w}"
        )
    };
    let plan = |ws: &[&str]| {
        let stmts: Vec<String> = ws.iter().map(|w| stmt(w)).collect();
        let refs: Vec<&str> = stmts.iter().map(String::as_str).collect();
        Derivatives::plan(&refs, &GenericDialect {})
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .pushdown
    };
    let restricted = |predicate: &str| {
        vec![Restriction {
            table: "w".into(),
            query: format!("SELECT * FROM w WHERE {predicate}"),
        }]
    };
    assert_eq!(plan(&[" WHERE layer = 0"]), restricted("layer = 0"));
    assert_eq!(
        plan(&[" WHERE layer = 0", " WHERE layer = 1"]),
        restricted("(layer = 0) OR (layer = 1)")
    );
    assert_eq!(plan(&[" WHERE layer = 0", ""]), vec![]);
}

#[test]
fn a_derivative_within_a_row_is_refused_with_the_usage() {
    // The names are ddx's: a call that is not one of these, such as the
    // derivative of an expression within a row, is refused, not left to
    // the engine to guess at.
    for sql in [
        "SELECT grad(x * x, x) FROM t",
        "SELECT AVG(grad(loss, theta)) FROM batch",
        "SELECT jvp(x * x, x, y) FROM t",
        "WITH v AS (SELECT 1 AS a) SELECT grad(a * a, a) FROM v",
    ] {
        let e = Calls::find(sql, &GenericDialect {}).unwrap_err();
        assert!(e.to_string().contains("write "), "{sql}: {e}");
    }
}

#[test]
fn a_statement_without_a_call_is_not_parsed() {
    assert!(Calls::find("not even SQL", &GenericDialect {})
        .unwrap()
        .is_none());
    assert!(Calls::find("SELECT gradient FROM t", &GenericDialect {})
        .unwrap()
        .is_none());
    assert!(Calls::find("SELECT grad ( FROM", &GenericDialect {})
        .unwrap()
        .is_none());
}

#[test]
fn misuse_is_refused_with_the_usage() {
    let loss = "WITH loss AS (SELECT SUM(a * b) AS l FROM w), v AS (SELECT 1 AS a) ";
    for (sql, why) in [
        ("SELECT grad(loss) FROM w", "write grad"),
        ("SELECT grad(loss, a, b) FROM w", "takes one"),
        (
            "SELECT grad(loss, a) FROM w JOIN x ON w.i = x.i",
            "ambiguous",
        ),
        ("SELECT grad(loss, y.a) FROM w", "not a column of a table"),
        ("SELECT grad(loss, a) FROM v", "is a CTE"),
        (
            "SELECT grad(loss, a) FROM (SELECT * FROM w) s",
            "not a table's column",
        ),
        ("SELECT * FROM grad(loss, w.a)", "not a relation"),
        ("SELECT vjp(loss, a) FROM w", "write vjp"),
        ("SELECT vjp(loss, a, 1.0) FROM w", "last argument"),
        ("SELECT jvp(a, w.a, v) + 1 FROM w", "column of its own"),
        ("SELECT jvp(a * b, w.a, v) FROM w", "name the tangent"),
        ("SELECT jvp(a, a, v) AS t FROM w", "not a table column"),
        (
            "SELECT x FROM (SELECT jvp(a, w.a, v) AS t FROM w) s",
            "move it into a CTE",
        ),
    ] {
        let e = Calls::find(&format!("{loss}{sql}"), &GenericDialect {}).unwrap_err();
        assert!(e.to_string().contains(why), "{sql}: {e}");
    }
    for call in [
        "jvp(a, w.a, x.b, v) AS t",
        "jvp(a, w.a, v) AS t, jvp(b, w.b, v) AS u",
    ] {
        let e =
            Calls::find(&format!("{loss}SELECT {call} FROM w"), &GenericDialect {}).unwrap_err();
        assert!(matches!(e, AdError::NotImplemented(_)), "{call}: {e}");
    }
}

const JVP: &str = "WITH h AS (SELECT x.n, w.o, w.val FROM x JOIN w ON x.i = w.i), \
                   dw AS (SELECT i, o, 0.5 AS val FROM w) \
                   SELECT n, SUM(val) AS l, jvp(SUM(val), w.val, dw) AS dl, \
                   JVP(MAX(val), w.val, dw) FROM h GROUP BY n ORDER BY n";

#[test]
fn a_tangent_is_a_column_of_its_select_s_rows() {
    // The second call has no alias, and no name to give its tangent.
    let e = Calls::find(JVP, &GenericDialect {}).unwrap_err();
    assert!(e.to_string().contains("name the tangent"), "{e}");

    let sql = JVP.replace("dw) FROM", "dw) AS dm FROM");
    let found = find(&sql);
    let [a, b] = &found.calls[..] else {
        unreachable!()
    };
    // f is the SELECT with each jvp written as its expression, after
    // the CTEs it reads.
    assert_eq!(a.derivative.of(), b.derivative.of());
    let f = a.derivative.of().to_string();
    assert!(f.starts_with("WITH h AS ("), "{f}");
    assert!(
        f.ends_with(
            "SELECT n, SUM(val) AS l, SUM(val) AS \"__ddx_jvp_0\", MAX(val) AS \
             \"__ddx_jvp_1\" FROM h GROUP BY n"
        ),
        "{f}"
    );
    let dw = a.derivative.seed_query().unwrap().to_string();
    assert!(dw.ends_with("SELECT * FROM dw"), "{dw}");
    assert_eq!(a.derivative.wrt(), vec![ColumnRef::new("w", "val")]);
    // The SELECT becomes a read of the value: the tangent in the call's
    // place, under its name; a column with none is 0. ORDER BY stays.
    let out = rewritten(&found);
    assert!(
        out.ends_with(
            "SELECT \"l\", \"t1\" AS \"dl\", CAST(0 AS DOUBLE) AS \"dm\" FROM \"v\" ORDER BY n"
        ),
        "{out}"
    );
}

#[test]
fn a_query_carries_only_the_ctes_it_reads() {
    // The second jvp's SELECT reads f, not j: j's call is not its.
    let sql = "WITH f AS (SELECT SUM(a * b) AS l FROM w), v AS (SELECT 1 AS a), \
               u AS (SELECT * FROM v), \
               j AS (SELECT jvp(l, w.a, v) AS t FROM f), \
               k AS (SELECT jvp(l, w.b, u) AS t FROM f) \
               SELECT * FROM j CROSS JOIN k";
    let found = find(sql);
    let k = found.calls[1].derivative.of().to_string();
    assert!(k.starts_with("WITH f AS ("), "{k}");
    assert!(!k.contains("j AS"), "{k}");
    let planned = Derivatives::plan(&[sql], &GenericDialect {}).unwrap();
    assert!((0..planned.derivatives().len()).all(|j| planned.query_needs(j).is_empty()));
    // A cotangent's CTE carries what it reads.
    assert!(found.calls[1]
        .derivative
        .seed_query()
        .unwrap()
        .to_string()
        .starts_with("WITH v AS ("));
}

#[test]
fn jvps_along_different_columns_or_tangents_are_different_jobs() {
    let f = "WITH v AS (SELECT 1 AS a), u AS (SELECT 2 AS a) ";
    let stmts = [
        format!("{f}SELECT jvp(SUM(a * b), w.a, v) AS t FROM w"),
        format!("{f}SELECT jvp(SUM(a * b), w.b, v) AS t FROM w"),
        format!("{f}SELECT jvp(SUM(a * b), w.a, u) AS t FROM w"),
        format!("{f}SELECT jvp(SUM(a * b), w.a, v) AS t FROM w"),
    ];
    let refs: Vec<&str> = stmts.iter().map(String::as_str).collect();
    let planned = Derivatives::plan(&refs, &GenericDialect {}).unwrap();
    assert_eq!(planned.derivatives().len(), 3);
    assert_eq!(planned.statement_needs(3), vec![0]);
}

/// A program that takes `inputs`, as if `planned` had built it for
/// its next job.
fn built(planned: &mut Derivatives, inputs: Vec<InputTable>) -> Job {
    let job = planned.next().unwrap().unwrap();
    planned.push_built(Box::new(crate::ForwardProgram {
        inputs,
        checks: Vec::new(),
        steps: Vec::new(),
        value: OutputTable {
            step: "__ddx_1_value".into(),
            of: crate::tables::Of::Output,
            columns: Vec::new(),
            tangents: Vec::new(),
        },
        gradients: Vec::new(),
    }));
    job
}

#[test]
fn a_job_s_seed_has_the_columns_its_program_asks_for() {
    let jvp = "SELECT jvp(SUM(val), w.val, db) AS t FROM w";
    let mut planned = Derivatives::plan(&[jvp], &GenericDialect {}).unwrap();
    let input = InputTable {
        name: "__ddx_1_tangent_0_w".into(),
        of: crate::tables::Of::Table(vec!["w".into()]),
        columns: vec!["i".into(), "o".into(), "val".into()],
        keys: 2,
    };
    let job = built(&mut planned, vec![input.clone()]);
    assert_eq!(
        planned.seeds(&job).unwrap(),
        vec![(
            input.name.clone(),
            "SELECT \"i\", \"o\", \"val\" FROM (SELECT * FROM db) AS __ddx_seed".into()
        )]
    );
    // A grad job writes its cotangent, 1, into its plan: it has none.
    let grad = "WITH l AS (SELECT SUM(val) AS l FROM w) SELECT grad(l, val) FROM w";
    let mut planned = Derivatives::plan(&[grad], &GenericDialect {}).unwrap();
    let job = built(&mut planned, vec![input]);
    assert!(planned.seeds(&job).is_err());
}

#[test]
fn a_call_inside_another_s_query_is_a_job_before_it_and_its_query_reads_it() {
    // The grad is inside the jvp's SELECT, after the jvp in the text.
    let sql = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w), \
                    v AS (SELECT i, 1.0 AS val FROM w) \
               SELECT i, jvp(grad(loss, val), w.val, v) AS hv FROM w";
    let mut planned = Derivatives::plan(&[sql], &GenericDialect {}).unwrap();
    assert_eq!(planned.derivatives().len(), 2);
    assert_eq!(planned.derivatives()[0].transform(), Transform::Grad);
    assert_eq!(planned.query_needs(1), vec![0]);
    let first = planned.next().unwrap().unwrap();
    assert!(!first.query.contains("grad("), "{}", first.query);
    // The outer job's query cannot be written until the inner one is
    // built: it reads its program's table.
    planned.push_built(Box::new(crate::BackwardProgram {
        inputs: Vec::new(),
        checks: Vec::new(),
        forward_steps: Vec::new(),
        backward_steps: Vec::new(),
        value: OutputTable {
            step: "__ddx_1_value".into(),
            of: crate::tables::Of::Output,
            columns: vec!["l".into()],
            tangents: Vec::new(),
        },
        gradients: vec![OutputTable {
            step: "__ddx_1_grad_0_w".into(),
            of: crate::tables::Of::Table(vec!["w".into()]),
            columns: vec!["i".into(), "val".into()],
            tangents: Vec::new(),
        }],
    }));
    let second = planned.next().unwrap().unwrap();
    assert!(
        second.query.contains(
            "SELECT i, w.\"__ddx_d0\" AS \"__ddx_jvp_0\" FROM (SELECT __ddx_t.*, \
             __ddx_g0.\"val\" AS \"__ddx_d0\" FROM w AS __ddx_t LEFT JOIN \
             \"__ddx_1_grad_0_w\" AS __ddx_g0"
        ),
        "{}",
        second.query
    );
}

#[test]
fn a_tangent_column_is_named_after_its_column_and_never_twice() {
    let value = OutputTable {
        step: "__ddx_1_jvp".into(),
        of: crate::tables::Of::Output,
        columns: vec![
            "l".into(),
            "l_tangent".into(),
            "__ddx_tangent_0".into(),
            "__ddx_tangent_1".into(),
        ],
        tangents: vec![
            crate::tables::Tangent {
                column: "l".into(),
                tangent: "__ddx_tangent_0".into(),
            },
            crate::tables::Tangent {
                column: "l_tangent".into(),
                tangent: "__ddx_tangent_1".into(),
            },
        ],
    };
    assert_eq!(
        value.query(),
        "SELECT \"l\", \"l_tangent\", \"__ddx_tangent_0\" AS \"l_tangent_2\", \
         \"__ddx_tangent_1\" AS \"l_tangent_tangent\" FROM \"__ddx_1_jvp\""
    );
}

#[test]
fn a_number_is_a_tangent_the_same_at_every_row() {
    // The derivative of an expression, row by row: jvp along 1.
    let found = find("SELECT x, jvp(sin(x), t.x, 1) AS dx FROM t");
    let Derivative::Jvp { wrt, tangent, .. } = &found.calls[0].derivative else {
        panic!("{:?}", found.calls[0].derivative)
    };
    assert_eq!(wrt, &vec![ColumnRef::new("t", "x")]);
    assert_eq!(
        tangent,
        &Tangent::Literal {
            value: 1.0,
            table: "t".into()
        }
    );
    let negative = find("SELECT jvp(sin(x), t.x, -0.5) AS dx FROM t");
    assert!(matches!(
        &negative.calls[0].derivative,
        Derivative::Jvp { tangent: Tangent::Literal { value: v, .. }, .. } if *v == -0.5
    ));
    // Its tangent table is the table's own dims, with the number under the
    // column's name.
    let mut planned = Derivatives::plan(
        &["SELECT x, jvp(sin(x), t.x, 1) AS dx FROM t"],
        &GenericDialect {},
    )
    .unwrap();
    let input = InputTable {
        name: "__ddx_1_tangent_0_t".into(),
        of: crate::tables::Of::Table(vec!["t".into()]),
        columns: vec!["i".into(), "x".into()],
        keys: 1,
    };
    let job = built(&mut planned, vec![input.clone()]);
    assert_eq!(
        planned.seeds(&job).unwrap(),
        vec![(
            input.name,
            "SELECT \"i\", CAST(1.0 AS DOUBLE) AS \"x\" FROM t".into()
        )]
    );
    // A cotangent is a relation, never a number.
    let e = Calls::find(
        "WITH f AS (SELECT x AS y FROM t) SELECT vjp(f, x, 1) FROM t",
        &GenericDialect {},
    )
    .unwrap_err();
    assert!(e.to_string().contains("name of its cotangent"), "{e}");
}

#[test]
fn an_unnamed_gradient_column_is_named_for_its_column() {
    // As a jvp column is `{e}_tangent`: not a name of ddx's, nor one that
    // moves when another call comes first.
    let out = rewritten(&find(
        "WITH loss AS (SELECT SUM(val) AS l FROM w) \
         SELECT i, grad(loss, val), 2 * grad(loss, val) AS twice FROM w",
    ));
    assert!(
        out.contains("SELECT i, w.\"__ddx_d0\" AS \"val_grad\", 2 * w.\"__ddx_d0\" AS twice"),
        "{out}"
    );
}

#[test]
fn a_filter_on_a_column_a_merged_program_differentiates_is_not_pushed() {
    // Each call alone may push the other's column, a dim of its table;
    // merged into one program along both, it is no dim, and only `i < 10`
    // holds for both.
    let sql = "WITH loss AS (SELECT SUM(a * b) AS l FROM w) \
               SELECT i, grad(loss, a) AS ga, grad(loss, b) AS gb FROM w \
               WHERE a > 0 AND i < 10";
    let planned = Derivatives::plan(&[sql], &GenericDialect {}).unwrap();
    let job = planned.next().unwrap().unwrap();
    assert_eq!(
        job.pushdown,
        vec![Restriction {
            table: "w".into(),
            query: "SELECT * FROM w WHERE i < 10".into(),
        }]
    );
}

#[test]
fn sql_ddx_writes_over_a_table_names_it_as_the_statement_does() {
    // A quoted name keeps its quotes, and an unquoted one stays unquoted,
    // for the engine to fold as it folds the statement's.
    let found = find("SELECT i, jvp(val * val, \"W\".val, 1) AS d FROM \"W\"");
    assert!(matches!(
        &found.calls[0].derivative,
        Derivative::Jvp { tangent: Tangent::Literal { table, .. }, .. } if table == "\"W\""
    ));
    let sql = "WITH loss AS (SELECT SUM(val) AS l FROM \"W\") \
               SELECT i, grad(loss, val) AS g FROM \"W\" WHERE i < 2";
    let planned = Derivatives::plan(&[sql], &GenericDialect {}).unwrap();
    assert_eq!(
        planned.next().unwrap().unwrap().pushdown[0].query,
        "SELECT * FROM \"W\" WHERE i < 2"
    );
}

#[test]
fn a_function_that_reads_its_own_derivative_is_refused() {
    // `loss` reads `g`, which is grad of `loss`: no order runs one first.
    // So does a loss that holds grad of itself, or a tangent that is its own
    // jvp's SELECT.
    for sql in [
        "WITH g AS (SELECT i, grad(loss, val) AS gv FROM w), \
              loss AS (SELECT SUM(gv * gv) AS l FROM g) \
         SELECT * FROM g",
        "WITH loss AS (SELECT SUM(val * val) AS l, grad(loss, val) AS g FROM w) \
         SELECT * FROM loss",
        // The tangent of `jvp` is the SELECT it is a column of.
        "WITH v AS (SELECT i, jvp(val, w.val, v) AS val FROM w) SELECT * FROM v",
    ] {
        let e = Derivatives::plan(&[sql], &GenericDialect {})
            .err()
            .unwrap_or_else(|| panic!("planned {sql}"));
        assert!(
            matches!(&e, AdError::InvalidPlan(m) if m.contains("cannot be part of what it differentiates")),
            "{sql}: {e}"
        );
    }
}

#[test]
fn a_jvp_select_orders_by_its_output_only() {
    let base = "SELECT x, jvp(sin(x), t.x, 1) AS dx FROM t ORDER BY ";
    for ok in ["x", "dx DESC", "2", "x, dx"] {
        Calls::find(&format!("{base}{ok}"), &GenericDialect {})
            .unwrap_or_else(|e| panic!("{ok}: {e}"));
    }
    for refused in ["t.x", "y", "x + 1"] {
        let e = Calls::find(&format!("{base}{refused}"), &GenericDialect {}).unwrap_err();
        assert!(
            matches!(&e, AdError::NotImplemented(m) if m.contains("order by a column of the SELECT's output")),
            "{refused}: {e}"
        );
    }
    // With a *, any bare column of the table is in the output.
    Calls::find(
        "SELECT *, jvp(sin(x), t.x, 1) AS dx FROM t ORDER BY y",
        &GenericDialect {},
    )
    .unwrap();
}

#[test]
fn a_call_written_twice_reads_one_gradient_column() {
    // A SELECT and its GROUP BY must say the same thing after the rewrite,
    // and a * excludes the one column once.
    let out = rewritten(&find(
        "WITH loss AS (SELECT SUM(val) AS l FROM w) \
         SELECT grad(loss, val) > 0 AS up, COUNT(*) AS n FROM w GROUP BY grad(loss, val) > 0",
    ));
    assert!(
        out.contains("SELECT w.\"__ddx_d0\" > 0 AS up, COUNT(*) AS n FROM")
            && out.ends_with("GROUP BY w.\"__ddx_d0\" > 0"),
        "{out}"
    );
    let out = rewritten(&find(
        "WITH loss AS (SELECT SUM(val) AS l FROM w) \
         SELECT *, grad(loss, val) AS a, grad(loss, val) AS b FROM w",
    ));
    assert!(out.contains("SELECT * EXCLUDE (\"__ddx_d0\"),"), "{out}");
}

#[test]
fn a_call_in_an_order_by_is_a_column_of_that_query_s_rows() {
    let loss = "WITH loss AS (SELECT SUM(val) AS l FROM w) ";
    let out = rewritten(&find(&format!(
        "{loss}SELECT i FROM w ORDER BY grad(loss, val) DESC LIMIT 2"
    )));
    assert!(
        out.contains("ORDER BY w.\"__ddx_d0\" DESC LIMIT 2"),
        "{out}"
    );
    // A subquery's ORDER BY orders the subquery's rows: its call is a column
    // of x's, never of the outer SELECT's w.
    let found = find(&format!(
        "{loss}SELECT w.i FROM w WHERE w.i IN (SELECT i FROM x ORDER BY grad(loss, val) LIMIT 1)"
    ));
    assert_eq!(
        found.calls[0].derivative.wrt(),
        &[ColumnRef::new("x", "val")]
    );
    // So is a call in a subquery within the outer ORDER BY.
    let found = find(&format!(
        "{loss}SELECT i FROM w ORDER BY (SELECT grad(loss, val) FROM x LIMIT 1)"
    ));
    assert_eq!(
        found.calls[0].derivative.wrt(),
        &[ColumnRef::new("x", "val")]
    );
    // Anywhere else outside a SELECT's body, a call is in no SELECT.
    for sql in [
        format!("{loss}SELECT i FROM w LIMIT grad(loss, val)"),
        format!("{loss}SELECT i FROM w UNION ALL SELECT i FROM w ORDER BY grad(loss, val)"),
    ] {
        let e = Calls::find(&sql, &GenericDialect {}).unwrap_err();
        assert!(
            matches!(&e, AdError::InvalidPlan(m) if m.contains("outside a SELECT's columns")),
            "{sql}: {e}"
        );
    }
}

#[test]
fn a_column_may_name_its_table_in_full() {
    let found = find(
        "WITH loss AS (SELECT SUM(val) AS l FROM public.w) \
         SELECT i, grad(loss, public.w.val) AS g FROM public.w",
    );
    assert_eq!(
        found.calls[0].derivative.wrt(),
        &[ColumnRef::new("public.w", "val")]
    );
    // An alias hides the table's name, as it does in SQL.
    let e = Calls::find(
        "WITH loss AS (SELECT SUM(val) AS l FROM public.w) \
         SELECT i, grad(loss, public.w.val) AS g FROM public.w AS p",
        &GenericDialect {},
    )
    .unwrap_err();
    assert!(matches!(e, AdError::InvalidPlan(_)), "{e}");
}
