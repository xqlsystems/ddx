// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Prepared statements (`ad::prepare`, `ddx_ad::Prepared`): the first run
//! computes the calls as `ad::sql_all` does, and every later run runs the
//! same plans again, on the tables' rows as they are then, asking the
//! engine for no plan, key or schema: only whether its views have changed,
//! in which case the statements are prepared again.

use std::sync::{Arc, Mutex};

use datafusion::arrow::array::{AsArray, RecordBatch};
use datafusion::arrow::datatypes::{Float64Type, Int64Type};
use datafusion::error::DataFusionError;
use datafusion::prelude::{DataFrame, SessionContext};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use ddx_ad::substrait::proto::{NamedStruct, Plan};
use ddx_ad::{Backend, Bound};
use ddx_datafusion::ad;
use ddx_datafusion::engine::{Compiled, DataFusion};

const STEP: &str = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w) \
                    SELECT i, val - 0.1 * grad(loss, val) AS val FROM w ORDER BY i";

/// `H·v` of the same loss, along the table `v`: `jvp` of its gradient,
/// whose tangent is a seed the program registers on every run.
const HVP: &str = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w), \
                        g AS (SELECT i, grad(loss, val) AS val FROM w) \
                   SELECT i, jvp(val, w.val, v) AS hv FROM g ORDER BY i";

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// The table `name` (`i BIGINT, val DOUBLE`), registered again with `vals`
/// at `i = 0, 1, …`.
async fn put(ctx: &SessionContext, name: &str, vals: &[f64]) {
    exec(ctx, &format!("DROP TABLE IF EXISTS {name}")).await;
    let rows: Vec<String> = vals
        .iter()
        .enumerate()
        .map(|(i, v)| format!("({i}, {v:?})"))
        .collect();
    exec(
        ctx,
        &format!(
            "CREATE TABLE {name} (i BIGINT NOT NULL, val DOUBLE) AS VALUES {}",
            rows.join(", ")
        ),
    )
    .await;
}

/// Each frame's second column, in order.
async fn values(frames: Vec<DataFrame>) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for df in frames {
        let batches: Vec<RecordBatch> = df.collect().await.unwrap();
        let mut column = Vec::new();
        for b in &batches {
            let v = b.column(1).as_primitive::<Float64Type>();
            column.extend(v.values().iter().copied());
        }
        out.push(column);
    }
    out
}

fn ddx_tables(ctx: &SessionContext) -> Vec<String> {
    let mut names = ctx
        .catalog("datafusion")
        .unwrap()
        .schema("public")
        .unwrap()
        .table_names();
    names.retain(|n| n.starts_with("__ddx_"));
    names
}

fn close(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.len() == y.len() && x.iter().zip(y).all(|(p, q)| (p - q).abs() < 1e-12))
}

#[tokio::test]
async fn a_prepared_step_reads_the_tables_rows_as_they_are_at_each_run() {
    let ctx = SessionContext::new();
    put(&ctx, "w", &[1.0, -2.0, 0.5]).await;
    put(&ctx, "v", &[1.0, 1.0, 1.0]).await;
    let step = ad::prepare(&ctx, &[STEP, HVP]);
    for k in 0..4 {
        let got = values(step.run().await.unwrap()).await;
        let want = values(ad::sql_all(&ctx, &[STEP, HVP]).await.unwrap()).await;
        assert!(close(&got, &want), "run {k}: {got:?}, not {want:?}");
        assert!(ddx_tables(&ctx).is_empty(), "{:?}", ddx_tables(&ctx));
        // The next step's weights, and a new direction: registered again,
        // as a training loop registers them.
        let w = got[0].clone();
        put(&ctx, "w", &w).await;
        put(&ctx, "v", &[1.0, k as f64, -1.0]).await;
    }
}

#[tokio::test]
async fn a_failed_run_leaves_nothing_and_the_next_runs_the_same_plans() {
    let ctx = SessionContext::new();
    put(&ctx, "w", &[1.0, -2.0, 0.5]).await;
    let step = ad::prepare(&ctx, &[STEP]);
    step.run().await.unwrap();
    // Two rows share i = 0: the gradient has no row to be at, and the
    // program's check refuses.
    exec(&ctx, "DROP TABLE w").await;
    exec(
        &ctx,
        "CREATE TABLE w (i BIGINT NOT NULL, val DOUBLE) AS VALUES (0, 1.0), (0, 2.0)",
    )
    .await;
    let err = step.run().await.unwrap_err().to_string();
    assert!(err.contains("invalid wrt"), "{err}");
    assert!(ddx_tables(&ctx).is_empty(), "{:?}", ddx_tables(&ctx));
    put(&ctx, "w", &[3.0, 0.0, -1.0]).await;
    let got = values(step.run().await.unwrap()).await;
    // val - 0.1 · 3 val²
    assert!(close(&got, &[vec![3.0 - 2.7, 0.0, -1.0 - 0.3]]), "{got:?}");
}

/// The engine, counting what it is asked.
struct Counting {
    engine: DataFusion,
    asked: Arc<Mutex<Vec<&'static str>>>,
}

impl Counting {
    fn ask(&self, what: &'static str) {
        self.asked.lock().unwrap().push(what);
    }
}

impl Backend for Counting {
    type Error = DataFusionError;
    type Compiled = Compiled;

    async fn table_schema(&mut self, name: &str) -> Result<NamedStruct, DataFusionError> {
        self.ask("table_schema");
        self.engine.table_schema(name).await
    }
    async fn returns_rows(&mut self, plans: &[&Bound]) -> Result<Vec<bool>, DataFusionError> {
        self.ask("returns_rows");
        self.engine.returns_rows(plans).await
    }
    async fn materialize(&mut self, name: &str, plan: &Bound) -> Result<(), DataFusionError> {
        self.ask("materialize");
        self.engine.materialize(name, plan).await
    }
    async fn drop_table(&mut self, name: &str) -> Result<(), DataFusionError> {
        self.engine.drop_table(name).await
    }
    async fn plan_sql(&mut self, sql: &str) -> Result<Plan, DataFusionError> {
        self.ask("plan_sql");
        self.engine.plan_sql(sql).await
    }
    async fn plan_key(&mut self, sql: &str) -> Result<Option<String>, DataFusionError> {
        self.ask("plan_key");
        self.engine.plan_key(sql).await
    }
    async fn view_sql(&mut self, name: &str) -> Result<Option<String>, DataFusionError> {
        self.ask("view_sql");
        self.engine.view_sql(name).await
    }
    async fn session_key(&mut self) -> Result<Option<String>, DataFusionError> {
        self.ask("session_key");
        self.engine.session_key().await
    }
    async fn compile(&mut self, plan: &Bound) -> Result<Option<Compiled>, DataFusionError> {
        self.ask("compile");
        self.engine.compile(plan).await
    }
    async fn materialize_compiled(
        &mut self,
        name: &str,
        plan: &Bound,
        compiled: &Compiled,
    ) -> Result<(), DataFusionError> {
        self.ask("materialize_compiled");
        self.engine.materialize_compiled(name, plan, compiled).await
    }
    async fn returns_rows_compiled(
        &mut self,
        plans: &[(&Bound, &Compiled)],
    ) -> Result<Vec<bool>, DataFusionError> {
        self.ask("returns_rows_compiled");
        self.engine.returns_rows_compiled(plans).await
    }
}

#[tokio::test]
async fn a_later_run_asks_the_engine_only_to_run_plans() {
    let ctx = SessionContext::new();
    put(&ctx, "w", &[1.0, -2.0, 0.5]).await;
    put(&ctx, "v", &[1.0, 1.0, 1.0]).await;
    exec(&ctx, "CREATE VIEW m AS SELECT i, val FROM w").await;
    let asked = Arc::new(Mutex::new(Vec::new()));
    let mut backend = Counting {
        engine: DataFusion(ctx.clone()),
        asked: Arc::clone(&asked),
    };
    let model = "WITH loss AS (SELECT SUM(val * val) AS l FROM m) \
                 SELECT i, grad(loss, val) FROM w";
    let (prepared, first) = ddx_ad::prepare(&mut backend, &[model, HVP], &GenericDialect {})
        .await
        .unwrap();
    let firsts = asked.lock().unwrap().clone();
    for what in ["plan_sql", "view_sql", "table_schema"] {
        assert!(firsts.contains(&what), "{what} not in {firsts:?}");
    }
    assert!(
        !firsts.contains(&"plan_key"),
        "prepared, and keyed: {firsts:?}"
    );
    first
        .release(&mut backend, &ddx_ad::Programs::new())
        .await
        .unwrap();
    asked.lock().unwrap().clear();

    let again = prepared
        .run(&mut backend)
        .await
        .unwrap()
        .expect("no view changed");
    assert_eq!(again.statements, prepared.statements());
    // Every plan was compiled, and runs as it was compiled: none is planned again
    // (but by the session, from SQL, with `DDX_STEPS_AS_SQL=1`).
    let later = asked.lock().unwrap().clone();
    let run = if std::env::var_os("DDX_STEPS_AS_SQL").is_some_and(|v| v == "1") {
        ["returns_rows", "materialize"]
    } else {
        ["returns_rows_compiled", "materialize_compiled"]
    };
    assert!(
        later.iter().all(|w| *w == "session_key" || run.contains(w)),
        "{later:?}"
    );
    // The gradient of Σ val², 2·val, read through the view.
    let df = ctx.sql(&again.statements[0]).await.unwrap();
    let batches = df.collect().await.unwrap();
    let mut got: Vec<(i64, f64)> = Vec::new();
    for b in &batches {
        let i = b.column(0).as_primitive::<Int64Type>();
        let g = b.column(1).as_primitive::<Float64Type>();
        got.extend((0..b.num_rows()).map(|r| (i.value(r), g.value(r))));
    }
    got.sort_by_key(|p| p.0);
    assert_eq!(got, [(0, 2.0), (1, -4.0), (2, 1.0)]);
    again
        .release(&mut backend, &ddx_ad::Programs::new())
        .await
        .unwrap();
    assert!(ddx_tables(&ctx).is_empty(), "{:?}", ddx_tables(&ctx));
}

/// The gradient of `SUM(val * val)` over the view `m`, at each `i` of `w`,
/// computed by `step` and by `ad::sql_all`.
async fn grads(ctx: &SessionContext, step: &ad::Prepared) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let got = values(step.run().await.unwrap()).await;
    let want = values(ad::sql_all(ctx, &[THROUGH_M]).await.unwrap()).await;
    (got, want)
}

const THROUGH_M: &str = "WITH loss AS (SELECT SUM(val * val) AS l FROM m) \
                         SELECT i, grad(loss, val) FROM w ORDER BY i";

#[tokio::test]
async fn a_view_redefined_is_differentiated_as_it_is_now() {
    let ctx = SessionContext::new();
    put(&ctx, "w", &[1.0, -2.0, 0.5]).await;
    exec(&ctx, "CREATE VIEW m AS SELECT i, val FROM w").await;
    let step = ad::prepare(&ctx, &[THROUGH_M]);
    let (got, _) = grads(&ctx, &step).await;
    assert_eq!(got, [vec![2.0, -4.0, 1.0]]);
    // Made again as it was: the same plans run.
    let before = DataFusion(ctx.clone()).session_key().await.unwrap();
    exec(&ctx, "CREATE OR REPLACE VIEW m AS SELECT i, val FROM w").await;
    assert_eq!(DataFusion(ctx.clone()).session_key().await.unwrap(), before);
    let (got, _) = grads(&ctx, &step).await;
    assert_eq!(got, [vec![2.0, -4.0, 1.0]]);
    exec(
        &ctx,
        "CREATE OR REPLACE VIEW m AS SELECT i, 2 * val AS val FROM w",
    )
    .await;
    let (got, _) = grads(&ctx, &step).await;
    assert_eq!(got, [vec![8.0, -16.0, 4.0]]);
    assert!(ddx_tables(&ctx).is_empty(), "{:?}", ddx_tables(&ctx));
}

#[tokio::test]
async fn a_view_made_again_over_a_redefined_view_is_differentiated_as_it_runs() {
    // DataFusion keeps the plan a view was made with: `m` reads `inner` as
    // it was until `m` is made again, with the same SQL.
    let ctx = SessionContext::new();
    put(&ctx, "w", &[1.0, -2.0, 0.5]).await;
    exec(&ctx, "CREATE VIEW inner_m AS SELECT i, val FROM w").await;
    exec(&ctx, "CREATE VIEW m AS SELECT i, val FROM inner_m").await;
    let step = ad::prepare(&ctx, &[THROUGH_M]);
    grads(&ctx, &step).await;
    exec(
        &ctx,
        "CREATE OR REPLACE VIEW inner_m AS SELECT i, 2 * val AS val FROM w",
    )
    .await;
    let (got, want) = grads(&ctx, &step).await;
    assert!(close(&got, &want), "{got:?}, not {want:?}");
    exec(
        &ctx,
        "CREATE OR REPLACE VIEW m AS SELECT i, val FROM inner_m",
    )
    .await;
    let (got, want) = grads(&ctx, &step).await;
    assert!(close(&got, &want), "{got:?}, not {want:?}");
    assert_eq!(got, [vec![8.0, -16.0, 4.0]]);
}

#[tokio::test]
async fn a_view_over_a_table_registered_again_is_refused_as_sql_all_refuses_it() {
    let ctx = SessionContext::new();
    put(&ctx, "w", &[1.0, -2.0, 0.5]).await;
    exec(&ctx, "CREATE VIEW m AS SELECT i, val FROM w").await;
    let step = ad::prepare(&ctx, &[THROUGH_M]);
    step.run().await.unwrap();
    // `m` still reads the old `w`, and ddx would read the new one.
    put(&ctx, "w", &[3.0, 0.0, -1.0]).await;
    let err = step.run().await.unwrap_err().to_string();
    assert!(err.contains("registered again"), "{err}");
    exec(&ctx, "CREATE OR REPLACE VIEW m AS SELECT i, val FROM w").await;
    let (got, _) = grads(&ctx, &step).await;
    assert_eq!(got, [vec![6.0, 0.0, -2.0]]);
}

#[tokio::test]
async fn a_run_after_the_default_schema_changes_matches_sql_all() {
    let ctx = SessionContext::new();
    exec(&ctx, "CREATE SCHEMA s").await;
    put(&ctx, "w", &[1.0, 2.0]).await;
    put(&ctx, "s.w", &[10.0, 20.0]).await;
    let step = ad::prepare(&ctx, &[STEP]);
    let first = values(step.run().await.unwrap()).await;
    exec(&ctx, "SET datafusion.catalog.default_schema = 's'").await;
    let got = values(step.run().await.unwrap()).await;
    let want = values(ad::sql_all(&ctx, &[STEP]).await.unwrap()).await;
    assert!(close(&got, &want), "{got:?}, not {want:?}");
    assert!(!close(&got, &first), "{got:?} read the old schema's w");
}

#[tokio::test]
async fn a_run_after_a_table_changes_its_types_matches_sql_all() {
    let ctx = SessionContext::new();
    put(&ctx, "w", &[1.0, -2.0, 0.5]).await;
    let step = ad::prepare(&ctx, &[STEP]);
    step.run().await.unwrap();
    // The same rows, as FLOAT: the plans were bound to DOUBLE.
    exec(&ctx, "DROP TABLE w").await;
    exec(
        &ctx,
        "CREATE TABLE w (i BIGINT NOT NULL, val FLOAT) AS VALUES (0, 1.0), (1, -2.0), (2, 0.5)",
    )
    .await;
    let got = step.run().await.unwrap();
    let want = ad::sql_all(&ctx, &[STEP]).await.unwrap();
    let (got, want) = (
        got[0].clone().collect().await.unwrap(),
        want[0].clone().collect().await.unwrap(),
    );
    assert_eq!(
        datafusion::arrow::util::pretty::pretty_format_batches(&got)
            .unwrap()
            .to_string(),
        datafusion::arrow::util::pretty::pretty_format_batches(&want)
            .unwrap()
            .to_string()
    );
}

/// `sql_all`'s and a prepared run's rows of each statement, as text.
async fn both(ctx: &SessionContext, step: &ad::Prepared, statements: &[&str]) -> (String, String) {
    let text = |batches: Vec<RecordBatch>| {
        datafusion::arrow::util::pretty::pretty_format_batches(&batches)
            .unwrap()
            .to_string()
    };
    let (mut got, mut want) = (String::new(), String::new());
    for df in step.run().await.unwrap() {
        got += &text(df.collect().await.unwrap());
    }
    for df in ad::sql_all(ctx, statements).await.unwrap() {
        want += &text(df.collect().await.unwrap());
    }
    (got, want)
}

#[tokio::test]
async fn a_compiled_plan_counts_the_rows_its_table_has_at_each_run() {
    // An ungrouped COUNT(*) of a table whose statistics are exact folds to
    // the table's row count when it is planned.
    let ctx = SessionContext::new();
    let mean = "WITH loss AS (SELECT SUM(val * val) / COUNT(*) AS l FROM w) \
                SELECT i, grad(loss, val) FROM w ORDER BY i";
    put(&ctx, "w", &[1.0, -2.0, 0.5]).await;
    let step = ad::prepare(&ctx, &[mean]);
    for vals in [&[1.0, -2.0][..], &[1.0, -2.0, 0.5, 4.0, 3.0], &[2.0]] {
        let (got, want) = both(&ctx, &step, &[mean]).await;
        assert_eq!(got, want);
        put(&ctx, "w", vals).await;
    }
}

#[tokio::test]
async fn a_compiled_join_reads_its_build_side_as_it_is_at_each_run() {
    // A join on keys a dynamic filter would narrow to the first run's, with
    // dynamic filter pushdown on, as it is by default.
    let ctx = SessionContext::new();
    let join = "WITH loss AS (SELECT SUM(w.val * k.val) AS l FROM w JOIN k ON w.i = k.i) \
                SELECT i, grad(loss, val) FROM w ORDER BY i";
    put(&ctx, "w", &[1.0, -2.0, 0.5, 3.0, 4.0]).await;
    exec(
        &ctx,
        "CREATE TABLE k (i BIGINT NOT NULL, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    let step = ad::prepare(&ctx, &[join]);
    for keys in [
        "(3, 5.0), (4, 6.0)",
        "(1, 7.0)",
        "(0, 1.0), (2, 1.0), (4, 1.0)",
    ] {
        let (got, want) = both(&ctx, &step, &[join]).await;
        assert_eq!(got, want);
        exec(&ctx, "DROP TABLE k").await;
        exec(
            &ctx,
            &format!("CREATE TABLE k (i BIGINT NOT NULL, val DOUBLE) AS VALUES {keys}"),
        )
        .await;
    }
}

#[tokio::test]
async fn a_compiled_plan_over_a_table_split_otherwise_is_planned_again() {
    use datafusion::arrow::array::{Float64Array, Int64Array};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionConfig;
    // One partition, so nothing repartitions the scan: the plan reads as
    // many partitions as it was planned for.
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, false),
        Field::new("val", DataType::Float64, true),
    ]));
    let batch = |i: Vec<i64>, v: Vec<f64>| {
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(i)),
                Arc::new(Float64Array::from(v)),
            ],
        )
        .unwrap()
    };
    let register = |partitions: Vec<Vec<RecordBatch>>| {
        ctx.deregister_table("w").unwrap();
        let table = MemTable::try_new(Arc::clone(&schema), partitions).unwrap();
        ctx.register_table("w", Arc::new(table)).unwrap();
    };
    register(vec![vec![batch(vec![0, 1, 2], vec![1.0, -2.0, 0.5])]]);
    let step = ad::prepare(&ctx, &[STEP]);
    let (got, want) = both(&ctx, &step, &[STEP]).await;
    assert_eq!(got, want);
    // The same rows, in three partitions, then two.
    register(vec![
        vec![batch(vec![0], vec![1.0])],
        vec![batch(vec![1], vec![-2.0])],
        vec![batch(vec![2], vec![0.5])],
    ]);
    let (got, want) = both(&ctx, &step, &[STEP]).await;
    assert_eq!(got, want);
    register(vec![
        vec![batch(vec![0, 1], vec![1.0, -2.0])],
        vec![batch(vec![2, 3], vec![0.5, 9.0])],
    ]);
    let (got, want) = both(&ctx, &step, &[STEP]).await;
    assert_eq!(got, want);
}

#[tokio::test]
async fn a_compiled_count_is_of_the_rows_its_table_has_when_it_runs() {
    // DataFusion folds an ungrouped COUNT(*) of a table whose statistics
    // are exact into the count when it plans it.
    let ctx = SessionContext::new();
    let mut engine = DataFusion(ctx.clone());
    put(&ctx, "w", &[1.0, -2.0, 0.5]).await;
    let plan = engine
        .plan_sql("SELECT COUNT(*) AS n FROM w")
        .await
        .unwrap();
    let plan = ddx_ad::Unbound::from_plan(plan)
        .bind(&mut |_| None)
        .unwrap();
    // A plan run as SQL (`DDX_STEPS_AS_SQL=1`) is not compiled.
    let Some(compiled) = engine.compile(&plan).await.unwrap() else {
        return;
    };
    for vals in [&[1.0, -2.0, 0.5][..], &[1.0], &[1.0, 2.0, 3.0, 4.0, 5.0]] {
        put(&ctx, "w", vals).await;
        engine
            .materialize_compiled("n", &plan, &compiled)
            .await
            .unwrap();
        let batches = ctx.table("n").await.unwrap().collect().await.unwrap();
        let n = batches[0].column(0).as_primitive::<Int64Type>().value(0);
        assert_eq!(n, vals.len() as i64);
        let rows = engine
            .returns_rows_compiled(&[(&plan, &compiled)])
            .await
            .unwrap();
        assert_eq!(rows, [true]);
    }
}
