// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Where a SQL statement's `grad`, `vjp` and `jvp` spend their time.
//!
//! Replays a recorded session (tables, then statements) and profiles each
//! statement that calls ddx: planning the queries ddx differentiates
//! (`plan_sql`), ddx-ad's own work (parsing, building the program,
//! binding), the checks, each step (split into consuming its Substrait plan,
//! DataFusion's planning, and execution), and the rewritten statements. It
//! also prints the steps' critical path: how long they would take if every
//! step ran as soon as the steps it reads were done.
//!
//! To record a session of one of xarray-sql's ML demos
//! (`benchmarks/ml/*.py`), from an xarray-sql checkout:
//!
//! ```text
//! python capture.py out/ benchmarks/ml/02_nanogpt.py --hosts datafusion --steps 1
//! python to_replay.py out/events.jsonl out/replay.txt
//! cargo run --release -p ddx-datafusion --example profile -- out/replay.txt [N]
//! ```
//!
//! `replay.txt` holds `@load name path.parquet`, `@execute`, `@query` and
//! `@query_all` blocks (statements of one batch separated by `@sep`, each
//! block closed by `@end`). The first `N` (default 1) blocks that call ddx
//! are profiled; the rest are run as they are. Options, as environment
//! variables:
//!
//! - `DDX_PROF_ANALYZE=k`: the physical plans of the `k` slowest steps;
//! - `DDX_PROF_STEPS=1`: every step, with the earlier steps it reads;
//! - `DDX_PROF_SHOW=1`: the rewritten statements;
//! - `DDX_PROF_DROP_WITH=1`: plan each rewritten statement without its
//!   `WITH` clause (where that still plans), to time CTEs nothing reads;
//! - `DDX_PROF_PASSES=n`: DataFusion's `optimizer.max_passes`;
//! - `DDX_PROF_BUILD=t.c,u.d`: instead, time planning the first block
//!   that does not call ddx (a loss) and building its gradient program with
//!   respect to those columns.
//!
//! `attention_by_hand.py` times demo 1's attention gradient against the
//! same gradient written by hand as SQL, step by step, as JAX's transpose
//! rules give it.

use std::sync::Arc;
use std::time::{Duration, Instant};

use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use ddx_ad::substrait::proto::{NamedStruct, Plan};
use ddx_ad::{Backend, Bound};
use ddx_datafusion::engine::{self, DataFusion};

#[derive(Default)]
struct Profile {
    plan_sql: Vec<(Duration, usize)>,
    schemas: Duration,
    checks: Vec<(Duration, usize)>,
    steps: Vec<StepTime>,
}

struct StepTime {
    name: String,
    consume: Duration,
    plan: Duration,
    exec: Duration,
    rows: usize,
    batches: usize,
    substrait_bytes: usize,
    operators: usize,
    analyzed: Option<String>,
    reads: Vec<String>,
}

/// DataFusion, timing everything a program's run asks of it.
struct Timed {
    inner: DataFusion,
    profile: Profile,
}

impl Backend for Timed {
    type Error = datafusion::error::DataFusionError;

    async fn table_schema(&mut self, name: &str) -> Result<NamedStruct> {
        let t = Instant::now();
        let s = self.inner.table_schema(name).await;
        self.profile.schemas += t.elapsed();
        s
    }

    async fn returns_rows(&mut self, plans: &[Bound]) -> Result<Vec<bool>> {
        let t = Instant::now();
        let r = self.inner.returns_rows(plans).await;
        self.profile.checks.push((t.elapsed(), plans.len()));
        r
    }

    async fn materialize(&mut self, name: &str, plan: &Bound) -> Result<()> {
        let ctx = &self.inner.0;
        let bytes = prost::Message::encoded_len(&**plan);
        let text = format!("{:?}", **plan);
        let reads: Vec<String> = self
            .profile
            .steps
            .iter()
            .map(|s| s.name.clone())
            .filter(|n| text.contains(&format!("\"{n}\"")))
            .collect();
        let t = Instant::now();
        let lp = engine::logical_plan(ctx, plan).await?;
        let consume = t.elapsed();
        let t = Instant::now();
        let df = ctx.execute_logical_plan(lp).await?;
        let task = Arc::new(df.task_ctx());
        let physical = df.create_physical_plan().await?;
        let plan_time = t.elapsed();
        let operators = displayable(physical.as_ref())
            .indent(false)
            .to_string()
            .lines()
            .count();
        let schema = physical.schema();
        let t = Instant::now();
        let batches = collect(Arc::clone(&physical), task).await?;
        let exec = t.elapsed();
        let analyzed = std::env::var("DDX_PROF_ANALYZE")
            .is_ok()
            .then(|| displayable(physical.as_ref()).indent(true).to_string());
        let rows = batches.iter().map(|b| b.num_rows()).sum();
        let n = batches.len();
        let table = MemTable::try_new(schema, vec![batches])?;
        ctx.deregister_table(name)?;
        ctx.register_table(name, Arc::new(table))?;
        self.profile.steps.push(StepTime {
            name: name.to_string(),
            consume,
            plan: plan_time,
            exec,
            rows,
            batches: n,
            substrait_bytes: bytes,
            operators,
            analyzed,
            reads,
        });
        Ok(())
    }

    async fn drop_table(&mut self, name: &str) -> Result<()> {
        self.inner.drop_table(name).await
    }

    async fn plan_sql(&mut self, sql: &str) -> Result<Plan> {
        let t = Instant::now();
        let p = self.inner.plan_sql(sql).await;
        let bytes = p.as_ref().map_or(0, prost::Message::encoded_len);
        self.profile.plan_sql.push((t.elapsed(), bytes));
        p
    }
}

enum Event {
    Load(String, String),
    Execute(String),
    Query(Vec<String>),
}

fn parse(text: &str) -> Vec<Event> {
    let mut events = Vec::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("@load ") {
            let (name, path) = rest.split_once(' ').expect("@load name path");
            events.push(Event::Load(name.into(), path.into()));
            continue;
        }
        let kind = line.trim();
        if kind.is_empty() {
            continue;
        }
        let mut stmts = vec![String::new()];
        for l in lines.by_ref() {
            match l {
                "@end" => break,
                "@sep" => stmts.push(String::new()),
                l => {
                    let s = stmts.last_mut().unwrap();
                    s.push_str(l);
                    s.push('\n');
                }
            }
        }
        events.push(match kind {
            "@execute" => Event::Execute(stmts.remove(0)),
            "@query" | "@query_all" => Event::Query(stmts),
            other => panic!("unknown block {other}"),
        });
    }
    events
}

/// `sql` without its WITH clause, to time a statement whose CTEs nothing reads.
fn without_with(sql: &str) -> Option<String> {
    use datafusion::sql::sqlparser::ast::Statement;
    use datafusion::sql::sqlparser::parser::Parser;
    let mut stmts = Parser::parse_sql(&GenericDialect {}, sql).ok()?;
    let Statement::Query(q) = stmts.get_mut(0)? else {
        return None;
    };
    q.with.take()?;
    Some(stmts[0].to_string())
}

fn calls_ad(sql: &str) -> bool {
    let lower = sql.to_lowercase();
    ["grad(", "vjp(", "jvp("].iter().any(|c| lower.contains(c))
}

fn ms(d: Duration) -> String {
    format!("{:.1}", d.as_secs_f64() * 1e3)
}

async fn profile(ctx: &SessionContext, statements: &[String]) -> Result<()> {
    let refs: Vec<&str> = statements.iter().map(String::as_str).collect();
    let mut timed = Timed {
        inner: DataFusion(ctx.clone()),
        profile: Profile::default(),
    };
    let total = Instant::now();
    let out = ddx_ad::rewrite(&mut timed, &refs, &GenericDialect {})
        .await
        .map_err(|e| match e {
            ddx_ad::RunError::Engine(e) => e,
            other => datafusion::error::DataFusionError::Plan(other.to_string()),
        })?;
    let rewrite = total.elapsed();
    let t = Instant::now();
    let mut rows = 0;
    let (mut st_plan, mut st_exec) = (Duration::ZERO, Duration::ZERO);
    for sql in &out.statements {
        if std::env::var("DDX_PROF_SHOW").is_ok() {
            println!(
                "-- rewritten ({} bytes):\n{}",
                sql.len(),
                sql.chars().take(3000).collect::<String>()
            );
        }
        let t = Instant::now();
        let df = match without_with(sql) {
            Some(bare) if std::env::var("DDX_PROF_DROP_WITH").is_ok() => match ctx.sql(&bare).await
            {
                Ok(df) => df,
                Err(_) => ctx.sql(sql).await?,
            },
            _ => ctx.sql(sql).await?,
        };
        st_plan += t.elapsed();
        let t = Instant::now();
        rows += df
            .collect()
            .await?
            .iter()
            .map(|b| b.num_rows())
            .sum::<usize>();
        st_exec += t.elapsed();
    }
    println!(
        "  statements: planning {} ms, executing {} ms",
        ms(st_plan),
        ms(st_exec)
    );
    let statements_time = t.elapsed();
    out.release(&mut timed).await?;
    let p = &timed.profile;
    let sum = |v: &[(Duration, usize)]| v.iter().map(|x| x.0).sum::<Duration>();
    let consume: Duration = p.steps.iter().map(|s| s.consume).sum();
    let plan: Duration = p.steps.iter().map(|s| s.plan).sum();
    let exec: Duration = p.steps.iter().map(|s| s.exec).sum();
    let accounted = sum(&p.plan_sql) + p.schemas + sum(&p.checks) + consume + plan + exec;
    println!(
        "\n== {} statement(s), total {} ms",
        statements.len(),
        ms(total.elapsed())
    );
    println!(
        "  plan_sql   {:>9} ms  ({} calls, {} KB of Substrait)",
        ms(sum(&p.plan_sql)),
        p.plan_sql.len(),
        p.plan_sql.iter().map(|x| x.1).sum::<usize>() / 1024
    );
    println!("  schemas    {:>9} ms", ms(p.schemas));
    println!(
        "  checks     {:>9} ms  ({} plans in {} batches)",
        ms(sum(&p.checks)),
        p.checks.iter().map(|x| x.1).sum::<usize>(),
        p.checks.len()
    );
    println!(
        "  steps      {:>9} ms  ({} steps: consume {} + plan {} + exec {})",
        ms(consume + plan + exec),
        p.steps.len(),
        ms(consume),
        ms(plan),
        ms(exec)
    );
    println!(
        "  ddx-ad     {:>9} ms  (rewrite minus the engine's calls: parsing, building, binding)",
        ms(rewrite.saturating_sub(accounted))
    );
    println!(
        "  statements {:>9} ms  ({} rows)",
        ms(statements_time),
        rows
    );
    // The critical path: each step starts when the steps it reads finish.
    let mut finish: std::collections::HashMap<&str, Duration> = Default::default();
    let mut longest = Duration::ZERO;
    for s in &p.steps {
        let start = s
            .reads
            .iter()
            .filter_map(|r| finish.get(r.as_str()))
            .max()
            .copied()
            .unwrap_or_default();
        let f = start + s.consume + s.plan + s.exec;
        longest = longest.max(f);
        finish.insert(s.name.as_str(), f);
    }
    println!(
        "  critical path of the steps: {} ms of {} ms",
        ms(longest),
        ms(consume + plan + exec)
    );
    let mut by_time: Vec<&StepTime> = p.steps.iter().collect();
    by_time.sort_by_key(|s| std::cmp::Reverse(s.consume + s.plan + s.exec));
    println!(
        "  {:<44} {:>8} {:>8} {:>8} {:>9} {:>6} {:>5} {:>7}",
        "slowest steps", "consume", "plan", "exec", "rows", "batch", "ops", "KB"
    );
    for s in by_time.iter().take(15) {
        println!(
            "  {:<44} {:>8} {:>8} {:>8} {:>9} {:>6} {:>5} {:>7}",
            s.name.chars().take(44).collect::<String>(),
            ms(s.consume),
            ms(s.plan),
            ms(s.exec),
            s.rows,
            s.batches,
            s.operators,
            s.substrait_bytes / 1024
        );
    }
    let k: usize = std::env::var("DDX_PROF_ANALYZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    for s in by_time.iter().take(k) {
        if let Some(a) = &s.analyzed {
            println!("\n-- {} ({} ms exec)\n{a}", s.name, ms(s.exec));
        }
    }
    if std::env::var("DDX_PROF_STEPS").is_ok() {
        for s in &p.steps {
            println!(
                "  step {:<36} {:>8} {:>9} reads {}",
                s.name,
                ms(s.consume + s.plan + s.exec),
                s.rows,
                s.reads
                    .iter()
                    .map(|r| r
                        .rsplit('_')
                        .take(2)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<Vec<_>>()
                        .join("_"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    Ok(())
}

/// Time planning `loss` and building its gradient program with respect to
/// `wrt` (`t.c,u.d`), apart from running anything.
async fn build(ctx: &SessionContext, loss: &str, wrt: &str) -> Result<()> {
    let wrt: Vec<ddx_ad::ColumnRef> = wrt
        .split(',')
        .map(|tc| {
            let (t, c) = tc.split_once('.').expect("table.column");
            ddx_ad::ColumnRef::new(t, c)
        })
        .collect();
    for _ in 0..2 {
        let t = Instant::now();
        let lp = ctx.sql(loss).await?.into_unoptimized_plan();
        let sql_t = t.elapsed();
        let t = Instant::now();
        let opt = ctx.state().optimize(&lp)?;
        let opt_t = t.elapsed();
        let t = Instant::now();
        let plan =
            datafusion_substrait::logical_plan::producer::to_substrait_plan(&opt, &ctx.state())?;
        let sub_t = t.elapsed();
        let bytes = prost::Message::encoded_len(&*plan);
        println!(
            "sql→plan {} ms, optimize {} ms, to_substrait {} ms ({} KB)",
            ms(sql_t),
            ms(opt_t),
            ms(sub_t),
            bytes / 1024
        );
        let t = Instant::now();
        let program = match ddx_ad::grad(&*plan, &wrt) {
            Ok(p) => p,
            Err(e) => {
                println!("  grad refused after {} ms: {e}", ms(t.elapsed()));
                continue;
            }
        };
        let grad_t = t.elapsed();
        let steps = ddx_ad::Program::step_count(&program);
        let mut aggs = (0usize, 0usize);
        let t_enc = Instant::now();
        count_aggs(
            plan.relations.first().and_then(|r| match &r.rel_type {
                Some(ddx_ad::substrait::proto::plan_rel::RelType::Root(root)) => {
                    root.input.as_ref()
                }
                _ => None,
            }),
            &mut aggs,
        );
        let enc_t = t_enc.elapsed();
        println!("  encoding every aggregate once: {} ms", ms(enc_t));
        let t = Instant::now();
        let unopt =
            datafusion_substrait::logical_plan::producer::to_substrait_plan(&lp, &ctx.state());
        let unopt_info = match unopt {
            Ok(u) => {
                let b = prost::Message::encoded_len(&*u);
                let t2 = Instant::now();
                let r = ddx_ad::grad(&*u, &wrt);
                format!(
                    "unoptimized: {} KB, grad {} ms ({})",
                    b / 1024,
                    ms(t2.elapsed()),
                    r.map(|p| format!("{} steps", ddx_ad::Program::step_count(&p)))
                        .unwrap_or_else(|e| e.to_string().chars().take(80).collect())
                )
            }
            Err(e) => format!("unoptimized: {e}"),
        };
        let _ = t;
        println!(
            "  grad {} ms ({} steps); {} aggregates, re-encoding each costs {} KB; {}",
            ms(grad_t),
            steps,
            aggs.0,
            aggs.1 / 1024,
            unopt_info
        );
    }
    Ok(())
}

fn count_aggs(rel: Option<&ddx_ad::substrait::proto::Rel>, out: &mut (usize, usize)) {
    use ddx_ad::substrait::proto::rel::RelType as R;
    let Some(rel) = rel else { return };
    let kids: Vec<Option<&ddx_ad::substrait::proto::Rel>> = match rel.rel_type.as_ref() {
        Some(R::Project(p)) => vec![p.input.as_deref()],
        Some(R::Filter(f)) => vec![f.input.as_deref()],
        Some(R::Sort(f)) => vec![f.input.as_deref()],
        Some(R::Fetch(f)) => vec![f.input.as_deref()],
        Some(R::Aggregate(a)) => {
            out.0 += 1;
            out.1 += prost::Message::encode_to_vec(&**a).len();
            vec![a.input.as_deref()]
        }
        Some(R::Join(j)) => vec![j.left.as_deref(), j.right.as_deref()],
        Some(R::Cross(j)) => vec![j.left.as_deref(), j.right.as_deref()],
        Some(R::Set(s)) => s.inputs.iter().map(Some).collect(),
        _ => vec![],
    };
    for k in kids {
        count_aggs(k, out);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("a replay file");
    let mut left: usize = args.next().map_or(1, |n| n.parse().expect("N"));
    let events = parse(&std::fs::read_to_string(path)?);
    let mut config = datafusion::prelude::SessionConfig::new();
    if let Ok(p) = std::env::var("DDX_PROF_PASSES") {
        config = config.set_usize("datafusion.optimizer.max_passes", p.parse().unwrap());
    }
    let ctx = SessionContext::new_with_config(config);
    ddx_datafusion::register_stop_gradient(&ctx);
    for event in events {
        match event {
            Event::Load(name, path) => {
                if let Some((schema, _)) = name.split_once('.') {
                    ctx.sql(&format!("CREATE SCHEMA IF NOT EXISTS {schema}"))
                        .await?
                        .collect()
                        .await?;
                }
                let df = ctx
                    .read_parquet(path, ParquetReadOptions::default())
                    .await?;
                let schema = Arc::new(df.schema().as_arrow().clone());
                let batches = df.collect().await?;
                ctx.deregister_table(name.as_str())?;
                ctx.register_table(
                    name.as_str(),
                    Arc::new(MemTable::try_new(schema, vec![batches])?),
                )?;
            }
            Event::Execute(sql) => {
                ctx.sql(&sql).await?.collect().await?;
            }
            Event::Query(stmts) if stmts.iter().any(|s| calls_ad(s)) => {
                if left == 0 {
                    break;
                }
                left -= 1;
                profile(&ctx, &stmts).await?;
            }
            Event::Query(stmts) => {
                if let Ok(wrt) = std::env::var("DDX_PROF_BUILD") {
                    build(&ctx, &stmts[0], &wrt).await?;
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}
