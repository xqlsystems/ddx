// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `jvp` of a query: its output and the output's directional derivative.
//!
//! As in JAX, [`jvp`] pushes a tangent of the `wrt` columns forward to the
//! query's output, `J·t`. A tangent is shaped like its primal: for each `wrt`
//! table, a relation with the table's dims and a tangent for each of its
//! `wrt` columns: one input table of the program per `wrt` table
//! ([`ForwardProgram::inputs`]), which the caller registers before it runs. A row
//! the tangent table lacks has tangent 0.
//!
//! Forward mode needs no transposes and no tape: tangents travel beside
//! values through the same operators (see [`crate::tangent`]'s table), so a
//! query's `jvp` is one plan, the query rewritten. Its result holds the
//! query's own output and, after it, the tangent of each output column that
//! depends on a `wrt` column, as `jax.jvp` returns both. A tangent is NULL
//! where its value is.
//!
//! [`jvp`] of a program does the same for the program's steps: of a `grad`
//! program, forward mode over reverse mode, which gives a Hessian-vector
//! product.
//!
//! The run protocol is a [`BackwardProgram`]'s ([`crate::run`], with
//! [`crate::Runner`]): checks first, then the steps.

use std::collections::{BTreeSet, HashMap};

use substrait::proto::{NamedStruct, Plan};

use crate::compose::{Differentiable, Parts, Subject};
use crate::emit::{plan as plan_of, project_emit, read_step};
use crate::error::{AdError, Result};
use crate::expr::field;
use crate::forward::{check_every_wrt_was_read, inline_references, root_of, wrt_table};
use crate::functions::{Extensions, Functions};
use crate::program::{dims_check, readable, repeated_keys, Check, ForwardProgram, Options, Step};
use crate::prune::prune_plan;
use crate::relation::{table_matches, ColumnRef, Table};
use crate::tables::{InputTable, Of, OutputTable, Tangent};
use crate::tangent::{Dualizer, Source, Tan};

/// The tangent table of `wrt` table `i`, in a program's namespace.
fn tangent_name(namespace: &str, i: usize, table: &[String]) -> String {
    format!("{namespace}tangent_{i}_{}", readable(table))
}

/// The column holding column `c`'s tangent, `__ddx_tangent_{c}`, unless
/// one of `taken` is called that (a column of a `jvp` program's step, whose
/// tangent `jvp` of the program takes again): then `__ddx_tangent_{c}_{n}`
/// for the first `n` that is free.
fn tangent_column(c: usize, taken: &[String]) -> String {
    let name = format!("__ddx_tangent_{c}");
    if !taken.contains(&name) {
        return name;
    }
    (2..)
        .map(|n| format!("{name}_{n}"))
        .find(|n| !taken.contains(n))
        .expect("a free name")
}

/// The Jacobian-vector product of `of`, a query's [`Plan`] or a program,
/// with respect to the `wrt` columns, along the tangents the caller
/// registers as [`ForwardProgram::inputs`] name.
///
/// Of a query, the program computes the query's output and, beside it, the
/// output's tangent.
///
/// Of a program of either kind, each step is rewritten as a query is, and
/// reads the rewritten steps before it, so every intermediate relation of the
/// program carries its tangent. Of a [`grad`](crate::grad) program, that is
/// forward over reverse: the tangent of each gradient along `v` is the
/// Hessian-vector product `H·v` ([`ForwardProgram::gradients`]), and the
/// value's tangent is the loss's directional derivative. There is no `hvp`:
/// `H·v` is `jvp(&grad(&plan, wrt)?, wrt)`, as in JAX it is
/// `jax.jvp(jax.grad(f), …)`, and one `grad` program serves every direction
/// (conjugate gradient runs it once per direction). The program's own input
/// tables (a [`vjp`](crate::vjp) program's cotangent, a `jvp` program's
/// tangent) are constant, and come first in [`ForwardProgram::inputs`].
pub fn jvp<D: Differentiable + ?Sized>(of: &D, wrt: &[ColumnRef]) -> Result<ForwardProgram> {
    jvp_with(of, wrt, &Options::new())
}

/// [`jvp`] with [`Options`]: a caller's `ddx-core` engine, for custom scalar
/// rules, or a fixed namespace. [`Options::restrict`] is refused: a jvp
/// computes every row's tangent.
pub fn jvp_with<D: Differentiable + ?Sized>(
    of: &D,
    wrt: &[ColumnRef],
    options: &Options,
) -> Result<ForwardProgram> {
    options.refuse_restrict("jvp")?;
    match of.subject() {
        Subject::Query(plan) => jvp_of_query(plan, wrt, options),
        Subject::Program(parts) => jvp_of_steps(&parts, wrt, options),
    }
}

/// [`jvp`] of a query.
fn jvp_of_query(plan: &Plan, wrt: &[ColumnRef], options: &Options) -> Result<ForwardProgram> {
    let namespace = options.namespace_or_new()?;
    let ddx = options.ddx_or_default();
    let mut sources = Sources::new(wrt, &namespace)?;
    let functions = Functions::from_plan(plan)?;
    let ext = Extensions::new(&functions);
    let mut dp = dual_plan(plan, &ddx, &functions, ext, &mut sources)?;
    sources.check_every_wrt_was_read()?;
    let name = format!("{namespace}jvp");
    let value = output_of(&name, Of::Output, &dp)?;
    if value.tangents.is_empty() {
        return Err(AdError::NotScalar(
            "no output column depends on a wrt column".into(),
        ));
    }
    let (inputs, checks) = sources.inputs(&mut dp.ext, Vec::new());
    Ok(ForwardProgram {
        inputs,
        checks,
        steps: vec![Step {
            name,
            plan: dp.plan,
        }],
        value,
        gradients: Vec::new(),
    })
}

/// [`jvp`] of a program's steps, of either kind.
fn jvp_of_steps(parts: &Parts, wrt: &[ColumnRef], options: &Options) -> Result<ForwardProgram> {
    let namespace = options.namespace_or_new()?;
    let ddx = options.ddx_or_default();
    let old = parts.namespace()?;
    let rename = |name: &str| -> Result<String> {
        name.strip_prefix(old)
            .map(|rest| format!("{namespace}{rest}"))
            .ok_or_else(|| {
                AdError::Internal(format!("step `{name}` is outside its program's namespace"))
            })
    };
    let mut sources = Sources::new(wrt, &namespace)?;
    // One set of declarations across the steps, so an anchor names one
    // function in all of them, as in every program, and a program built
    // from this one can read them together.
    let functions = Functions::union(parts.plans())?;
    let mut ext = Extensions::new(&functions);
    let mut steps = Vec::new();
    // Only the value's and the gradients' tangents are outputs, refused if
    // one is; an intermediate step's refused tangent matters only if a
    // later step reads it, which refuses then.
    let wanted: HashMap<&str, &Of> = std::iter::once(parts.value)
        .chain(parts.gradients)
        .map(|o| (o.step.as_str(), &o.of))
        .collect();
    let mut outputs: HashMap<String, OutputTable> = HashMap::new();
    for step in &parts.steps {
        let mut dp = dual_plan(&step.plan, &ddx, &functions, ext, &mut sources)?;
        let name = rename(&step.name)?;
        if let Some(&of) = wanted.get(step.name.as_str()) {
            outputs.insert(step.name.clone(), output_of(&name, of.clone(), &dp)?);
        }
        sources.steps.insert(
            step.name.clone(),
            (name.clone(), dp.columns.clone(), dp.tans.clone()),
        );
        steps.push(Step {
            name,
            plan: std::mem::take(&mut dp.plan),
        });
        ext = dp.ext;
    }
    sources.check_every_wrt_was_read()?;
    let mut take = |name: &str| {
        outputs
            .remove(name)
            .ok_or_else(|| AdError::Internal(format!("no step `{name}`")))
    };
    let value = take(&parts.value.step)?;
    let gradients = parts
        .gradients
        .iter()
        .map(|g| take(&g.step))
        .collect::<Result<Vec<_>>>()?;
    let (mut inputs, mut checks) = sources.inputs(&mut ext, Vec::new());
    parts.carry(&mut inputs, &mut checks);
    Ok(ForwardProgram {
        inputs,
        checks,
        steps,
        value,
        gradients,
    })
}

/// A plan's dual, as a plan.
struct DualPlan {
    plan: Plan,
    /// Its columns: the plan's own, then the tangents.
    columns: Vec<String>,
    /// For each of the plan's columns, its tangent: [`Tan::Col`] names one of
    /// `columns`.
    tans: Vec<Tan>,
    /// The declarations the plan uses.
    ext: Extensions,
}

/// `plan`'s dual as a plan: its columns, then the tangent of each that has
/// one, NULL where its value is NULL, named `__ddx_tangent_{c}`.
/// `functions` declares every function `plan` calls, and `ext` starts from
/// them.
fn dual_plan(
    plan: &Plan,
    ddx: &ddx_core::Ddx,
    functions: &Functions,
    ext: Extensions,
    sources: &mut Sources,
) -> Result<DualPlan> {
    let (root, names) = root_of(plan)?;
    let root = inline_references(root, &plan.relations)?;
    let mut source = |n: &[String], s: Option<&NamedStruct>| sources.source(n, s);
    let mut dz = Dualizer::with_extensions(ddx, functions, ext, &mut source);
    let d = dz.dual(&root)?;
    let big = d.rel_width();
    let mut exprs = Vec::new();
    let mut columns = names.clone();
    let mut tans = Vec::with_capacity(d.width);
    for (c, t) in d.tans.iter().enumerate() {
        tans.push(match t {
            Tan::Col(j) => {
                exprs.push(dz.masked(field(c), field(*j)));
                columns.push(tangent_column(c, &columns));
                Tan::Col(columns.len() - 1)
            }
            other => other.clone(),
        });
    }
    let emit: Vec<usize> = (0..d.width).chain(big..big + exprs.len()).collect();
    let rel = project_emit(d.rel, exprs, Some(emit));
    let mut plan = plan_of(rel, columns.clone(), &dz.ext);
    prune_plan(&mut plan);
    Ok(DualPlan {
        plan,
        columns,
        tans,
        ext: dz.ext,
    })
}

/// A dual plan materialized as `step`, as an output: refused if a column's
/// tangent was refused.
fn output_of(step: &str, of: Of, dp: &DualPlan) -> Result<OutputTable> {
    let mut tangents = Vec::new();
    for (c, t) in dp.tans.iter().enumerate() {
        match t {
            Tan::Col(j) => tangents.push(Tangent {
                column: dp.columns[c].clone(),
                tangent: dp.columns[*j].clone(),
            }),
            Tan::Refused(why) => return Err(why.clone()),
            Tan::Zero => {}
        }
    }
    Ok(OutputTable {
        step: step.to_string(),
        of,
        columns: dp.columns.clone(),
        tangents,
    })
}

/// Where each read of a jvp program gets its tangents: the `wrt` tables,
/// registered on first read, and the steps already rewritten.
struct Sources<'a> {
    wrt: &'a [ColumnRef],
    namespace: &'a str,
    tables: Vec<Table>,
    seen: BTreeSet<String>,
    /// Each rewritten step, by its name in the program: its dual's name,
    /// columns and tangents.
    steps: HashMap<String, (String, Vec<String>, Vec<Tan>)>,
}

impl<'a> Sources<'a> {
    fn new(wrt: &'a [ColumnRef], namespace: &'a str) -> Result<Self> {
        if wrt.is_empty() {
            return Err(AdError::UnknownWrt(
                "no wrt columns were given; name at least one table column".into(),
            ));
        }
        Ok(Sources {
            wrt,
            namespace,
            tables: Vec::new(),
            seen: BTreeSet::new(),
            steps: HashMap::new(),
        })
    }

    fn source(&mut self, names: &[String], schema: Option<&NamedStruct>) -> Result<Option<Source>> {
        if let [name] = names {
            if let Some((dual, columns, tans)) = self.steps.get(name) {
                return Ok(Some(Source::Step {
                    dual: dual.clone(),
                    columns: columns.clone(),
                    tans: tans.clone(),
                }));
            }
        }
        self.seen.insert(names.join("."));
        if !self.wrt.iter().any(|w| table_matches(&w.table, names)) {
            return Ok(None);
        }
        let schema = schema.filter(|s| s.r#struct.is_some()).ok_or_else(|| {
            AdError::InvalidPlan(format!("the read of `{}` has no schema", names.join(".")))
        })?;
        let i = match self.tables.iter().position(|t| t.names == names) {
            Some(i) if self.tables[i].schema != *schema => {
                return Err(AdError::InvalidPlan(format!(
                    "table `{}` is read with two different schemas",
                    names.join(".")
                )))
            }
            Some(i) => i,
            None => {
                self.tables.push(wrt_table(names, schema, self.wrt)?);
                self.tables.len() - 1
            }
        };
        Ok(Some(Source::Table {
            table: self.tables[i].clone(),
            tangent: tangent_name(self.namespace, i, names),
        }))
    }

    fn check_every_wrt_was_read(&self) -> Result<()> {
        check_every_wrt_was_read(self.wrt, &self.seen, &self.tables)
    }

    /// The tangent tables to register, and the checks to run first:
    /// `checks`, then each `wrt` table's dims check and each tangent
    /// table's, once each.
    fn inputs(
        &self,
        ext: &mut Extensions,
        mut checks: Vec<Check>,
    ) -> (Vec<InputTable>, Vec<Check>) {
        let tangent_tables: Vec<InputTable> = self
            .tables
            .iter()
            .enumerate()
            .map(|(i, t)| InputTable {
                name: tangent_name(self.namespace, i, &t.names),
                of: Of::Table(t.names.clone()),
                columns: t
                    .dims
                    .iter()
                    .chain(&t.values)
                    .map(|&c| t.columns()[c].clone())
                    .collect(),
                keys: t.dims.len(),
            })
            .collect();
        for t in &self.tables {
            let check = dims_check(ext, t);
            if !checks.iter().any(|c| c.message == check.message) {
                checks.push(check);
            }
        }
        for (t, tt) in self.tables.iter().zip(&tangent_tables) {
            checks.push(tangent_check(ext, t, tt));
        }
        (tangent_tables, checks)
    }
}

/// The check that a tangent table has one row per dim tuple: a repeated one
/// would join twice and double its rows.
fn tangent_check(ext: &mut Extensions, table: &Table, tt: &InputTable) -> Check {
    let k = table.dims.len();
    let keys: Vec<usize> = (0..k).collect();
    let repeated = repeated_keys(ext, read_step(&tt.name, tt.columns.clone()), &keys);
    Check {
        table: vec![tt.name.clone()],
        keys: tt.columns[..k].to_vec(),
        plan: plan_of(repeated, tt.columns[..k].to_vec(), ext),
        message: format!(
            "the tangent table `{}` has rows that share their dims ({}); it needs one row per \
             row of `{}`",
            tt.name,
            tt.columns[..k].join(", "),
            table.names.join(".")
        ),
    }
}
