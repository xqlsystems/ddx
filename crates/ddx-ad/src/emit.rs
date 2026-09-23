// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Writing the plans of a backward program.
//!
//! Every step ddx emits is an ordinary, unmarked Substrait plan (design.md
//! §4.3), built from six relations: read, project, filter, join, aggregate and
//! union. The builders here produce them in the shapes DataFusion's own
//! producer does, since that is the consumer they are tested against.
//!
//! # Reading an earlier step
//!
//! A step often reads a table an earlier step materialized: the tape of the
//! forward pass, or a cotangent. A Substrait read must state the types of the
//! columns it reads, and ddx does not know them. It could derive them, but only
//! by re-implementing each engine's typing rules (what `SUM` of a `REAL` is, what
//! `ROW_NUMBER()` returns). Instead a read of a step is emitted **unbound**: it
//! names its columns and leaves their types out. The engine materializes the
//! steps in order, so by the time it consumes a plan, every step it reads
//! exists and has a schema, and [`bind_reads`] fills the types in from it. The
//! types are then the engine's own by construction.

use substrait::proto::aggregate_function::AggregationInvocation;
use substrait::proto::aggregate_rel::{Grouping, Measure};
use substrait::proto::expression::MaskExpression;
use substrait::proto::join_rel::JoinType;
use substrait::proto::plan_rel::RelType as PlanRelType;
use substrait::proto::read_rel::{NamedTable, ReadType};
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::{Emit, EmitKind};
use substrait::proto::set_rel::SetOp;
use substrait::proto::{
    r#type, AggregateFunction, AggregateRel, Expression, FilterRel, FunctionArgument, JoinRel,
    NamedStruct, Plan, PlanRel, ProjectRel, ReadRel, Rel, RelCommon, RelRoot, SetRel,
};

use crate::error::{AdError, Result};
use crate::functions::Extensions;

/// A read of every column of a table whose schema is known.
pub fn read_table(names: Vec<String>, schema: NamedStruct) -> Rel {
    read(names, schema, None, None)
}

/// A read of a table with a schema, a pushed-down filter and a projection
/// mask, as a producer wrote it.
pub fn read(
    names: Vec<String>,
    schema: NamedStruct,
    filter: Option<Expression>,
    projection: Option<MaskExpression>,
) -> Rel {
    Rel {
        rel_type: Some(RelType::Read(Box::new(ReadRel {
            base_schema: Some(schema),
            filter: filter.map(Box::new),
            projection,
            read_type: Some(ReadType::NamedTable(NamedTable {
                names,
                advanced_extension: None,
            })),
            ..Default::default()
        }))),
    }
}

/// An unbound read of the step `name`, whose columns are `columns`. See the
/// module docs: its types are filled in by [`bind_reads`] once the step exists.
pub fn read_step(name: &str, columns: Vec<String>) -> Rel {
    read_table(
        vec![name.to_string()],
        NamedStruct {
            names: columns,
            r#struct: None,
        },
    )
}

/// `input`'s columns followed by `exprs`.
pub fn project(input: Rel, exprs: Vec<Expression>) -> Rel {
    project_emit(input, exprs, None)
}

/// A projection whose output is the columns `emit` picks from `input`'s
/// columns followed by `exprs`.
pub fn project_emit(input: Rel, exprs: Vec<Expression>, emit: Option<Vec<usize>>) -> Rel {
    Rel {
        rel_type: Some(RelType::Project(Box::new(ProjectRel {
            common: emit.map(common_emit),
            input: Some(Box::new(input)),
            expressions: exprs,
            advanced_extension: None,
        }))),
    }
}

/// Only the columns `emit` picks from `input`.
pub fn select(input: Rel, emit: Vec<usize>) -> Rel {
    project_emit(input, vec![], Some(emit))
}

fn common_emit(emit: Vec<usize>) -> RelCommon {
    RelCommon {
        emit_kind: Some(EmitKind::Emit(Emit {
            output_mapping: emit.into_iter().map(|i| i as i32).collect(),
        })),
        ..Default::default()
    }
}

/// The rows of `input` where `condition` holds.
pub fn filter(input: Rel, condition: Expression) -> Rel {
    Rel {
        rel_type: Some(RelType::Filter(Box::new(FilterRel {
            common: None,
            input: Some(Box::new(input)),
            condition: Some(Box::new(condition)),
            advanced_extension: None,
        }))),
    }
}

/// A join. For an inner join the output is `left`'s columns then `right`'s.
pub fn join(left: Rel, right: Rel, condition: Expression, kind: JoinType) -> Rel {
    Rel {
        rel_type: Some(RelType::Join(Box::new(JoinRel {
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            expression: Some(Box::new(condition)),
            r#type: kind as i32,
            ..Default::default()
        }))),
    }
}

/// A grouped aggregate: the output is one column per grouping expression, then
/// one per measure. With no groupings it is one row.
///
/// Each measure is `(function anchor, arguments)`.
pub fn aggregate(
    input: Rel,
    groupings: Vec<Expression>,
    measures: Vec<(u32, Vec<Expression>)>,
) -> Rel {
    let references = (0..groupings.len() as u32).collect();
    Rel {
        rel_type: Some(RelType::Aggregate(Box::new(AggregateRel {
            input: Some(Box::new(input)),
            // Both spellings of the groupings, as DataFusion writes them: the
            // top-level list the current spec uses, and the deprecated
            // per-grouping copy older consumers read.
            #[allow(deprecated)]
            groupings: vec![Grouping {
                grouping_expressions: groupings.clone(),
                expression_references: references,
            }],
            grouping_expressions: groupings,
            measures: measures
                .into_iter()
                .map(|(anchor, args)| Measure {
                    measure: Some(AggregateFunction {
                        function_reference: anchor,
                        arguments: args
                            .into_iter()
                            .map(|e| FunctionArgument {
                                arg_type: Some(
                                    substrait::proto::function_argument::ArgType::Value(e),
                                ),
                            })
                            .collect(),
                        invocation: AggregationInvocation::All as i32,
                        ..Default::default()
                    }),
                    filter: None,
                })
                .collect(),
            ..Default::default()
        }))),
    }
}

/// `UNION ALL` of `inputs`, which must have the same columns.
pub fn union_all(inputs: Vec<Rel>) -> Rel {
    Rel {
        rel_type: Some(RelType::Set(SetRel {
            common: None,
            inputs,
            op: SetOp::UnionAll as i32,
            advanced_extension: None,
        })),
    }
}

/// A plan whose root is `root`, with output columns named `names`.
pub fn plan(root: Rel, names: Vec<String>, ext: &Extensions) -> Plan {
    Plan {
        version: Some(substrait::version::version_with_producer("ddx")),
        extensions: ext.declarations(),
        relations: vec![PlanRel {
            rel_type: Some(PlanRelType::Root(RelRoot {
                input: Some(root),
                names,
            })),
        }],
        ..Default::default()
    }
}

/// The names of the tables `plan` reads without stating their types, in the
/// order they appear. These are the steps it depends on.
pub fn unbound_reads(plan: &Plan) -> Vec<String> {
    let mut out = Vec::new();
    for_each_read(plan, &mut |read| {
        if let (Some(ReadType::NamedTable(t)), Some(s)) = (&read.read_type, &read.base_schema) {
            if s.r#struct.is_none() {
                out.push(t.names.join("."));
            }
        }
    });
    out
}

/// Fill in the types of every unbound read in `plan` (see the module docs).
///
/// `schema` is asked for each unbound table by name and returns its column
/// types, in any order; each column the read names is looked up in it by name.
/// A missing table or column is an error, since the plan cannot run without it.
pub fn bind_reads(
    plan: &mut Plan,
    schema: &mut dyn FnMut(&str) -> Option<NamedStruct>,
) -> Result<()> {
    let mut failure = None;
    for_each_read_mut(plan, &mut |read| {
        if failure.is_some() {
            return;
        }
        let (Some(ReadType::NamedTable(t)), Some(base)) = (&read.read_type, &mut read.base_schema)
        else {
            return;
        };
        if base.r#struct.is_some() {
            return;
        }
        let name = t.names.join(".");
        let Some(actual) = schema(&name) else {
            failure = Some(AdError::InvalidPlan(format!(
                "step `{name}` is read before it exists"
            )));
            return;
        };
        let types = actual.r#struct.map(|s| s.types).unwrap_or_default();
        let mut picked = Vec::with_capacity(base.names.len());
        for col in &base.names {
            match actual.names.iter().position(|n| n == col) {
                Some(i) if i < types.len() => picked.push(types[i].clone()),
                _ => {
                    failure = Some(AdError::InvalidPlan(format!(
                        "step `{name}` has no column `{col}` (it has {:?})",
                        actual.names
                    )));
                    return;
                }
            }
        }
        base.r#struct = Some(r#type::Struct {
            types: picked,
            type_variation_reference: 0,
            nullability: r#type::Nullability::Required as i32,
        });
    });
    match failure {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn for_each_read(plan: &Plan, f: &mut dyn FnMut(&ReadRel)) {
    let mut plan = plan.clone();
    for_each_read_mut(&mut plan, &mut |r| f(r));
}

fn for_each_read_mut(plan: &mut Plan, f: &mut dyn FnMut(&mut ReadRel)) {
    for rel in plan.relations.iter_mut() {
        match rel.rel_type.as_mut() {
            Some(PlanRelType::Root(root)) => {
                if let Some(r) = root.input.as_mut() {
                    walk_rel(r, f);
                }
            }
            Some(PlanRelType::Rel(r)) => walk_rel(r, f),
            None => {}
        }
    }
}

/// Visit every read in a relation tree. Covers the relations ddx emits and
/// the ones it copies from a forward plan.
fn walk_rel(rel: &mut Rel, f: &mut dyn FnMut(&mut ReadRel)) {
    let Some(kind) = rel.rel_type.as_mut() else {
        return;
    };
    let mut visit = |r: &mut Option<Box<Rel>>| {
        if let Some(r) = r.as_mut() {
            walk_rel(r, f);
        }
    };
    match kind {
        RelType::Read(r) => f(r),
        RelType::Filter(r) => visit(&mut r.input),
        RelType::Fetch(r) => visit(&mut r.input),
        RelType::Aggregate(r) => visit(&mut r.input),
        RelType::Sort(r) => visit(&mut r.input),
        RelType::Project(r) => visit(&mut r.input),
        RelType::Join(r) => {
            visit(&mut r.left);
            visit(&mut r.right);
        }
        RelType::Cross(r) => {
            visit(&mut r.left);
            visit(&mut r.right);
        }
        RelType::Set(r) => {
            for i in r.inputs.iter_mut() {
                walk_rel(i, f);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{call, field, fp64};
    use crate::functions::Functions;
    use substrait::proto::r#type::{Kind, I64};

    fn i64_type() -> substrait::proto::Type {
        substrait::proto::Type {
            kind: Some(Kind::I64(I64 {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Nullable as i32,
            })),
        }
    }

    #[test]
    fn an_unbound_read_is_listed_and_then_bound_by_column_name() {
        let ext = Extensions::new(&Functions::default());
        let mut p = plan(
            join(
                read_step("__ddx_fwd_0", vec!["c0".into(), "c1".into()]),
                read_step("__ddx_bwd_0", vec!["c1".into()]),
                call(0, vec![field(0), field(2)]),
                JoinType::Inner,
            ),
            vec!["a".into(), "b".into(), "c".into()],
            &ext,
        );
        assert_eq!(unbound_reads(&p), vec!["__ddx_fwd_0", "__ddx_bwd_0"]);

        let mut asked = Vec::new();
        bind_reads(&mut p, &mut |name| {
            asked.push(name.to_string());
            Some(NamedStruct {
                // Deliberately in a different order from the read's columns.
                names: vec!["c1".into(), "c0".into()],
                r#struct: Some(r#type::Struct {
                    types: vec![fp64(), i64_type()],
                    ..Default::default()
                }),
            })
        })
        .unwrap();
        assert_eq!(asked, vec!["__ddx_fwd_0", "__ddx_bwd_0"]);
        assert!(unbound_reads(&p).is_empty());

        let mut types = Vec::new();
        for_each_read(&p, &mut |r| {
            types.push(
                r.base_schema
                    .as_ref()
                    .unwrap()
                    .r#struct
                    .clone()
                    .unwrap()
                    .types,
            )
        });
        assert_eq!(types, vec![vec![i64_type(), fp64()], vec![fp64()]]);
    }

    #[test]
    fn binding_a_missing_step_or_column_is_an_error() {
        let ext = Extensions::new(&Functions::default());
        let mut p = plan(read_step("s", vec!["x".into()]), vec!["x".into()], &ext);
        let err = bind_reads(&mut p.clone(), &mut |_| None).unwrap_err();
        assert!(err.to_string().contains("read before it exists"), "{err}");
        let err = bind_reads(&mut p, &mut |_| {
            Some(NamedStruct {
                names: vec!["y".into()],
                r#struct: Some(r#type::Struct {
                    types: vec![fp64()],
                    ..Default::default()
                }),
            })
        })
        .unwrap_err();
        assert!(err.to_string().contains("no column `x`"), "{err}");
    }
}
