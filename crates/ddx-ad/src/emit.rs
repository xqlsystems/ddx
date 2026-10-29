// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Writing the plans of a backward program.
//!
//! Every step ddx emits is an ordinary Substrait plan, with nothing in it only
//! ddx understands, built from seven relations: read, project, filter, join,
//! aggregate, union and fetch. The builders here produce them in the shapes
//! DataFusion's own producer does, since that is the consumer they are tested
//! against.
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
use substrait::proto::fetch_rel::CountMode;
use substrait::proto::join_rel::JoinType;
use substrait::proto::plan_rel::RelType as PlanRelType;
use substrait::proto::read_rel::{NamedTable, ReadType};
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::{Emit, EmitKind};
use substrait::proto::set_rel::SetOp;
use substrait::proto::{
    r#type, AggregateFunction, AggregateRel, Expression, FetchRel, FilterRel, FunctionArgument,
    JoinRel, NamedStruct, Plan, PlanRel, ProjectRel, ReadRel, Rel, RelCommon, RelRoot, SetRel,
};

use crate::error::{AdError, Result};
use crate::expr::lit_bool;
use crate::functions::Extensions;
use crate::optics::{AsJoin, AsRead, Compose, Deep, Fold, PlanRels, RelKind, Setter};

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

/// Does `emit` pick each of `width` columns once, in order, so it changes
/// nothing?
pub(crate) fn is_identity(emit: &[usize], width: usize) -> bool {
    emit.len() == width && emit.iter().enumerate().all(|(i, &c)| i == c)
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

/// None of `input`'s rows: its columns, typed as the engine types them,
/// with nothing in them. Joined to with a FULL JOIN, it gives another
/// relation's rows NULL in those columns.
pub fn no_rows(input: Rel) -> Rel {
    Rel {
        rel_type: Some(RelType::Fetch(Box::new(FetchRel {
            input: Some(Box::new(input)),
            #[allow(deprecated)]
            count_mode: Some(CountMode::Count(0)),
            ..Default::default()
        }))),
    }
}

/// A plan whose root is `root`, with output columns named `names`.
pub fn plan(mut root: Rel, names: Vec<String>, ext: &Extensions) -> Plan {
    condition_joins(&mut root);
    Plan {
        version: Some(substrait::version::version_with_producer("ddx")),
        extension_urns: ext.urns(),
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

/// `root` with each join but an inner one that has no condition given the
/// condition `true`, which means the same. DataFusion plans an `EXISTS` or
/// `IN` whose subquery reads nothing of the outer row as such a semi or
/// anti join, and its Substrait consumer refuses one with no condition
/// (ddx issue #171).
fn condition_joins(root: &mut Rel) {
    Deep.then(RelKind).then(AsJoin).modify(root, |j| {
        let typed = !matches!(j.r#type(), JoinType::Inner | JoinType::Unspecified);
        if typed && j.expression.is_none() {
            j.expression = Some(Box::new(lit_bool(true)));
        }
    })
}

/// Every read in a plan, its shared relations' and its subqueries'
/// included.
fn reads() -> impl Fold<Plan, ReadRel> + Setter<Plan, ReadRel> {
    PlanRels.then(Deep).then(RelKind).then(AsRead)
}

/// Each read in `plan` that names its columns without their types (an
/// earlier step's table), with the columns it names.
pub(crate) fn for_each_unbound_read(plan: &Plan, f: &mut dyn FnMut(&str, &[String])) {
    reads().each(plan, |read| {
        if let (Some(ReadType::NamedTable(t)), Some(s)) = (&read.read_type, &read.base_schema) {
            if s.r#struct.is_none() {
                f(&t.names.join("."), &s.names);
            }
        }
    });
}

/// The names of the tables `plan` reads without stating their types, in the
/// order they appear. These are the steps it depends on.
pub fn unbound_reads(plan: &Plan) -> Vec<String> {
    let mut out = Vec::new();
    reads().each(plan, |read| {
        if let (Some(ReadType::NamedTable(t)), Some(s)) = (&read.read_type, &read.base_schema) {
            if s.r#struct.is_none() {
                out.push(t.names.join("."));
            }
        }
    });
    out
}

/// The name of every table `plan` reads, each its parts joined by `.`, in
/// the order they appear, subqueries' included.
pub fn tables_read(plan: &Plan) -> Vec<String> {
    let mut out = Vec::new();
    reads().each(plan, |read| {
        if let Some(ReadType::NamedTable(t)) = &read.read_type {
            out.push(t.names.join("."));
        }
    });
    out
}

/// `plan` with each table it reads named as `rename` says, or as it was
/// where `rename` returns `None`: for a producer that writes a table by a
/// name the engine would not find it by (DuckDB's drops its schema).
pub fn rename_tables(plan: &mut Plan, rename: &mut dyn FnMut(&[String]) -> Option<Vec<String>>) {
    reads().modify(plan, |read| {
        if let Some(ReadType::NamedTable(t)) = &mut read.read_type {
            if let Some(names) = rename(&t.names) {
                t.names = names;
            }
        }
    });
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
    reads().try_modify(plan, &mut |read| {
        let (Some(ReadType::NamedTable(t)), Some(base)) = (&read.read_type, &mut read.base_schema)
        else {
            return Ok(());
        };
        if base.r#struct.is_some() {
            return Ok(());
        }
        let name = t.names.join(".");
        let Some(actual) = schema(&name) else {
            return Err(AdError::InvalidPlan(format!(
                "step `{name}` is read before it exists"
            )));
        };
        let types = actual.r#struct.map(|s| s.types).unwrap_or_default();
        let mut picked = Vec::with_capacity(base.names.len());
        for col in &base.names {
            match actual.names.iter().position(|n| n == col) {
                Some(i) if i < types.len() => picked.push(types[i].clone()),
                // The schema names the column but types fewer columns than
                // it names: an engine's schema with no struct, say.
                Some(_) => {
                    return Err(AdError::InvalidPlan(format!(
                        "the schema given for step `{name}` names {} columns but types {}, \
                         so `{col}` has no type",
                        actual.names.len(),
                        types.len()
                    )));
                }
                None => {
                    return Err(AdError::InvalidPlan(format!(
                        "step `{name}` has no column `{col}` (it has {:?})",
                        actual.names
                    )));
                }
            }
        }
        base.r#struct = Some(r#type::Struct {
            types: picked,
            type_variation_reference: 0,
            nullability: r#type::Nullability::Required as i32,
        });
        Ok(())
    })
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

    /// A join of `kind` with no condition.
    fn bare_join(kind: JoinType) -> Rel {
        let side = || Box::new(read_step("t", vec!["c".into()]));
        Rel {
            rel_type: Some(RelType::Join(Box::new(JoinRel {
                left: Some(side()),
                right: Some(side()),
                r#type: kind as i32,
                ..Default::default()
            }))),
        }
    }

    /// The conditions of the joins in `rel`, subqueries included, in the
    /// order [`Deep`] meets them.
    fn join_conditions(rel: &Rel) -> Vec<Option<Expression>> {
        Deep.then(RelKind)
            .then(AsJoin)
            .to_vec(rel)
            .into_iter()
            .map(|j| j.expression.as_deref().cloned())
            .collect()
    }

    #[test]
    fn a_join_but_an_inner_one_with_no_condition_is_given_true_subqueries_included() {
        use substrait::proto::expression::nested::{self, NestedType};
        use substrait::proto::expression::subquery::{InPredicate, Scalar, SubqueryType};
        use substrait::proto::expression::{Nested, RexType, Subquery};
        use substrait::proto::function_argument::ArgType;
        let subquery = |t: SubqueryType| Expression {
            rex_type: Some(RexType::Subquery(Box::new(Subquery {
                subquery_type: Some(t),
            }))),
        };
        // An `IN` whose needle is a struct literal holding a subquery over a
        // semi join.
        let needle = Expression {
            rex_type: Some(RexType::Nested(Nested {
                nested_type: Some(NestedType::Struct(nested::Struct {
                    fields: vec![subquery(SubqueryType::Scalar(Box::new(Scalar {
                        input: Some(Box::new(bare_join(JoinType::LeftSemi))),
                    })))],
                })),
                ..Default::default()
            })),
        };
        let in_anti = subquery(SubqueryType::InPredicate(Box::new(InPredicate {
            needles: vec![needle],
            haystack: Some(Box::new(bare_join(JoinType::LeftAnti))),
        })));
        // ... as the argument of an aggregate over an inner join with none.
        let aggregate = Rel {
            rel_type: Some(RelType::Aggregate(Box::new(AggregateRel {
                input: Some(Box::new(bare_join(JoinType::Inner))),
                measures: vec![Measure {
                    measure: Some(AggregateFunction {
                        arguments: vec![FunctionArgument {
                            arg_type: Some(ArgType::Value(in_anti)),
                        }],
                        ..Default::default()
                    }),
                    filter: None,
                }],
                ..Default::default()
            }))),
        };
        let ext = Extensions::new(&Functions::default());
        let p = plan(aggregate, vec!["s".into()], &ext);
        let (root, _) = crate::forward::root_of(&p).unwrap();
        // The anti and semi joins in the subqueries get `true` (a
        // relation's subqueries come before its inputs); the inner join
        // keeps no condition.
        assert_eq!(
            join_conditions(root),
            vec![Some(lit_bool(true)), Some(lit_bool(true)), None]
        );
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
        reads().each(&p, |r| {
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
