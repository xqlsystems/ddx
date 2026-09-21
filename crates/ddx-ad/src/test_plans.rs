// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Small builders for Substrait plans written by hand, for use in unit tests.
//! The tests in `ddx-datafusion` cover the output of a real producer, because
//! that crate can plan SQL.

use substrait::proto::aggregate_rel::{Grouping, Measure};
use substrait::proto::expression::field_reference::{ReferenceType, RootReference, RootType};
use substrait::proto::expression::literal::LiteralType;
use substrait::proto::expression::mask_expression::{StructItem, StructSelect};
use substrait::proto::expression::reference_segment::{self, StructField};
use substrait::proto::expression::{
    if_then, FieldReference, IfThen, Literal, MaskExpression, ReferenceSegment, RexType,
    ScalarFunction,
};
use substrait::proto::extensions::simple_extension_declaration::{ExtensionFunction, MappingType};
use substrait::proto::extensions::SimpleExtensionDeclaration;
use substrait::proto::function_argument::ArgType;
use substrait::proto::join_rel::JoinType;
use substrait::proto::read_rel::{NamedTable, ReadType};
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::{Emit, EmitKind};
use substrait::proto::{
    plan_rel, AggregateFunction, AggregateRel, Expression, FilterRel, FunctionArgument, JoinRel,
    NamedStruct, Plan, PlanRel, ProjectRel, ReadRel, Rel, RelCommon, RelRoot,
};

use crate::markers::Functions;

fn rel(t: RelType) -> Rel {
    Rel { rel_type: Some(t) }
}

/// A named table with these columns.
pub fn read(name: &str, cols: &[&str]) -> Rel {
    rel(RelType::Read(Box::new(ReadRel {
        base_schema: Some(NamedStruct {
            names: cols.iter().map(|c| c.to_string()).collect(),
            r#struct: None,
        }),
        read_type: Some(ReadType::NamedTable(NamedTable {
            names: vec![name.into()],
            advanced_extension: None,
        })),
        ..Default::default()
    })))
}

/// A read that projects the base schema down to the fields in `select`.
pub fn masked_read(name: &str, cols: &[&str], select: &[i32]) -> Rel {
    let mut r = read(name, cols);
    if let Some(RelType::Read(r)) = r.rel_type.as_mut() {
        r.projection = Some(MaskExpression {
            select: Some(StructSelect {
                struct_items: select
                    .iter()
                    .map(|&field| StructItem { field, child: None })
                    .collect(),
            }),
            maintain_singular_struct: false,
        });
    }
    r
}

pub fn join_typed(l: Rel, r: Rel, t: JoinType) -> Rel {
    rel(RelType::Join(Box::new(JoinRel {
        left: Some(Box::new(l)),
        right: Some(Box::new(r)),
        expression: Some(Box::new(lit(true))),
        r#type: t.into(),
        ..Default::default()
    })))
}

pub fn join(l: Rel, r: Rel) -> Rel {
    join_typed(l, r, JoinType::Inner)
}

pub fn filter(input: Rel, condition: Expression) -> Rel {
    rel(RelType::Filter(Box::new(FilterRel {
        input: Some(Box::new(input)),
        condition: Some(Box::new(condition)),
        ..Default::default()
    })))
}

pub fn project(input: Rel, expressions: Vec<Expression>) -> Rel {
    rel(RelType::Project(Box::new(ProjectRel {
        input: Some(Box::new(input)),
        expressions,
        ..Default::default()
    })))
}

/// `GROUP BY keys`, with each measure given as a function anchor and its
/// arguments.
pub fn aggregate(input: Rel, keys: &[Expression], measures: &[(u32, Vec<Expression>)]) -> Rel {
    rel(RelType::Aggregate(Box::new(AggregateRel {
        input: Some(Box::new(input)),
        groupings: vec![Grouping {
            expression_references: (0..keys.len() as u32).collect(),
            ..Default::default()
        }],
        grouping_expressions: keys.to_vec(),
        measures: measures
            .iter()
            .map(|(f, args)| Measure {
                measure: Some(AggregateFunction {
                    function_reference: *f,
                    arguments: args.iter().cloned().map(value_arg).collect(),
                    ..Default::default()
                }),
                filter: None,
            })
            .collect(),
        ..Default::default()
    })))
}

/// Set the `emit` output mapping of a relation.
pub fn emit(mut r: Rel, mapping: &[i32]) -> Rel {
    let common = Some(RelCommon {
        emit_kind: Some(EmitKind::Emit(Emit {
            output_mapping: mapping.to_vec(),
        })),
        ..Default::default()
    });
    match r.rel_type.as_mut() {
        Some(RelType::Read(x)) => x.common = common,
        Some(RelType::Project(x)) => x.common = common,
        Some(RelType::Join(x)) => x.common = common,
        Some(RelType::Aggregate(x)) => x.common = common,
        Some(RelType::Filter(x)) => x.common = common,
        _ => panic!("emit() not wired up for this relation"),
    }
    r
}

pub fn plan(root: Rel) -> Plan {
    Plan {
        relations: vec![PlanRel {
            rel_type: Some(plan_rel::RelType::Root(RelRoot {
                input: Some(root),
                names: vec![],
            })),
        }],
        ..Default::default()
    }
}

fn declarations(fns: &[(u32, &str)]) -> Vec<SimpleExtensionDeclaration> {
    fns.iter()
        .map(|(anchor, name)| SimpleExtensionDeclaration {
            mapping_type: Some(MappingType::ExtensionFunction(ExtensionFunction {
                function_anchor: *anchor,
                name: name.to_string(),
                extension_urn_reference: u32::MAX,
            })),
        })
        .collect()
}

pub fn plan_with_functions(root: Rel, fns: &[(u32, &str)]) -> Plan {
    Plan {
        extensions: declarations(fns),
        ..plan(root)
    }
}

pub fn functions(fns: &[(u32, &str)]) -> Functions {
    Functions::from_plan(&Plan {
        extensions: declarations(fns),
        ..Default::default()
    })
    .unwrap()
}

/// Input-row field `i`.
pub fn field(i: i32) -> Expression {
    Expression {
        rex_type: Some(RexType::Selection(Box::new(FieldReference {
            reference_type: Some(ReferenceType::DirectReference(ReferenceSegment {
                reference_type: Some(reference_segment::ReferenceType::StructField(Box::new(
                    StructField {
                        field: i,
                        child: None,
                    },
                ))),
            })),
            root_type: Some(RootType::RootReference(RootReference {})),
        }))),
    }
}

fn value_arg(e: Expression) -> FunctionArgument {
    FunctionArgument {
        arg_type: Some(ArgType::Value(e)),
    }
}

/// A call to the function declared at `anchor`.
pub fn call(anchor: u32, args: Vec<Expression>) -> Expression {
    Expression {
        rex_type: Some(RexType::ScalarFunction(ScalarFunction {
            function_reference: anchor,
            arguments: args.into_iter().map(value_arg).collect(),
            ..Default::default()
        })),
    }
}

/// `CASE WHEN cond THEN then ELSE els END`.
pub fn if_then(cond: Expression, then: Expression, els: Expression) -> Expression {
    Expression {
        rex_type: Some(RexType::IfThen(Box::new(IfThen {
            ifs: vec![if_then::IfClause {
                r#if: Some(cond),
                then: Some(then),
            }],
            r#else: Some(Box::new(els)),
        }))),
    }
}

fn literal(t: LiteralType) -> Expression {
    Expression {
        rex_type: Some(RexType::Literal(Literal {
            literal_type: Some(t),
            ..Default::default()
        })),
    }
}

pub fn lit(b: bool) -> Expression {
    literal(LiteralType::Boolean(b))
}

pub fn lit_f64(x: f64) -> Expression {
    literal(LiteralType::Fp64(x))
}
