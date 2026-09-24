// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Relations as ddx sees them: dims and values.
//!
//! JAX differentiates functions of arrays, and a gradient has the shape of
//! what it is taken with respect to. ddx differentiates queries over
//! relations, and plays the same game with the XQL data model (design.md §1): a
//! relation is an N-dimensional array stored long, one row per coordinate
//! tuple. Its columns are of two kinds.
//!
//! - **Dims** are the coordinates. Together they identify a row, and they are
//!   never differentiated: a gradient does not move a coordinate.
//! - **Values** are the numbers at a coordinate. They are what a gradient is
//!   taken with respect to, and what a tangent or cotangent is a number for.
//!
//! A tangent or cotangent of a relation is a relation with the same dims and
//! the same value columns, so a gradient comes back shaped like the table it is
//! the gradient of, as `jax.grad` returns a pytree shaped like its argument.
//!
//! For a table the query reads, the values are the columns named in `wrt`, and
//! the dims are all the others. A table whose every column is named in `wrt`
//! has no dims and is refused. That the dims identify the rows (no two rows
//! share a dim tuple) is the XQL model's promise; ddx cannot see it in a plan,
//! and a table that breaks it gets each shared tuple's rows' gradients summed. For a relation the query computes, the plan
//! says which is which: a `GROUP BY` key is a dim, an aggregate is a value, and
//! a join's dims are both sides' dims.

use substrait::proto::NamedStruct;

/// A column of a table the query reads: what a gradient is taken with respect
/// to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ColumnRef {
    /// The table, as the plan names it: `weights`, or `schema.weights`. A bare
    /// name also matches a qualified one with that last part.
    pub table: String,
    /// The column.
    pub column: String,
}

impl ColumnRef {
    /// `table.column`.
    pub fn new(table: impl Into<String>, column: impl Into<String>) -> Self {
        ColumnRef {
            table: table.into(),
            column: column.into(),
        }
    }
}

/// A table the query reads that a gradient is taken with respect to.
#[derive(Debug, Clone)]
pub struct Table {
    /// Its name parts, as the plan's reads give them.
    pub names: Vec<String>,
    /// Its full schema.
    pub schema: NamedStruct,
    /// The positions of its dims: every column not named in `wrt`.
    pub dims: Vec<usize>,
    /// The positions of its values: the columns named in `wrt`.
    pub values: Vec<usize>,
}

impl Table {
    /// The column names.
    pub fn columns(&self) -> &[String] {
        &self.schema.names
    }
}

/// Does the `wrt` table name `wanted` name the table `names`?
pub(crate) fn table_matches(wanted: &str, names: &[String]) -> bool {
    names.join(".") == wanted
        || (!wanted.contains('.') && names.last().is_some_and(|n| n == wanted))
}

/// The column called `wanted`: an exact match, else the only
/// case-insensitive one.
pub(crate) fn find_column(columns: &[String], wanted: &str) -> Option<usize> {
    if let Some(i) = columns.iter().position(|c| c == wanted) {
        return Some(i);
    }
    let mut folded = columns
        .iter()
        .enumerate()
        .filter(|(_, c)| c.eq_ignore_ascii_case(wanted));
    match (folded.next(), folded.next()) {
        (Some((i, _)), None) => Some(i),
        _ => None,
    }
}
