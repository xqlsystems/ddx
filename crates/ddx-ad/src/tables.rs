// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The tables a program shares with its caller.
//!
//! Every kind of program, `grad`'s, `vjp`'s and `jvp`'s, has the same
//! anatomy: tables the caller registers before it runs ([`InputTable`]),
//! checks, steps, and tables it leaves for the caller once it has run
//! ([`OutputTable`]). Each table is *of* something ([`Of`]): the query's
//! output, or a `wrt` table. So:
//!
//! | | input | outputs |
//! |---|---|---|
//! | [`grad`](crate::grad) | none | the value; each `wrt` table's gradient |
//! | [`vjp`](crate::vjp) | the output's cotangent | the value; each `wrt` table's gradient |
//! | [`jvp`](crate::jvp) | each `wrt` table's tangent | the value, with its tangents |
//! | [`jvp`](crate::jvp) of a `grad` program | each `wrt` table's tangent | the value and each gradient, with their tangents (`H·v`) |
//!
//! A cotangent and a tangent are shaped like what they are of, as in JAX:
//! its keys, then a number for each of its values, under the values' names.

/// What an [`InputTable`] or an [`OutputTable`] is of.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Of {
    /// The query's output.
    Output,
    /// A `wrt` table, by its name parts as the plan gives them.
    Table(Vec<String>),
}

impl Of {
    /// The `wrt` table's name parts, if it is of one.
    pub fn table(&self) -> Option<&[String]> {
        match self {
            Of::Table(names) => Some(names),
            Of::Output => None,
        }
    }
}

/// A table the caller registers before a program runs: [`vjp`](crate::vjp)'s
/// cotangent of the output, or [`jvp`](crate::jvp)'s tangent of a `wrt`
/// table.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InputTable {
    /// The name to register it under.
    pub name: String,
    /// What it is a cotangent or tangent of.
    pub of: Of,
    /// The columns it must have, named as in what it is of: its keys, then
    /// its values.
    pub columns: Vec<String>,
    /// How many of `columns` (the first) are keys. A key tuple may occur at
    /// most once, which a check of the program confirms; a key tuple it lacks
    /// is a row whose cotangent or tangent is 0.
    pub keys: usize,
}

/// A table a program leaves for the caller once it has run.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OutputTable {
    /// The step that writes it.
    pub step: String,
    /// What it is: the query's output (the value), or a `wrt` table's
    /// gradient.
    pub of: Of,
    /// Its columns: for the value, the query's own; for a gradient, the
    /// table's dims then its `wrt` values, named as in the table. The
    /// tangents, if any, come after them.
    pub columns: Vec<String>,
    /// For a [`jvp`](crate::jvp) program, each column that has a tangent and
    /// where it is; empty otherwise.
    pub tangents: Vec<Tangent>,
}

/// Where a column's tangent is, in an [`OutputTable`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Tangent {
    /// The column, by name.
    pub column: String,
    /// The column holding its tangent.
    pub tangent: String,
}
