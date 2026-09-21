// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The function-name table: the names an engine gives to the operations that
//! ddx reasons about.
//!
//! Substrait identifies a function by an anchor into a declaration in the plan,
//! and that declaration carries a name. An extension URN can qualify the name
//! and pin it to a published definition. Neither target engine emits a URN
//! (decision log `[S7]`). The name is therefore all that ddx has, and the name
//! belongs to the producer rather than to ddx.
//!
//! The map from a name to a concept lives here, in one table, and not as string
//! literals in the rules. The recognizer needs the table now. The backward
//! emitter needs the same table in the other direction, to call *multiply* by
//! the name that the target engine uses (design.md §4.2, `[S7]`).

use crate::markers::base_name;

/// An aggregate function, in the terms that differentiation needs.
///
/// Not every aggregate has a transpose rule. This enum names the aggregates that
/// ddx recognizes, so that ddx can differentiate them or refuse them by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggKind {
    /// Summation. This is the only aggregate with a transpose rule, and both the
    /// contraction marker and the reduce marker sit inside it (design.md §4.3).
    Sum,
    /// A mean. It is not a transposable primitive. design.md §4.3 uses a `SUM`
    /// and then an elementwise divide.
    Mean,
    /// A maximum or a minimum. ddx differentiates one only through the Route
    /// idiom for argmax. A `ddx_stop_gradient` call removes one from the backward
    /// pass, as a softmax stability shift needs.
    Extremum,
    /// A count of rows. It never carries gradient, because it does not read a
    /// value.
    Count,
}

impl AggKind {
    /// The aggregate that a declared function name refers to, if ddx knows the
    /// name. The match ignores case and the Substrait signature suffix.
    ///
    /// These spellings are the ones that the target engines emit. DataFusion and
    /// DuckDB agree on all of them, because both follow the standard Substrait
    /// names in `functions_arithmetic`. To support a third engine that spells an
    /// aggregate differently, add the spelling here. Nothing else changes.
    pub fn from_name(name: &str) -> Option<AggKind> {
        let base = base_name(name);
        for (kind, spellings) in [
            // `sum0` differs from `sum` only on empty input (0 rather than
            // NULL), which no transpose rule can distinguish: the derivative of
            // a sum over no rows is empty either way.
            (AggKind::Sum, &["sum", "sum0"][..]),
            (AggKind::Mean, &["avg", "mean"][..]),
            (AggKind::Extremum, &["max", "min"][..]),
            (AggKind::Count, &["count"][..]),
        ] {
            if spellings.iter().any(|s| s.eq_ignore_ascii_case(base)) {
                return Some(kind);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_the_spellings_both_engines_emit() {
        assert_eq!(AggKind::from_name("sum"), Some(AggKind::Sum));
        assert_eq!(AggKind::from_name("SUM:fp64"), Some(AggKind::Sum));
        assert_eq!(AggKind::from_name("avg"), Some(AggKind::Mean));
        assert_eq!(AggKind::from_name("max"), Some(AggKind::Extremum));
        assert_eq!(AggKind::from_name("min"), Some(AggKind::Extremum));
        assert_eq!(AggKind::from_name("count"), Some(AggKind::Count));
        assert_eq!(AggKind::from_name("stddev"), None);
    }
}
