// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The function-name table: what an engine calls the operations ddx reasons
//! about.
//!
//! Substrait identifies a function by an anchor into a plan-local declaration,
//! and that declaration carries a *name*. Ideally the name would be qualified by
//! an extension URN, pinning it to a published definition; in practice neither
//! target engine emits one (decision log `[S7]`), so the name is all there is,
//! and it is the *producer's* name — engine vocabulary, not ddx's.
//!
//! So the mapping from a name to a concept lives here, in one table, rather than
//! as string literals in the rules. The recognizer needs it now; the backward
//! emitter will need the same table in the other direction — to call *multiply*
//! by whatever the target engine calls it (design.md §4.2, `[S7]`).

use crate::markers::base_name;

/// An aggregate function, as far as differentiation cares.
///
/// Not every aggregate has a transpose rule; this names the ones ddx must
/// recognize in order to either differentiate them or refuse them by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggKind {
    /// Summation — the only aggregate with a transpose rule, and the one both
    /// the contraction and reduce markers must sit inside (design.md §4.3).
    Sum,
    /// A mean. Deliberately *not* a transposable primitive: design.md §4.3 has
    /// `SUM` then an elementwise divide.
    Mean,
    /// Maximum or minimum. Differentiable only as Route's argmax idiom, or
    /// removable with `ddx_stop_gradient` (softmax's shift).
    Extremum,
    /// A row count: never carries gradient, since it doesn't read a value.
    Count,
}

impl AggKind {
    /// The aggregate a producer's declared function name refers to, if ddx knows
    /// it. Matched on the name without its Substrait signature suffix,
    /// case-insensitively.
    ///
    /// The spellings are the ones the target engines' producers actually emit.
    /// DataFusion and DuckDB agree on all of these (both follow Substrait's
    /// standard `functions_arithmetic` names); a third engine that spells one
    /// differently adds a spelling here, and nothing else changes.
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
