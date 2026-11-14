// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A shift that cancels: the maximum a softmax or a log-sum-exp subtracts.
//!
//! A softmax is written `exp(x − m) / SUM(exp(x − m))`, with `m` the
//! group's maximum, so that `exp` can't overflow; a log-sum-exp is written
//! `m + ln(SUM(exp(x − m)))`. Neither depends on `m`. Raise `m` by `δ`, and
//! every `exp(x − m)` of the group, so their sum too, scales by `e^−δ`: the
//! ratio stays, and `ln` of the sum falls by `δ` as `m` rises by it. So
//! `m̄ = 0`, and ddx reads `m` as a constant, as `jax.nn.softmax` holds its
//! maximum with `stop_gradient`. Computing `m̄` would cost a sum over every
//! row of every group to get zero, up to rounding (#243, #251).
//!
//! A saved value `m` is held constant when every use of it is one of these,
//! where "a group" means one that holds `m`'s dims equal to its own keys,
//! so `m` is the same throughout it:
//!
//! | use | gives |
//! |---|---|
//! | `x − m`, `x` not computed from `m` | a shifted value |
//! | `exp` of a shifted value | a term |
//! | `SUM` of a term over a group | a normalizer `z` |
//! | `MAX`, `MIN` or `AVG` of `m` over a group | `m` again |
//! | `ln z` | a log-normalizer |
//! | a term divided by `z`, of the same group | a value `m` cancels from |
//! | `m` plus a log-normalizer, of the same group | a value `m` cancels from |
//!
//! Each of these must itself be used only as the table says, up to the
//! values `m` cancels from, which may be used anyhow. "Of the same group"
//! means the region holds the two operands' copies of `m`'s dims equal.
//! Anything else (`m` in the output, in a grouping key, a window, a union,
//! or a subquery) leaves `m` varied, and its gradient is computed as
//! before.

use std::collections::{BTreeMap, BTreeSet};

use substrait::proto::expression::RexType;
use substrait::proto::read_rel::ReadType;
use substrait::proto::Expression;

use crate::equal::{held_equal, Classes};
use crate::error::Result;
use crate::expr::{as_field, fields_of, scalar_args};
use crate::forward::{saved_name, Def, Forward, Input, Output, Region};
use crate::functions::Functions;
use crate::optics::{AsRead, Compose, Cosmos, Deep, Fold, RelKind};

/// What a column or an expression is, as far as one shift `m` goes. Each
/// but `Clean` carries its group: the region's columns that hold `m`'s
/// dims, one per dim.
#[derive(Debug, Clone, PartialEq)]
enum Kind {
    /// Not computed from `m`, or computed from it in a way it cancels from.
    Clean,
    /// `m`, or `MAX`, `MIN` or `AVG` of it over a group.
    Shift(Vec<usize>),
    /// `x − m`.
    Shifted(Vec<usize>),
    /// `exp(x − m)`.
    Term(Vec<usize>),
    /// `SUM` of terms over a group.
    Normalizer(Vec<usize>),
    /// `ln` of a normalizer.
    Log(Vec<usize>),
    /// Computed from `m` some other way.
    Bad,
}

/// A saved value computed from a shift: `m` itself, `m` again, or a
/// normalizer. `keys` are the saved aggregate's output columns that hold
/// `m`'s dims, one per dim.
#[derive(Debug, Clone)]
struct Taint {
    normalizer: bool,
    keys: Vec<usize>,
}

/// The saved values `(aggregate, output column)` that are shifts that
/// cancel (see the module docs), for `forward` to read as constants.
pub(crate) fn cancelling(f: &Forward) -> Result<BTreeSet<(usize, usize)>> {
    let mut held = BTreeSet::new();
    for (n, v) in candidates(f)? {
        if cancels(f, n, v)? {
            held.insert((n, v));
        }
    }
    Ok(held)
}

/// Every varied saved value, with dims that aren't varied, that some
/// region's column or some aggregate's argument subtracts: through columns
/// that only copy it, as a window read as an aggregate is copied.
fn candidates(f: &Forward) -> Result<BTreeSet<(usize, usize)>> {
    let mut out = BTreeSet::new();
    let regions = f
        .saved
        .iter()
        .map(|s| (&s.input, Some(s)))
        .chain([(&f.output, None)]);
    for (region, saved) in regions {
        let mut exprs: Vec<&Expression> = region
            .defs
            .iter()
            .filter_map(|d| match d {
                Def::Expr(e) => Some(e),
                _ => None,
            })
            .collect();
        if let Some(s) = saved {
            for m in &s.measures {
                exprs.extend(crate::expr::value_args(&m.arguments)?);
            }
        }
        for e in exprs {
            let mut subtracted = Vec::new();
            Cosmos.each(e, |x| {
                if let Some(RexType::ScalarFunction(g)) = &x.rex_type {
                    if f.functions.name(g.function_reference).ok() == Some("subtract") {
                        if let Ok([_, b]) =
                            <[&Expression; 2]>::try_from(scalar_args(g).unwrap_or_default())
                        {
                            subtracted.extend(as_field(b));
                        }
                    }
                }
            });
            for mut c in subtracted {
                while let Def::Expr(e) = &region.defs[c] {
                    match as_field(e) {
                        Some(below) => c = below,
                        None => break,
                    }
                }
                if let Def::Input { slot, col } = region.defs[c] {
                    if let Input::Saved(n) = region.slots[slot].input {
                        let s = &f.saved[n];
                        let fixed = s.dims().iter().all(|&d| !s.varied[d]);
                        if s.varied[col] && fixed && matches!(s.outputs[col], Output::Value(_)) {
                            out.insert((n, col));
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Is every use of saved value `v` of aggregate `n` one it cancels from?
fn cancels(f: &Forward, n: usize, v: usize) -> Result<bool> {
    let mut taints: BTreeMap<(usize, usize), Taint> = BTreeMap::new();
    taints.insert(
        (n, v),
        Taint {
            normalizer: false,
            keys: f.saved[n].dims(),
        },
    );
    let later = (n + 1..f.saved.len()).map(Some).chain([None]);
    for k in later {
        let region = match k {
            Some(k) => &f.saved[k].input,
            None => &f.output,
        };
        let Some((kinds, mut held)) = kinds(f, region, &taints)? else {
            return Ok(false);
        };
        let Some(k) = k else {
            // The output: every column the query returns must be clean.
            return Ok(region.outputs.iter().all(|&c| kinds[c] == Kind::Clean));
        };
        let saved = &f.saved[k];
        let classify = |e: &Expression, held: &mut Classes| kind(&f.functions, e, &kinds, held);
        for g in &saved.groupings {
            if classify(g, &mut held)? != Kind::Clean {
                return Ok(false);
            }
        }
        for (col, out) in saved.outputs.iter().enumerate() {
            let Output::Value(m) = *out else {
                continue;
            };
            let measure = &saved.measures[m];
            let args = crate::expr::value_args(&measure.arguments)?;
            let [arg] = args[..] else {
                for a in args {
                    if classify(a, &mut held)? != Kind::Clean {
                        return Ok(false);
                    }
                }
                continue;
            };
            let name = f.functions.name(measure.function_reference)?;
            let (group, normalizer) = match (classify(arg, &mut held)?, name) {
                (Kind::Clean, _) => continue,
                (Kind::Term(g), "sum") => (g, true),
                (Kind::Shift(g), "max" | "min" | "avg" | "mean") => (g, false),
                _ => return Ok(false),
            };
            // Each of m's dims, as a grouping key: the group is one m holds.
            let mut keys = Vec::new();
            for g in group {
                let dim = saved.dims().into_iter().find(|&d| {
                    let Output::Dim(gi) = saved.outputs[d] else {
                        return false;
                    };
                    as_field(&saved.groupings[gi]).is_some_and(|key| held.same(key, g))
                });
                match dim {
                    Some(d) => keys.push(d),
                    None => return Ok(false),
                }
            }
            taints.insert((k, col), Taint { normalizer, keys });
        }
    }
    Ok(true)
}

/// Each column's [`Kind`] in `region`, given the saved values `taints`
/// computed from the shift, and the columns the region holds equal; `None`
/// if a column uses the shift in a way it doesn't cancel from, or a
/// tainted value is read where its uses can't be seen.
fn kinds(
    f: &Forward,
    region: &Region,
    taints: &BTreeMap<(usize, usize), Taint>,
) -> Result<Option<(Vec<Kind>, Classes)>> {
    let mut held = Classes::new(region.defs.len());
    let tainted: BTreeSet<usize> = taints.keys().map(|&(s, _)| s).collect();
    let read = region
        .slots
        .iter()
        .filter_map(|s| match s.input {
            Input::Saved(n) if tainted.contains(&n) => Some((n, s.offset)),
            _ => None,
        })
        .collect::<Vec<_>>();
    if read.is_empty() {
        return Ok(Some((vec![Kind::Clean; region.defs.len()], held)));
    }
    // Every read of a tainted aggregate is a slot whose columns are here.
    if read.iter().any(|(_, offset)| offset.is_none()) {
        return Ok(None);
    }
    // Subqueries' reads included.
    let mut names = Vec::new();
    Deep.then(RelKind).then(AsRead).each(&region.rel, |read| {
        if let Some(ReadType::NamedTable(t)) = &read.read_type {
            names.push(t.names.join("."));
        }
    });
    for &n in &tainted {
        let name = saved_name(&f.namespace, n);
        let slots = read.iter().filter(|(s, _)| *s == n).count();
        if names.iter().filter(|r| **r == name).count() != slots {
            return Ok(None);
        }
    }
    held_equal(&f.functions, &region.rel, 0, &mut held)?;
    let mut kinds = Vec::with_capacity(region.defs.len());
    for d in &region.defs {
        let k = match d {
            Def::Input { slot, col } => {
                let s = &region.slots[*slot];
                match (s.input, s.offset) {
                    (Input::Saved(n), Some(offset)) => match taints.get(&(n, *col)) {
                        Some(t) => {
                            let group = t.keys.iter().map(|&d| offset + d).collect();
                            if t.normalizer {
                                Kind::Normalizer(group)
                            } else {
                                Kind::Shift(group)
                            }
                        }
                        None => Kind::Clean,
                    },
                    _ => Kind::Clean,
                }
            }
            Def::Expr(e) => match kind(&f.functions, e, &kinds, &mut held)? {
                Kind::Bad => return Ok(None),
                k => k,
            },
            Def::Window { function, .. } => {
                let fs = fields_of(function)?;
                if fs.iter().any(|&c| kinds[c] != Kind::Clean) {
                    return Ok(None);
                }
                Kind::Clean
            }
            Def::Union(arms) => {
                if arms.iter().any(|a| kinds[a.column] != Kind::Clean) {
                    return Ok(None);
                }
                Kind::Clean
            }
            Def::Const | Def::Branch => Kind::Clean,
        };
        kinds.push(k);
    }
    Ok(Some((kinds, held)))
}

/// `e`'s [`Kind`], over columns of kinds `cols`, in a region that holds
/// the columns `held` joins equal.
fn kind(functions: &Functions, e: &Expression, cols: &[Kind], held: &mut Classes) -> Result<Kind> {
    let same = |a: &[usize], b: &[usize], held: &mut Classes| {
        a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| held.same(x, y))
    };
    if let Some(c) = as_field(e) {
        return Ok(cols[c].clone());
    }
    if let Some(RexType::ScalarFunction(g)) = &e.rex_type {
        let mut kinds = Vec::new();
        for a in scalar_args(g)? {
            kinds.push(kind(functions, a, cols, held)?);
        }
        use Kind::*;
        return Ok(match (functions.name(g.function_reference)?, &kinds[..]) {
            ("subtract", [Clean, Shift(m)]) => Shifted(m.clone()),
            ("exp", [Shifted(m)]) => Term(m.clone()),
            ("ln", [Normalizer(m)]) => Log(m.clone()),
            ("divide", [Term(a), Normalizer(b)]) if same(a, b, held) => Clean,
            ("add", [Shift(a), Log(b)] | [Log(b), Shift(a)]) if same(a, b, held) => Clean,
            _ if kinds.iter().all(|k| *k == Clean) => Clean,
            _ => Bad,
        });
    }
    // Anything else is clean when it reads nothing computed from the shift.
    Ok(match fields_of(e) {
        Ok(fs) if fs.iter().all(|&c| cols[c] == Kind::Clean) => Kind::Clean,
        _ => Kind::Bad,
    })
}
