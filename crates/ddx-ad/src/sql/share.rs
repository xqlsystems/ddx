// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Planning each CTE a query reads more than once on its own, so the plan
//! ddx differentiates holds it once.
//!
//! An engine inlines a CTE at every read, and Substrait plans are trees, so
//! a model whose layers read the last one twice reaches ddx, and the
//! engine's optimizer, at a size exponential in its depth. Instead, each
//! CTE the query would copy is planned alone, in order, reading the ones
//! planned before it as *placeholders*: tables under the program's
//! namespace with the CTE's columns and no rows, which the caller creates
//! from the engine's plan of the CTE with `LIMIT 0`. The query is planned
//! the same way. Then [`stitch`] joins the plans into one, each CTE a
//! relation of its own that every read reaches through a `ReferenceRel`,
//! which reverse and forward mode read once each (see
//! [`crate::forward::shared_relations`]).
//!
//! A CTE is planned alone only when that is plainly the same query: not
//! for a `WITH RECURSIVE`, nor a CTE read inside an expression's subquery
//! (where ddx refuses a reference), nor one whose name a nested `WITH`
//! defines again, nor one read under another spelling (`x` and `"X"`),
//! which engines resolve differently. Such a CTE is left for the engine to
//! inline, as before, which is always right.
//!
//! Planning a CTE alone keeps the engine from pushing a reader's filter
//! into it, or pruning a column only some readers use; ddx prunes the
//! steps it writes, and a filter on a gradient's dims still restricts it,
//! being planned against the table.

use std::collections::HashMap;
use std::ops::ControlFlow;

use ddx_core::sqlparser::ast::{
    Cte, Expr, Ident, ObjectName, ObjectNamePart, Query, Visit, VisitMut, Visitor,
};
use substrait::proto::read_rel::ReadType;
use substrait::proto::rel::RelType;
use substrait::proto::{Plan, PlanRel, ReadRel, ReferenceRel, Rel};

use super::build::{quoted, rename_reads};
use crate::compose::redeclare_rel;
use crate::emit::{filter, plan as plan_of, select};
use crate::error::{AdError, Result};
use crate::forward::{
    apply_emit, inline_references, read_outputs, root_of, windows_as_projections,
};
use crate::functions::{Extensions, Functions};
use crate::optics::{AsRead, Compose, Deep, Fold, RelKind, Setter, Tree};

/// A query whose CTEs read more than once are planned one by one.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Split {
    /// Each such CTE, in order: the placeholder its readers read, and the
    /// query for its rows, which reads earlier ones' placeholders.
    pub ctes: Vec<Piece>,
    /// The query, reading each such CTE's placeholder.
    pub root: String,
    /// The namespace the placeholders are named under: fresh to the call,
    /// so calls that overlap make tables of their own.
    pub namespace: String,
}

/// One CTE planned on its own.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Piece {
    pub placeholder: String,
    pub sql: String,
}

impl Split {
    /// The query for a placeholder's columns and no rows: what the caller
    /// materializes as it before planning a piece that reads it.
    pub fn empty(piece: &Piece) -> String {
        format!("SELECT * FROM ({}) AS __ddx_empty LIMIT 0", piece.sql)
    }
}

/// `query` split so that each CTE an engine would copy is planned once,
/// its placeholders named under `namespace`; `None` if no CTE would be
/// copied, or none can be split off safely.
pub(super) fn split(query: &Query, namespace: &str) -> Option<Split> {
    let with = query.with.as_ref()?;
    if with.recursive {
        return None;
    }
    let ctes = &with.cte_tables;
    let n = ctes.len();
    // How often each scope reads each CTE it can see: CTE `j` sees the ones
    // before it; the query's body (scope `n`) sees all.
    let mut reads = vec![vec![0usize; n]; n + 1];
    let mut unsafe_ = vec![false; n];
    for j in 0..=n {
        let visible = &ctes[..j.min(n)];
        let mut refs = Refs::new(visible);
        if j < n {
            let _ = ctes[j].query.visit(&mut refs);
        } else {
            let mut body = query.clone();
            body.with = None;
            let _ = body.visit(&mut refs);
        }
        reads[j][..j.min(n)].copy_from_slice(&refs.counts);
        for (k, bad) in refs.unsafe_.iter().enumerate() {
            unsafe_[k] |= *bad;
        }
    }
    // A CTE left to the engine is inlined wherever it is read, a subquery
    // included, where a placeholder read could not become a reference; so
    // what it reads is left to the engine too.
    for j in (0..n).rev() {
        if unsafe_[j] {
            for k in 0..j {
                if reads[j][k] > 0 {
                    unsafe_[k] = true;
                }
            }
        }
    }
    // How many copies of each CTE the engine's plan would hold, readers
    // first; a CTE planned on its own is one copy for what it reads.
    let mut copies = vec![0usize; n];
    let mut shared = vec![false; n];
    for k in (0..n).rev() {
        let mut c = reads[n][k];
        for j in k + 1..n {
            let each = if shared[j] { 1 } else { copies[j] };
            c += reads[j][k] * each;
        }
        copies[k] = c;
        shared[k] = c >= 2 && !unsafe_[k];
    }
    if !shared.contains(&true) {
        return None;
    }
    let placeholders: HashMap<usize, String> = (0..n)
        .filter(|&k| shared[k])
        .map(|k| (k, format!("{namespace}cte_{k}")))
        .collect();
    let mut pieces = Vec::new();
    for k in (0..n).filter(|&k| shared[k]) {
        let mut body = (*ctes[k].query).clone();
        replace(&mut body, &ctes[..k], &placeholders);
        let alias = &ctes[k].alias;
        let columns = if alias.columns.is_empty() {
            String::new()
        } else {
            let names: Vec<String> = alias.columns.iter().map(|c| c.to_string()).collect();
            format!(" ({})", names.join(", "))
        };
        let with = with_of(ctes, k, &reads, &shared, &placeholders);
        pieces.push(Piece {
            placeholder: placeholders[&k].clone(),
            sql: format!("{with}SELECT * FROM ({body}) AS __ddx_cte{columns}"),
        });
    }
    let mut root = query.clone();
    root.with = None;
    replace(&mut root, ctes, &placeholders);
    let with = with_of(ctes, n, &reads, &shared, &placeholders);
    Some(Split {
        ctes: pieces,
        root: format!("{with}{root}"),
        namespace: namespace.to_string(),
    })
}

/// The `WITH` clause scope `upto` needs: each CTE before it that it reads,
/// directly or through such CTEs, that is not planned on its own, its reads
/// of those that are replaced by their placeholders; `""` if none.
fn with_of(
    ctes: &[Cte],
    upto: usize,
    reads: &[Vec<usize>],
    shared: &[bool],
    placeholders: &HashMap<usize, String>,
) -> String {
    let mut wanted = vec![false; upto];
    for k in 0..upto {
        wanted[k] = reads[upto][k] > 0 && !shared[k];
    }
    for j in (0..upto).rev() {
        if wanted[j] {
            for k in 0..j {
                if reads[j][k] > 0 && !shared[k] {
                    wanted[k] = true;
                }
            }
        }
    }
    let kept: Vec<String> = (0..upto)
        .filter(|&k| wanted[k])
        .map(|k| {
            let mut cte = ctes[k].clone();
            replace(&mut cte.query, &ctes[..k], placeholders);
            cte.to_string()
        })
        .collect();
    if kept.is_empty() {
        String::new()
    } else {
        format!("WITH {} ", kept.join(", "))
    }
}

/// Counts of a scope's reads of the CTEs it sees, and which of them it
/// reads in a way that rules out planning them alone.
struct Refs<'c> {
    ctes: &'c [Cte],
    counts: Vec<usize>,
    unsafe_: Vec<bool>,
    /// How many expression subqueries the walk is inside.
    subqueries: usize,
}

impl<'c> Refs<'c> {
    fn new(ctes: &'c [Cte]) -> Self {
        Refs {
            ctes,
            counts: vec![0; ctes.len()],
            unsafe_: vec![false; ctes.len()],
            subqueries: 0,
        }
    }
}

impl Visitor for Refs<'_> {
    type Break = ();

    fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<()> {
        // A nested WITH that defines a CTE's name again: a read under it
        // may be of either, so the CTE is left to the engine.
        if let Some(with) = &query.with {
            for inner in &with.cte_tables {
                for (k, cte) in self.ctes.iter().enumerate() {
                    if inner
                        .alias
                        .name
                        .value
                        .eq_ignore_ascii_case(&cte.alias.name.value)
                    {
                        self.unsafe_[k] = true;
                    }
                }
            }
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        if is_subquery(expr) {
            self.subqueries += 1;
        }
        ControlFlow::Continue(())
    }

    fn post_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        if is_subquery(expr) {
            self.subqueries -= 1;
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_relation(&mut self, name: &ObjectName) -> ControlFlow<()> {
        let matches = ctes_of(name, self.ctes);
        let alone = matches.len() == 1;
        for (k, same) in matches {
            // Every read counts, against each CTE it may be of, so each
            // stays in the `WITH` of a scope that reads it, however it
            // reads it.
            self.counts[k] += 1;
            if !same || !alone || self.subqueries > 0 {
                self.unsafe_[k] = true;
            }
        }
        ControlFlow::Continue(())
    }
}

fn is_subquery(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Subquery(_) | Expr::InSubquery { .. } | Expr::Exists { .. }
    )
}

/// Which of `ctes` a one-part relation name may read, and whether it is
/// spelled as each is (both unquoted, which every engine compares without
/// case, or both quoted the same). Engines differ on how case and quotes
/// resolve, so every CTE whose name matches but for case may be the one
/// read.
fn ctes_of(name: &ObjectName, ctes: &[Cte]) -> Vec<(usize, bool)> {
    let [ObjectNamePart::Identifier(id)] = name.0.as_slice() else {
        return Vec::new();
    };
    ctes.iter()
        .enumerate()
        .filter(|(_, c)| c.alias.name.value.eq_ignore_ascii_case(&id.value))
        .map(|(k, c)| (k, same_spelling(id, &c.alias.name)))
        .collect()
}

/// The CTE a relation name surely reads: the one CTE its name may read,
/// spelled as it is.
fn cte_of(name: &ObjectName, ctes: &[Cte]) -> Option<usize> {
    match ctes_of(name, ctes)[..] {
        [(k, true)] => Some(k),
        _ => None,
    }
}

fn same_spelling(a: &Ident, b: &Ident) -> bool {
    match (a.quote_style, b.quote_style) {
        (None, None) => a.value.eq_ignore_ascii_case(&b.value),
        (Some(_), Some(_)) => a.value == b.value,
        _ => false,
    }
}

/// `query` with each read of one of `ctes` planned alone replaced by a read
/// of its placeholder.
fn replace<V: VisitMut>(node: &mut V, ctes: &[Cte], placeholders: &HashMap<usize, String>) {
    rename_reads(node, |name| {
        let k = cte_of(name, ctes)?;
        placeholders.get(&k).map(|p| quoted(p))
    });
}

/// One plan from `split`'s: `pieces`, the engine's plan of each CTE in
/// order, and `root`, of the query. Each CTE becomes a relation of its own,
/// after the root, and each read of its placeholder a `ReferenceRel` to it,
/// with the read's filter and projection kept. Functions are declared
/// again from one set, as a program's plans are.
pub(super) fn stitch(split: &Split, pieces: &[Plan], root: &Plan) -> Result<Plan> {
    let mut ext = Extensions::new(&Functions::from_plan(&Plan::default())?);
    let ordinals: HashMap<&str, i32> = split
        .ctes
        .iter()
        .enumerate()
        .map(|(i, p)| (p.placeholder.as_str(), i as i32 + 1))
        .collect();
    let mut relations = Vec::with_capacity(pieces.len());
    for plan in pieces {
        relations.push(PlanRel {
            rel_type: Some(substrait::proto::plan_rel::RelType::Rel(one(
                plan, &ordinals, &mut ext,
            )?)),
        });
    }
    let (_, names) = root_of(root)?;
    let rel = one(root, &ordinals, &mut ext)?;
    let mut plan = plan_of(rel, names, &ext);
    plan.relations.extend(relations);
    Ok(plan)
}

/// `plan`'s relation, its functions declared in `ext` and each read of a
/// placeholder in `ordinals` a reference to it.
fn one(plan: &Plan, ordinals: &HashMap<&str, i32>, ext: &mut Extensions) -> Result<Rel> {
    // DuckDB's window relations, as the projections ddx's rules read, so
    // their functions are declared again as a projection's are.
    let plan = &*windows_as_projections(plan);
    let (root, _) = root_of(plan)?;
    let mut rel = inline_references(root, &plan.relations)?;
    redeclare_rel(&mut rel, &Functions::from_plan(plan)?, &HashMap::new(), ext)?;
    // Not inside subqueries, where a reference can't sit: a read left there
    // is refused below.
    Tree.try_modify(&mut rel, &mut |r| {
        if let Some(RelType::Read(read)) = &r.rel_type {
            if let Some(ordinal) = placeholder(read, ordinals) {
                *r = referenced_as_read(read, ordinal)?;
            }
        }
        Ok::<(), AdError>(())
    })?;
    if reads_placeholder(&rel, ordinals) {
        return Err(AdError::NotImplemented(
            "a read of a CTE planned on its own that is not a relation's input".into(),
        ));
    }
    Ok(rel)
}

fn placeholder(read: &ReadRel, ordinals: &HashMap<&str, i32>) -> Option<i32> {
    let Some(ReadType::NamedTable(t)) = &read.read_type else {
        return None;
    };
    // By the last part: a producer may qualify the name with the schema.
    t.names
        .last()
        .and_then(|name| ordinals.get(name.as_str()).copied())
}

/// Does `rel` still read a placeholder anywhere, subqueries included? The
/// program never reads one, so such a plan is refused, and the query is
/// planned whole.
fn reads_placeholder(rel: &Rel, ordinals: &HashMap<&str, i32>) -> bool {
    Deep.then(RelKind)
        .then(AsRead)
        .any(rel, |read| placeholder(read, ordinals).is_some())
}

/// A reference to relation `ordinal` in place of `read`, a read of its
/// placeholder: the read's filters, then its projection and emit.
fn referenced_as_read(read: &ReadRel, ordinal: i32) -> Result<Rel> {
    let mut rel = Rel {
        rel_type: Some(RelType::Reference(ReferenceRel {
            subtree_ordinal: ordinal,
        })),
    };
    for f in [&read.filter, &read.best_effort_filter]
        .into_iter()
        .flatten()
    {
        rel = filter(rel, (**f).clone());
    }
    let outputs = apply_emit(read.common.as_ref(), read_outputs(read)?)?;
    if read.advanced_extension.is_some() {
        return Err(AdError::NotImplemented(
            "a read of a CTE planned on its own with an extension".into(),
        ));
    }
    Ok(select(rel, outputs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddx_core::sqlparser::ast::Statement;
    use ddx_core::sqlparser::dialect::GenericDialect;
    use ddx_core::sqlparser::parser::Parser;

    fn query(sql: &str) -> Query {
        match Parser::parse_sql(&GenericDialect {}, sql)
            .unwrap()
            .remove(0)
        {
            Statement::Query(q) => *q,
            other => panic!("{other}"),
        }
    }

    /// The CTEs `sql` plans on its own, by index into its `WITH`.
    fn shared(sql: &str) -> Vec<usize> {
        let Some(s) = split(&query(sql), "__ddx_t_") else {
            return Vec::new();
        };
        s.ctes
            .iter()
            .map(|p| p.placeholder["__ddx_t_cte_".len()..].parse().unwrap())
            .collect()
    }

    #[test]
    fn a_cte_read_twice_is_planned_alone_and_one_read_once_is_not() {
        assert_eq!(
            shared("WITH a AS (SELECT 1 AS x), b AS (SELECT 2 AS y) SELECT * FROM a, a, b"),
            vec![0]
        );
        assert_eq!(
            shared("WITH a AS (SELECT 1 AS x) SELECT * FROM a"),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn copies_multiply_through_ctes_left_to_the_engine() {
        // c is read twice, so it is planned alone, and reads a once.
        assert_eq!(
            shared("WITH a AS (SELECT 1 AS x), c AS (SELECT x FROM a) SELECT * FROM c, c"),
            vec![1]
        );
        // d reads a once and is read once, but d's reader e is read twice:
        // e is planned alone, and a, d once each inside it.
        assert_eq!(
            shared(
                "WITH a AS (SELECT 1 AS x), d AS (SELECT x FROM a), e AS (SELECT x FROM d) \
                 SELECT * FROM e, e"
            ),
            vec![2]
        );
        // a read once by each of two readers that are each read once: two
        // copies, so a is planned alone.
        assert_eq!(
            shared(
                "WITH a AS (SELECT 1 AS x), p AS (SELECT x FROM a), q AS (SELECT x FROM a) \
                 SELECT * FROM p, q"
            ),
            vec![0]
        );
    }

    #[test]
    fn a_cte_that_might_be_misread_is_left_to_the_engine() {
        // A recursive WITH.
        assert!(shared("WITH RECURSIVE a AS (SELECT 1 AS x) SELECT * FROM a, a").is_empty());
        // Read inside an expression's subquery, where ddx refuses a reference.
        assert!(shared(
            "WITH a AS (SELECT 1 AS x) SELECT * FROM a, a WHERE 1 IN (SELECT x FROM a)"
        )
        .is_empty());
        // Defined again by a nested WITH.
        assert!(shared(
            "WITH a AS (SELECT 1 AS x) \
             SELECT * FROM a, a, (WITH a AS (SELECT 2 AS x) SELECT * FROM a) AS n"
        )
        .is_empty());
        // Read under another spelling, which engines resolve differently.
        assert!(shared("WITH a AS (SELECT 1 AS x) SELECT * FROM a, \"A\"").is_empty());
        // A later CTE's name, read before it is defined: a table, not the CTE.
        assert!(
            shared("WITH a AS (SELECT * FROM b, b), b AS (SELECT 1 AS y) SELECT * FROM a")
                .is_empty()
        );
    }

    #[test]
    fn a_cte_read_only_where_it_might_be_misread_stays_in_the_with() {
        // b is read only inside a subquery, so it stays in the root's WITH;
        // a, read twice, would be planned alone, but b reads it and is
        // inlined into the subquery, so a is left to the engine too.
        let sql = "WITH a AS (SELECT 1 AS x), b AS (SELECT x FROM a) \
                   SELECT * FROM a, a WHERE 1 IN (SELECT x FROM b)";
        assert!(shared(sql).is_empty());
        // c, read twice, is planned alone; b reads only a, so the root
        // keeps b and a in its WITH and reads c's placeholder.
        let s = split(
            &query(
                "WITH a AS (SELECT 1 AS x), b AS (SELECT x FROM a), c AS (SELECT 2 AS y) \
                 SELECT * FROM c, c WHERE 1 IN (SELECT x FROM b)",
            ),
            "__ddx_t_",
        )
        .unwrap();
        assert_eq!(
            s.root,
            "WITH a AS (SELECT 1 AS x), b AS (SELECT x FROM a) \
             SELECT * FROM \"__ddx_t_cte_2\" AS c, \"__ddx_t_cte_2\" AS c \
             WHERE 1 IN (SELECT x FROM b)"
        );
        // Read under another spelling: kept, for the engine to resolve.
        let s = split(
            &query("WITH a AS (SELECT 1 AS x), c AS (SELECT 2 AS y) SELECT * FROM c, c, \"A\""),
            "__ddx_t_",
        )
        .unwrap();
        assert!(
            s.root.starts_with("WITH a AS (SELECT 1 AS x) "),
            "{}",
            s.root
        );
        // A read that may be of either of two CTEs whose names differ only
        // in case keeps both, and neither is planned alone.
        let s = split(
            &query(
                "WITH a AS (SELECT 1 AS x), \"A\" AS (SELECT 2 AS x), c AS (SELECT 3 AS y) \
                 SELECT * FROM c, c, \"A\", \"A\"",
            ),
            "__ddx_t_",
        )
        .unwrap();
        assert!(
            s.root
                .starts_with("WITH a AS (SELECT 1 AS x), \"A\" AS (SELECT 2 AS x) "),
            "{}",
            s.root
        );
        assert_eq!(s.ctes.len(), 1);
    }

    #[test]
    fn a_placeholder_read_a_reference_cannot_replace_is_refused() {
        use substrait::proto::expression::subquery::{Scalar, SubqueryType};
        use substrait::proto::expression::{RexType, Subquery};
        use substrait::proto::read_rel::NamedTable;
        use substrait::proto::{Expression, FilterRel};
        let read = |names: &[&str]| Rel {
            rel_type: Some(RelType::Read(Box::new(ReadRel {
                read_type: Some(ReadType::NamedTable(NamedTable {
                    names: names.iter().map(|n| n.to_string()).collect(),
                    ..Default::default()
                })),
                ..Default::default()
            }))),
        };
        let ordinals = HashMap::from([("__ddx_t_cte_0", 1)]);
        // Qualified by a schema, it is still the placeholder.
        assert!(reads_placeholder(
            &read(&["public", "__ddx_t_cte_0"]),
            &ordinals
        ));
        assert!(!reads_placeholder(&read(&["t"]), &ordinals));
        // Inside a filter's scalar subquery.
        let filtered = Rel {
            rel_type: Some(RelType::Filter(Box::new(FilterRel {
                input: Some(Box::new(read(&["t"]))),
                condition: Some(Box::new(Expression {
                    rex_type: Some(RexType::Subquery(Box::new(Subquery {
                        subquery_type: Some(SubqueryType::Scalar(Box::new(Scalar {
                            input: Some(Box::new(read(&["__ddx_t_cte_0"]))),
                        }))),
                    }))),
                })),
                ..Default::default()
            }))),
        };
        assert!(reads_placeholder(&filtered, &ordinals));
    }

    #[test]
    fn each_piece_reads_the_placeholders_of_those_before_it() {
        let s = split(
            &query(
                "WITH a (x) AS (SELECT 1), m AS (SELECT x FROM a), b AS (SELECT p.x FROM m p, a q) \
                 SELECT * FROM b, b, a",
            ),
            "__ddx_t_",
        )
        .unwrap();
        let names: Vec<&str> = s.ctes.iter().map(|p| p.placeholder.as_str()).collect();
        assert_eq!(names, ["__ddx_t_cte_0", "__ddx_t_cte_2"]);
        // a keeps its column list; b reads a's placeholder, and m, read once
        // and left to the engine, in its own WITH.
        assert_eq!(s.ctes[0].sql, "SELECT * FROM (SELECT 1) AS __ddx_cte (x)");
        assert_eq!(
            s.ctes[1].sql,
            "WITH m AS (SELECT x FROM \"__ddx_t_cte_0\" AS a) \
             SELECT * FROM (SELECT p.x FROM m p, \"__ddx_t_cte_0\" q) AS __ddx_cte"
        );
        assert_eq!(
            s.root,
            "SELECT * FROM \"__ddx_t_cte_2\" AS b, \"__ddx_t_cte_2\" AS b, \"__ddx_t_cte_0\" AS a"
        );
    }
}
