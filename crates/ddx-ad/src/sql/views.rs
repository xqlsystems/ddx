// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Writing the views an objective reads as CTEs, so [`super::share`] can
//! plan each of their CTEs read more than once on its own.
//!
//! A model is often a view (`CREATE VIEW logits AS WITH x0 AS (…), … SELECT
//! …`), and its layers, the CTEs a residual connection or a layer norm reads
//! twice, are inside it, where the objective's text does not show them. So
//! each view the objective reads, whose SQL the engine gives
//! ([`crate::Backend::view_sql`]), becomes CTEs of the objective: the view's
//! own CTEs, renamed `__ddx_view_{k}_{name}` so they cannot meet another
//! name, then its query as the CTE `__ddx_view_{k}`, which the objective
//! reads in the view's place. Views a view reads are written the same way,
//! before it. They all come before the objective's own CTEs, so a view's
//! reads, which the engine resolves in its catalog, cannot reach one.
//!
//! A view is left as it is, for the engine to inline as before, where
//! writing it out could change what a read means: under a name a CTE of
//! the objective also has, with SQL that does not parse, or with a
//! `WITH RECURSIVE` or a CTE name defined again inside it.

use std::collections::HashMap;
use std::ops::ControlFlow;

use ddx_core::sqlparser::ast::{
    Cte, Ident, ObjectName, ObjectNamePart, Query, Statement, TableAlias, TableAliasColumnDef,
    Visit, VisitMut, Visitor, VisitorMut, With,
};
use ddx_core::sqlparser::dialect::GenericDialect;
use ddx_core::sqlparser::parser::Parser;

use super::build::quoted;
use crate::run::Backend;

/// How many views deep a view may read others: past it, the rest are left
/// to the engine. Engines refuse a view that reads itself; this only stops
/// a backend that does not.
const DEPTH: usize = 64;

/// `query` with each view it reads written as CTEs (see the module docs);
/// `None` if it reads none the engine gives the SQL of.
pub(super) async fn expand<B: Backend>(backend: &mut B, query: &Query) -> Option<Query> {
    let with = query.with.as_ref();
    if with.is_some_and(|w| w.recursive) {
        return None;
    }
    // Any name a CTE has, anywhere in the query, is not read as a view.
    let mut defined = Vec::new();
    let _ = query.visit(&mut Defined(&mut defined));
    let mut views = Views {
        defined,
        written: HashMap::new(),
        ctes: Vec::new(),
        named: 0,
    };
    let mut out = query.clone();
    let mut body_names = names(&out);
    body_names.retain(|n| !views.is_cte(n));
    let mut any = false;
    for name in body_names {
        if let Some(alias) = views.write(backend, &name, 0).await {
            rename(&mut out, &name, &alias);
            any = true;
        }
    }
    if !any {
        return None;
    }
    let mut ctes = std::mem::take(&mut views.ctes);
    if let Some(w) = out.with.take() {
        ctes.extend(w.cte_tables);
    }
    out.with = Some(With {
        with_token: ddx_core::sqlparser::ast::helpers::attached_token::AttachedToken::empty(),
        recursive: false,
        cte_tables: ctes,
    });
    Some(out)
}

struct Views {
    /// Every CTE name in the objective, at any depth, without case.
    defined: Vec<String>,
    /// Each view written so far, by its name as read: its CTE's name, or
    /// `None` if it is left to the engine.
    written: HashMap<String, Option<String>>,
    /// The CTEs written, each after those it reads.
    ctes: Vec<Cte>,
    /// How many views have taken a name: each takes the next as it starts,
    /// before the views it reads take theirs, so no two share one.
    named: usize,
}

impl Views {
    fn is_cte(&self, name: &ObjectName) -> bool {
        match name.0.as_slice() {
            [ObjectNamePart::Identifier(id)] => self
                .defined
                .iter()
                .any(|d| d.eq_ignore_ascii_case(&id.value)),
            _ => false,
        }
    }

    /// The CTE the view `name` is written as, writing it (and the views it
    /// reads) if it is not yet; `None` if `name` is not a view whose SQL the
    /// engine gives, or it is left to the engine.
    async fn write<B: Backend>(
        &mut self,
        backend: &mut B,
        name: &ObjectName,
        depth: usize,
    ) -> Option<String> {
        let key = name.to_string();
        if let Some(done) = self.written.get(&key) {
            return done.clone();
        }
        // Left to the engine unless it is written below.
        self.written.insert(key.clone(), None);
        if depth > DEPTH {
            return None;
        }
        let sql = backend.view_sql(&key).await.ok().flatten()?;
        let (mut query, columns) = definition(&sql)?;
        let alias = format!("__ddx_view_{}", self.named);
        self.named += 1;
        // Its own CTEs, hoisted beside it under names of their own.
        let mut hoisted = Vec::new();
        if let Some(w) = query.with.take() {
            if w.recursive {
                return None;
            }
            let mut inner = Vec::new();
            for c in &w.cte_tables {
                let _ = c.query.visit(&mut Defined(&mut inner));
            }
            for (i, mut c) in w.cte_tables.into_iter().enumerate() {
                let name = c.alias.name.value.clone();
                if inner.iter().any(|d| d.eq_ignore_ascii_case(&name)) {
                    return None;
                }
                let new = format!("{alias}_{i}_{name}");
                let from = ObjectName::from(vec![c.alias.name.clone()]);
                rename(&mut query, &from, &new);
                c.alias.name = quoted(&new);
                hoisted.push((from, c));
            }
            // Each CTE reads those before it by their new names.
            let renames: Vec<(ObjectName, String)> = hoisted
                .iter()
                .map(|(from, c)| (from.clone(), c.alias.name.value.clone()))
                .collect();
            for (i, (_, c)) in hoisted.iter_mut().enumerate() {
                for (from, new) in &renames[..i] {
                    rename(&mut c.query, from, new);
                }
            }
        }
        // The views it reads, written before it. A name its own CTEs had is
        // renamed already, so what is left reads the catalog.
        let hoisted_names: Vec<String> = hoisted
            .iter()
            .map(|(_, c)| c.alias.name.value.clone())
            .collect();
        let mut reads = names(&query);
        for (_, c) in &hoisted {
            reads.extend(names(&*c.query));
        }
        reads.retain(|n| match n.0.as_slice() {
            [ObjectNamePart::Identifier(id)] => !hoisted_names.contains(&id.value),
            _ => true,
        });
        for read in reads {
            if let Some(inner) = Box::pin(self.write(backend, &read, depth + 1)).await {
                rename(&mut query, &read, &inner);
                for (_, c) in hoisted.iter_mut() {
                    rename(&mut c.query, &read, &inner);
                }
            }
        }
        self.ctes.extend(hoisted.into_iter().map(|(_, c)| c));
        self.ctes.push(Cte {
            alias: TableAlias {
                explicit: false,
                name: quoted(&alias),
                columns,
                at: None,
            },
            query: Box::new(query),
            from: None,
            materialized: None,
            closing_paren_token:
                ddx_core::sqlparser::ast::helpers::attached_token::AttachedToken::empty(),
        });
        self.written.insert(key, Some(alias.clone()));
        Some(alias)
    }
}

/// The query of a view's SQL, `CREATE VIEW … AS <query>` or the query
/// alone, and its column list; `None` if it does not parse as either.
fn definition(sql: &str) -> Option<(Query, Vec<TableAliasColumnDef>)> {
    let mut statements = Parser::parse_sql(&GenericDialect {}, sql).ok()?;
    if statements.len() != 1 {
        return None;
    }
    match statements.remove(0) {
        Statement::CreateView(v) => {
            let columns = v
                .columns
                .into_iter()
                .map(|c| TableAliasColumnDef {
                    name: c.name,
                    data_type: None,
                })
                .collect();
            Some((*v.query, columns))
        }
        Statement::Query(q) => Some((*q, Vec::new())),
        _ => None,
    }
}

/// Every relation name `node` reads, each once.
fn names<V: Visit>(node: &V) -> Vec<ObjectName> {
    struct Names(Vec<ObjectName>);
    impl Visitor for Names {
        type Break = ();
        fn pre_visit_relation(&mut self, name: &ObjectName) -> ControlFlow<()> {
            if !self.0.contains(name) {
                self.0.push(name.clone());
            }
            ControlFlow::Continue(())
        }
    }
    let mut n = Names(Vec::new());
    let _ = node.visit(&mut n);
    n.0
}

/// Pushes the name of every CTE in what it visits.
struct Defined<'a>(&'a mut Vec<String>);

impl Visitor for Defined<'_> {
    type Break = ();
    fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<()> {
        if let Some(w) = &query.with {
            self.0
                .extend(w.cte_tables.iter().map(|c| c.alias.name.value.clone()));
        }
        ControlFlow::Continue(())
    }
}

/// `node` with each read of `from`, as written, a read of `to`.
fn rename<V: VisitMut>(node: &mut V, from: &ObjectName, to: &str) {
    struct Rename<'a> {
        from: &'a ObjectName,
        to: Ident,
    }
    impl VisitorMut for Rename<'_> {
        type Break = ();
        fn pre_visit_relation(&mut self, name: &mut ObjectName) -> ControlFlow<()> {
            if same(name, self.from) {
                *name = ObjectName::from(vec![self.to.clone()]);
            }
            ControlFlow::Continue(())
        }
    }
    let _ = node.visit(&mut Rename {
        from,
        to: quoted(to),
    });
}

/// Whether two names are written alike: part for part, unquoted parts
/// without case, quoted ones exactly.
fn same(a: &ObjectName, b: &ObjectName) -> bool {
    a.0.len() == b.0.len()
        && a.0.iter().zip(&b.0).all(|(x, y)| match (x, y) {
            (ObjectNamePart::Identifier(x), ObjectNamePart::Identifier(y)) => {
                match (x.quote_style, y.quote_style) {
                    (None, None) => x.value.eq_ignore_ascii_case(&y.value),
                    (Some(_), Some(_)) => x.value == y.value,
                    _ => false,
                }
            }
            _ => false,
        })
}
