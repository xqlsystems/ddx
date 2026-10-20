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
    Cte, ObjectName, ObjectNamePart, Query, Statement, TableAlias, TableAliasColumnDef, Visit,
    VisitMut, Visitor, With,
};
use ddx_core::sqlparser::dialect::GenericDialect;
use ddx_core::sqlparser::parser::Parser;

use super::build::{quoted, rename_reads};
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

    /// The next name for a view's CTE that neither it nor the names of its
    /// own CTEs, `{name}_{i}_{cte}`, can make the same as one the objective
    /// defines.
    fn fresh(&mut self) -> String {
        loop {
            let name = format!("__ddx_view_{}", self.named);
            self.named += 1;
            let prefix = format!("{name}_");
            let taken = self.defined.iter().any(|d| {
                d.eq_ignore_ascii_case(&name)
                    || d.get(..prefix.len())
                        .is_some_and(|p| p.eq_ignore_ascii_case(&prefix))
            });
            if !taken {
                return name;
            }
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
        let alias = self.fresh();
        // Its own CTEs, hoisted beside it under names of their own.
        let mut hoisted = Vec::new();
        if let Some(w) = query.with.take() {
            if w.recursive {
                return None;
            }
            // A name defined again inside the view, in a CTE or its query,
            // may be read as either: the view is left to the engine.
            let mut inner = Vec::new();
            for c in &w.cte_tables {
                let _ = c.query.visit(&mut Defined(&mut inner));
            }
            let _ = query.visit(&mut Defined(&mut inner));
            // So is one read under another spelling (`a` and `"a"`), which
            // engines resolve differently.
            let mut reads = names(&query);
            for c in &w.cte_tables {
                reads.extend(names(&*c.query));
            }
            for c in &w.cte_tables {
                let own = ObjectName::from(vec![c.alias.name.clone()]);
                let misspelled = reads.iter().any(|r| match r.0.as_slice() {
                    [ObjectNamePart::Identifier(id)] => {
                        id.value.eq_ignore_ascii_case(&c.alias.name.value) && !same(r, &own)
                    }
                    _ => false,
                });
                if misspelled {
                    return None;
                }
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
        // Nor is a name a CTE nested in it has, at any depth: that read may
        // be of the CTE, and renaming it would read the catalog instead.
        let mut nested = Vec::new();
        let _ = query.visit(&mut Defined(&mut nested));
        for (_, c) in &hoisted {
            let _ = c.query.visit(&mut Defined(&mut nested));
        }
        reads.retain(|n| match n.0.as_slice() {
            [ObjectNamePart::Identifier(id)] => {
                !hoisted_names.contains(&id.value)
                    && !nested.iter().any(|d| d.eq_ignore_ascii_case(&id.value))
            }
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
    rename_reads(node, |name| same(name, from).then(|| quoted(to)));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::Bound;
    use substrait::proto::{NamedStruct, Plan};

    /// A backend that knows only views' SQL.
    struct Views(Vec<(&'static str, &'static str)>);

    impl Backend for Views {
        type Error = String;
        async fn table_schema(&mut self, _: &str) -> Result<NamedStruct, String> {
            Err("no tables".into())
        }
        async fn returns_rows(&mut self, _: &[Bound]) -> Result<Vec<bool>, String> {
            Err("no runs".into())
        }
        async fn materialize(&mut self, _: &str, _: &Bound) -> Result<(), String> {
            Err("no runs".into())
        }
        async fn drop_table(&mut self, _: &str) -> Result<(), String> {
            Err("no runs".into())
        }
        async fn plan_sql(&mut self, _: &str) -> Result<Plan, String> {
            Err("no plans".into())
        }
        async fn view_sql(&mut self, name: &str) -> Result<Option<String>, String> {
            Ok(self
                .0
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, sql)| sql.to_string()))
        }
    }

    fn query(sql: &str) -> Query {
        match Parser::parse_sql(&GenericDialect {}, sql)
            .unwrap()
            .remove(0)
        {
            Statement::Query(q) => *q,
            other => panic!("{other}"),
        }
    }

    fn expanded(views: Vec<(&'static str, &'static str)>, sql: &str) -> Option<String> {
        let mut backend = Views(views);
        let q = query(sql);
        let fut = expand(&mut backend, &q);
        ready(fut).map(|q| q.to_string())
    }

    /// The futures here never wait: each backend call is ready at once.
    fn ready<F: std::future::Future>(f: F) -> F::Output {
        use std::pin::pin;
        use std::task::{Context, Poll, Waker};
        match pin!(f).poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(out) => out,
            Poll::Pending => unreachable!("a stub backend never waits"),
        }
    }

    #[test]
    fn a_name_a_cte_nested_in_a_view_has_is_not_read_as_a_view() {
        // x is a view, but v's subquery defines a CTE x and reads that.
        let got = expanded(
            vec![
                (
                    "v",
                    "CREATE VIEW v AS SELECT * FROM (WITH x AS (SELECT 1 AS c) SELECT c FROM x) s",
                ),
                ("x", "CREATE VIEW x AS SELECT 2 AS c"),
            ],
            "SELECT * FROM v",
        )
        .unwrap();
        assert_eq!(
            got,
            "WITH \"__ddx_view_0\" AS (SELECT * FROM (WITH x AS (SELECT 1 AS c) SELECT c FROM x) s) \
             SELECT * FROM \"__ddx_view_0\" AS v"
        );
    }

    #[test]
    fn a_view_takes_no_name_the_objective_defines() {
        // The objective defines the first view name, and one the second
        // view's own CTE would take: each view takes the next free name.
        let got = expanded(
            vec![(
                "v",
                "CREATE VIEW v AS WITH a AS (SELECT 1 AS x) SELECT x FROM a",
            )],
            "WITH \"__ddx_view_0\" AS (SELECT 2 AS x), \"__ddx_view_1_0_a\" AS (SELECT 3 AS x) \
             SELECT * FROM v, \"__ddx_view_0\", \"__ddx_view_1_0_a\"",
        )
        .unwrap();
        assert!(
            got.starts_with(
                "WITH \"__ddx_view_2_0_a\" AS (SELECT 1 AS x), \"__ddx_view_2\" AS \
                 (SELECT x FROM \"__ddx_view_2_0_a\" AS a), \"__ddx_view_0\" AS"
            ),
            "{got}"
        );
    }

    #[test]
    fn a_view_is_written_as_its_ctes_then_its_query() {
        let got = expanded(
            vec![(
                "v",
                "CREATE VIEW v AS WITH a AS (SELECT 1 AS x) SELECT x FROM a",
            )],
            "SELECT * FROM v",
        );
        assert_eq!(
            got.as_deref(),
            Some(
                "WITH \"__ddx_view_0_0_a\" AS (SELECT 1 AS x), \"__ddx_view_0\" AS \
                 (SELECT x FROM \"__ddx_view_0_0_a\" AS a) SELECT * FROM \"__ddx_view_0\" AS v"
            )
        );
    }

    #[test]
    fn a_view_whose_query_defines_a_cte_s_name_again_is_left_to_the_engine() {
        // Its query's inner `a` is read as itself, which renaming every read
        // of `a` would break.
        let got = expanded(
            vec![(
                "v",
                "CREATE VIEW v AS WITH a AS (SELECT 1 AS x) \
                 SELECT * FROM (WITH a AS (SELECT 2 AS x) SELECT * FROM a) s",
            )],
            "SELECT * FROM v",
        );
        assert_eq!(got, None);
    }

    #[test]
    fn a_view_that_reads_its_cte_under_another_spelling_is_left_to_the_engine() {
        let got = expanded(
            vec![(
                "v",
                "CREATE VIEW v AS WITH a AS (SELECT 1 AS x) SELECT * FROM \"a\"",
            )],
            "SELECT * FROM v",
        );
        assert_eq!(got, None);
    }

    #[test]
    fn views_that_read_views_take_names_of_their_own() {
        let got = expanded(
            vec![
                (
                    "outer_v",
                    "CREATE VIEW outer_v AS SELECT 2 * x AS x FROM inner_v",
                ),
                ("inner_v", "CREATE VIEW inner_v AS SELECT 1 AS x"),
            ],
            "SELECT * FROM outer_v",
        )
        .unwrap();
        assert_eq!(
            got,
            "WITH \"__ddx_view_1\" AS (SELECT 1 AS x), \"__ddx_view_0\" AS \
             (SELECT 2 * x AS x FROM \"__ddx_view_1\" AS inner_v) \
             SELECT * FROM \"__ddx_view_0\" AS outer_v"
        );
    }
}
