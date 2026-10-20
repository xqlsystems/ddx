// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Nodes ddx builds into the queries it differentiates and the statements
//! it rewrites, rather than writing SQL text and parsing it.
//!
//! Every node built here has no place in any statement: its tokens and
//! names are unplaced ([`Span::empty`]). The statement's nodes are named by
//! where they start ([`super::find::Site`]), and an unplaced node never is,
//! so one ddx built is never taken for one of the statement's, wherever it
//! ends up.

use ddx_core::sqlparser::ast::helpers::attached_token::AttachedToken;
use std::ops::ControlFlow;

use ddx_core::sqlparser::ast::{
    Cte, Expr, GroupByExpr, Ident, ObjectName, ObjectNamePart, Query, Select, SelectFlavor,
    SelectItem, SetExpr, TableAlias, TableFactor, TableWithJoins, VisitMut, VisitorMut,
    WildcardAdditionalOptions, With,
};
use ddx_core::sqlparser::tokenizer::Span;

/// `SELECT <projection> FROM <from>`.
pub(super) fn select(projection: Vec<SelectItem>, from: TableWithJoins) -> Select {
    Select {
        select_token: AttachedToken::empty(),
        optimizer_hints: Vec::new(),
        distinct: None,
        select_modifiers: None,
        top: None,
        top_before_distinct: false,
        projection,
        exclude: None,
        into: None,
        from: vec![from],
        lateral_views: Vec::new(),
        prewhere: None,
        selection: None,
        connect_by: Vec::new(),
        group_by: GroupByExpr::Expressions(Vec::new(), Vec::new()),
        cluster_by: Vec::new(),
        distribute_by: Vec::new(),
        sort_by: Vec::new(),
        having: None,
        named_window: Vec::new(),
        qualify: None,
        window_before_qualify: false,
        value_table_mode: None,
        flavor: SelectFlavor::Standard,
    }
}

/// `SELECT * FROM <table>`.
pub(super) fn select_star(table: ObjectName) -> Select {
    select(
        vec![SelectItem::Wildcard(WildcardAdditionalOptions::default())],
        TableWithJoins {
            relation: self::table(table, None),
            joins: Vec::new(),
        },
    )
}

/// `<with> <select>`: a query with nothing after its `SELECT`.
pub(super) fn query(with: Option<With>, select: Select) -> Query {
    Query {
        with,
        body: Box::new(SetExpr::Select(Box::new(select))),
        order_by: None,
        limit_clause: None,
        fetch: None,
        locks: Vec::new(),
        for_clause: None,
        settings: None,
        format_clause: None,
        pipe_operators: Vec::new(),
    }
}

/// `WITH <ctes>`, or `None` for no CTEs.
pub(super) fn with(ctes: Vec<Cte>) -> Option<With> {
    (!ctes.is_empty()).then(|| With {
        with_token: AttachedToken::empty(),
        recursive: false,
        cte_tables: ctes,
    })
}

/// The table `name`, read as it is.
pub(super) fn table(name: ObjectName, alias: Option<TableAlias>) -> TableFactor {
    TableFactor::Table {
        name,
        alias,
        args: None,
        with_hints: Vec::new(),
        version: None,
        with_ordinality: false,
        partitions: Vec::new(),
        json_path: None,
        sample: None,
        index_hints: Vec::new(),
    }
}

/// `AS <name>`.
pub(super) fn named(name: Ident) -> Option<TableAlias> {
    Some(TableAlias {
        explicit: true,
        name,
        columns: Vec::new(),
        at: None,
    })
}

/// `name` as a quoted identifier.
pub(super) fn quoted(name: &str) -> Ident {
    Ident::with_quote('"', name)
}

/// `node` with each table it reads that `to` names anew read under that
/// name instead. A read with no alias keeps its old name, the last part,
/// as one, so a column qualified by it (`model.v`) still resolves, and a
/// column qualified by all of a longer one (`public.model.v`) is qualified
/// by the alias instead.
pub(super) fn rename_reads<V: VisitMut>(
    node: &mut V,
    to: impl FnMut(&ObjectName) -> Option<Ident>,
) {
    struct Rename<F> {
        to: F,
        /// Each name of more than one part read with no alias, renamed.
        qualified: Vec<ObjectName>,
    }
    impl<F: FnMut(&ObjectName) -> Option<Ident>> VisitorMut for Rename<F> {
        type Break = ();
        fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<()> {
            if let TableFactor::Table { name, alias, .. } = factor {
                if let Some(new) = (self.to)(name) {
                    if alias.is_none() {
                        if let Some(ObjectNamePart::Identifier(old)) = name.0.last() {
                            *alias = Some(TableAlias {
                                explicit: true,
                                name: unplaced(old),
                                columns: Vec::new(),
                                at: None,
                            });
                        }
                        if name.0.len() > 1 && !self.qualified.iter().any(|q| same(q, name)) {
                            self.qualified.push(name.clone());
                        }
                    }
                    *name = ObjectName::from(vec![new]);
                }
            }
            ControlFlow::Continue(())
        }
    }
    let mut rename = Rename {
        to,
        qualified: Vec::new(),
    };
    let _ = node.visit(&mut rename);
    if rename.qualified.is_empty() {
        return;
    }
    // A select's columns come before its FROM, so they are requalified
    // once every read is renamed.
    struct Requalify(Vec<ObjectName>);
    impl VisitorMut for Requalify {
        type Break = ();
        fn pre_visit_expr(&mut self, expr: &mut Expr) -> ControlFlow<()> {
            if let Expr::CompoundIdentifier(parts) = expr {
                for q in &self.0 {
                    let n = q.0.len();
                    let prefix = ObjectName::from(parts.get(..n).unwrap_or_default().to_vec());
                    if parts.len() > n && same(&prefix, q) {
                        parts.drain(..n - 1);
                        break;
                    }
                }
            }
            ControlFlow::Continue(())
        }
    }
    let _ = node.visit(&mut Requalify(rename.qualified));
}

/// Whether two names are written alike: part for part, unquoted parts
/// without case, quoted ones exactly.
pub(super) fn same(a: &ObjectName, b: &ObjectName) -> bool {
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

/// `ident`, unplaced: as the statement writes it, quotes included, but
/// with no place in it.
pub(super) fn unplaced(ident: &Ident) -> Ident {
    Ident {
        span: Span::empty(),
        ..ident.clone()
    }
}
