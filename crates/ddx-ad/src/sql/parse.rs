// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Parsing a statement, with a CTE's materialization hint (`AS MATERIALIZED`
//! or `AS NOT MATERIALIZED`) in any dialect. DuckDB and Postgres both take
//! it, but `sqlparser` parses it only in its Postgres dialect, so in another
//! a statement with one would not parse, and its calls would go unfound. A
//! hint changes how the engine runs a CTE, not what it means, so the
//! statement is parsed with each hint blanked out (every position kept),
//! and the hint is put back on its CTE for the statement the engine runs.

use std::collections::HashMap;
use std::ops::ControlFlow;

use ddx_core::sqlparser::ast::{CteAsMaterialized, Query, Statement, VisitMut, VisitorMut};
use ddx_core::sqlparser::dialect::Dialect;
use ddx_core::sqlparser::keywords::Keyword;
use ddx_core::sqlparser::parser::{Parser, ParserError};
use ddx_core::sqlparser::tokenizer::{Location, Token, TokenWithSpan, Tokenizer};

/// `sql`'s statements, as `Parser::parse_sql` gives them, a CTE's
/// materialization hint parsed in any dialect (see the module docs).
pub(super) fn parse(dialect: &dyn Dialect, sql: &str) -> Result<Vec<Statement>, ParserError> {
    let error = match Parser::parse_sql(dialect, sql) {
        Ok(statements) => return Ok(statements),
        Err(e) => e,
    };
    let Some((blanked, hints)) = without_hints(dialect, sql) else {
        return Err(error);
    };
    let mut statements = Parser::parse_sql(dialect, &blanked).map_err(|_| error)?;
    let _ = statements.visit(&mut Hints(&hints));
    Ok(statements)
}

/// `sql` with each CTE's hint blanked out, and each hint by where its CTE's
/// name starts; `None` if it has none.
fn without_hints(
    dialect: &dyn Dialect,
    sql: &str,
) -> Option<(String, HashMap<Location, CteAsMaterialized>)> {
    let tokens = Tokenizer::new(dialect, sql).tokenize_with_location().ok()?;
    // The tokens that mean something, without whitespace and comments.
    let solid: Vec<&TokenWithSpan> = tokens
        .iter()
        .filter(|t| !matches!(t.token, Token::Whitespace(_)))
        .collect();
    let keyword = |i: usize, k: Keyword| matches!(solid.get(i), Some(TokenWithSpan { token: Token::Word(w), .. }) if w.keyword == k);
    let mut hints = HashMap::new();
    let mut blank: Vec<&TokenWithSpan> = Vec::new();
    for at in 0..solid.len() {
        if !keyword(at, Keyword::AS) {
            continue;
        }
        let (hint, end) = if keyword(at + 1, Keyword::MATERIALIZED) {
            (CteAsMaterialized::Materialized, at + 1)
        } else if keyword(at + 1, Keyword::NOT) && keyword(at + 2, Keyword::MATERIALIZED) {
            (CteAsMaterialized::NotMaterialized, at + 2)
        } else {
            continue;
        };
        if !matches!(solid.get(end + 1).map(|t| &t.token), Some(Token::LParen)) {
            continue;
        }
        let Some(name) = cte_name(&solid, at) else {
            continue;
        };
        hints.insert(name, hint);
        blank.extend(&solid[at + 1..=end]);
    }
    if hints.is_empty() {
        return None;
    }
    let lines = line_starts(sql);
    let mut blanked = sql.to_string();
    for token in blank {
        let (start, end) = (
            offset(sql, &lines, token.span.start)?,
            offset(sql, &lines, token.span.end)?,
        );
        // Keywords are ASCII, so the spaces keep every byte where it was.
        blanked.replace_range(start..end, &" ".repeat(end - start));
    }
    Some((blanked, hints))
}

/// Where the name of the CTE whose `AS` is `solid[at]` starts: the word
/// before it, or before its column list.
fn cte_name(solid: &[&TokenWithSpan], at: usize) -> Option<Location> {
    let mut i = at.checked_sub(1)?;
    if solid[i].token == Token::RParen {
        let mut depth = 0usize;
        loop {
            match solid[i].token {
                Token::RParen => depth += 1,
                Token::LParen => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            i = i.checked_sub(1)?;
        }
        i = i.checked_sub(1)?;
    }
    matches!(solid[i].token, Token::Word(_)).then_some(solid[i].span.start)
}

/// The byte each line of `sql` starts at.
fn line_starts(sql: &str) -> Vec<usize> {
    std::iter::once(0)
        .chain(sql.match_indices('\n').map(|(i, _)| i + 1))
        .collect()
}

/// The byte offset of `at`, a 1-based line and character column.
fn offset(sql: &str, lines: &[usize], at: Location) -> Option<usize> {
    let start = *lines.get(usize::try_from(at.line).ok()?.checked_sub(1)?)?;
    let column = usize::try_from(at.column).ok()?.checked_sub(1)?;
    let line = &sql[start..];
    Some(
        start
            + line
                .char_indices()
                .nth(column)
                .map_or(line.len(), |(i, _)| i),
    )
}

/// Puts each hint back on the CTE whose name starts where it was found.
struct Hints<'a>(&'a HashMap<Location, CteAsMaterialized>);

impl VisitorMut for Hints<'_> {
    type Break = ();

    fn pre_visit_query(&mut self, query: &mut Query) -> ControlFlow<()> {
        if let Some(with) = &mut query.with {
            for cte in &mut with.cte_tables {
                if let Some(hint) = self.0.get(&cte.alias.name.span.start) {
                    cte.materialized = Some(*hint);
                }
            }
        }
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddx_core::sqlparser::dialect::{DuckDbDialect, GenericDialect};

    fn printed(dialect: &dyn Dialect, sql: &str) -> String {
        parse(dialect, sql).unwrap()[0].to_string()
    }

    #[test]
    fn a_hint_parses_in_any_dialect_and_is_kept() {
        let sql = "WITH s AS MATERIALIZED (SELECT 1 AS x), t (y) AS NOT MATERIALIZED (SELECT x FROM s) SELECT * FROM t";
        for dialect in [&DuckDbDialect {} as &dyn Dialect, &GenericDialect {}] {
            assert_eq!(
                printed(dialect, sql),
                "WITH s AS MATERIALIZED (SELECT 1 AS x), t (y) AS NOT MATERIALIZED (SELECT x FROM s) SELECT * FROM t"
            );
        }
    }

    #[test]
    fn a_hint_in_a_nested_query_or_on_another_line_is_kept() {
        let sql =
            "SELECT * FROM (WITH é AS\n  MATERIALIZED (SELECT 'ü' AS x)\nSELECT x FROM é) AS q";
        assert_eq!(
            printed(&DuckDbDialect {}, sql),
            "SELECT * FROM (WITH é AS MATERIALIZED (SELECT 'ü' AS x) SELECT x FROM é) AS q"
        );
    }

    #[test]
    fn other_syntax_errors_are_the_parser_s() {
        let sql = "WITH s AS MATERIALIZED (SELECT 1 AS x) SELECT * FROM";
        let direct = Parser::parse_sql(&DuckDbDialect {}, sql).unwrap_err();
        assert_eq!(parse(&DuckDbDialect {}, sql).unwrap_err(), direct);
        assert!(parse(&DuckDbDialect {}, "SELECT FROM WHERE").is_err());
    }

    #[test]
    fn a_column_named_materialized_is_not_a_hint() {
        let sql = "SELECT x AS materialized FROM t";
        assert_eq!(printed(&DuckDbDialect {}, sql), sql);
    }
}
