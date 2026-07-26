// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

use sqlparser::ast::{Expr, BinaryOperator, Ident, Spanned};
use sqlparser::tokenizer::Span;

fn ident(s: &str) -> Box<Expr> { Box::new(Expr::Identifier(Ident::new(s))) }
fn bin(l: Box<Expr>, op: BinaryOperator, r: Box<Expr>) -> Box<Expr> {
    Box::new(Expr::BinaryOp { left: l, op, right: r })
}
fn nest(e: Box<Expr>) -> Box<Expr> { Box::new(Expr::Nested(e)) }

fn main() {
    let a_plus_b = bin(ident("a"), BinaryOperator::Plus, ident("b"));
    let expr = bin(a_plus_b.clone(), BinaryOperator::Multiply, ident("c"));
    println!("G1 constructed (a+b)*c   Display => {}", expr);
    let fixed = bin(nest(a_plus_b), BinaryOperator::Multiply, ident("c"));
    println!("G1 fixed  Nested(a+b)*c  Display => {}", fixed);

    let b_plus_c = bin(ident("b"), BinaryOperator::Plus, ident("c"));
    let sub = bin(ident("a"), BinaryOperator::Minus, b_plus_c.clone());
    println!("G1 constructed a-(b+c)   Display => {}", sub);
    let subf = bin(ident("a"), BinaryOperator::Minus, nest(b_plus_c));
    println!("G1 fixed  a-Nested(b+c)  Display => {}", subf);

    let sql = "SELECT 'héllo', grad(x, x) FROM t";
    let dialect = sqlparser::dialect::DuckDbDialect {};
    let stmts = sqlparser::parser::Parser::parse_sql(&dialect, sql).unwrap();
    let sp: Span = stmts[0].span();
    println!("G3 stmt span start: line={} column={}", sp.start.line, sp.start.column);
    let byte_of_grad = sql.find("grad").unwrap();
    let char_of_grad = sql[..byte_of_grad].chars().count();
    println!("G3 'grad' byte offset={}, char offset={} (differ by the multibyte é)", byte_of_grad, char_of_grad);
}
