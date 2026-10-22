//! Spike (not for merging): how faithfully polyglot-sql regenerates the SQL
//! `ddx_ad::to_sql` writes.
//!
//! Reads `query<TAB>sql` lines, where `sql` is `to_sql`'s DuckDB SQL for
//! `query`, and prints `query<TAB>sql'`, with `sql'` parsed by polyglot as
//! DuckDB and generated for the dialect named by the second argument
//! (`duckdb` or `datafusion`). Run `sql'` on that engine and compare its rows
//! with `query`'s. On the 55 queries of `crates/ddx-datafusion/tests/to_sql.rs`
//! (2026-10-21): DuckDB to DuckDB kept every result; DuckDB to DataFusion
//! kept 53 of 55, refusing `//` with operand types it couldn't infer, and
//! leaving `sign` unrenamed (DataFusion spells it `signum`).
//!
//! `cargo run -p ddx-ad --example polyglot_roundtrip -- cases.tsv datafusion`
use polyglot_sql::{transpile, DialectType};

fn main() {
    let path = std::env::args().nth(1).expect("a file of query<TAB>sql lines");
    let to = match std::env::args().nth(2).as_deref() {
        Some("datafusion") => DialectType::DataFusion,
        _ => DialectType::DuckDB,
    };
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let Some((query, sql)) = line.split_once('\t') else {
            continue;
        };
        match transpile(sql, DialectType::DuckDB, to) {
            Ok(v) => println!("{query}\t{}", v.join(" ").replace('\n', " ")),
            Err(e) => println!("{query}\tERROR {}", e.to_string().replace('\n', " ")),
        }
    }
}
