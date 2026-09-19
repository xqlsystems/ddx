// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The four marker functions, and the plan's function table they're found in.
//!
//! A marker is an identity scalar function whose only job is to survive planning
//! and tag the operation around it (design.md §4.2). In a Substrait plan it's an
//! ordinary `scalar_function` whose `function_reference` points at an extension
//! declaration carrying the marker's name.
//!
//! Markers are recognized **by name**. Neither target engine emits an extension
//! URN for a user-registered function: DataFusion 54 declares every function —
//! its own built-ins and `ddx_contract_mark` alike — with a bare name and
//! `extension_urn_reference = u32::MAX` (checked against real producer output
//! while building this; design.md `[S2]` found the same of DuckDB). A URN can't
//! identify a marker when there is none to read.

use std::collections::HashMap;

use substrait::proto::extensions::simple_extension_declaration::MappingType;
use substrait::proto::Plan;

use crate::error::{AdError, Result};

/// One of ddx's extension-function markers (design.md §4.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Marker {
    /// `SUM(ddx_contract_mark(a.val * b.val))` — this aggregate is a
    /// contraction of the join feeding it.
    Contract,
    /// `SUM(ddx_reduce_mark(val))` — this aggregate is a plain reduction over
    /// the dims it drops.
    Reduce,
    /// `ddx_route_mark(val)` — this value is selected by a top-1-per-group
    /// window (argmax/argmin routing).
    Route,
    /// `ddx_stop_gradient(x)` — no cotangent flows into `x`.
    StopGradient,
}

impl Marker {
    /// Every marker, in a fixed order.
    pub const ALL: [Marker; 4] = [
        Marker::Contract,
        Marker::Reduce,
        Marker::Route,
        Marker::StopGradient,
    ];

    /// The function name a user writes in SQL.
    pub fn name(self) -> &'static str {
        match self {
            Marker::Contract => "ddx_contract_mark",
            Marker::Reduce => "ddx_reduce_mark",
            Marker::Route => "ddx_route_mark",
            Marker::StopGradient => "ddx_stop_gradient",
        }
    }

    /// The marker a declared function name refers to, if any.
    ///
    /// Case-folded, because SQL function names are case-insensitive. A Substrait
    /// compound name (`ddx_contract_mark:fp64`) is matched on the part before
    /// the signature.
    pub fn from_name(name: &str) -> Option<Marker> {
        let base = base_name(name);
        Marker::ALL
            .into_iter()
            .find(|m| m.name().eq_ignore_ascii_case(base))
    }
}

/// A function name without its Substrait signature suffix: `add:i64_i64` → `add`.
pub(crate) fn base_name(name: &str) -> &str {
    name.split(':').next().unwrap_or(name)
}

/// The plan's scalar/aggregate/window function declarations, by anchor.
#[derive(Debug, Clone, Default)]
pub struct Functions {
    by_anchor: HashMap<u32, String>,
}

impl Functions {
    /// Read the function declarations out of `plan.extensions`.
    pub fn from_plan(plan: &Plan) -> Result<Self> {
        let mut by_anchor = HashMap::new();
        for ext in &plan.extensions {
            if let Some(MappingType::ExtensionFunction(f)) = &ext.mapping_type {
                if by_anchor
                    .insert(f.function_anchor, f.name.clone())
                    .is_some()
                {
                    return Err(AdError::InvalidPlan(format!(
                        "function anchor {} is declared twice",
                        f.function_anchor
                    )));
                }
            }
        }
        Ok(Functions { by_anchor })
    }

    /// The declared name of the function at `anchor`, signature suffix and all.
    pub fn name(&self, anchor: u32) -> Result<&str> {
        self.by_anchor
            .get(&anchor)
            .map(String::as_str)
            .ok_or_else(|| {
                AdError::InvalidPlan(format!("function anchor {anchor} is never declared"))
            })
    }

    /// The declared name without its signature suffix, lower-cased — the form
    /// to compare against (`sum`, `multiply`, `tanh`).
    pub fn base_name(&self, anchor: u32) -> Result<String> {
        Ok(base_name(self.name(anchor)?).to_ascii_lowercase())
    }

    /// The marker the function at `anchor` is, if it is one.
    pub fn marker(&self, anchor: u32) -> Result<Option<Marker>> {
        Ok(Marker::from_name(self.name(anchor)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use substrait::proto::extensions::simple_extension_declaration::ExtensionFunction;
    use substrait::proto::extensions::SimpleExtensionDeclaration;

    fn declare(anchor: u32, name: &str) -> SimpleExtensionDeclaration {
        SimpleExtensionDeclaration {
            mapping_type: Some(MappingType::ExtensionFunction(ExtensionFunction {
                function_anchor: anchor,
                name: name.into(),
                extension_urn_reference: u32::MAX,
            })),
        }
    }

    #[test]
    fn names_round_trip() {
        for m in Marker::ALL {
            assert_eq!(Marker::from_name(m.name()), Some(m));
        }
    }

    #[test]
    fn names_are_matched_case_insensitively_and_without_a_signature() {
        assert_eq!(
            Marker::from_name("DDX_Contract_Mark"),
            Some(Marker::Contract)
        );
        assert_eq!(
            Marker::from_name("ddx_reduce_mark:fp64"),
            Some(Marker::Reduce)
        );
        assert_eq!(Marker::from_name("ddx_contract"), None);
        assert_eq!(Marker::from_name("sum"), None);
    }

    #[test]
    fn reads_the_function_table() {
        let plan = Plan {
            extensions: vec![declare(0, "sum"), declare(7, "ddx_stop_gradient")],
            ..Default::default()
        };
        let fns = Functions::from_plan(&plan).unwrap();
        assert_eq!(fns.base_name(0).unwrap(), "sum");
        assert_eq!(fns.marker(0).unwrap(), None);
        assert_eq!(fns.marker(7).unwrap(), Some(Marker::StopGradient));
        assert!(matches!(fns.name(3), Err(AdError::InvalidPlan(_))));
    }

    #[test]
    fn refuses_a_doubly_declared_anchor() {
        let plan = Plan {
            extensions: vec![declare(1, "sum"), declare(1, "ddx_contract_mark")],
            ..Default::default()
        };
        assert!(matches!(
            Functions::from_plan(&plan),
            Err(AdError::InvalidPlan(_))
        ));
    }
}
