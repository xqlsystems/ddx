// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Function names in a plan.
//!
//! A Substrait plan refers to every function, `add` and `sum` as much as a
//! user's own, by a plan-local integer anchor. The plan's `extensions` list maps
//! each anchor to a name. [`Functions`] reads that list once.
//!
//! Functions are recognized **by name**. The spec would have each declared
//! against an extension URN, but DataFusion 54 declares every function, its
//! own built-ins included, with a bare name and no URN
//! (`extension_urn_reference = u32::MAX`). Names are compared case-folded, and a
//! `:signature` suffix (`add:fp64_fp64`, the compound-name form of the spec) is
//! ignored.
//!
//! # `ddx_stop_gradient`
//!
//! ddx claims one function name in the forward query: [`STOP_GRADIENT`], the
//! identity at runtime and a constant to differentiation, like JAX's
//! `lax.stop_gradient`. It is the only thing a query says to ddx. Everything
//! else, which aggregates sum and which joins broadcast, ddx reads off the
//! operators themselves, since each operator's derivative follows from what it
//! computes and needs no label.

use std::collections::HashMap;

use substrait::proto::extensions::simple_extension_declaration::{ExtensionFunction, MappingType};
use substrait::proto::Plan;

use crate::error::{AdError, Result};

/// The SQL name of the stop-gradient function: `ddx_stop_gradient(x)` is `x`,
/// and no gradient flows into `x` through it.
pub const STOP_GRADIENT: &str = "ddx_stop_gradient";

/// A function name as ddx compares it: lower-cased, without a `:signature`
/// suffix.
pub fn normalize(name: &str) -> String {
    let base = name.split(':').next().unwrap_or(name);
    base.to_ascii_lowercase()
}

/// The function anchors a plan declares, and their names.
#[derive(Debug, Clone, Default)]
pub struct Functions {
    names: HashMap<u32, String>,
    declarations: Vec<ExtensionFunction>,
}

impl Functions {
    /// Read the declarations out of `plan`.
    ///
    /// Two declarations of the same anchor are refused: every later lookup
    /// would otherwise depend on which one won.
    pub fn from_plan(plan: &Plan) -> Result<Self> {
        let mut names = HashMap::new();
        let mut declarations = Vec::new();
        for ext in &plan.extensions {
            if let Some(MappingType::ExtensionFunction(f)) = &ext.mapping_type {
                declarations.push(f.clone());
                if names
                    .insert(f.function_anchor, normalize(&f.name))
                    .is_some()
                {
                    return Err(AdError::InvalidPlan(format!(
                        "function anchor {} is declared twice",
                        f.function_anchor
                    )));
                }
            }
        }
        Ok(Functions {
            names,
            declarations,
        })
    }

    /// The normalized name behind `anchor`.
    pub fn name(&self, anchor: u32) -> Result<&str> {
        self.names.get(&anchor).map(String::as_str).ok_or_else(|| {
            AdError::InvalidPlan(format!("function anchor {anchor} is not declared"))
        })
    }

    /// Is `anchor` [`STOP_GRADIENT`]?
    pub fn is_stop_gradient(&self, anchor: u32) -> Result<bool> {
        Ok(self.name(anchor)? == STOP_GRADIENT)
    }

    /// The declarations, as the plan gave them.
    pub fn declarations(&self) -> &[ExtensionFunction] {
        &self.declarations
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use substrait::proto::extensions::SimpleExtensionDeclaration;

    pub(crate) fn plan_declaring(names: &[(u32, &str)]) -> Plan {
        Plan {
            extensions: names
                .iter()
                .map(|(anchor, name)| SimpleExtensionDeclaration {
                    mapping_type: Some(MappingType::ExtensionFunction(ExtensionFunction {
                        extension_urn_reference: u32::MAX,
                        function_anchor: *anchor,
                        name: name.to_string(),
                    })),
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn names_are_normalized_case_folded_and_without_a_signature() {
        assert_eq!(normalize("DDX_Stop_Gradient"), STOP_GRADIENT);
        assert_eq!(normalize("sum:fp64"), "sum");
    }

    #[test]
    fn the_function_table_maps_anchors_to_normalized_names() {
        let plan = plan_declaring(&[(0, "multiply"), (7, "DDX_STOP_GRADIENT"), (3, "sum:fp64")]);
        let f = Functions::from_plan(&plan).unwrap();
        assert_eq!(f.name(0).unwrap(), "multiply");
        assert_eq!(f.name(3).unwrap(), "sum");
        assert!(f.is_stop_gradient(7).unwrap());
        assert!(!f.is_stop_gradient(0).unwrap());
    }

    #[test]
    fn an_undeclared_anchor_is_an_error() {
        let f = Functions::from_plan(&plan_declaring(&[(0, "add")])).unwrap();
        assert!(matches!(f.name(9), Err(AdError::InvalidPlan(_))));
    }

    #[test]
    fn a_doubly_declared_anchor_is_an_error() {
        let plan = plan_declaring(&[(1, "add"), (1, "ddx_stop_gradient")]);
        assert!(matches!(
            Functions::from_plan(&plan),
            Err(AdError::InvalidPlan(_))
        ));
    }
}
