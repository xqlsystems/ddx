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
use substrait::proto::extensions::{SimpleExtensionDeclaration, SimpleExtensionUrn};
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
    /// The plan's other declarations (types, type variations), which an
    /// expression copied out of it may still refer to.
    others: Vec<SimpleExtensionDeclaration>,
    /// The extension URNs the plan's declarations point into.
    urns: Vec<SimpleExtensionUrn>,
}

impl Functions {
    /// Read the declarations out of `plan`.
    ///
    /// Two declarations of the same anchor are refused: every later lookup
    /// would otherwise depend on which one won.
    pub fn from_plan(plan: &Plan) -> Result<Self> {
        let mut names = HashMap::new();
        let mut declarations = Vec::new();
        let mut others = Vec::new();
        for ext in &plan.extensions {
            let Some(MappingType::ExtensionFunction(f)) = &ext.mapping_type else {
                others.push(ext.clone());
                continue;
            };
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
        Ok(Functions {
            names,
            declarations,
            others,
            urns: plan.extension_urns.clone(),
        })
    }

    /// The declarations of every plan in `plans`, which must agree: an anchor
    /// two of them declare names one function. The plans of a program do:
    /// each is written from one [`Extensions`], which only grows.
    pub fn union<'p>(plans: impl IntoIterator<Item = &'p Plan>) -> Result<Self> {
        let mut all = Plan::default();
        for plan in plans {
            for ext in &plan.extensions {
                if !all.extensions.contains(ext) {
                    all.extensions.push(ext.clone());
                }
            }
            for urn in &plan.extension_urns {
                if !all.extension_urns.contains(urn) {
                    all.extension_urns.push(urn.clone());
                }
            }
        }
        Functions::from_plan(&all)
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

    /// Can `anchor` give a different value each time a query is run
    /// (`random()`, `now()`)? A recomputation would not repeat it.
    pub fn is_volatile(&self, anchor: u32) -> Result<bool> {
        const VOLATILE: &[&str] = &[
            "random",
            "rand",
            "uuid",
            "gen_random_uuid",
            "now",
            "current_timestamp",
            "current_time",
            "localtimestamp",
            "localtime",
        ];
        Ok(VOLATILE.contains(&self.name(anchor)?))
    }

    /// Can the aggregate or window function `anchor` round differently from
    /// one run to the next? A sum adds in the order its partitions arrive, so
    /// its last bits can change; a maximum, a count or a rank cannot. Any
    /// function not known to be exact is assumed to round.
    pub fn rounds(&self, anchor: u32) -> Result<bool> {
        const EXACT: &[&str] = &[
            "max",
            "min",
            "count",
            "any_value",
            "first_value",
            "last_value",
            "nth_value",
            "bool_and",
            "bool_or",
            "row_number",
            "rank",
            "dense_rank",
            "ntile",
            "lag",
            "lead",
        ];
        Ok(!EXACT.contains(&self.name(anchor)?))
    }

    /// The declarations, as the plan gave them.
    pub fn declarations(&self) -> &[ExtensionFunction] {
        &self.declarations
    }
}

/// The functions a plan ddx writes declares.
///
/// It starts from the input plan's declarations, anchors unchanged, so an
/// expression copied out of the forward query keeps meaning the same thing
/// without being rewritten. A function ddx introduces (the `multiply` of a
/// chain rule, the `sum` of a transpose) reuses the input's anchor when the
/// input declares that name, and gets a fresh one otherwise.
#[derive(Debug, Clone)]
pub struct Extensions {
    declarations: Vec<ExtensionFunction>,
    others: Vec<SimpleExtensionDeclaration>,
    urns: Vec<SimpleExtensionUrn>,
    by_name: HashMap<String, u32>,
    next: u32,
}

impl Extensions {
    /// Start from `functions`' declarations.
    pub fn new(functions: &Functions) -> Self {
        let mut by_name = HashMap::new();
        for d in functions.declarations() {
            by_name
                .entry(normalize(&d.name))
                .or_insert(d.function_anchor);
        }
        let next = functions
            .declarations()
            .iter()
            .map(|d| d.function_anchor + 1)
            .max()
            .unwrap_or(0);
        Extensions {
            declarations: functions.declarations().to_vec(),
            others: functions.others.clone(),
            urns: functions.urns.clone(),
            by_name,
            next,
        }
    }

    /// The anchor for function `name`, declaring it if needed.
    ///
    /// A new declaration has no extension URN, which is how DataFusion 54
    /// declares every function, its own included; its consumer resolves
    /// functions by name.
    pub fn anchor(&mut self, name: &str) -> u32 {
        if let Some(a) = self.by_name.get(&normalize(name)) {
            return *a;
        }
        let anchor = self.next;
        self.next += 1;
        self.by_name.insert(normalize(name), anchor);
        self.declarations.push(ExtensionFunction {
            extension_urn_reference: u32::MAX,
            function_anchor: anchor,
            name: name.to_string(),
        });
        anchor
    }

    /// The declarations, for a plan's `extensions` list: the input plan's,
    /// functions and otherwise, then those ddx added.
    pub fn declarations(&self) -> Vec<SimpleExtensionDeclaration> {
        self.declarations
            .iter()
            .map(|d| SimpleExtensionDeclaration {
                mapping_type: Some(MappingType::ExtensionFunction(d.clone())),
            })
            .chain(self.others.iter().cloned())
            .collect()
    }

    /// The input plan's extension URNs, for a plan's `extension_urns` list,
    /// so a declaration copied from it still points at its URN.
    pub fn urns(&self) -> Vec<SimpleExtensionUrn> {
        self.urns.clone()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

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
    fn a_plan_written_from_another_keeps_its_extension_urns_and_declarations() {
        // From the composability review (#72): emitted plans had no
        // `extension_urns`, so a function copied from the input pointed into a
        // table that was not there, and a strict consumer rejects that.
        use substrait::proto::extensions::simple_extension_declaration::ExtensionType;
        let mut plan = plan_declaring(&[(0, "sum:fp64")]);
        let MappingType::ExtensionFunction(f) = plan.extensions[0].mapping_type.as_mut().unwrap()
        else {
            unreachable!()
        };
        f.extension_urn_reference = 1;
        plan.extension_urns = vec![SimpleExtensionUrn {
            extension_urn_anchor: 1,
            urn: "extension:io.substrait:functions_arithmetic".into(),
        }];
        let ty = SimpleExtensionDeclaration {
            mapping_type: Some(MappingType::ExtensionType(ExtensionType {
                extension_urn_reference: 1,
                type_anchor: 5,
                name: "point".into(),
            })),
        };
        plan.extensions.push(ty.clone());
        let functions = Functions::from_plan(&plan).unwrap();
        let mut ext = Extensions::new(&functions);
        ext.anchor("multiply");
        assert_eq!(ext.urns(), plan.extension_urns);
        let declared = ext.declarations();
        assert_eq!(declared[0], plan.extensions[0]);
        assert!(declared.contains(&ty));
        assert_eq!(declared.len(), 3);
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
    fn extensions_keep_the_input_anchors_and_add_new_ones_after_them() {
        let plan = plan_declaring(&[(0, "multiply"), (4, "sum:fp64")]);
        let mut ext = Extensions::new(&Functions::from_plan(&plan).unwrap());
        assert_eq!(ext.anchor("multiply"), 0);
        assert_eq!(ext.anchor("SUM"), 4);
        assert_eq!(ext.anchor("add"), 5);
        assert_eq!(ext.anchor("add"), 5);
        let out = Plan {
            extensions: ext.declarations(),
            ..Default::default()
        };
        let f = Functions::from_plan(&out).unwrap();
        assert_eq!(f.name(4).unwrap(), "sum");
        assert_eq!(f.name(5).unwrap(), "add");
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
