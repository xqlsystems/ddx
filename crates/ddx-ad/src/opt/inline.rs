// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Inlining a step that one later step reads, once.
//!
//! A step's table is written, then read back by the steps after it. When
//! exactly one later step reads it, and reads it once, that step can compute
//! it in place of the read: the program computes the same thing, once, and
//! saves the step's write, its read and its own planning (#186).
//!
//! Some steps stay steps:
//! - a result, which the caller reads (the value, a gradient);
//! - a step read in two places, or twice by one step (a contraction's sum
//!   that each side reads, a window's summed cotangents beneath a stack of
//!   windows), which a copy would compute twice;
//! - a step whose reader would then be too deep for a consumer
//!   ([`super::split::MAX_DEPTH`]), or name its columns at more than
//!   [`BUDGET`] expression nodes: a step boundary is where DataFusion's
//!   Substrait consumer starts its column names afresh ([`inlined_size`]).

use std::collections::{BTreeSet, HashMap};

use substrait::proto::extensions::SimpleExtensionDeclaration;
use substrait::proto::read_rel::ReadType;
use substrait::proto::rel::RelType;
use substrait::proto::{Plan, Rel};

use crate::emit::{select, unbound_reads};
use crate::forward::root_of;
use crate::inlined::inlined_size;
use crate::optics::{Compose, Deep, PlanRels, Setter};
use crate::program::Step;

/// The most expression nodes a column of a step may take written out down to
/// the tables it reads, for a step to be inlined into it. ddxdb runs a step
/// above this as SQL rather than Substrait, since DataFusion plans such
/// names slowly.
pub(crate) const BUDGET: u64 = 2_000;

/// `steps`, in the order they run, with each step that one later step reads
/// once computed inside that step instead. `keep` names the steps a caller
/// reads, which stay.
pub(crate) fn inline_steps(steps: &mut Vec<Step>, keep: &BTreeSet<String>) {
    // Each step's readers, a reader once per read.
    let mut readers: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, s) in steps.iter().enumerate() {
        for read in unbound_reads(&s.plan.0) {
            readers.entry(read).or_default().push(i);
        }
    }
    let mut gone = vec![false; steps.len()];
    // From the last step back, so a step's readers are final before it is
    // looked at: a step inlined into its reader makes that reader the reader
    // of everything it read.
    for k in (0..steps.len()).rev() {
        let name = steps[k].name.clone();
        if keep.contains(&name) {
            continue;
        }
        let [j] = readers.get(&name).map(Vec::as_slice).unwrap_or(&[]) else {
            continue;
        };
        let j = *j;
        let Some(merged) = inlined(&steps[j].plan.0, &name, &steps[k].plan.0) else {
            continue;
        };
        if super::split::depth_of(&merged) >= super::split::MAX_DEPTH
            || !inlined_size(&merged).is_some_and(|n| n <= BUDGET)
        {
            continue;
        }
        steps[j].plan.0 = merged;
        gone[k] = true;
        readers.remove(&name);
        for read in unbound_reads(&steps[k].plan.0) {
            if let Some(r) = readers.get_mut(&read) {
                for reader in r.iter_mut().filter(|r| **r == k) {
                    *reader = j;
                }
            }
        }
    }
    let mut i = 0;
    steps.retain(|_| {
        i += 1;
        !gone[i - 1]
    });
}

/// `reader` with its one read of step `name` replaced by `step`'s relation,
/// picking the columns the read named, and declaring `step`'s functions.
/// `None` if the read filters or masks its columns, or names a column
/// `step` doesn't give, or `step` shares relations through `ReferenceRel`,
/// which its root alone doesn't carry.
fn inlined(reader: &Plan, name: &str, step: &Plan) -> Option<Plan> {
    if step.relations.len() != 1 {
        return None;
    }
    let (root, columns) = root_of(step).ok()?;
    let mut out = reader.clone();
    let mut ok = true;
    PlanRels.then(Deep).modify(&mut out, |rel: &mut Rel| {
        let Some(RelType::Read(read)) = &rel.rel_type else {
            return;
        };
        let (Some(ReadType::NamedTable(t)), Some(schema)) = (&read.read_type, &read.base_schema)
        else {
            return;
        };
        if schema.r#struct.is_some() || t.names.join(".") != name {
            return;
        }
        if read.filter.is_some() || read.best_effort_filter.is_some() || read.projection.is_some() {
            ok = false;
            return;
        }
        let picked: Option<Vec<usize>> = schema
            .names
            .iter()
            .map(|c| columns.iter().position(|n| n == c))
            .collect();
        match picked {
            Some(picked) => *rel = select(root.clone(), picked),
            None => ok = false,
        }
    });
    if !ok {
        return None;
    }
    // The program's plans share one table of functions, each plan declaring
    // those it had when it was written: the reader may lack some of the
    // step's.
    let declared: BTreeSet<(u8, u32)> = out.extensions.iter().filter_map(anchor).collect();
    out.extensions.extend(
        step.extensions
            .iter()
            .filter(|d| anchor(d).is_some_and(|a| !declared.contains(&a)))
            .cloned(),
    );
    for urn in &step.extension_urns {
        if !out.extension_urns.contains(urn) {
            out.extension_urns.push(urn.clone());
        }
    }
    Some(out)
}

/// What a declaration declares: its kind (function, type, type variation),
/// each numbered on its own, and its anchor.
fn anchor(d: &SimpleExtensionDeclaration) -> Option<(u8, u32)> {
    use substrait::proto::extensions::simple_extension_declaration::MappingType;
    match &d.mapping_type {
        Some(MappingType::ExtensionFunction(f)) => Some((0, f.function_anchor)),
        Some(MappingType::ExtensionType(t)) => Some((1, t.type_anchor)),
        Some(MappingType::ExtensionTypeVariation(v)) => Some((2, v.type_variation_anchor)),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{plan, project, read_step};
    use crate::expr::{call, field, lit_f64};
    use crate::functions::{Extensions, Function, Functions};
    use crate::program::Unbound;

    /// A step `name` reading `reads` (each a step and the columns it names)
    /// side by side, adding 1 to its first column with `ext`'s `add`, with
    /// output columns `out`.
    fn step(ext: &mut Extensions, name: &str, reads: &[(&str, &[&str])], out: &[&str]) -> Step {
        let add = ext.anchor(Function::Add);
        let mut rels = reads
            .iter()
            .map(|(r, cols)| read_step(r, cols.iter().map(|c| c.to_string()).collect()));
        let first = rels.next().expect("a read");
        let rel = rels.fold(first, |l, r| {
            crate::emit::join(l, r, crate::expr::lit_bool(true), JoinType::Inner)
        });
        let width: usize = reads.iter().map(|(_, c)| c.len()).sum();
        let rel = project(rel, vec![call(add, vec![field(0), lit_f64(1.0)])]);
        let keep: Vec<usize> = (width + 1 - out.len()..=width).collect();
        Step {
            name: name.into(),
            plan: Unbound(plan(
                select(rel, keep),
                out.iter().map(|c| c.to_string()).collect(),
                ext,
            )),
        }
    }

    fn names(steps: &[Step]) -> Vec<&str> {
        steps.iter().map(|s| s.name.as_str()).collect()
    }

    use substrait::proto::join_rel::JoinType;

    fn keep(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn a_chain_of_steps_each_read_once_becomes_its_last() {
        let mut ext = Extensions::new(&Functions::default());
        let mut steps = vec![
            step(&mut ext, "a", &[("t", &["x"])], &["c0"]),
            step(&mut ext, "b", &[("a", &["c0"])], &["c0"]),
            step(&mut ext, "value", &[("b", &["c0"])], &["l"]),
        ];
        inline_steps(&mut steps, &keep(&["value"]));
        assert_eq!(names(&steps), vec!["value"]);
        // It reads what `a` read.
        assert_eq!(steps[0].plan.reads(), vec!["t"]);
    }

    #[test]
    fn a_step_read_twice_or_by_two_steps_stays() {
        let mut ext = Extensions::new(&Functions::default());
        let mut twice = vec![
            step(&mut ext, "a", &[("t", &["x"])], &["c0"]),
            step(&mut ext, "value", &[("a", &["c0"]), ("a", &["c0"])], &["l"]),
        ];
        inline_steps(&mut twice, &keep(&["value"]));
        assert_eq!(names(&twice), vec!["a", "value"]);

        let mut two = vec![
            step(&mut ext, "a", &[("t", &["x"])], &["c0"]),
            step(&mut ext, "b", &[("a", &["c0"])], &["c0"]),
            step(&mut ext, "value", &[("a", &["c0"]), ("b", &["c0"])], &["l"]),
        ];
        inline_steps(&mut two, &keep(&["value"]));
        // `b` is read once and goes; then `value` reads `a` twice.
        assert_eq!(names(&two), vec!["a", "value"]);
    }

    #[test]
    fn a_result_stays_though_one_step_reads_it() {
        let mut ext = Extensions::new(&Functions::default());
        let mut steps = vec![
            step(&mut ext, "value", &[("t", &["x"])], &["c0"]),
            step(&mut ext, "grad", &[("value", &["c0"])], &["g"]),
        ];
        inline_steps(&mut steps, &keep(&["value", "grad"]));
        assert_eq!(names(&steps), vec!["value", "grad"]);
    }

    #[test]
    fn the_inlined_step_gives_the_columns_its_read_named() {
        let mut ext = Extensions::new(&Functions::default());
        // `a` gives (c0, c1); `value` reads only c1.
        let a = {
            let add = ext.anchor(Function::Add);
            let rel = project(
                read_step("t", vec!["x".into()]),
                vec![call(add, vec![field(0), lit_f64(1.0)]), lit_f64(7.0)],
            );
            Step {
                name: "a".into(),
                plan: Unbound(plan(
                    select(rel, vec![1, 2]),
                    vec!["c0".into(), "c1".into()],
                    &ext,
                )),
            }
        };
        let mut steps = vec![a, step(&mut ext, "value", &[("a", &["c1"])], &["l"])];
        inline_steps(&mut steps, &keep(&["value"]));
        assert_eq!(names(&steps), vec!["value"]);
        // value's select, over its projection, over a select of a's
        // column 1 (the 7.0).
        let (root, _) = root_of(&steps[0].plan.0).unwrap();
        let input = |r: &Rel| match &r.rel_type {
            Some(RelType::Project(p)) => p.input.as_deref().cloned().unwrap(),
            other => panic!("{other:?}"),
        };
        let read = input(&input(root));
        let Some(RelType::Project(read)) = &read.rel_type else {
            panic!("{read:?}")
        };
        let emit = match read.common.as_ref().and_then(|c| c.emit_kind.as_ref()) {
            Some(substrait::proto::rel_common::EmitKind::Emit(e)) => e.output_mapping.clone(),
            other => panic!("{other:?}"),
        };
        assert_eq!(emit, vec![1]);
    }

    #[test]
    fn a_function_only_the_inlined_step_declared_is_declared_in_its_reader() {
        // The program's plans share one table of functions, and the reader
        // was written before `a` declared Subtract.
        let mut ext = Extensions::new(&Functions::default());
        let value = step(&mut ext, "value", &[("a", &["c0"])], &["l"]);
        let sub = ext.anchor(Function::Subtract);
        let a = Step {
            name: "a".into(),
            plan: Unbound(plan(
                project(
                    read_step("t", vec!["x".into()]),
                    vec![call(sub, vec![field(0), lit_f64(1.0)])],
                ),
                vec!["x".into(), "c0".into()],
                &ext,
            )),
        };
        let before = value.plan.0.extensions.len();
        let mut steps = vec![a, value];
        inline_steps(&mut steps, &keep(&["value"]));
        assert_eq!(names(&steps), vec!["value"]);
        assert_eq!(steps[0].plan.0.extensions.len(), before + 1);
    }
}
