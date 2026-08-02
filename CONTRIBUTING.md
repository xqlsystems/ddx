# Contributing to `ddx`

Thanks for your interest in `ddx` — *[JAX](https://docs.jax.dev/)-style automatic
differentiation in SQL*. This guide covers how to get set up, the conventions we
hold contributions to, and the checks your change needs to pass.

Whether you're fixing a typo, closing an issue, or proposing a new capability,
you're welcome here. If anything below is unclear or out of date, that itself is
worth an issue or PR.

## Start with the design doc

`ddx` is a **numerical-correctness** project first and a convenience second, and
it is designed on paper before it is written in code. Two documents are the
source of truth — read the parts relevant to your change before you start:

- [`docs/design.md`](docs/design.md) — the full design: what `ddx` is, the v1
  (scalar) and v2 (query-level) layers, the differentiation surface, the
  per-engine integration story, and the milestone plan. It ends with a decision
  log (`F#`/`G#`/`R#`/`S#`/`Q#`) recording *why* each choice was made.
- [`docs/spikes/`](docs/spikes/) — small runnable programs that back every
  deciding claim in the design (each cited by tag). New direction-changing claims
  should come with a spike.

The guiding principles (design.md §2) shape almost every review comment:

1. **Fail loud, never silently wrong.** An unsupported construct is a *typed
   error*, never an approximate or silently-zero derivative. A wrong number in
   valid-looking SQL is the worst possible outcome — most of our test effort
   exists to prevent exactly that.
2. **Rewrite, don't execute.** `grad`/`jvp` are compile-time markers, rewritten
   away before the query runs — never row functions.
3. **Tag explicitly; never infer.** Meaning is marked by the user, not guessed
   from plan/AST shape.
4. **Prove it.** A claim that can be checked with a small program gets one (a
   spike, a property test, a regression test).

## Getting set up

You need a Rust toolchain. The **minimum supported Rust version (MSRV) is
1.88** — this is enforced in CI and declared in `Cargo.toml`. (The floor comes
from a transitive build dependency; `ddx`'s own code needs far less.)

```bash
rustup toolchain install stable      # for day-to-day work
rustup component add rustfmt clippy
git clone https://github.com/xqlsystems/ddx
cd ddx
cargo build --workspace
cargo test  --workspace              # runs every test, including the fuzz suite
```

### Repository layout

```
crates/
  ddx-core/          # the v1 engine — implemented. sqlparser only. Start here.
  ddx-ad/            # v2 query-level reverse-mode AD — scaffold (M3/M4)
  ddx-datafusion/    # DataFusion adapter — scaffold (M2)
python/ddxdb/        # PyO3/maturin wheel — scaffold (M2)
docs/design.md       # the design (source of truth)
docs/spikes/         # runnable evidence behind the design
tests/               # cross-engine numeric-agreement suites (vs JAX) — scaffold
.github/workflows/   # CI (PR gates) + the nightly property-fuzz soak
```

`ddx-core` is where nearly all current work happens. The other crates are
honest, compile-checked scaffolds (no hidden stubs) awaiting their milestone.

## The dependency policy (important)

The differentiation cores are deliberately minimal so any engine can drive them:

- **`ddx-core` depends on `sqlparser` only** — no `datafusion`, no `duckdb`, no
  `protoc`. Please don't add dependencies to it. Heavy, engine-specific
  dependencies belong in the adapter crates (`ddx-datafusion`, `ddx-duckdb`,
  …), which quarantine them.
- `sqlparser` is **pinned exactly** (`=0.62.0`) and re-exported as
  `ddx_core::sqlparser`. A `sqlparser` bump is a breaking change to `ddx-core`
  and needs its own discussion (see design.md §6, `G2`).

## Development workflow

### 1. Open (or find) an issue

For anything beyond a trivial fix, please
[open an issue](https://github.com/xqlsystems/ddx/issues) describing the bug or
proposal first, so we can agree on the approach before you invest time — and, for
a bug, so there's a place to record the reproducer. Bug reports are most useful
with a **minimal failing SQL input** and the expected vs. actual output.

### 2. Make the change

- Match the surrounding code: comment density, naming, and idiom. The existing
  code is heavily commented for *why*, not *what* — keep that up, especially at
  any spot that could otherwise become silently wrong.
- Add a **test with your change** (see below). A bug fix should come with the
  regression test that would have caught it; a new rule with the derivative it
  produces and the cases it refuses.
- Every new `.rs`/`.py` file needs an SPDX header (see [Licensing](#licensing)).

### 3. Run the checks locally

These are exactly what CI gates on — running them first saves a round-trip:

```bash
# formatting and lints (CI treats warnings as errors)
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings

# tests — the full suite (deterministic tests + the bounded property fuzz)
cargo test --workspace
```

> **Tip:** clippy caches per-crate. To get exactly what CI gets, run it from a
> clean state:
> `cargo clean -p ddx-core && cargo clippy --workspace --all-targets -- -D warnings`.

If you touched the **engine** (`ddx-core`'s differentiation or rewrite logic),
also give the property-fuzz soak a spin — it explores far past the bounded seeds
and is where several real bugs were first found:

```bash
DDX_SOAK_SECS=120 cargo test -p ddx-core --test simulation --release \
  -- --ignored --nocapture soak_continuous_property_fuzz
```

### 4. Open a pull request

- Keep PRs focused; one logical change per PR is easiest to review.
- Write a description that says *what* changed and *why*, and links the issue it
  closes. If the change is subtle, say what could have gone wrong and how the
  tests cover it.
- **CI must be green.** The pull-request checks are:
  - **`rustfmt + clippy`** — `cargo fmt --all --check` and
    `cargo clippy --workspace --all-targets -- -D warnings`.
  - **`tests (…)`** — the deterministic test suite (unit + integration +
    doctests, excluding the property fuzz) run on **stable, beta, nightly, and
    the MSRV (1.88)**. `nightly` is informational; stable, beta, and MSRV must
    pass. (The property/fuzz *soak* runs on a nightly schedule, not on every PR.)

Maintainers merge; you don't need to (and can't) push to protected branches.

## Testing conventions

Differentiation is a numerical-correctness feature, so tests are layered
(design.md §5). In `crates/ddx-core/tests/`:

- **`rules.rs`** — per-rule math and the errors the engine *refuses* (each
  unsupported case asserts a typed error, not a wrong number).
- **`rewrite.rs`** — the end-to-end marker path: span splicing (UTF-8/multibyte,
  multiple/nested markers), the ambiguity and projection-boundary guards,
  identifier folding, and `explain`.
- **`roundtrip.rs`** — the §5 property invariant: a constructed derivative,
  rendered and re-parsed, is the same AST modulo parentheses.
- **`simulation.rs`** — dependency-free, seed-reproducible property/fuzz tests: a
  finite-difference oracle for numeric agreement, render fidelity, and
  self-consumption (the engine must re-parse and re-differentiate its own
  output). Every failure prints the seed that produced it, so it reproduces
  exactly. New invariants are welcome here.

When you find a bug, the ideal contribution is: a failing test first (an
`#[ignore]`-d "known bug" test is a fine way to record it), then the fix, then
un-`#[ignore]` it.

## Releases

Releases to [crates.io](https://crates.io) are automated with
[release-plz](https://release-plz.dev/) and follow [SemVer](https://semver.org/),
one version per crate. You don't do anything special in a normal PR — just land
your change with a clear description.

How it works: every push to `main` updates a **Release PR** that bumps the
changed crates' versions and appends to their `CHANGELOG.md` from the commit
history. **Merging that PR cuts the release** — it tags the version and publishes
to crates.io. Releases are meant to be frequent, so that PR is merged often.

Two things help the automation pick the right version bump:

- Write commit/PR titles that hint at the change type — a
  [Conventional Commits](https://www.conventionalcommits.org/) prefix (`fix:`,
  `feat:`, `feat!:`/`BREAKING CHANGE:` for a breaking API change) is ideal.
- Every PR runs `cargo publish --dry-run`, so packaging problems surface before
  release, not during it. `cargo-semver-checks` also runs at release time and
  will force a major bump if a public API changed incompatibly.

Maintainers only: the release workflow needs a `CARGO_REGISTRY_TOKEN` repository
secret (a crates.io API token) — see the comments in
[`.github/workflows/release.yml`](.github/workflows/release.yml), including the
optional upgrade to crates.io Trusted Publishing.

## Licensing

`ddx` is licensed under **Apache-2.0** and follows the
[REUSE](https://reuse.software/) specification: every `.rs` and `.py` file
carries an SPDX header. New files need one:

```rust
// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0
```

The [`license.sh`](license.sh) script annotates files for you (it uses
[`uv`](https://docs.astral.sh/uv/)'s `uvx`):

```bash
./license.sh
```

By submitting a contribution, you agree to license it under Apache-2.0.

## Where to ask

Open a [GitHub issue](https://github.com/xqlsystems/ddx/issues) for bugs,
questions, and proposals. For a change you're unsure about, an issue (or a draft
PR) is the best way to get early feedback before investing in the full
implementation.

Thank you for helping build differentiable databases. 🎉

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md), standard to all [xql.systems](https://xql.systems) projects. All participants are expected to uphold it.