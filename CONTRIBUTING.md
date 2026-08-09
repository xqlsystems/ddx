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
cargo test  --workspace              # the deterministic suite (the soak is #[ignore]-d)
```

### Repository layout

```
crates/
  ddx-core/          # the v1 engine — sqlparser only. Start here.
  ddx-datafusion/    # DataFusion adapter: AnalyzerRule (bare grad) + ddx_sql
  ddx-ad/            # v2 query-level reverse-mode AD — scaffold (M3/M4)
python/ddxdb/        # PyO3/maturin wheel: rewrite_sql + a DataFusion Context
tests/               # cross-engine numeric-agreement suites (vs JAX)
docs/design.md       # the design (source of truth)
docs/spikes/         # runnable evidence behind the design
.github/workflows/   # CI (PR gates) + the nightly property-fuzz soak
```

`ddx-core` is where most work happens; everything else is a thin layer over it.
`ddx-ad` is an honest, compile-checked scaffold (no hidden stubs) awaiting its
milestone.

## Working on the Python code

There are two Python surfaces, and they are separate environments on purpose.

| | what it is | where |
|---|---|---|
| **`ddxdb`** | the wheel published to PyPI — a thin PyO3 binding over `ddx-core` | `python/ddxdb` |
| **the oracle suite** | ddx's derivatives checked against `jax.grad` on real engines | `tests/` |

Both need [`uv`](https://docs.astral.sh/uv/) and a Rust toolchain, because both
build the extension module from this repo's own crates.

### The wheel

```bash
cd python/ddxdb
uv venv --python 3.12
uv pip install maturin '.[test]'
uv run maturin develop --uv        # compiles ddx-core + the binding into the venv
uv run pytest tests/ -q
```

`python/ddxdb` is **deliberately its own cargo workspace**, not a member of the
repo's. `pyo3` links against libpython, so folding it in would make every
`cargo build` at the repo root require a correctly configured Python interpreter
— for a crate only the wheel build needs.

### The oracle suite

```bash
uv sync --project tests --reinstall-package ddxdb
uv run --project tests pytest tests/ -q
```

`tests/` is a `uv` project with `ddxdb` as a *path dependency*, so `uv sync`
builds the wheel through maturin's build backend. There is no separate
`maturin develop` step to run in the wrong directory or point at the wrong
virtualenv.

> **`--reinstall-package ddxdb` is load-bearing, not defensive.** A plain
> `uv sync` sees `ddxdb` already installed and audits it in milliseconds —
> *including after you have edited `crates/ddx-core`*, because uv watches the
> Python package and the Rust source is not in it. Leave it off and the suite
> passes against the engine you had **before** your change, which is the most
> expensive kind of green. Measured, not hypothetical:
>
> ```
> $ uv sync                             # after editing the cos rule
> Audited 17 packages in 2ms
> >>> ddxdb.rewrite_sql("SELECT grad(cos(x), x) ...")
> 'SELECT (-sin(x)) AS d FROM t'        # the OLD rule
> ```

`tests/uv.lock` is committed, so CI resolves the environment the repo describes.
That also means a JAX release cannot change the oracle underneath a green build:
upgrading is a deliberate `uv lock --upgrade --project tests`, and
`tests/test_conventions.py` is what reports whether anything JAX promises moved.

### How the oracle suite is built, and why it matters when editing it

Read [`tests/README.md`](tests/README.md) before changing `tests/oracle.py`. The
short version: a generated Python function is traced by JAX into a **jaxpr**, and
that *one* object is then interpreted three ways — `jax.grad` for the oracle,
`to_sql` for the SQL ddx rewrites, and `trace` for the numeric conditioning
gates. Rendering the SQL and the oracle function separately would let the two
drift apart and compare *different* functions while reporting the result as a
fact about ddx. Keep that property.

Two conventions to know before a failure confuses you:

- **Skipped points are normal.** A generated expression is often in-domain almost
  nowhere, and at a near-cancellation point two correct computations of the same
  derivative disagree in every digit. Points are screened; the *retention rate*
  is asserted, so a suite that quietly began rejecting everything fails rather
  than passing vacuously.
- **Some cases are pinned, not compared.** Where ddx and JAX differ on purpose —
  `abs` at its kink, missing values, domain edges — both sides are asserted, so
  changing either is visible. Do not "fix" those by making ddx match JAX.

### Python conventions

- Every new `.py` file needs an SPDX header (see [Licensing](#licensing)).
- Tests assert on **executed numbers** wherever an engine is involved, not on
  rewritten SQL text. A rewrite that looks right but plans to the wrong
  expression is the failure worth catching, and a string comparison cannot see
  it.
- Gate an engine-dependent test with the fixture that needs it, never a
  module-level `pytest.importorskip`: that raises during collection and silently
  skips *everything below it in the file*.

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
  - **`ddxdb wheel (py3.10 / py3.12)`** — builds the wheel and runs its tests on
    the `requires-python` floor and the current version, so the floor is tested
    rather than asserted.
  - **`JAX numeric-agreement oracle`** — the `tests/` suite against DuckDB and
    DataFusion. Generated, but seeded, so a failure reproduces exactly — which is
    what makes it a blocking gate rather than a soak finding.
  - **`package (publish dry-run)`** — `cargo publish --dry-run` for each
    publishable crate, so packaging breakage surfaces on a PR instead of when a
    release first tries to upload.
  - **`ci-wheel`** (path-filtered, runs only when `python/` or `crates/ddx-core/`
    changes) — builds the wheel on all five platforms we ship to. This is where a
    `pyo3`/linker problem shows up, and no amount of Linux testing catches it.

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

### The Python wheel

`ddxdb` ships to [PyPI](https://pypi.org/project/ddxdb/) on a **separate train**,
cut by pushing a tag:

```bash
git tag ddxdb-v0.1.0 && git push origin ddxdb-v0.1.0
```

It is deliberately not `on: release`. release-plz cuts a GitHub Release for every
crate it publishes, so a release trigger would fire on `ddx-core-v0.2.1` and try
to publish a Python package that had not changed.

Before tagging, run [`publish-pypi.yml`](.github/workflows/publish-pypi.yml) via
**`workflow_dispatch`**. That builds every wheel and the sdist, installs each one
and exercises it, then stops — publishing is gated on the tag — so the whole
pipeline can be checked while a mistake is still free. A bad PyPI release can
only be yanked, never replaced.

Maintainers only:

- The crates.io release workflow needs a `CARGO_REGISTRY_TOKEN` repository secret
  — see the comments in
  [`.github/workflows/release.yml`](.github/workflows/release.yml), including the
  optional upgrade to crates.io Trusted Publishing.
- PyPI uses **Trusted Publishing (OIDC)**, so there is no token to leak or
  rotate. It needs a one-time pending publisher on PyPI; the exact values are in
  the header of `publish-pypi.yml`.
- **Read the changelog entry release-plz generates before merging its PR.** It is
  derived from the squashed commit subject, which is the pull-request title — so
  a PR titled for one package can produce a changelog entry describing that
  package in a *different* crate's `CHANGELOG.md`. This has happened. Scoping the
  PR title (`fix(ddx-core): …`) prevents it; reading the entry catches the rest.

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