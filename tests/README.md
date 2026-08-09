# Cross-engine tests (vs JAX)

The **numeric-agreement suites**: the same function, rewritten by ddx and
evaluated on DuckDB and DataFusion, must produce the numbers `jax.grad` produces.

JAX is the natural oracle rather than a convenient one — ddx mirrors its
forward/reverse structure and the same seed/cotangent semantics — so a
disagreement is a finding on one side or the other, not a units mismatch.

## Running it

```sh
uv sync --project tests --reinstall-package ddxdb
uv run --project tests pytest tests/ -q
```

`ddxdb` is a path dependency in `pyproject.toml`, so `uv sync` builds the wheel
from this repo's own crates through maturin's build backend. There is no separate
`maturin develop` step to forget, run from the wrong directory, or point at the
wrong virtualenv.

**`--reinstall-package ddxdb` is the load-bearing part.** A plain `uv sync` sees
that `ddxdb` is already installed and audits it in milliseconds — including after
you have edited `crates/ddx-core`, because uv is watching the Python package and
the Rust source is not in it. The suite would then pass against the engine you
had *before* your change, which is the most expensive kind of green. Verified,
not assumed:

```
$ uv sync                            # after editing the cos rule
Audited 17 packages in 2ms
>>> ddxdb.rewrite_sql("SELECT grad(cos(x), x) ...")   # the OLD rule
'SELECT (-sin(x)) AS d FROM t'

$ uv sync --reinstall-package ddxdb
'SELECT (sin(x)) AS d FROM t'                          # the edited one
```

Every suite skips cleanly if `jax` or `ddxdb` is missing, so a partial
environment reports "skipped" rather than failing for the wrong reason.

`uv.lock` is committed, so CI resolves the environment this file describes. That
also means a JAX release cannot silently change the oracle underneath us — run
`uv lock --upgrade --project tests` deliberately, and let `test_conventions.py`
tell you whether anything JAX promises has moved.

## How it avoids testing itself

The failure to design against is not a wrong answer but a **false agreement**. If
the SQL handed to ddx and the function handed to JAX were rendered separately
from some shared description, the two could drift apart — or share a
misconception — and the suite would compare two different functions while
reporting whatever it found as a fact about ddx.

So there is only one function. A generated Python callable is traced by JAX into
a **jaxpr**, its own let-bound IR of primitives, and everything downstream is an
interpreter over that single object:

| interpreter | produces | written by |
|---|---|---|
| `jax.grad` / `jax.jvp` | the oracle | JAX |
| `oracle.to_sql` | the SQL ddx rewrites | this repo |
| `oracle.trace` | intermediates for the conditioning gates | this repo |

`to_sql` is the only hand-written translation, and a bug in it *cannot* cause a
false pass: rendering the wrong expression makes the engine compute something JAX
did not differentiate, which is a mismatch, which is a failure. The dangerous
direction is closed by construction.

Reusing JAX's IR costs one thing, stated rather than hidden: a jaxpr is already
lowered, so rules JAX does not keep as primitives are unreachable. `jnp.log2`
traces to `log(x)/log(2)`, so no generated expression will ever render SQL's
`log2`. Those rules are covered by name in `test_rules.py`.

## The files

- **`oracle.py`** — the jaxpr interpreters, the conditioning gates, the generator.
- **`engines.py`** — runs a statement on DuckDB or DataFusion and returns the
  column. Rows arrive as Arrow, not as a `VALUES` literal, because DuckDB reads a
  bare `1.5` as `DECIMAL(2,1)` and the two engines would otherwise be handed
  different types for the same fixture.
- **`test_jax_agreement.py`** — generated: `grad` vs `jax.vmap(jax.grad)`, `jvp`
  vs `jax.jvp`, second derivatives vs nested `jax.grad`, and central differences
  as an independent cross-check. JAX and ddx share a structure, so a misconception
  common to both would be invisible between them; a finite difference shares
  nothing with either, because it only ever evaluates the primal.
- **`test_rules.py`** — every v1 rule by name, so a failure names the rule rather
  than a seed, and so `log2`/`log10`/constant-base `power` are covered at all.
- **`test_conventions.py`** — the places ddx and JAX differ *on purpose*, pinned
  from both sides rather than compared.

## Why points get skipped, and why the rate is asserted

A generated expression is often in-domain almost nowhere, and at a
near-cancellation point two correct computations of the same derivative disagree
in every digit. Comparing there would report correct code as broken, so points
are screened: domain margin, a magnitude cap, and a divide-by-noise threshold
derived from the tolerance rather than picked.

Screening is also how an oracle loses its teeth, so the retention rate is
**asserted** — a suite that quietly began rejecting everything would otherwise
keep passing while testing nothing. `test_jax_agreement.py` asserts *rule
coverage* for the same reason: the generator's weights are tuned for admissible
points, not for coverage, and the two can drift apart in silence.

## Conventions, not comparisons

At a kink the derivative does not exist, so every autodiff system picks a value
and none is more correct. Comparing there would report a deliberate decision as a
bug — and worse, would let a change to that decision pass unnoticed as long as it
still matched JAX. So both sides are pinned:

- **`abs` at 0** — ddx gives `0`, JAX gives `1`. ddx emits a portable `CASE`
  rather than an engine `signum`/`sign` builtin, which is what makes the pin hold
  everywhere: DuckDB has only `sign`, DataFusion only `signum`, and `signum(0)`
  is `1`.
- **The second derivative of `abs`** — refused with a typed error, not answered
  with `0`. Its first derivative is a `CASE`, and ddx does not differentiate
  `CASE`. JAX takes the other branch and returns `0`.
- **Domain widening** — `sqrt(x)` is defined at 0 and `1/(2*sqrt(x))` is not, so
  the derivative must not come back as a finite number.
- **NULL** — a missing input stays missing rather than becoming `0`, which would
  be indistinguishable from a genuine zero gradient.

## Still to come

Cross-engine *equivalence* as its own axis — DuckDB against DataFusion directly
rather than each against JAX (GitHub #30) — and the NULL/folding agreement cases
(#32).

The Rust unit and integration tests for the engine itself live with the crate in
[`../crates/ddx-core/tests`](../crates/ddx-core/tests): the ported rule tests,
span splicing, the guards, identifier folding, and the semantic round-trip
property test. The runnable spikes behind each design claim are in
[`../docs/spikes`](../docs/spikes).
