# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Every v1 rule, one named test each, against `jax.grad`.

The generated suite in `test_jax_agreement.py` is the stronger check — it finds
compositions nobody thought to write — but it can only test rules that survive
JAX's tracing, and it reports a failure as a random seed rather than as a rule.
This file is the complement: the SQL is written out by hand, so a failure names
the rule that broke, and rules the generator cannot reach are still covered.

Two kinds of rule are only here:

- **`log2` / `log10`** — JAX lowers these before a jaxpr exists (`jnp.log2`
  traces to `log(x)/log(2)`), so no generated expression will ever render SQL's
  `log2`. Writing the SQL by hand is the only way to exercise ddx's rule for it.
- **`power` with a constant base** — the generator only produces a constant
  *exponent*, because choosing which side is constant interacts with keeping the
  point in-domain. The other half of the rule is checked here.
"""

from __future__ import annotations

import numpy as np
import pytest

import oracle
from engines import ENGINES

jax = pytest.importorskip("jax", reason="the JAX oracle needs jax installed")
ddxdb = pytest.importorskip("ddxdb", reason="needs the ddxdb wheel built")

jnp = jax.numpy

# (SQL body, the same function for JAX, a sampling interval inside its domain).
# The interval is the derivative's domain, which is the stricter one: `sqrt` is
# defined at 0 and `1/(2*sqrt(x))` is not.
RULES = [
    ("x + y", lambda x, y: x + y, (-3.0, 3.0)),
    ("x - y", lambda x, y: x - y, (-3.0, 3.0)),
    ("x * y", lambda x, y: x * y, (-3.0, 3.0)),
    ("x / y", lambda x, y: x / y, (0.5, 3.0)),
    ("sin(x)", lambda x, y: jnp.sin(x), (-3.0, 3.0)),
    ("cos(x)", lambda x, y: jnp.cos(x), (-3.0, 3.0)),
    ("tan(x)", lambda x, y: jnp.tan(x), (-1.2, 1.2)),
    ("asin(x)", lambda x, y: jnp.arcsin(x), (-0.9, 0.9)),
    ("acos(x)", lambda x, y: jnp.arccos(x), (-0.9, 0.9)),
    ("atan(x)", lambda x, y: jnp.arctan(x), (-3.0, 3.0)),
    ("exp(x)", lambda x, y: jnp.exp(x), (-3.0, 3.0)),
    ("ln(x)", lambda x, y: jnp.log(x), (0.3, 3.0)),
    ("log2(x)", lambda x, y: jnp.log2(x), (0.3, 3.0)),
    ("log10(x)", lambda x, y: jnp.log10(x), (0.3, 3.0)),
    ("sqrt(x)", lambda x, y: jnp.sqrt(x), (0.3, 3.0)),
    ("sinh(x)", lambda x, y: jnp.sinh(x), (-3.0, 3.0)),
    ("cosh(x)", lambda x, y: jnp.cosh(x), (-3.0, 3.0)),
    ("tanh(x)", lambda x, y: jnp.tanh(x), (-3.0, 3.0)),
    # `abs` away from its kink; the kink itself is a pinned convention where ddx
    # and JAX deliberately differ (see test_conventions.py).
    ("abs(x)", lambda x, y: jnp.abs(x), (0.5, 3.0)),
    ("power(x, 3)", lambda x, y: x**3, (-3.0, 3.0)),
    ("power(x, -2)", lambda x, y: x**-2, (0.5, 3.0)),
    ("power(x, 1.5)", lambda x, y: x**1.5, (0.3, 3.0)),
    ("power(2.0, x)", lambda x, y: 2.0**x, (-3.0, 3.0)),
    # Compositions that exercise the chain rule against a nested restricted
    # domain, where a rule that forgot the inner derivative would still look
    # plausible at a single point.
    ("sin(x * y)", lambda x, y: jnp.sin(x * y), (-2.0, 2.0)),
    ("ln(x * x + 1.0)", lambda x, y: jnp.log(x * x + 1.0), (-3.0, 3.0)),
    ("sqrt(exp(x))", lambda x, y: jnp.sqrt(jnp.exp(x)), (-3.0, 2.0)),
]

POINTS = 16


@pytest.mark.parametrize("engine", ENGINES, ids=str)
@pytest.mark.parametrize("body, fn, domain", RULES, ids=[r[0] for r in RULES])
@pytest.mark.parametrize("wrt", ["x", "y"])
def test_rule_agrees_with_jax(engine, body, fn, domain, wrt):
    """ddx's derivative of `body` matches `jax.grad` across the rule's domain.

    Differentiating with respect to `y` as well as `x` is not padding: most of
    these bodies do not mention `y`, so it checks that a rule contributes `0`
    for a variable it does not contain rather than leaking a term.
    """
    lo, hi = domain
    xs = np.linspace(lo, hi, POINTS)
    ys = np.linspace(0.5, 2.5, POINTS)

    sql = ddxdb.rewrite_sql(
        f"SELECT grad({body}, {wrt}) AS d FROM t ORDER BY i", engine.dialect
    )
    got = engine.column(sql, xs, ys)

    grad = jax.grad(fn, argnums=0 if wrt == "x" else 1)
    want = np.asarray(jax.vmap(grad)(jnp.asarray(xs), jnp.asarray(ys)))

    for i, (x, y) in enumerate(zip(xs, ys)):
        scale = max(abs(float(want[i])), abs(float(fn(float(x), float(y)))), 1.0)
        assert oracle.close_at_scale(float(got[i]), float(want[i]), scale), (
            f"{engine.name}: d/d{wrt} {body} disagrees with JAX\n"
            f"  rewritten : {sql}\n"
            f"  at        : x={float(x)!r}, y={float(y)!r}\n"
            f"  ddx={float(got[i])!r}  jax={float(want[i])!r}"
        )


def test_every_rule_the_engine_implements_has_a_case_here():
    """The rule table must not fall behind the rule set it claims to cover.

    ddx's built-in unary rules are the list below. Adding a rule to the engine
    without adding it here would leave it untested while every existing test
    still passed, so the omission is asserted against rather than noticed.
    """
    implemented = {
        "sin", "cos", "tan", "asin", "acos", "atan",
        "exp", "ln", "log2", "log10", "sqrt",
        "sinh", "cosh", "tanh", "abs",
    }
    covered = {name for name in implemented if f"{name}(" in " ".join(r[0] for r in RULES)}
    assert covered == implemented, f"no case for: {sorted(implemented - covered)}"
