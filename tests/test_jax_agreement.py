# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Numeric agreement between ddx and `jax.grad`, on a real engine.

Each test generates a function, hands the *same* traced jaxpr to both sides —
rendered as SQL for ddx, differentiated by JAX for the oracle — runs the rewrite
on DuckDB and DataFusion, and compares the two columns of numbers. See
`oracle.py` for why there is only one representation of the function.

JAX is the right oracle rather than a convenient one: ddx mirrors its
forward/reverse structure and its seed/cotangent semantics. It is not
*authoritative* everywhere, though. Where the two conventions genuinely differ
rather than one being wrong — `abs` at its kink — the convention is pinned in
`test_conventions.py` and the generated suite steers clear of it.

Everything is seeded, so a failure reproduces exactly.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

import oracle
from engines import ENGINES
from oracle import candidate, sample_points, trace

jax = pytest.importorskip("jax", reason="the JAX oracle needs jax installed")
ddxdb = pytest.importorskip("ddxdb", reason="needs the ddxdb wheel built")

SEED = 20260809
EXPRESSIONS = 120
POINTS = 12
MIN_POINTS = 4

# A generated expression is often in-domain almost nowhere — three nested
# restricted-domain functions can have no admissible region at all — so some are
# skipped for want of comparable points. That is expected; a suite where it
# quietly became *most* of them would report success while testing nothing, so
# the retention floor is asserted rather than assumed.
MIN_RETENTION = 0.5


def _judge(cand, got, xs, ys, want, sql, engine):
    """Compare two columns row by row, failing with enough detail to reproduce."""
    assert len(got) == len(xs), (
        f"{engine}: engine returned {len(got)} rows for {len(xs)} inputs\n  {sql}"
    )
    for i, (x, y) in enumerate(zip(xs, ys)):
        x, y = float(x), float(y)
        scale = max(trace(cand.closed, x, y).magnitude, abs(float(want[i])))
        if not oracle.close_at_scale(float(got[i]), float(want[i]), scale):
            tolerated = oracle.ATOL + oracle.RTOL * max(abs(scale), 1.0)
            pytest.fail(
                f"{engine}: ddx and JAX disagree\n"
                f"  jaxpr      : {cand.closed}\n"
                f"  expression : {cand.sql}\n"
                f"  wrt        : {cand.wrt}\n"
                f"  rewritten  : {sql}\n"
                f"  at         : x={x!r}, y={y!r}\n"
                f"  ddx        : {float(got[i])!r}\n"
                f"  jax        : {float(want[i])!r}\n"
                f"  difference : {abs(float(got[i]) - float(want[i]))!r}"
                f" (tolerated {tolerated!r} at scale {scale!r})"
            )


def _draw(rng, depth=3):
    """Generate candidates until one is usable, yielding it with its points."""
    cand = candidate(rng, depth)
    if cand is None:
        return None, None, None
    xs, ys = sample_points(rng, cand, POINTS)
    if len(xs) < MIN_POINTS:
        return None, None, None
    return cand, xs, ys


@pytest.mark.parametrize("engine", ENGINES, ids=str)
def test_grad_agrees_with_jax_grad(engine):
    """`grad(f, v)` evaluated per row equals `jax.vmap(jax.grad(f))`."""
    rng = random.Random(SEED)
    judged = 0

    for _ in range(EXPRESSIONS):
        cand, xs, ys = _draw(rng)
        if cand is None:
            continue

        sql = ddxdb.rewrite_sql(
            f"SELECT grad({cand.sql}, {cand.wrt}) AS d FROM t ORDER BY i",
            engine.dialect,
        )
        got = engine.column(sql, xs, ys)

        grad = jax.grad(cand.fn, argnums=cand.argnum)
        want = np.asarray(jax.vmap(grad)(jax.numpy.asarray(xs), jax.numpy.asarray(ys)))

        _judge(cand, got, xs, ys, want, sql, engine.name)
        judged += 1

    assert judged / EXPRESSIONS >= MIN_RETENTION, (
        f"only {judged}/{EXPRESSIONS} expressions had {MIN_POINTS} comparable "
        f"points; the generator or the conditioning gates have drifted and this "
        f"suite is no longer testing much"
    )


@pytest.mark.parametrize("engine", ENGINES, ids=str)
def test_jvp_agrees_with_jax_jvp(engine):
    """`jvp(f, v, t)` equals JAX's forward-mode directional derivative.

    ddx seeds one variable's tangent from a column and leaves the other at zero,
    which is `jax.jvp` with a one-hot tangent — the same semantics, so this is a
    direct comparison rather than an analogy.
    """
    rng = random.Random(SEED + 1)
    judged = 0

    for _ in range(EXPRESSIONS):
        cand, xs, ys = _draw(rng)
        if cand is None:
            continue

        # The tangent is the other column, so it varies per row: a rule that
        # dropped it would not merely rescale the answer, it would change shape.
        tangent = "y" if cand.wrt == "x" else "x"
        sql = ddxdb.rewrite_sql(
            f"SELECT jvp({cand.sql}, {cand.wrt}, {tangent}) AS d FROM t ORDER BY i",
            engine.dialect,
        )
        got = engine.column(sql, xs, ys)

        def directional(x, y, _f=cand.fn, _wrt=cand.wrt):
            seed = (y, 0.0) if _wrt == "x" else (0.0, x)
            return jax.jvp(_f, (x, y), seed)[1]

        want = np.asarray(
            jax.vmap(directional)(jax.numpy.asarray(xs), jax.numpy.asarray(ys))
        )

        _judge(cand, got, xs, ys, want, sql, engine.name)
        judged += 1

    assert judged / EXPRESSIONS >= MIN_RETENTION


@pytest.mark.parametrize("engine", ENGINES, ids=str)
def test_second_derivative_agrees_with_nested_jax_grad(engine):
    """Nesting the marker gives the second derivative, as nesting `jax.grad` does.

    Higher order in ddx is not a separate mechanism — the rewrite is closed over
    its own output, so `grad(grad(f, x), x)` is the first rule applied twice.
    This checks the composition, which no single-order test can.
    """
    rng = random.Random(SEED + 2)
    judged = 0

    for _ in range(EXPRESSIONS):
        cand, xs, ys = _draw(rng, depth=2)
        if cand is None:
            continue
        if "abs" in oracle.primitives(cand.closed):
            # `abs` is differentiable once and not twice: its derivative is a
            # CASE, and ddx does not differentiate CASE. That is a deliberate
            # refusal rather than a gap — see
            # test_the_second_derivative_of_abs_is_refused_rather_than_guessed.
            continue

        sql = ddxdb.rewrite_sql(
            f"SELECT grad(grad({cand.sql}, {cand.wrt}), {cand.wrt}) AS d "
            f"FROM t ORDER BY i",
            engine.dialect,
        )
        got = engine.column(sql, xs, ys)

        second = jax.grad(jax.grad(cand.fn, argnums=cand.argnum), argnums=cand.argnum)
        want = np.asarray(
            jax.vmap(second)(jax.numpy.asarray(xs), jax.numpy.asarray(ys))
        )

        # A second derivative squares the conditioning problem: its intermediates
        # are the first derivative's, differentiated again. Points merely adequate
        # for one order need not be for two, so drop rows where either side has
        # left the finite range rather than comparing infinities.
        keep = np.isfinite(want) & np.isfinite(got)
        if keep.sum() < MIN_POINTS:
            continue

        _judge(cand, got[keep], xs[keep], ys[keep], want[keep], sql, engine.name)
        judged += 1

    assert judged / EXPRESSIONS >= 0.3, (
        f"only {judged}/{EXPRESSIONS} second derivatives were comparable"
    )


@pytest.mark.parametrize("engine", ENGINES, ids=str)
def test_finite_differences_cross_check_the_oracle(engine):
    """A central difference agrees with ddx too — independently of JAX.

    JAX and ddx share a structure (forward rules plus the chain rule), so a
    misconception common to both would be invisible to a comparison between them.
    A finite difference shares nothing with either: it only ever evaluates the
    primal. It is far less precise, which is why it cross-checks rather than
    replaces the oracle.
    """
    rng = random.Random(SEED + 3)
    h = 1e-5
    judged = 0

    for _ in range(EXPRESSIONS):
        cand, xs, ys = _draw(rng, depth=2)
        if cand is None:
            continue

        sql = ddxdb.rewrite_sql(
            f"SELECT grad({cand.sql}, {cand.wrt}) AS d FROM t ORDER BY i",
            engine.dialect,
        )
        got = engine.column(sql, xs, ys)

        for i, (x, y) in enumerate(zip(xs, ys)):
            x, y = float(x), float(y)
            hx, hy = (h, 0.0) if cand.wrt == "x" else (0.0, h)
            # The step has to stay inside the region the point was admitted for,
            # or the quotient straddles a boundary ddx never claimed anything about.
            plus = trace(cand.closed, x + hx, y + hy)
            minus = trace(cand.closed, x - hx, y - hy)
            if not (plus.admissible and minus.admissible):
                continue
            approx = (plus.value - minus.value) / (2 * h)

            # A central difference is second-order accurate, so its own error is
            # about h^2 relative to the scale — nine or ten digits at h = 1e-5,
            # nowhere near the 1e-9 the JAX comparison uses. Comparing at the
            # oracle's tolerance would only be measuring the approximation.
            scale = max(plus.magnitude, abs(approx), 1.0)
            assert abs(float(got[i]) - approx) <= 1e-4 * scale, (
                f"{engine.name}: ddx disagrees with a central difference\n"
                f"  expression: {cand.sql}   wrt {cand.wrt}\n"
                f"  at x={x!r}, y={y!r}\n"
                f"  ddx={float(got[i])!r}  finite-difference={approx!r}"
            )
        judged += 1

    assert judged / EXPRESSIONS >= MIN_RETENTION


def test_the_generator_reaches_every_rule_the_jaxpr_can_carry():
    """Coverage, asserted rather than hoped for.

    A generated suite proves nothing about a rule it never emitted, and the
    weights in `oracle.gen` are tuned for admissible *points*, not for coverage —
    so the two can drift apart silently. This fails when they do.

    `log2`/`log10` are absent because JAX lowers them away before a jaxpr exists
    (`jnp.log2` traces to `log(x)/log(2)`); they are covered by name in
    `test_rules.py`.
    """
    rng = random.Random(SEED)
    seen: set[str] = set()
    for _ in range(EXPRESSIONS * 4):
        cand = candidate(rng)
        if cand is not None:
            seen |= oracle.primitives(cand.closed)

    expected = {
        "add", "sub", "mul", "div",
        "sin", "cos", "tan", "asin", "acos", "atan",
        "exp", "log", "sqrt", "sinh", "cosh", "tanh", "abs",
        "integer_pow", "pow",
    }
    assert expected <= seen, f"never generated: {sorted(expected - seen)}"
