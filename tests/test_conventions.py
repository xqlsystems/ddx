# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Points where ddx and JAX differ on purpose, pinned rather than compared.

An oracle is only useful where one side can be wrong. At a kink the derivative
does not exist, so every autodiff system picks a value by convention and no
choice is more correct than another — comparing there would report a deliberate
decision as a bug, and worse, would let a *change* to that decision pass
unnoticed as long as it kept matching whatever JAX does today.

So these are pinned from both sides: what ddx emits, what JAX returns, and the
statement that the two differ. If JAX changes its convention, the test that
records JAX's value fails and someone reads this file instead of quietly
inheriting the change.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from engines import ENGINES

jax = pytest.importorskip("jax", reason="the JAX oracle needs jax installed")
ddxdb = pytest.importorskip("ddxdb", reason="needs the ddxdb wheel built")

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("engine", ENGINES, ids=str)
def test_abs_at_its_kink_is_zero_on_every_engine(engine):
    """`d/dx |x|` at 0 is 0 in ddx — a pinned convention, identical everywhere.

    ddx emits a portable `CASE` rather than an engine `signum`/`sign` builtin,
    for two reasons that both bite here: the builtins are not portable (DuckDB
    has only `sign`, DataFusion only `signum`), and `signum(0)` is 1, so a bare
    builtin would silently break this pin on the engine that had it.
    """
    xs = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    ys = np.zeros_like(xs)
    sql = ddxdb.rewrite_sql(
        "SELECT grad(abs(x), x) AS d FROM t ORDER BY i", engine.dialect
    )
    got = engine.column(sql, xs, ys)
    assert list(got) == [-1.0, -1.0, 0.0, 1.0, 1.0]


def test_jax_disagrees_with_ddx_at_the_kink():
    """JAX returns 1 where ddx returns 0. Recorded, so a drift is visible.

    Neither is wrong: |x| has no derivative at 0, and the subgradient is the
    whole interval [-1, 1]. ddx picks the symmetric point because it is the only
    choice that makes `grad(abs(x), x)` an odd function like `abs` itself is
    even; JAX picks the right-hand limit. The generated oracle suite stays clear
    of this point, which is why the kink is a domain-margin constraint in
    `oracle.py` rather than a special case in the comparison.
    """
    assert float(jax.grad(jax.numpy.abs)(0.0)) == 1.0
    assert float(jax.grad(jax.numpy.abs)(-0.0)) == 1.0

    ddx_at_zero = ddxdb.rewrite_sql("SELECT grad(abs(x), x) AS d FROM t")
    assert "CASE" in ddx_at_zero.upper(), (
        "the abs rule must emit a portable CASE, not an engine signum/sign "
        f"builtin: {ddx_at_zero}"
    )


def test_the_second_derivative_of_abs_is_refused_rather_than_guessed():
    """`abs` differentiates once and not twice, and ddx says so.

    The first derivative is a `CASE` — the portable sign — and ddx does not
    differentiate `CASE`, so the second derivative is a typed error. The honest
    answer is not `0`: `|x|` has no second derivative at 0, and returning the
    almost-everywhere value would hand back a number that is wrong at exactly
    the point anyone would ask about.

    This is pinned because the failure is *useful*, and a later change that made
    it return `0` instead would look like added coverage while removing the
    warning. JAX takes the other branch and returns 0 here, which is why this is
    a convention rather than a bug on either side.
    """
    once = ddxdb.rewrite_sql("SELECT grad(abs(x), x) AS d FROM t")
    assert "CASE" in once.upper()

    with pytest.raises(ddxdb.UnsupportedExpression) as raised:
        ddxdb.rewrite_sql("SELECT grad(grad(abs(x), x), x) AS d FROM t")
    assert "CASE" in str(raised.value), (
        "the error must name what it could not differentiate, so a caller can "
        f"tell why: {raised.value}"
    )

    assert float(jax.grad(jax.grad(jax.numpy.abs))(1.5)) == 0.0


@pytest.mark.parametrize("engine", ENGINES, ids=str)
def test_a_derivative_can_leave_the_domain_its_primal_stayed_inside(engine):
    """`sqrt(x)` is defined at 0; `1/(2*sqrt(x))` is not.

    Differentiation widens the domain requirement, and this is the reason the
    generated suite screens on the *derivative's* domain rather than on whether
    the primal evaluated. Engines are not required to agree on what comes back
    from the edge — only that it is not a finite number pretending to be an
    answer.
    """
    xs = np.array([0.0, 1.0, 4.0])
    ys = np.zeros_like(xs)

    primal = engine.column("SELECT sqrt(x) AS v FROM t ORDER BY i", xs, ys)
    assert list(primal) == [0.0, 1.0, 2.0], "the primal is fine at 0"

    sql = ddxdb.rewrite_sql(
        "SELECT grad(sqrt(x), x) AS d FROM t ORDER BY i", engine.dialect
    )
    got = engine.column(sql, xs, ys)
    assert not np.isfinite(got[0]), (
        f"{engine.name}: the derivative at the domain edge came back as "
        f"{got[0]!r}, which would be a finite answer to a question with none"
    )
    assert np.allclose(got[1:], [0.5, 0.25])


@pytest.mark.parametrize("engine", ENGINES, ids=str)
@pytest.mark.parametrize(
    "body",
    [
        "x * x",      # plain arithmetic: NULL propagates for free
        "abs(x)",     # a CASE, where it does not propagate for free
        "sqrt(x)",    # a quotient
        "sin(x)",     # a chain-rule factor
    ],
)
def test_a_null_input_stays_null_rather_than_becoming_zero(engine, body):
    """A missing value propagates through the derivative as it does the primal.

    A derivative of NULL that came back `0` would be indistinguishable from a
    genuine zero gradient — the difference between "no data here" and "this
    parameter does not move". In a training loop the second is a converged
    weight and the first is a hole in the batch.

    Most rules propagate NULL for free, because arithmetic does. `abs` is the
    exception worth parametrizing over: its derivative is a `CASE`, and under
    three-valued logic a comparison against NULL is NULL rather than false — so a
    row with no value answers none of the branches and falls into `ELSE`. A rule
    written with `ELSE 0.0` therefore hands back a confident zero gradient for
    missing data, on every engine, while looking perfectly reasonable. The sign
    rule states `u = 0` as its own branch so that `ELSE` means only "none of the
    comparisons answered".
    """
    # `None`, not `np.nan`: `pa.array([1.0, np.nan, 3.0])` has null_count 0,
    # because NaN is a float value in Arrow and not a null. A fixture built that
    # way tests NaN propagation while appearing to test this.
    xs = pa.array([1.0, None, 3.0], type=pa.float64())
    assert xs.null_count == 1, "the fixture must actually contain a NULL"
    assert pa.array(np.array([1.0, np.nan, 3.0])).null_count == 0, (
        "...and NaN is not one, which is why this is built from Arrow"
    )
    ys = pa.array([0.0, 0.0, 0.0], type=pa.float64())

    sql = ddxdb.rewrite_sql(
        f"SELECT grad({body}, x) AS d FROM t ORDER BY i", engine.dialect
    )
    assert list(engine.nulls(sql, xs, ys)) == [False, True, False], (
        f"{engine.name}: d/dx {body} did not keep the missing row missing\n"
        f"  rewritten: {sql}\n"
        f"  returned : {engine.values(sql, xs, ys)}"
    )


@pytest.mark.parametrize("engine", ENGINES, ids=str)
def test_the_derivative_of_abs_is_a_double_not_a_decimal(engine):
    """The sign CASE must not come back in a type of its own.

    Its branches are bare literals, which DuckDB reads as `DECIMAL(2,1)` — so
    without a typed branch this one derivative would arrive in a different type
    from every other one ddx emits, with different arithmetic under it. The
    typed NULL in `ELSE` fixes the whole CASE at DOUBLE, which is the same
    branch that carries missingness.
    """
    sql = ddxdb.rewrite_sql(
        "SELECT grad(abs(x), x) AS d FROM t ORDER BY i", engine.dialect
    )
    values = engine.values(sql, np.array([2.0, -2.0, 0.0]), np.zeros(3))
    assert [type(v) for v in values] == [float, float, float], (
        f"{engine.name}: expected float64, got {[type(v).__name__ for v in values]} "
        f"from {sql}"
    )
    assert values == [1.0, -1.0, 0.0]
