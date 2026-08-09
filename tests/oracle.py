# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Render a jaxpr as SQL, so ddx and JAX differentiate provably the same function.

The design's central claim is that rewriting `grad(f, x)` into derivative SQL and
evaluating it per row is the relational equivalent of ``jax.vmap(jax.grad(f))``.
Testing that needs the *same* `f` on both sides, and the failure mode to design
against is not a wrong answer but a false agreement: two hand-written renderings
of one expression — text for SQL, arithmetic for the oracle — can drift apart, or
share a misconception, and then the suite compares two different functions and
reports whatever it finds as a fact about ddx.

So there is only ever one function here. A generated Python callable is traced by
JAX into a **jaxpr** — its own typed IR, a let-bound DAG of primitives — and
everything downstream is an interpreter over that one object:

- `jax.grad` / `jax.jvp` differentiate it. That is JAX's own machinery, not a
  reimplementation, so the oracle cannot be wrong in a way this repo controls.
- :func:`to_sql` renders it as a SQL scalar expression, for ddx.
- :func:`trace` evaluates it numerically, reusing the jaxpr's let-bindings as
  exactly the intermediates the conditioning gates need to look at.

That leaves :func:`to_sql` as the only hand-written translation, and a bug in it
cannot cause a false pass: rendering the wrong expression makes the engine
compute a different function than JAX differentiated, which is a mismatch, which
is a failure. The dangerous direction is closed off by construction.

**What this costs.** A jaxpr is already lowered, so ddx rules JAX does not keep
as primitives are unreachable from here: `jnp.log2` traces to `log(x)/log(2)` and
`jnp.log10` to `log(x)*0.4342…`, so no amount of generation will emit SQL's
`log2`/`log10`. Every other v1 rule survives tracing one-to-one. Those two are
covered by name in `test_rules.py` instead of being quietly missing.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any, Callable, Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal, Var

# SQL DOUBLE is float64. Without this JAX traces in float32 and every comparison
# would be measuring JAX's truncation rather than ddx's math.
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# jaxpr -> SQL
# ---------------------------------------------------------------------------

# Fully parenthesized on purpose: this text is ddx's input, so its shape should
# be what the jaxpr says and not what SQL precedence would recover from it.
BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/"}

# jaxpr primitive -> the SQL function that means the same thing. Where the two
# spell it differently, SQL's name wins (`log` is `ln`).
UNARY = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "exp": "exp",
    "log": "ln",
    "sqrt": "sqrt",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "abs": "abs",
}

# Parameters a primitive may carry that this interpreter is entitled to ignore,
# and the only value of each it is entitled to ignore. `accuracy=None` means
# "exact", `out_dtype=None` means "no dtype change" — anything else changes what
# the primitive computes, and silently dropping it would make the SQL and the
# oracle disagree about the function while the harness reported neither.
IGNORABLE = {"accuracy": None, "out_dtype": None}


class Unrenderable(Exception):
    """A jaxpr contains something this interpreter does not model.

    Raised rather than skipped. An unknown primitive means the generator has
    started producing something the SQL side cannot express, and treating that as
    "nothing to test here" is how a suite quietly stops testing.
    """


# Primitives whose parameters this interpreter reads rather than ignores, so the
# blanket check below does not have to know what each one means.
PARAMETERISED = ("integer_pow", "convert_element_type")


def _check_params(eqn) -> None:
    for name, value in eqn.params.items():
        if name in IGNORABLE:
            if value != IGNORABLE[name]:
                raise Unrenderable(f"{eqn.primitive}: {name}={value!r} changes semantics")
        elif eqn.primitive.name not in PARAMETERISED:
            raise Unrenderable(f"{eqn.primitive}: unmodelled parameter {name}={value!r}")


def _is_identity_convert(eqn) -> bool:
    """Is this `convert_element_type` a no-op at float64?

    JAX inserts one wherever a weakly-typed Python scalar meets a float64 array.
    Every value in this harness is float64, so such a node computes nothing and
    renders as its operand — but a conversion to any *other* dtype would change
    the arithmetic (an integer cast is how a derivative silently becomes a
    truncated one), so only this exact case is treated as transparent.
    """
    return (
        np.dtype(eqn.params["new_dtype"]) == np.dtype(np.float64)
        and eqn.params.get("sharding") is None
    )


def is_real(closed: ClosedJaxpr) -> bool:
    """Does every value in this jaxpr stay in float64?

    JAX will happily leave the reals: raising a negative constant to a fractional
    power constant-folds to a *complex* number at trace time, and from there the
    whole graph is `complex128`. SQL has no such type — `power(-2.5, 0.7)` is
    NULL — so such a draw is not a rewriting ddx got wrong, it is a function that
    cannot be posed to it at all.

    Checked as a property of the traced graph rather than guessed at during
    generation, because the promotion happens inside JAX's constant folding and
    is not visible from the callable that produced it.
    """
    aval_dtypes = [v.aval.dtype for v in closed.jaxpr.invars]
    aval_dtypes += [
        v.aval.dtype for eqn in closed.jaxpr.eqns for v in eqn.outvars
    ]
    aval_dtypes += [np.asarray(c).dtype for c in closed.consts]
    aval_dtypes += [
        np.asarray(a.val).dtype
        for eqn in closed.jaxpr.eqns
        for a in eqn.invars
        if isinstance(a, Literal)
    ]
    return all(np.dtype(d) == np.dtype(np.float64) for d in aval_dtypes)


def _literal(value: Any) -> str:
    # repr() round-trips a float exactly, so the SQL literal and the number JAX
    # traced are the same value rather than nearly the same.
    return repr(float(value))


def to_sql(closed: ClosedJaxpr, names: dict[Var, str]) -> str:
    """Render a jaxpr as one SQL scalar expression.

    Let-bindings are inlined by substitution, because a `grad(...)` marker takes
    a single expression — there is nowhere to put a binding. A shared
    subexpression is therefore duplicated at each use, which is why callers
    bound the size of what they generate.
    """
    env: dict[Var, str] = dict(names)
    jaxpr: Jaxpr = closed.jaxpr
    for var, const in zip(jaxpr.constvars, closed.consts):
        env[var] = _literal(const)

    def read(atom) -> str:
        if isinstance(atom, Literal):
            return _literal(atom.val)
        return env[atom]

    for eqn in jaxpr.eqns:
        _check_params(eqn)
        name = eqn.primitive.name
        args = [read(a) for a in eqn.invars]
        if name in BINARY:
            rendered = f"({args[0]} {BINARY[name]} {args[1]})"
        elif name in UNARY:
            rendered = f"{UNARY[name]}({args[0]})"
        elif name == "neg":
            rendered = f"(-{args[0]})"
        elif name == "integer_pow":
            rendered = f"power({args[0]}, {eqn.params['y']})"
        elif name == "pow":
            rendered = f"power({args[0]}, {args[1]})"
        elif name == "convert_element_type" and _is_identity_convert(eqn):
            rendered = args[0]
        else:
            raise Unrenderable(f"no SQL rendering for primitive {name!r}")
        if len(eqn.outvars) != 1:
            raise Unrenderable(f"{name} returns {len(eqn.outvars)} values")
        env[eqn.outvars[0]] = rendered

    (out,) = jaxpr.outvars
    return read(out)


def primitives(closed: ClosedJaxpr) -> set[str]:
    """Every primitive the jaxpr uses — the unit of rule coverage."""
    return {eqn.primitive.name for eqn in closed.jaxpr.eqns}


# ---------------------------------------------------------------------------
# Numeric conditioning: is this point fit to compare floats at?
# ---------------------------------------------------------------------------
#
# These are not tolerances. They decide whether a *point* can be compared at all,
# and skipping an unfit one is the difference between an oracle with teeth and
# one that reports correct derivatives as bugs.

DOMAIN_EPS = 1e-3
"""Stay this far clear of every restricted-domain boundary.

Differentiation widens the domain requirement: `sqrt` is defined at 0 and its
derivative is not, `asin` is defined at 1 and its derivative is not. Every
constraint below is stated on the derivative's domain, the stricter one.
"""

MAG_CAP = 1e8
"""Skip points where any intermediate exceeds this.

Two algebraically identical expressions evaluated in different orders agree only
to about `eps * (magnitude of the intermediates)`. Past this cap that slop swamps
any result worth comparing.
"""

RTOL = 1e-9
ATOL = 1e-12


@dataclasses.dataclass
class Trace:
    """What one evaluation of a jaxpr says about its own fitness to compare."""

    value: float
    """The primal at this point."""

    magnitude: float
    """The largest absolute value any intermediate took — the comparison scale."""

    margin: float
    """Distance to the nearest restricted-domain boundary; negative if outside."""

    noisy_division: bool
    """Whether some division here amplifies rounding noise past `RTOL`.

    A subtraction can annihilate its operands — `tan(u) - sin(u)/cos(u)` is
    identically zero — leaving a residue of a few machine epsilons whose digits
    are pure rounding. Divide by that and the noise is scaled by the reciprocal,
    without bound, and two correct computations of the same derivative disagree
    wildly because neither is computing anything meaningful.

    The magnitude cap does not see this: the operands are small and well behaved,
    and what was lost is *significance*. The threshold is derived rather than
    picked — a denominator carries absolute error about `eps * scale`, so
    dividing inflates relative error by `eps * scale / |den|`; requiring that to
    stay under `RTOL` gives `|den| > eps * scale / RTOL`.
    """

    @property
    def admissible(self) -> bool:
        return (
            math.isfinite(self.value)
            and math.isfinite(self.magnitude)
            and self.magnitude <= MAG_CAP
            and self.margin >= DOMAIN_EPS
            and not self.noisy_division
        )


def trace(closed: ClosedJaxpr, x: float, y: float) -> Trace:
    """Evaluate the jaxpr at one point, watching its intermediates go by.

    The jaxpr's let-bindings are precisely the intermediates the gates care
    about, so this is one pass rather than a re-walk per question — and it sees
    the same values the engine will, because it interprets the same graph.
    """
    jaxpr = closed.jaxpr
    # Each variable carries its value and the largest magnitude anywhere in the
    # subgraph that produced it; the latter is what a division needs to know
    # about its denominator to tell cancellation from a genuinely small number.
    env: dict[Any, tuple[float, float]] = {}
    for var, const in zip(jaxpr.constvars, closed.consts):
        env[var] = (float(const), abs(float(const)))
    for var, value in zip(jaxpr.invars, (x, y)):
        env[var] = (value, abs(value))

    def read(atom) -> tuple[float, float]:
        if isinstance(atom, Literal):
            v = float(atom.val)
            return v, abs(v)
        return env[atom]

    worst_magnitude = max(abs(x), abs(y))
    margin = math.inf
    noisy = False
    eps = float(np.finfo(np.float64).eps)

    with np.errstate(all="ignore"):
        for eqn in jaxpr.eqns:
            name = eqn.primitive.name
            operands = [read(a) for a in eqn.invars]
            values = [v for v, _ in operands]
            submax = max((m for _, m in operands), default=0.0)

            # Domain constraints, stated on the derivative rather than the
            # primal wherever the two differ.
            if name in ("sqrt", "log"):
                margin = min(margin, values[0])
            elif name in ("asin", "acos"):
                margin = min(margin, 1.0 - abs(values[0]))
            elif name == "tan":
                margin = min(margin, abs(math.cos(values[0])))
            elif name == "abs":
                # Not a domain edge but a kink, and the one place ddx and JAX
                # deliberately disagree: ddx pins abs'(0) = 0, JAX returns 1.
                # Pinned by its own test, so the generated suite steers clear.
                margin = min(margin, abs(values[0]))
            elif name == "div":
                den, den_scale = operands[1]
                margin = min(margin, abs(den))
                if abs(den) <= eps * den_scale / RTOL:
                    noisy = True
            elif name == "pow":
                # A fractional power needs a positive base in real arithmetic;
                # engines return NULL or NaN rather than erroring.
                margin = min(margin, values[0])
            elif name == "integer_pow":
                if eqn.params["y"] <= 1:
                    # The rule emits base^(y - 1), so a zero base is a division
                    # by zero once the exponent drops below 1.
                    margin = min(margin, abs(values[0]))

            out = _apply(name, values, eqn.params)
            if math.isnan(out) and not any(math.isnan(v) for v in values):
                # The primitive left the reals on operands that were fine, so
                # some constraint above is missing rather than merely violated.
                margin = min(margin, -math.inf)
            magnitude = max(submax, abs(out))
            worst_magnitude = max(worst_magnitude, magnitude)
            env[eqn.outvars[0]] = (out, magnitude)

        value, _ = read(jaxpr.outvars[0])

    if math.isnan(margin):
        margin = -math.inf
    return Trace(value, worst_magnitude, margin, noisy)


_APPLY: dict[str, Callable[[list[float]], float]] = {
    "add": lambda a: a[0] + a[1],
    "sub": lambda a: a[0] - a[1],
    "mul": lambda a: a[0] * a[1],
    "div": lambda a: a[0] / a[1] if a[1] != 0 else math.inf * (1 if a[0] >= 0 else -1),
    "neg": lambda a: -a[0],
    "sin": lambda a: math.sin(a[0]),
    "cos": lambda a: math.cos(a[0]),
    "tan": lambda a: math.tan(a[0]),
    "asin": lambda a: math.asin(a[0]) if abs(a[0]) <= 1 else math.nan,
    "acos": lambda a: math.acos(a[0]) if abs(a[0]) <= 1 else math.nan,
    "atan": lambda a: math.atan(a[0]),
    "exp": lambda a: math.exp(a[0]) if a[0] < 709 else math.inf,
    "log": lambda a: math.log(a[0]) if a[0] > 0 else math.nan,
    "sqrt": lambda a: math.sqrt(a[0]) if a[0] >= 0 else math.nan,
    "sinh": lambda a: math.sinh(a[0]) if abs(a[0]) < 709 else math.copysign(math.inf, a[0]),
    "cosh": lambda a: math.cosh(a[0]) if abs(a[0]) < 709 else math.inf,
    "tanh": lambda a: math.tanh(a[0]),
    "abs": lambda a: abs(a[0]),
}


def _apply(name: str, values: list[float], params: dict) -> float:
    """Evaluate one primitive in plain float64.

    Deliberately not `jnp`: this runs per candidate point during rejection
    sampling, where a JAX dispatch per primitive would dominate the runtime, and
    nothing here needs a tracer. `float(np.power(...))` matches SQL's `power`,
    which returns NULL/NaN for a negative base with a fractional exponent where
    Python's `**` would return a complex number.
    """
    if name in _APPLY:
        try:
            return float(_APPLY[name](values))
        except (ValueError, OverflowError, ZeroDivisionError):
            return math.nan
    if name == "integer_pow":
        return float(np.power(values[0], params["y"]))
    if name == "pow":
        return float(np.power(values[0], values[1]))
    if name == "convert_element_type":
        return values[0]
    raise Unrenderable(f"no numeric evaluation for primitive {name!r}")


def close_at_scale(a: float, b: float, scale: float) -> bool:
    """Do two floats agree to within `ATOL + RTOL * scale`?

    Tolerance is relative to the *computation's* scale, not the result's. Two
    expressions that reach the same value by different associations agree only to
    about `eps * (magnitude of the intermediates)`, which a result-relative
    tolerance would reject at any cancellation point. A real bug perturbs the
    result by a finite fraction of its own scale, so it still clears
    `RTOL * scale` wherever the point is well conditioned — the check sheds float
    noise without losing its teeth.
    """
    return abs(a - b) <= ATOL + RTOL * max(abs(scale), 1.0)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

# Functions whose domain is the whole real line, so nesting them does not drive
# every sample out of range. The restricted ones are still generated, just less
# often: an expression nesting three of them is in-domain almost nowhere, and a
# generator that mostly produces unusable points tests almost nothing.
TOTAL = [jnp.sin, jnp.cos, jnp.arctan, jnp.exp, jnp.tanh, jnp.sinh, jnp.cosh, jnp.abs]
PARTIAL = [jnp.tan, jnp.arcsin, jnp.arccos, jnp.log, jnp.sqrt]


def gen(rng: random.Random, depth: int) -> Callable:
    """A random callable of `(x, y)` built from `jnp` operations.

    Composed as closures rather than as a tree with a renderer, because the
    jaxpr *is* the tree — building a second representation to render from is the
    duplication this module exists to avoid.
    """
    if depth <= 0 or rng.random() < 0.25:
        if rng.random() < 0.75:
            return (lambda x, y: x) if rng.random() < 0.5 else (lambda x, y: y)
        # A plain Python float, not `jnp.asarray(...)`: a weakly-typed scalar
        # traces to a `Literal` sitting inline in the equation that uses it,
        # where an array would trace to a `convert_element_type` node wrapping
        # a constant. Both render correctly; the literal renders more readably.
        constant = round(rng.uniform(-3.0, 3.0), 3)
        return lambda x, y, _c=constant: _c

    roll = rng.random()
    if roll < 0.45:
        left, right = gen(rng, depth - 1), gen(rng, depth - 1)
        op = rng.choice(("+", "-", "*", "/"))
        if op == "+":
            return lambda x, y: left(x, y) + right(x, y)
        if op == "-":
            return lambda x, y: left(x, y) - right(x, y)
        if op == "*":
            return lambda x, y: left(x, y) * right(x, y)
        return lambda x, y: left(x, y) / right(x, y)
    if roll < 0.85:
        inner = gen(rng, depth - 1)
        fn = rng.choice(TOTAL) if rng.random() < 0.75 else rng.choice(PARTIAL)
        return lambda x, y, _f=fn, _i=inner: _f(_i(x, y))

    base = gen(rng, depth - 1)
    # An integer exponent leaves the base unrestricted and traces to
    # `integer_pow`; a fractional one needs a positive base and traces to `pow`.
    # Both are ddx rules, so both are generated.
    exponent = (
        rng.choice((-2, -1, 2, 3))
        if rng.random() < 0.75
        else round(rng.uniform(0.2, 2.5), 2)
    )
    return lambda x, y, _b=base, _e=exponent: _b(x, y) ** _e


@dataclasses.dataclass
class Candidate:
    """A generated function, its jaxpr, its SQL, and which variable to vary."""

    fn: Callable
    closed: ClosedJaxpr
    sql: str
    wrt: str

    @property
    def argnum(self) -> int:
        return 0 if self.wrt == "x" else 1


MAX_SQL = 4000
"""Reject a rendering past this many characters.

Inlining a let-bound DAG duplicates every shared subexpression, so a jaxpr that
reuses a value at several depths can render to something exponentially larger
than itself. Such a statement tests the engine's parser, not ddx's calculus.
"""


def candidate(rng: random.Random, depth: int = 3) -> Optional[Candidate]:
    """Generate one function and prepare both of its interpretations.

    Returns `None` when the draw is not a question SQL can be asked — constant in
    both variables, complex-valued, or rendering past :data:`MAX_SQL`. Callers
    count those, and the suites assert the rate stays low.

    A draw being unusable is *not* the same as a primitive this module cannot
    render, which stays an :class:`Unrenderable` and fails the run: one means the
    generator wandered somewhere SQL does not go, the other means the harness has
    fallen behind what the generator emits, and silently skipping the second is
    how a suite stops testing without anyone noticing.
    """
    fn = gen(rng, depth)
    closed = jax.make_jaxpr(fn)(1.0, 1.0)
    if not is_real(closed):
        return None
    invars = closed.jaxpr.invars
    names = {invars[0]: "x", invars[1]: "y"}

    # `Literal` is unhashable, so it cannot be looked up — and a jaxpr carries
    # its constants as literal atoms sitting right alongside the variables.
    used = {
        names[v]
        for eqn in closed.jaxpr.eqns
        for v in eqn.invars
        if isinstance(v, Var) and v in names
    }
    if not used:
        # Constant in both variables: `jax.grad` is 0 everywhere and ddx folds
        # the marker away, so the comparison has no content.
        return None

    sql = to_sql(closed, names)
    if len(sql) > MAX_SQL:
        return None
    return Candidate(fn, closed, sql, rng.choice(sorted(used)))


def sample_points(
    rng: random.Random, cand: Candidate, count: int, tries: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """`count` points at which `cand` is fit to compare, or as many as found.

    Rejection sampling: the admissible region of a generated expression has no
    closed form, so points are drawn and screened. Returning fewer than asked is
    normal, and the caller decides whether that is enough to judge on.
    """
    xs: list[float] = []
    ys: list[float] = []
    for _ in range(tries):
        if len(xs) >= count:
            break
        x = round(rng.uniform(-3.0, 3.0), 6)
        y = round(rng.uniform(-3.0, 3.0), 6)
        if trace(cand.closed, x, y).admissible:
            xs.append(x)
            ys.append(y)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
