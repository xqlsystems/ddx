# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Backward programs, for any engine (ddx v2).

A :class:`BackwardProgram` is the gradient of a query as plain Substrait plans
and the table names to materialize their results under. :func:`grad_plan` and
:func:`vjp_plan` build one from the serialized Substrait plan of the query, as
any engine's producer writes it, and :func:`run` runs one on a
:class:`Backend`: the four things only an engine can do. The order of a run,
and what it drops when, are ddx-ad's (``ddx_ad::Runner``), the same protocol
the Rust adapters follow::

    class MyEngine:                      # a Backend
        def select_all(self, name): ...  # Substrait plan of SELECT * FROM name
        def returns_rows(self, plan): ...
        def materialize(self, name, plan): ...
        def drop_table(self, name): ...

    program = grad_plan(engine_plan_bytes, [("w", "val")])
    run(MyEngine(), program)

Nothing here imports an engine. :mod:`ddxdb.ad` builds on it for DataFusion,
from SQL.
"""

from __future__ import annotations

import dataclasses
from typing import Iterator, Optional, Protocol, Sequence

from ._ddxdb import _bind_reads, _grad, _unbound_reads, _vjp

__all__ = [
    "Backend",
    "BackwardProgram",
    "Check",
    "Gradient",
    "Step",
    "grad_plan",
    "run",
    "vjp_plan",
]


@dataclasses.dataclass(frozen=True)
class Step:
    """A plan to run, and the table name to register its result as."""

    name: str
    plan: bytes  # a serialized Substrait plan


@dataclasses.dataclass(frozen=True)
class Check:
    """A plan that must return no rows, and what a row means."""

    plan: bytes  # a serialized Substrait plan
    message: str


@dataclasses.dataclass(frozen=True)
class Gradient:
    """Where a ``wrt`` table's gradient lands.

    ``columns`` are the table's dims, then its ``wrt`` values, named as in the
    table; each value column holds the gradient: ``0`` where none reached, and
    ``NULL`` where the value itself is ``NULL``.
    """

    table: str
    step: str
    columns: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class BackwardProgram:
    """The steps that compute a query's value and its gradient."""

    forward_steps: tuple[Step, ...]
    value: str  # the step holding the query's own result
    cotangent_table: str  # vjp: the table the caller registers the cotangent as
    cotangent: tuple[str, ...]  # vjp: that table's columns
    checks: tuple[Check, ...]  # run before the steps; each must return no rows
    backward_steps: tuple[Step, ...]
    gradients: tuple[Gradient, ...]
    # The same program on the Rust side, which runs the run protocol
    # (ddx_ad::Runner) and plans sql_all's statements.
    _handle: object = dataclasses.field(default=None, repr=False, compare=False)

    def steps(self) -> Iterator[Step]:
        """Every step, in the order they must run."""
        yield from self.forward_steps
        yield from self.backward_steps

    def intermediate_steps(self) -> Iterator[Step]:
        """The steps only other steps read: the saved aggregates and the
        cotangents. :func:`run` drops them once the gradients are written."""
        keep = {self.value, *(g.step for g in self.gradients)}
        return (s for s in self.steps() if s.name not in keep)


class Backend(Protocol):
    """An engine that runs programs: the four primitives of ``ddx_ad::Backend``.

    Plans arrive as serialized Substrait, with every read of an earlier step
    already bound to that table's schema as the engine states it.
    """

    def select_all(self, name: str) -> bytes:
        """The engine's own serialized Substrait plan of ``SELECT * FROM
        name``. ddx binds a plan's reads of ``name`` with the schema of its
        read, so the names and types are the ones the engine's consumer
        expects. Asked once per table, and again after the table is
        rewritten."""
        ...

    def returns_rows(self, plan: bytes) -> bool:
        """Run ``plan`` and say whether it returned any row."""
        ...

    def materialize(self, name: str, plan: bytes) -> None:
        """Run ``plan`` and store its rows as the table ``name``, replacing
        one."""
        ...

    def drop_table(self, name: str) -> None:
        """Drop the table ``name`` if it exists."""
        ...


def _program(raw_and_handle) -> BackwardProgram:
    raw, handle = raw_and_handle
    forward, value, cotangent_table, cotangent, checks, backward, gradients = raw
    return BackwardProgram(
        forward_steps=tuple(Step(n, p) for n, p in forward),
        value=value,
        cotangent_table=cotangent_table,
        cotangent=tuple(cotangent),
        checks=tuple(Check(p, m) for p, m in checks),
        backward_steps=tuple(Step(n, p) for n, p in backward),
        gradients=tuple(Gradient(t, s, tuple(c)) for t, s, c in gradients),
        _handle=handle,
    )


def grad_plan(
    plan: bytes, wrt: Sequence[tuple[str, str]], *, namespace: Optional[str] = None
) -> BackwardProgram:
    """The gradient of the loss the serialized Substrait ``plan`` computes,
    with respect to the ``(table, column)`` pairs in ``wrt``.

    ``namespace`` names every table the program writes ``{namespace}…``
    instead of under a fresh prefix, so the same plan gives the same program:
    it must start with ``__ddx_``, end with ``_``, and hold only lower-case
    ASCII letters, digits and ``_``.
    """
    return _program(_grad(plan, [tuple(w) for w in wrt], namespace))


def vjp_plan(
    plan: bytes, wrt: Sequence[tuple[str, str]], *, namespace: Optional[str] = None
) -> BackwardProgram:
    """The vector-Jacobian product of the query the serialized Substrait
    ``plan`` computes (see :func:`grad_plan`)."""
    return _program(_vjp(plan, [tuple(w) for w in wrt], namespace))


def run(backend: Backend, program: BackwardProgram) -> None:
    """Run ``program`` on ``backend``: its checks, then every step. A run that
    succeeds drops the intermediate tables and leaves the value and the
    gradients; one that fails, at a check or a step, leaves none of the
    program's tables. Raises :class:`ddxdb.InvalidColumn` when a check
    returns a row, or what the backend raised."""
    # The order of a run, and what it drops when, are ddx_ad::Runner's: this
    # loop only does what it is asked.
    runner = program._handle.runner()
    steps = list(program.steps())
    schemas: dict[str, bytes] = {}

    def bound(plan: bytes) -> bytes:
        for name in _unbound_reads(plan):
            if name not in schemas:
                schemas[name] = backend.select_all(name)
        return _bind_reads(plan, {n: schemas[n] for n in _unbound_reads(plan)})

    while (action := runner.next()) is not None:
        kind, arg = action
        try:
            if kind == "check":
                runner.checked(backend.returns_rows(bound(program.checks[arg].plan)))
            elif kind == "step":
                step = steps[arg]
                plan = bound(step.plan)
                schemas.pop(step.name, None)  # rewritten: its schema is this run's
                backend.materialize(step.name, plan)
                runner.done()
            else:
                schemas.pop(arg, None)
                backend.drop_table(arg)
                runner.done()
        except Exception as e:  # handed to the runner, which raises it in finish()
            runner.fail(e)
    runner.finish()
