# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""ddx v2 against `jax.grad`: the MLP, attention and max-pool fixtures.

Each test builds the fixture of one of the design's spikes
(`docs/spikes/relational_ad_spike.py`, `attention_ad_spike.py`,
`route_ad_spike.py`) with the same seeds, writes its forward pass and loss as
plain SQL, takes `grad(loss, table.val)` in SQL on DataFusion, and compares
every gradient entry with `jax.grad` of the same function written in JAX. The
spikes showed hand-applied transpose rules match JAX to machine precision;
these show `grad` in SQL does too.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

TOLERANCE = 1e-12


@pytest.fixture
def env():
    """jax, the ddxdb.ad module and a fresh DataFusion context.

    A fixture rather than module-level skips, so a missing dependency skips
    these tests and nothing else (CONTRIBUTING.md, Python conventions).
    """
    jax = pytest.importorskip("jax", reason="the JAX oracle needs jax installed")
    datafusion = pytest.importorskip("datafusion")
    pytest.importorskip("ddxdb", reason="needs the ddxdb wheel built")
    from ddxdb import ad

    jax.config.update("jax_enable_x64", True)
    return jax, ad, datafusion.SessionContext()


def register(ctx, name: str, array: np.ndarray, dims: tuple[str, ...]) -> None:
    """Register `array` as the tidy table `name(dims..., val)`."""
    index = np.indices(array.shape).reshape(array.ndim, -1)
    columns = {d: pa.array(index[i], pa.int64()) for i, d in enumerate(dims)}
    columns["val"] = pa.array(array.reshape(-1).astype(np.float64))
    ctx.register_record_batches(name, [pa.table(columns).to_batches()])


def gradients(ad, ctx, with_loss: str, tables: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """`SELECT * FROM grad(loss, t.val)` for each table `t`, back as dense
    arrays shaped like the table. One statement per table, sharing one
    program."""
    statements = [f"{with_loss} SELECT * FROM grad(loss, {t}.val)" for t in tables]
    out = {}
    for (name, value), df in zip(tables.items(), ad.sql_all(ctx, statements)):
        t = df.to_arrow_table()
        got = np.full(value.shape, np.nan)
        dims = t.column_names[:-1]
        got[tuple(t.column(d).to_numpy() for d in dims)] = t.column("val").to_numpy()
        assert not np.isnan(got).any(), f"{name}: a gradient row is missing"
        out[name] = got
    return out


def test_mlp_matches_jax_grad(env):
    jax, ad, ctx = env
    jnp = jax.numpy
    # relational_ad_spike.py's fixture, drawn in the same order.
    rng = np.random.default_rng(0)
    N, D, H1, H2, C = 8, 6, 5, 4, 3
    x = rng.standard_normal((N, D))
    y = rng.integers(0, C, size=N)
    W0 = rng.standard_normal((D, H1)) * 0.1
    b0 = rng.standard_normal(H1) * 0.1
    W1 = rng.standard_normal((H1, H2)) * 0.1
    b1 = rng.standard_normal(H2) * 0.1
    W2 = rng.standard_normal((H2, C)) * 0.1
    b2 = rng.standard_normal(C) * 0.1

    register(ctx, "x", x, ("sample", "inp"))
    labels = pa.table({"sample": pa.array(np.arange(N), pa.int64()), "label": pa.array(y, pa.int64())})
    ctx.register_record_batches("y", [labels.to_batches()])
    params = {"w0": W0, "b0": b0, "w1": W1, "b1": b1, "w2": W2, "b2": b2}
    for name, value in params.items():
        register(ctx, name, value, ("inp", "out") if name.startswith("w") else ("out",))

    # One layer: contract with the weights, then add the bias with a join, as
    # nn.py does. The loss is nn.py's: a max-shifted softmax and -AVG(ln p).
    def layer(i, src):
        return f"""
         c{i} AS (SELECT a.sample, w.out, SUM(a.val * w.val) AS z
                  FROM {src} a JOIN w{i} w ON a.inp = w.inp GROUP BY a.sample, w.out),
         z{i} AS (SELECT c.sample, c.out, c.z + b.val AS z
                  FROM c{i} c JOIN b{i} b ON c.out = b.out)"""

    with_loss = f"""
    WITH {layer(0, "x")},
         h0 AS (SELECT sample, out AS inp, tanh(z) AS val FROM z0),
         {layer(1, "h0")},
         h1 AS (SELECT sample, out AS inp, tanh(z) AS val FROM z1),
         {layer(2, "h1")},
         m AS (SELECT sample, MAX(z) AS m FROM z2 GROUP BY sample),
         e AS (SELECT z2.sample, z2.out, exp(z2.z - m.m) AS e
               FROM z2 JOIN m ON z2.sample = m.sample),
         s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample),
         loss AS (SELECT -AVG(ln(e.e / s.s)) AS loss
                  FROM e JOIN s ON e.sample = s.sample JOIN y ON y.sample = e.sample
                  WHERE e.out = y.label)"""
    got = gradients(ad, ctx, with_loss, params)

    def jax_loss(W0, b0, W1, b1, W2, b2):
        a0 = jnp.tanh(jnp.array(x) @ W0 + b0)
        a1 = jnp.tanh(a0 @ W1 + b1)
        ll = jax.nn.log_softmax(a1 @ W2 + b2)
        return -(ll[jnp.arange(N), jnp.array(y)]).mean()

    values = list(params.values())
    expected = jax.grad(jax_loss, argnums=tuple(range(6)))(*map(jnp.array, values))
    loss = ctx.sql(f"{with_loss} SELECT loss FROM loss").to_arrow_table().column(0)[0].as_py()
    assert abs(loss - float(jax_loss(*map(jnp.array, values)))) < TOLERANCE
    for name, want in zip(params, expected):
        np.testing.assert_allclose(got[name], np.asarray(want), rtol=0, atol=TOLERANCE, err_msg=name)


def test_attention_matches_jax_grad(env):
    jax, ad, ctx = env
    jnp = jax.numpy
    # attention_ad_spike.py's fixture, drawn in the same order.
    rng = np.random.default_rng(1)
    n, dm, dh = 5, 6, 4
    X = rng.standard_normal((n, dm))
    Wq = rng.standard_normal((dm, dh)) * 0.3
    Wk = rng.standard_normal((dm, dh)) * 0.3
    Wv = rng.standard_normal((dm, dh)) * 0.3
    tgt = rng.standard_normal((n, dh))
    scale = float(1.0 / np.sqrt(dh))

    inputs = {"wq": Wq, "wk": Wk, "wv": Wv, "x": X}
    for name, value in inputs.items():
        register(ctx, name, value, ("t", "d") if name == "x" else ("d", "e"))
    register(ctx, "tgt", tgt, ("t", "e"))

    def project(w):
        return f"""SELECT x.t, w.e, SUM(x.val * w.val) AS val
                   FROM x JOIN {w} w ON x.d = w.d GROUP BY x.t, w.e"""

    with_loss = f"""
    WITH q AS ({project("wq")}), k AS ({project("wk")}), v AS ({project("wv")}),
         s AS (SELECT q.t, k.t AS u, SUM(q.val * k.val) * {scale!r} AS val
               FROM q JOIN k ON q.e = k.e GROUP BY q.t, k.t),
         m AS (SELECT t, MAX(val) AS m FROM s GROUP BY t),
         ex AS (SELECT s.t, s.u, exp(s.val - m.m) AS val FROM s JOIN m ON s.t = m.t),
         z AS (SELECT t, SUM(val) AS val FROM ex GROUP BY t),
         a AS (SELECT ex.t, ex.u, ex.val / z.val AS val FROM ex JOIN z ON ex.t = z.t),
         o AS (SELECT a.t, v.e, SUM(a.val * v.val) AS val
               FROM a JOIN v ON a.u = v.t GROUP BY a.t, v.e),
         loss AS (SELECT SUM(0.5 * power(o.val - tgt.val, 2)) AS loss
                  FROM o JOIN tgt ON o.t = tgt.t AND o.e = tgt.e)"""
    got = gradients(ad, ctx, with_loss, inputs)

    def jax_loss(Wq, Wk, Wv, X):
        Q, K, V = X @ Wq, X @ Wk, X @ Wv
        A = jax.nn.softmax((Q @ K.T) * scale, axis=1)
        return 0.5 * ((A @ V - jnp.array(tgt)) ** 2).sum()

    expected = jax.grad(jax_loss, argnums=(0, 1, 2, 3))(*map(jnp.array, inputs.values()))
    for name, want in zip(inputs, expected):
        np.testing.assert_allclose(got[name], np.asarray(want), rtol=0, atol=TOLERANCE, err_msg=name)


@pytest.mark.parametrize("tie", [False, True])
def test_max_pool_matches_jax_grad(env, tie):
    jax, ad, ctx = env
    jnp = jax.numpy
    # route_ad_spike.py's fixture: a max-pool over `item`, per group. MAX
    # shares the cotangent across tied maxima, as jax.grad(jnp.max) does, so
    # this holds at a tie too.
    rng = np.random.default_rng(2)
    G, N = 4, 5
    X = rng.standard_normal((G, N))
    w = rng.standard_normal(G)
    if tie:
        X[0, 1] = X[0, 3] = 5.0

    register(ctx, "x", X, ("g", "item"))
    register(ctx, "wt", w, ("g",))
    with_loss = """
    WITH p AS (SELECT g, MAX(val) AS m FROM x GROUP BY g),
         loss AS (SELECT SUM(p.m * wt.val) AS loss FROM p JOIN wt ON p.g = wt.g)"""
    got = gradients(ad, ctx, with_loss, {"x": X})

    want = jax.grad(lambda X: (jnp.array(w) * jnp.max(X, axis=1)).sum())(jnp.array(X))
    np.testing.assert_allclose(got["x"], np.asarray(want), rtol=0, atol=TOLERANCE)


def test_rank_pool_matches_jax_grad_away_from_ties(env):
    jax, ad, ctx = env
    jnp = jax.numpy
    # The same max-pool in nn.py's ranking idiom.
    rng = np.random.default_rng(2)
    X = rng.standard_normal((4, 5))
    w = rng.standard_normal(4)
    register(ctx, "x", X, ("g", "item"))
    register(ctx, "wt", w, ("g",))
    with_loss = """
    WITH r AS (SELECT g, item, val,
                      ROW_NUMBER() OVER (PARTITION BY g ORDER BY val DESC, item) AS rk
               FROM x),
         loss AS (SELECT SUM(r.val * wt.val) AS loss
                  FROM r JOIN wt ON r.g = wt.g WHERE r.rk = 1)"""
    got = gradients(ad, ctx, with_loss, {"x": X})

    want = jax.grad(lambda X: (jnp.array(w) * jnp.max(X, axis=1)).sum())(jnp.array(X))
    np.testing.assert_allclose(got["x"], np.asarray(want), rtol=0, atol=TOLERANCE)
