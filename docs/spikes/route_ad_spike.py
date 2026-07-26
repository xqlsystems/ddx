# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""De-risk spike for the Route (argmax/argmin routing) transpose rule
(design-relational-ad.md §3.4) -- the one rule not yet checked against jax.grad
when the first version of the design was written.

Route: for each group, keep the row that is argmax of `val` over `route_dim`
(nn.py's existing ROW_NUMBER()-then-filter pattern, nn.py:437-445 -- currently
used only for reporting accuracy, never differentiated; real CNNs need exactly
this operation, differentiated, for max-pooling). Its VJP is a SCATTER: the
cotangent flows only to the winning position, zero elsewhere.

Setup: a max-pool-style layer. X has shape (group, item). Y[group] = max_item X.
loss = sum(w * Y). Compare our rule-derived dL/dX against jax.grad(loss)(X),
using jnp.max as the reference max-pool/argmax gradient.

Result (2026-07-20): machine-exact match (0.00e+00) away from ties. AT AN EXACT
TIE, our rule's first-index tiebreak (matching numpy/SQL ROW_NUMBER's
deterministic ordering) diverges from jax.grad(jnp.max)'s convention, which
SPLITS the cotangent evenly across every tied-for-max entry. Both are defensible
conventions; they are NOT the same convention, and design-relational-ad.md's
Route rule must pin its own explicitly (same treatment §8/F12 already gives the
`abs`-at-0 kink) rather than claim agreement with jax.grad at ties.
"""
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

rng = np.random.default_rng(2)
G, N = 4, 5   # groups, items per group
X = rng.standard_normal((G, N))

# ---- forward: Route (argmax over `item`, per `group`) ----------------------
# Mirrors nn.py's ROW_NUMBER() OVER (PARTITION BY group ORDER BY val DESC) = 1:
winner = np.argmax(X, axis=1)            # per-group winning item index
Y = X[np.arange(G), winner]              # the routed (selected) value per group

w = rng.standard_normal(G)               # a downstream elementwise+reduce loss,
loss = float((w * Y).sum())              # so there's a real cotangent to route back

# ---- transpose (VJP) rule: scatter Ybar to the winner only, zero elsewhere -
def route_vjp(shape, winner_idx, Ybar):
    Xbar = np.zeros(shape)
    Xbar[np.arange(shape[0]), winner_idx] = Ybar
    return Xbar

Ybar = w                                  # d(loss)/dY = w  (loss = sum(w*Y))
Xbar = route_vjp(X.shape, winner, Ybar)   # == the SQL scatter in §3.4

# ---- gold standard: jax.grad, using jnp.max as the reference convention ----
def jax_loss(X):
    return (jnp.array(w) * jnp.max(X, axis=1)).sum()

Xbar_jax = jax.grad(jax_loss)(jnp.array(X))
err = float(np.max(np.abs(Xbar - np.asarray(Xbar_jax))))
print(f"loss (ours vs jax): {loss:.6f} vs {float(jax_loss(jnp.array(X))):.6f}")
print(f"Route VJP max|rule - jax.grad| = {err:.2e}   {'OK' if err < 1e-9 else 'MISMATCH'}")

# ---- tie-break convention check --------------------------------------------
Xtie = X.copy()
Xtie[0, 0] = 100.0
Xtie[0, 1] = 100.0        # items 0 & 1 jointly and unambiguously group 0's max
winner_tie = np.argmax(Xtie, axis=1)   # numpy argmax: deterministic first-index tiebreak
Xbar_tie = route_vjp(Xtie.shape, winner_tie, w)
Xbar_tie_jax = jax.grad(jax_loss)(jnp.array(Xtie))
print("\nTie-break check (group 0, items 0 & 1 forced to an exact tie for the max):")
print("  our rule   (first-index tiebreak):    ", Xbar_tie[0, :3])
print("  jax.grad(jnp.max) (splits the tie):   ", np.asarray(Xbar_tie_jax)[0, :3])
diverge = not np.allclose(Xbar_tie[0, :3], np.asarray(Xbar_tie_jax)[0, :3])
print(f"  conventions differ at ties: {diverge}   (expected: True -- PIN OURS EXPLICITLY, don't claim tie-agreement with jax.grad)")
