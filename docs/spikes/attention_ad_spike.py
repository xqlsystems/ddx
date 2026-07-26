# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Generality de-risk for query-level reverse-mode AD (§7.3): scaled dot-product
attention from the SAME four transpose rules, no new primitives.

Attention block:  X -> Q,K,V (proj)  ->  S = QKᵀ/√d  ->  A = softmax_key(S)  ->  O = A@V
Everything is one of the four relational primitives:
  - contraction (JOIN+GROUP BY SUM): the 3 projections, QKᵀ, A@V
  - elementwise: the 1/√d scale, exp in softmax, the loss residual
  - group-SUM (+ broadcast): softmax's per-query normaliser (over the KEY axis)
The softmax VJP is NOT a new rule — it composes from elementwise-mul + group-SUM +
broadcast, exactly as the MLP's softmax-CE delta did. If gradients w.r.t. Wq/Wk/Wv/X
match jax.grad machine-exact, attention is covered → the rule set generalises.
"""
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

rng = np.random.default_rng(1)
n, dm, dh = 5, 6, 4                    # positions, d_model, d_head
X  = rng.standard_normal((n, dm))
Wq = rng.standard_normal((dm, dh)) * .3
Wk = rng.standard_normal((dm, dh)) * .3
Wv = rng.standard_normal((dm, dh)) * .3
tgt = rng.standard_normal((n, dh))
scale = 1.0 / np.sqrt(dh)

# ---- primitives ----
def contract(A, B): return np.einsum("ij,jk->ik", A, B)   # C[i,k]=Σⱼ A[i,j]B[j,k]
def softmax_rows(S):
    m = S.max(1, keepdims=True); e = np.exp(S - m); return e / e.sum(1, keepdims=True)

# ---- transpose (VJP) rules — the ONLY backward machinery ----
def contract_vjp(A, B, Cbar):
    return np.einsum("ik,jk->ij", Cbar, B), np.einsum("ij,ik->jk", A, Cbar)   # Ā, B̄
def softmax_vjp(A, Abar):
    # S̄[i,j] = A[i,j] * (Abar[i,j] - Σ_k A[i,k] Abar[i,k])   — mul + group-SUM + broadcast
    return A * (Abar - (A * Abar).sum(1, keepdims=True))

# ---- forward (tape = keep Q,K,V,S,A) ----
Q = contract(X, Wq); K = contract(X, Wk); V = contract(X, Wv)
S = contract(Q, K.T) * scale                 # QKᵀ then elementwise scale
A = softmax_rows(S)                          # softmax over key axis j
O = contract(A, V)
loss = 0.5 * ((O - tgt) ** 2).sum()

# ---- backward: pure rule application, reverse order ----
Obar = (O - tgt)                                            # d(½‖O-tgt‖²)/dO
Abar, Vbar = contract_vjp(A, V, Obar)                       # O = A@V
Sbar = softmax_vjp(A, Abar)                                 # A = softmax(S)
Sbar_raw = Sbar * scale                                     # unscale (elementwise)
Qbar, Kt_bar = contract_vjp(Q, K.T, Sbar_raw)               # S_raw = Q@Kᵀ
Kbar = Kt_bar.T
Xq, Wq_bar = contract_vjp(X, Wq, Qbar)                      # Q = X@Wq
Xk, Wk_bar = contract_vjp(X, Wk, Kbar)                      # K = X@Wk
Xv, Wv_bar = contract_vjp(X, Wv, Vbar)                      # V = X@Wv
Xbar = Xq + Xk + Xv                                         # X feeds all three (sum paths)

# ---- gold standard ----
def jax_loss(Wq, Wk, Wv, X):
    Q = X @ Wq; K = X @ Wk; V = X @ Wv
    S = (Q @ K.T) * scale
    A = jax.nn.softmax(S, axis=1)
    O = A @ V
    return 0.5 * ((O - jnp.array(tgt)) ** 2).sum()
gWq, gWk, gWv, gX = jax.grad(jax_loss, argnums=(0, 1, 2, 3))(
    jnp.array(Wq), jnp.array(Wk), jnp.array(Wv), jnp.array(X))

def chk(name, ours, ref):
    err = float(np.max(np.abs(np.asarray(ours) - np.asarray(ref))))
    print(f"  {name:3s} max|rule - jax.grad| = {err:.2e}   {'OK' if err < 1e-9 else 'MISMATCH'}")

print(f"loss ours vs jax: {loss:.6f} vs {float(jax_loss(*map(jnp.array,(Wq,Wk,Wv,X)))):.6f}\n")
print("attention gradients — rule-derived vs jax.grad:")
for nm, o, r in [("Wq",Wq_bar,gWq),("Wk",Wk_bar,gWk),("Wv",Wv_bar,gWv),("X",Xbar,gX)]:
    chk(nm, o, r)
