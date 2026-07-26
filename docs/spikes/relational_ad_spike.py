# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""De-risk spike for "true AD" in ddx (G5 follow-up).

Claim under test (Fable): the manual backward pass in xarray-sql#196 (nn.py) is
exactly the mechanical application of *transpose (VJP) rules* for a handful of
relational primitives — so query-level reverse-mode AD is an engineering project,
not open research, and ddx-core's scalar `grad` survives as the elementwise leaf.

Method: build a 2-hidden-layer MLP forward pass out of FOUR relational primitives,
then define ONE transpose rule per primitive and compose them in reverse to get all
parameter gradients — using ONLY those rules (this mirrors nn.py's delta*/g* queries
one-to-one, noted inline). Verify every gradient against jax.grad to ~1e-6.

The four primitives (each maps directly to SQL, per nn.py):
  P1 contraction   C[i,k] = Σ_j A[i,j] B[j,k]      = JOIN on j + GROUP BY SUM
  P2 elementwise   Y = f(X)                          = scalar expr (ddx `grad`)
  P3 group-sum     S[k]   = Σ_i X[i,k]               = GROUP BY SUM  (mean = /N)
  P4 broadcast/add Z[i,k] = C[i,k] + b[k]            = JOIN bias on k
  (softmax-CE seed decomposes into exp/ln elementwise + group-sum + gather/scatter)
"""
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

rng = np.random.default_rng(0)
N, D, H1, H2, C = 8, 6, 5, 4, 3          # samples, in, hidden1, hidden2, classes
x  = rng.standard_normal((N, D))
y  = rng.integers(0, C, size=N)          # labels
W0 = rng.standard_normal((D,  H1)) * 0.1; b0 = rng.standard_normal(H1) * 0.1
W1 = rng.standard_normal((H1, H2)) * 0.1; b1 = rng.standard_normal(H2) * 0.1
W2 = rng.standard_normal((H2, C )) * 0.1; b2 = rng.standard_normal(C ) * 0.1
onehot = np.eye(C)[y]

# ---- primitives (forward) --------------------------------------------------
def contract(A, B):        return np.einsum("ij,jk->ik", A, B)   # P1  (JOIN+SUM)
def bias_add(Z, b):        return Z + b[None, :]                 # P4  (JOIN bias)
def tanh(Z):               return np.tanh(Z)                     # P2
def dtanh(Z):              return 1.0 - np.tanh(Z)**2            # P2 local deriv = grad(tanh(z),z)

# ---- transpose (VJP) rules -------------------------------------------------
# T1 contraction:  C=Σ_j A[i,j]B[j,k].  Given Cbar:
def contract_vjp(A, B, Cbar):
    Abar = np.einsum("ik,jk->ij", Cbar, B)   # Ā[i,j] = Σ_k Cbar[i,k] B[j,k]   (contraction!)
    Bbar = np.einsum("ij,ik->jk", A, Cbar)   # B̄[j,k] = Σ_i A[i,j] Cbar[i,k]   (contraction!)
    return Abar, Bbar
# T2 elementwise:  Xbar = Ybar * f'(X)                              (JOIN local deriv)
# T4 bias add:     Cbar_c = Zbar ;  bbar = Σ_i Zbar[i,k]            (group-sum over samples)
def biasadd_vjp(Zbar): return Zbar, Zbar.sum(axis=0)

# ---- forward (tape = keep every pre-activation, exactly nn.py's .cache()) ---
z0 = bias_add(contract(x,  W0), b0); a0 = tanh(z0)     # fwd0
z1 = bias_add(contract(a0, W1), b1); a1 = tanh(z1)     # fwd1
z2 = bias_add(contract(a1, W2), b2)                    # logits (linear)

# loss = mean softmax cross-entropy
def logsumexp(Z): m = Z.max(1, keepdims=True); return (m.squeeze(1) + np.log(np.exp(Z-m).sum(1)))
loss = (logsumexp(z2) - z2[np.arange(N), y]).mean()

# ---- backward: ONLY rule application, reverse order ------------------------
# T5 seed: cotangent on logits = (softmax - onehot)/N   [decomposes to exp/gsum/gather]
sm   = np.exp(z2 - z2.max(1, keepdims=True)); sm /= sm.sum(1, keepdims=True)
z2b  = (sm - onehot) / N                                        # == nn.py delta2 (÷N folded here)
# layer 2:  z2 = a1@W2 + b2
a1b, gb2 = None, None
a1b, W2b = contract_vjp(a1, W2, z2b)                            # ā1 = delta2@W2ᵀ ; W̄2 = a1ᵀ@delta2  == g2
_,   gb2 = biasadd_vjp(z2b)                                     # == gb2
# layer 1:  a1 = tanh(z1)
z1b = a1b * dtanh(z1)                                           # == delta1 (dc1 * grad(tanh,z1))
a0b, W1b = contract_vjp(a0, W1, z1b)                            # == g1
_,   gb1 = biasadd_vjp(z1b)                                     # == gb1
# layer 0:  a0 = tanh(z0)
z0b = a0b * dtanh(z0)                                           # == delta0
xb,  W0b = contract_vjp(x,  W0, z0b)                            # == g0  (xb discarded: x is data)
_,   gb0 = biasadd_vjp(z0b)                                     # == gb0

# ---- gold standard: jax.grad -----------------------------------------------
def jax_loss(W0,b0,W1,b1,W2,b2):
    a0 = jnp.tanh(jnp.array(x)@W0 + b0)
    a1 = jnp.tanh(a0@W1 + b1)
    z2 = a1@W2 + b2
    ll = jax.nn.log_softmax(z2)
    return -(ll[jnp.arange(N), jnp.array(y)]).mean()
gW0,gb0j,gW1,gb1j,gW2,gb2j = jax.grad(jax_loss, argnums=(0,1,2,3,4,5))(
    *[jnp.array(v) for v in (W0,b0,W1,b1,W2,b2)])

def chk(name, ours, ref):
    err = float(np.max(np.abs(np.asarray(ours) - np.asarray(ref))))
    print(f"  {name:5s} max|rule - jax.grad| = {err:.2e}   {'OK' if err < 1e-9 else 'MISMATCH'}")

print(f"loss (ours vs jax): {loss:.6f} vs {float(jax_loss(*[jnp.array(v) for v in (W0,b0,W1,b1,W2,b2)])):.6f}\n")
print("parameter gradients — rule-derived vs jax.grad:")
for n,o,r in [("W2",W2b,gW2),("b2",gb2,gb2j),("W1",W1b,gW1),("b1",gb1,gb1j),("W0",W0b,gW0),("b0",gb0,gb0j)]:
    chk(n,o,r)
