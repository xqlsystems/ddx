# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(2)
n, dm, dh = 5, 6, 4
X=rng.standard_normal((n,dm)); Wq=rng.standard_normal((dm,dh))*.3
Wk=rng.standard_normal((dm,dh))*.3; Wv=rng.standard_normal((dm,dh))*.3
tgt=rng.standard_normal((n,dh)); scale=1/np.sqrt(dh)
mask = np.tril(np.ones((n,n)))            # causal: j<=i allowed
def contract(A,B): return np.einsum("ij,jk->ik",A,B)
def contract_vjp(A,B,Cb): return np.einsum("ik,jk->ij",Cb,B), np.einsum("ij,ik->jk",A,Cb)
def softmax_rows_masked(S):
    Sm = np.where(mask>0, S, -1e30); m=Sm.max(1,keepdims=True); e=np.exp(Sm-m)
    return e/e.sum(1,keepdims=True)
def softmax_vjp(A,Ab): return A*(Ab-(A*Ab).sum(1,keepdims=True))
# forward
Q=contract(X,Wq);K=contract(X,Wk);V=contract(X,Wv)
S=contract(Q,K.T)*scale; A=softmax_rows_masked(S); O=contract(A,V)
loss=0.5*((O-tgt)**2).sum()
# backward (masking = elementwise: masked A=0 so softmax_vjp gives S̄=0 there; the
# mask contributes no gradient. No new rule.)
Ob=(O-tgt); Ab,Vb=contract_vjp(A,V,Ob); Sb=softmax_vjp(A,Ab)*scale
Qb,Ktb=contract_vjp(Q,K.T,Sb); Kb=Ktb.T
Xq,Wqb=contract_vjp(X,Wq,Qb); Xk,Wkb=contract_vjp(X,Wk,Kb); Xv,Wvb=contract_vjp(X,Wv,Vb)
def jl(Wq,Wk,Wv,X):
    Q=X@Wq;K=X@Wk;V=X@Wv; S=(Q@K.T)*scale
    S=jnp.where(jnp.array(mask)>0,S,-1e30); A=jax.nn.softmax(S,axis=1)
    return 0.5*((A@V-jnp.array(tgt))**2).sum()
gWq,gWk,gWv=jax.grad(jl,argnums=(0,1,2))(*map(jnp.array,(Wq,Wk,Wv,X)))
for nm,o,r in [("Wq",Wqb,gWq),("Wk",Wkb,gWk),("Wv",Wvb,gWv)]:
    e=float(np.max(np.abs(np.asarray(o)-np.asarray(r)))); print(f"  causal {nm}: {e:.2e} {'OK' if e<1e-9 else 'FAIL'}")
