# Notes for a fast linear algebra project

Seed notes for a separate xqlsystems project: an engine-side library that
makes tensor contractions fast when they are written relationally, as
joins and `SUM` over tables of `(index…, value)` rows. ddx would use it
without depending on it. These notes record why it is needed, what it
would contain, and how it would fit with ddx. They are not a design yet.

## Why a separate project

ddx differentiates relational plans. Its backward pass is mostly
contractions with the same shape as the forward ones: a forward
`Y = X·W` has the backward contractions `X̄ = Ȳ·Wᵀ` and `W̄ = Xᵀ·Ȳ`. The
work ddx emits is within a constant of reverse mode's lower bound (about 2-3x
the forward pass when contractions dominate; `design.md` §4.7, added with
the performance PR stacked on #102). What is far from the bound is how an
engine runs one contraction:

- **A contraction is a hash join, then an aggregate.** `X(N×D)·W(D×H)`
  as `SELECT x.n, w.h, SUM(x.val * w.val) … JOIN … GROUP BY x.n, w.h`
  makes the join emit `N·D·H` rows before the `SUM` folds them. A dense
  GEMM does the same arithmetic without materializing any of it, at BLAS
  or XPU speed.
- **Engines don't push a `SUM` below a join.** A product of three or more
  factors materializes the full join before anything is summed. The
  contraction order that keeps intermediates small is invisible to a join
  optimizer (Blacher et al., below).
- **Relational tables lose dense structure.** `(i, j, val)` rows don't say
  that the coordinates map arithmetically to offsets, so the engine can't
  choose a dense kernel even when the data is dense.

None of these is about differentiation, and all of them are about how a
query engine executes contractions. That is why they belong in a library of
their own, which accelerates any contraction: a user's forward query,
ddx's backward steps, or SQL that never touches ddx.

ddx's benchmark (`crates/ddx-datafusion/tests/ad_perf.rs`) gives a baseline
on DataFusion: a 50,000×16 by 16×8 matrix product takes 146 ms forward,
and its gradient with respect to both operands about 1.8 s, 12x as long,
although the work ddx emits is just the two backward contractions. The gap
is the hash joins: each contraction materializes all `N·D·H` join rows
before summing them.

## Components

### 1. EinFoldHashJoin: a join that sums as it matches

The highest-value piece. It fuses the `SUM` into the hash join:

- build a hash table on one operand, keyed by the contracted indices;
- probe with the other, and accumulate each product directly into a hash
  table keyed by the free (output) indices.

The `N·D·H` join rows never exist. In the database literature this is a
*groupjoin* (Moerkotte & Neumann, VLDB 2011) and, more generally, *eager
aggregation* (Yan & Larson, VLDB 1995). Specialized to einsum it is
Gustavson's algorithm for sparse matrix multiplication (row by row, with an
accumulator), which is what sparse BLAS libraries use. Missing rows are
zeros, so relational sparsity comes for free; dense blocks can go to a
dense kernel (component 2).

Correctness notes: `SUM` ignores NULL, while a product with a NULL operand
is NULL, so the fused operator must drop NULL products exactly as the
join-then-aggregate would. A group with no matches is absent in both forms.
The summation order changes, so floating-point results differ in the last
bits. ddx already tolerates this (its MAX/MIN tie rule treats a sum over
partitions as able to jitter).

### 2. Layout indexes (CuTe)

CuTe's layouts (a shape and a stride, composable) describe how coordinates
map to memory offsets; using them as an index for tensors is discussed in
<https://github.com/NVlabs/CuTe/issues/4>.
A table carrying a layout index tells the engine its rows are a dense (or
blocked) tensor. The contraction operator can then switch from hashing to
tiled dense kernels, and hand a tile to an XPU GEMM.

Open questions:
- How a layout attaches to a table: table metadata in DataFusion, and a
  Substrait extension on the read so other engines see it.
- How a layout survives relational operations: which projections, filters
  and joins preserve it, and where it must be recomputed or dropped.
- Mixed sparse-dense data: blocked layouts with a sparse block index.

### 3. Rechunking for the physical layer (Pangeo's rechunker)

For distributed or out-of-core contractions, both operands must be chunked
compatibly along the contracted indices. Pangeo's rechunker
(<https://rechunker.readthedocs.io/en/latest/algorithm.html>) converts an
array from one chunking to another under a memory budget, through an
intermediate chunking chosen so that no worker holds more than the budget
and the number of reads and writes stays small. Here it would be the
repartitioning (exchange) step in front of a contraction: realign the
operands' chunks, then contract chunk-locally. It connects naturally to
xarray-sql, where the data already arrives chunked.

### 4. Contraction-path optimization

Blacher, Giesen, Klaus, Staudt, Laue and Leis, *Efficient and Portable
Einstein Summation in SQL*
(<https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/einsteinsum.pdf>),
show that a large einsum written as one SQL query (one join and one
`GROUP BY … SUM`) runs far faster when it is decomposed into nested
pairwise contractions (CTEs, each with its own `GROUP BY`), in an order
chosen by a contraction-path optimizer (opt_einsum: greedy or optimal;
cotengra for harder cases). Summing an index out early keeps intermediates
small. Standard query optimizers do not do this: they reorder joins, but
"know nothing about the contraction order" and keep the one `GROUP BY` on
top. Even HyPer, the fastest engine in their tests, gained from the
explicit decomposition.

The paper notes the trade-off: an external path finder knows the
contraction structure, while the engine knows the sizes and sparsity. That
argues for running the path finder inside the engine extension, with the
engine's statistics. Doing it there also covers multi-way products in ddx's
backward steps: a contribution to one factor's gradient is a `SUM` of the
product of the other factors, which is itself an einsum.

## How it would fit with ddx

**As an engine extension, not a ddx dependency.** The integration point is
an optimizer rule that recognizes the contraction pattern in any plan:

```
Aggregate(group by free indices; SUM(a · b · …))
  over equi-Join(s) on index columns
```

It rewrites the pattern to a physical `EinsumExec` (component 1, choosing
dense kernels where layouts allow, component 2). On DataFusion that is an
`OptimizerRule` plus an `ExecutionPlan`; for other engines, a Substrait
extension relation (`einsum`, with the spec and layouts) that engines with
the library consume. This accelerates users' forward queries and ddx's
backward steps alike, with no change to ddx.

Two notes from ddx's emitted plans:
- The product often sits in a projection below the aggregate, not inside
  the `SUM` (ddx rebuilds a region as projections, then aggregates a
  column of it), so the rule must look through projections.
- ddx could later emit the `einsum` extension relation directly for its
  backward contributions, as a capability set in `ddx_ad::Options`, the
  same way a portable form of the `MAX`/`MIN` windows would be chosen
  (`design.md` §4.6).

**What stays in ddx:**
- Caching each step's physical plan across training steps. A program's
  plans do not change between steps, so they need planning once.
  Once contractions are fast, the roughly 3 ms of planning per step would
  otherwise dominate (`design.md` §4.7).
- Forward-mode tangents inside long row-local regions, so that plans grow
  linearly, not quadratically, in a chain's length.
- Emitting contributions in a canonical, einsum-shaped form, so the rule
  above recognizes them cleanly.

## A first milestone

1. EinFoldHashJoin as a DataFusion `ExecutionPlan`, with the rewrite rule,
   for two-operand contractions over sparse `(index…, value)` tables.
2. Correctness: the rule on and off give the same results (to rounding)
   over ddx's soak generator, which already produces random contractions,
   NULLs and ties; and ddx's gradients still match `jax.grad`
   (`tests/test_v2_jax.py`).
3. Speed: ddx's `matmul` and `attn` benchmark families (`ad_perf.rs`),
   forward and backward, rule on against rule off.

Then layouts and dense kernels, then path optimization for multi-way
products, then rechunking for out-of-core data.

## References

- Blacher, Giesen, Klaus, Staudt, Laue, Leis. *Efficient and Portable
  Einstein Summation in SQL.*
  <https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/einsteinsum.pdf>
- Moerkotte, Neumann. *Accelerating Queries with Group-By and Join by
  Groupjoin.* VLDB 2011.
- Yan, Larson. *Eager Aggregation and Lazy Aggregation.* VLDB 1995.
- Gustavson. *Two Fast Algorithms for Sparse Matrices: Multiplication and
  Permuted Transposition.* ACM TOMS 1978.
- CuTe layouts as tensor indexes: <https://github.com/NVlabs/CuTe/issues/4>
- Rechunker's algorithm: <https://rechunker.readthedocs.io/en/latest/algorithm.html>
- opt_einsum: <https://github.com/dgasmith/opt_einsum>
- cotengra: <https://github.com/jcmgray/cotengra>
