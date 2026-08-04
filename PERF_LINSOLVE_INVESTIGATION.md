# Linear-solver performance investigation — handoff

Branch `perf/linsolve-investigation`, based on `feat/amr-recovery-marker` (`d495e90b`).
Local, remote `mn` and xeon in sync. Working tree clean.

**Status: Phases 1 (instrument + capture) and 2 (offline benchmark) are complete. Phase 3 (implement)
has begun. Two cheap leads are now measured on the real model:**
- **Lead 1** ([§10](#10-phase-3-lead-1-measured)): implemented, safe, preserves the Newton path
  exactly, but **break-even** on this thrashing contact+damage model — not the 2–3× the offline
  benchmark projected. The reliable speed lever is now **Lead 2** (freeze the symbolic factorization),
  which is blocked on the §7 drift question.
- **Lead 3 step 1** ([§11](#11-phase-3-lead-3-step-1-measured)): a near-null-space (translations,
  with or without a coupled-block constant) **does not** rescue monolithic AMG — it stalls at
  0.2–0.65 residual regardless. Block structure, not the near-null-space, is the missing ingredient.
- **Lead 3 step 2** ([§12](#12-phase-3-lead-3-step-2--block-amg-is-feasible)): the block preconditioner
  (AMG per field inside block Gauss–Seidel) **converges** — 93–117 GMRES iterations where monolithic
  stalls. **Feasibility for the 1M+ goal is proven.** The count is bottlenecked by per-field AMG
  quality (not the coupling), so the production build (a parallel AMG + block driver + the 6 RBMs,
  which need nodal coordinates) is substantial but no longer a research risk.

Two measured leads plus one untested-but-not-excluded route; see
[§4](#4-recommendation), which ends with a suggested order of work. One open question blocks part of
it, see [§7](#7-the-open-question).

> **A verdict was reversed during this investigation.** An earlier version of this document
> concluded the iterative route was dead. That was an artefact of benchmarking GMRES at
> `rtol=1e-8`, far tighter than a Newton step needs. At an inexact-Newton tolerance the same data
> says the opposite. The retracted reasoning is kept in [§3.6](#36-what-the-first-reading-got-wrong)
> because the failure mode is easy to repeat.

---

## 1. The original question, and the answer

> Penalty contact, AMR with serendipity elements, gradient-enhanced damage. No saddle-point
> structure. PARDISO becomes the bottleneck past 500k dof. Considering an iterative solver (GMRES)
> + preconditioning. Caveat: no MPI.

Two independent wins, both measured, which stack:

1. **A lagged LU as a Krylov preconditioner, at an inexact-Newton tolerance.** One PARDISO
   factorization reused across ≥8 subsequent Newton iterations converges in **4–9 GMRES iterations
   to 1e-4** (~3–7 s) against **11.5 s** for a direct solve, and does **not** degrade with staleness.
   Roughly 2–3× on the solve.
2. **Freezing the symbolic factorization.** 35% of every Newton iteration is spent recomputing a
   reordering that never had to change. Worth **1.78×** on the direct solves that remain.

They compose: (1) removes most direct solves, (2) makes the periodic refactorization cheaper.

Neither is the thing originally proposed. Plain "GMRES + ILU" is not what pays off — the payoff comes
from reusing *exact* factorizations across Newton iterations.

**But note which problem that solves.** Both wins keep a direct factorization in the loop, so both
inherit PARDISO's memory ceiling: they make the present model faster, they do not raise the size limit.
The original framing was "PARDISO becomes the bottleneck past 500k dof" — if that means *time*, these
two wins are the answer; if it means *memory at 1M+ dof*, only a genuinely factorization-free
preconditioner (block AMG, §3.5) removes the wall, and that remains untested. At the current 280k dof
the model peaks at 18.9 GB of 187 GB, so the wall is not close. **Which of those two goals is meant
should be settled before Phase 3, because it changes the ordering in [§4](#4-recommendation).**

---

## 2. Measured baseline

xeon, `OMP_NUM_THREADS=16`, `PYTHON_GIL=0`, `PYTHONUNBUFFERED=1`. Anchor pry-out model:
280,155 dof; 14,036 active elements after AMR (from 11,176); 1,208 hanging nodes; 16,556 slave dofs
eliminated by MPC. Step 2, 44 linear solves. Peak RSS 18.9 GB of 187 GB — **not** memory-bound.

Field layout: `displacement [0, 214659)`, `nonlocal damage [214659, 280155)`.

| item | acc. | calls | per call | share |
|---|---|---|---|---|
| **linear solve** | 711.30 s | 44 | **16.17 s** | **82%** |
| → pardiso index preparation | 2.05 s | 44 | 0.047 s | 0.2% |
| → pardiso phase 11 (reorder + symbolic) | 306.08 s | 44 | **6.96 s** | **35%** |
| → pardiso phase 22 (numeric factorization) | 366.07 s | 44 | 8.32 s | 42% |
| → pardiso phase 33 (back substitution) | 30.48 s | 44 | 0.69 s | 3.5% |
| **mpc transform system matrix** | 121.12 s | 44 | **2.75 s** | **14%** |
| → `TᵀK` | 65.35 s | 44 | 1.485 s | 7.6% |
| → `(TᵀK)T` | 13.01 s | 44 | 0.296 s | 1.5% |
| → `+ C`, tocsr, sort indices | 42.67 s | 44 | 0.970 s | 4.9% |
| elements (assembly) | 19.03 s | 52 | 0.366 s | 1.9% |
| dirichlet K on CSR | 3.92 s | 44 | 0.089 s | 0.5% |
| assemble stiffness CSR | 3.42 s | 44 | 0.078 s | 0.4% |
| assemble constraints | 1.64 s | 49 | 0.034 s | 0.2% |
| mpc transform residual | 0.10 s | 49 | 0.002 s | — |
| convergence check | 0.08 s | 43 | 0.002 s | — |
| AMR | 0.33 s | 8 | 0.041 s | — |

≈ **19.6 s per Newton iteration**. The MPC condensation was untimed before this branch and appeared
only as a ~19% unexplained residue.

---

## 3. The measurements

### 3.1 `pattern` — the sparsity pattern churns every iteration

nnz across consecutive Newton iterations **of the same increment**:

| ord | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| nnz | 40,910,093 | 40,687,250 | 40,790,329 | 40,575,754 | 40,603,002 | 40,607,398 | 40,605,712 | 40,598,381 | 40,604,485 |
| Δ | — | −222,843 | +103,079 | −214,575 | +27,248 | +4,396 | −1,686 | −7,331 | +6,104 |

Verdict on every step: `CHANGED` (compared by index arrays, not just nnz). Last system:
144.9 nnz/row, **21,085,552 structurally asymmetric entries** (52%), **0 explicitly stored zeros**.

So `PardisoSolver._hasSamePattern()` never returns True and `reuseSymbolicFactorization=True` is a
**no-op on any model with ties or AMR**, even when switched on.

Cause: `nonlinearimplicitstatic.py:737` calls `K.eliminate_zeros()` on the condensed matrix, pruning
exactly the Dirichlet-zeroed off-diagonals that `dirichlet.pyx` deliberately keeps (hence also the
52% asymmetry). Two sources are mixed: the Dirichlet zeros (large but *constant* within a step) and
genuinely-zero values that vary with the iterate — contact activation, damage, plasticity — which is
the ±200k swing.

### 3.2 `reuse --unifyPattern` — what a stable pattern is worth

All systems re-expressed on the structural union of their patterns (40,918,420 nnz), 16 threads:

| | reuse OFF | reuse ON |
|---|---|---|
| first solve | 15.19 s | 15.00 s (must analyze) |
| subsequent | 14.91 – 15.06 s | **8.41 – 8.47 s** |

**1.78×, saving ~6.56 s/solve**, matching the 6.96 s phase-11 cost. Deviation from the no-reuse
solutions: **1e-13 … 1e-15** on every system. On this coupled tie + contact + gradient-damage
sequence, reuse with a genuinely fixed pattern is safe — a counter-example to the "silently wrong
results" warning in the wrapper docstring, and consistent with that wrongness having come from reuse
engaging when the pattern was *not* genuinely fixed.

### 3.3 `threads` — PARDISO saturates at one socket

One full solve (phases 11+22+33, reuse off):

| threads | 1 | 4 | 8 | 16 | 36 |
|---|---|---|---|---|---|
| time | 94.52 s | 40.53 s | 25.28 s | 17.91 s | 15.09 s |
| speedup | 1.00× | 2.33× | 3.74× | 5.28× | 6.26× |
| efficiency | 100% | 58% | 47% | 33% | 17% |

Xeon Gold 6140, **2 sockets × 18 cores, 2 NUMA nodes**. 16 threads fits one socket; 36 spans both for
only 1.19× more. **Suggested: `OMP_NUM_THREADS=18` with `numactl --cpunodebind=0 --membind=0`.**
Untested — worth one measurement.

The MPC condensation (2.75 s) is single-threaded and does not shrink, so it grows as a share with
every core added: 13% at 16 threads, 15% at 36.

### 3.4 `lagged` — the main result

GMRES preconditioned by an exact PARDISO LU of iterate `ord 2`, `OMP_NUM_THREADS=MKL_NUM_THREADS=16`
(`bench_lagged_tolerances.log`). Iterations needed to reach each relative tolerance:

| ord | staleness | **1e-2** | **1e-4** | 1e-6 | 1e-8 | wall to 1e-8 |
|---|---|---|---|---|---|---|
| 2 | 0 | 1 | 1 | 1 | 1 | 2.1 s |
| 3 | 1 | 40 | 57 | 71 | 115 | 140.5 s |
| 4 | 2 | **1** | **9** | 23 | 33 | 44.0 s |
| 5 | 3 | **1** | **5** | 29 | 51 | 85.4 s |
| 6 | 4 | **1** | **5** | 34 | 58 | 88.8 s |
| 7 | 5 | **1** | **4** | 26 | 60 | 126.8 s |
| 8 | 6 | **1** | **5** | 30 | 62 | 93.5 s |
| 9 | 7 | **1** | **4** | 28 | 60 | 93.5 s |
| 10 | 8 | **1** | **5** | 30 | 62 | 94.2 s |

Measured cost structure:

- preconditioner apply (PARDISO phase 33): **0.734 s, MKL-threaded**
- sparse matvec (SciPy CSR): **0.052 s, serial**
- per iteration **0.786 s**, only **6.6% of it serial**
- reference direct solve with reused symbolic factorization, same process: **11.52 s**
- ⇒ **break-even ≈ 15 iterations**

**Reading:** at a 1e-4 forcing tolerance, iterates 2–8 need 4–9 iterations ≈ **3–7 s versus 11.5 s**,
and the count is flat in staleness — one factorization serves at least 8 Newton iterations. At 1e-2
it is a single iteration.

**`ord 3` is the exception and matters for the design:** 40 iterations even at 1e-2. It is the
iterate right after the large first correction, it has the largest amplification (below), and the
smallest `‖b‖` (4.1e-2). **The first solve after a big state change should stay direct.**

### 3.5 `amgcl` — ILU0 loses; AMG was mis-configured and is still untested

AMGCL on its OpenMP `builtin` backend, 16 threads — matvec, smoother and orthogonalization all
threaded, unlike the SciPy path. Target to beat: **11.46 s** (`bench_amgcl_fixed.log`).

| configuration | rtol | iters | true rel. res | wall | dev. from direct |
|---|---|---|---|---|---|
| bicgstab + AMG(SA, ilu0) *(wrapper default)* | 1e-2 | 500 ✗ | 5.03e+00 | 133.2 s | 3.6e+00 |
| gmres(100) + AMG(SA, ilu0) | 1e-2 | 500 ✗ | 1.95e-01 | 72.6 s | 7.1e-01 |
| fgmres(100) + AMG(SA, spai0) | 1e-2 | 500 ✗ | 6.53e-01 | 18.8 s | 9.5e-01 |
| gmres(100) + ILU0 only | 1e-2 | 150 | 9.78e-03 | 13.7 s | 1.9e-01 |
| gmres(100) + ILU0 only | 1e-4 | 474 | 9.91e-05 | 31.7 s | 1.9e-03 |
| gmres(100) + ILU0 only | 1e-8 | 500 ✗ | 7.53e-05 | 33.1 s | 1.3e-03 |
| **idrs(4) + ILU0 only** | 1e-2 | 180 | 9.94e-03 | 14.8 s | 1.5e-01 |
| **idrs(4) + ILU0 only** | 1e-4 | 298 | 8.21e-05 | 20.6 s | 4.9e-04 |
| **idrs(4) + ILU0 only** | 1e-8 | 425 | 6.28e-09 | 27.2 s | 7.5e-09 |

✗ = hit maxiter (500) without converging.

**This measured the configuration that is already known not to work, so it does not license any
conclusion about AMG on this problem.** Alkmim, Gamnitzer, Dummer, Neuner & Hofstetter, *Algebraic
Multigrid Based Preconditioning Approaches for Generalized Continuum Models and Indirect Displacement
Control*, IJNME 127:e70309 (2026), §3.2, states it plainly: *"applying AMG directly to a multi-field
problem is ineffective"* — multi-field systems need AMG **inside a block preconditioner**. What was
run above is monolithic, block-unaware AMG on the coupled system. Two specific defects:

- **No near-null-space.** AMGCL's smoothed aggregation defaults to a single constant vector. For an
  elasticity block the near-null-space is the **rigid body modes** — 6 in 3D (that paper, Table 1 and
  §3.3). Withholding them is on its own enough to make AMG useless on an elasticity operator, and it
  is the most likely single cause of the failure above.
- **No block structure.** The paper's two working strategies are *nested AMG* (B-AMG): AMG per field
  inside a block Gauss–Seidel preconditioner, Eq. (44)/(46); and *monolithic AMG* (AMG-B): block
  transfer operators and block smoothers so the coupling is represented on every hierarchy level,
  Eq. (50)/(51). Both reach ~1e-7 with GMRES on 348k–1.95M-dof three-field systems of the same model
  class (gradient-enhanced micropolar, Marmot constitutive models).

The ILU0 numbers stand on their own — that path is block-agnostic by nature — but the AMG rows should
be read as "not yet tested", not as evidence against AMG.

**Single-level ILU0 works but is not competitive.** The best configuration, IDR(s=4) + ILU0, is
slower than the direct solve at every usable tolerance: 20.6 s at 1e-4 versus 11.46 s, i.e. **1.8×
slower**. The 1e-2 rows look close on wall time but are useless as Newton corrections — a 1e-2
residual leaves a **15–19% error** in the solution, another consequence of the conditioning
(residual is not error here).

**What this does settle: parallelism is not the missing ingredient.** AMGCL *is* threaded end to end
and still loses, because at the same 1e-4 tolerance:

| | iterations to 1e-4 | wall |
|---|---|---|
| lagged exact LU + SciPy GMRES (partly serial) | **4–9** | 3–7 s |
| AMGCL ILU0 + IDR(s), fully threaded | **298** | 20.6 s |

~35× more iterations. Preconditioner *quality* dominates by a margin threading cannot recover. That
conclusion is about **ILU0**, and it is what motivates trying a *good* preconditioner (block AMG with
rigid body modes) rather than abandoning the iterative route.

Two defects found and fixed along the way (`dca36610`, `731bb2b3`): AMGCL's iteration count and
error were computed and discarded, so an unconverged solve was indistinguishable from a converged
one (a finite wrong answer passes the nonlinear solvers' NaN check); and the AMG smoother key is
`relax`, not `relaxation`, so the wrapper's shipped "BiCGStab + ILU0" default had never actually used
ILU0 — AMGCL warns on stderr about the unknown key and silently substitutes its default.

**What a proper test needs** (none of it expressible through the current wrapper, which only forwards
a JSON parameter tree):

- **Rigid body modes** as `precond.coarsening.nullspace` — AMGCL takes these as a raw pointer in its
  property tree, so they cannot come through JSON. Needs an optional array argument threaded through
  `.pyx` / `.pxd` / `.hpp`. This is the highest-value single change.
- **Block/field-split**, either `precond.class = schur_pressure_correction` with a `pmask` over the
  `[0, 214659) | [214659, 280155)` split (also a pointer, same problem), or a hand-rolled block
  Gauss-Seidel following Eq. (44)/(46).
- Possibly a scaling/equilibration pass against the 1e8 dynamic range.

For reference, the paper's own stack is Trilinos — Belos (GMRES, no restarts), Teko (block
preconditioner), MueLu (AMG), Chebyshev(6, 20) smoothers, 3 levels — which is a plausible alternative
to extending the AMGCL wrapper if that turns out awkward.

### 3.6 What the first reading got wrong

Two mistakes, recorded so they are not repeated:

**Wrong tolerance.** Only the `rtol=1e-8` column was measured, giving 59–190 iterations and the
conclusion "4–12.5× slower than direct, the route is dead". A Newton step never needs 1e-8. With
`atol=0` it also interacts badly with right-hand sides whose norms span four orders of magnitude:
`ord 3` at `‖b‖ = 4.1e-2` was being asked for an absolute 4e-10.

**Wrong proxy for preconditioner quality.** `bench_drift.log` showed consecutive Jacobians differing
by only 0.3% in relative Frobenius norm, which was read as "lagged LU should be excellent" and then,
when it wasn't, as "these systems must be hopeless". Both readings were wrong, because the governing
quantity is `‖A₂⁻¹ΔA‖`, not `‖ΔA‖/‖A‖`:

| ord | ‖A₂⁻¹ΔA‖ (measured) | ‖ΔA‖_F/‖A‖_F |
|---|---|---|
| 3 | ≥ 3.32 | 3.9e-3 |
| 4 | ≥ 0.70 | 2.9e-3 |
| 5 | ≥ 1.71 | 3.1e-3 |

`A₂⁻¹` amplifies by **180–850×**, turning a 0.3% matrix perturbation into an **O(1)** perturbation of
the preconditioned operator. So `A₂⁻¹Aₖ` is *not* near the identity — hence no one-iteration
convergence, and hence the slow tail to 1e-8. But an O(1) perturbation with a fast-decaying spectrum
is exactly the case where GMRES kills most of the error in the first few iterations, which is why the
loose-tolerance columns are so cheap. Both facts are consistent; only the tight-tolerance column was
being looked at.

A relative Frobenius norm is dominated by the largest entries. With `‖diag(A)‖ ≈ 8e8` against
residuals of order 1, it is blind to the stiff directions that actually govern convergence.

---

## 4. Recommendation

**Lead 1 — lagged factorization + Krylov with inexact-Newton forcing.** The larger win, ~2–3× on the
solve. Design implied by the data:

- Factorize directly on the first iteration of an increment, and after any state change (AMR, tie
  re-detection, contact status flip) — `ord 3` shows the iterate after a large correction is not
  cheap to precondition.
- Reuse that factorization for subsequent iterations via `PardisoSolver.factorize()` /
  `.solveFactorized()` (already on this branch, `81d52193`).
- Eisenstat–Walker forcing rather than a fixed tolerance. **Unmeasured risk:** looser linear
  tolerances buy more *nonlinear* iterations, and none of this measures that. It is the first thing
  to check in Phase 3.
- Refactorize when the iteration count exceeds a threshold (~15, the break-even).

Use **1e-4** as the forcing tolerance, not 1e-2. 1e-2 looks tempting (a single GMRES iteration) but
leaves a **15–19% error in the correction** — residual is not error on this system. At 1e-4 the
deviation from the direct solution is ~5e-4.

**Lead 2 — deterministic pattern + frozen symbolic factorization.** 1.78× on the direct solves that
remain. Requires building `TᵀKT + C` on a cached pattern and scattering values into it instead of
running two SciPy SpGEMMs per iteration, which also recovers most of the 2.75 s condensation cost.
The pruning half is already switchable via `pruneCondensedMatrixZeros` (`b0ad20bf`); the pattern
caching is not written. Blocked on [§7](#7-the-open-question).

**Lead 3 — block AMG, if the goal is size rather than speed.** §3.5's AMG rows do *not* rule this out;
they measured the configuration the literature already calls ineffective. This is the only route that
removes the memory ceiling, and Alkmim et al. (2026) show it working on 348k–1.95M-dof systems of this
exact model class. It is also the most work, and the expensive part is not the AMGCL wrapper (~70 lines
for pointer-valued `nullspace`/`pmask`) but the fact that **rigid body modes need nodal coordinates,
which `createSolver(opts)` never receives** — a real widening of the linsolver interface.

So prove it offline before plumbing anything:

1. **Cheap partial test, no geometry needed. — DONE, negative, see [§11](#11-phase-3-lead-3-step-1-measured).**
   The three *translational* near-null-space vectors are constructible from the DOF layout alone
   (displacement is field-major, node-major, 3 components per node). Translations dominate the
   elasticity near-null-space; if AMG does not improve at all with 3 candidates instead of AMGCL's
   default 1, rotations will not rescue it. **Measured: it does not improve — monolithic AMG stalls at
   0.2–0.65 residual with 1 constant, 3 translations, or 3 translations + a coupled-block constant
   alike. So steps 2–3 (the real block preconditioner) are the only AMG path left, not an optional
   escalation.**
2. **If that is promising**, dump nodal coordinates + field slices alongside the matrices (small
   `matrixdump` extension), build all 6 RBMs offline, add the field-split `pmask`, and re-benchmark
   against the same 11.46 s target.
3. Only then consider widening the linsolver interface.

Before any of that, read the iteration counts off Alkmim et al.'s own tables. **Not extracted here** —
if block AMG needs ~30 V-cycles at ~0.3 s each on a 40M-nnz operator that is ~9 s, i.e. comparable to
the direct solve rather than clearly better at *this* size. That number should gate step 1.

**Also:** pin threads to one socket (§3.3).

### Suggested order of work

1. **Lead 1** — lagged LU + Eisenstat-Walker forcing. Measured, unblocked, code already present.
2. **Verify the nonlinear cost of a 1e-4 linear tolerance.** The one unmeasured thing that could eat
   Lead 1 outright: one run, compare Newton iteration counts against `baseline_omp16.log`.
3. **Lead 3 step 1** — the ~2 h AMGCL translations test, to settle the AMG question on evidence.
4. **Lead 2** — pattern caching, together with the drift validation in §7.

---

## 5. What is on the branch

| commit | what |
|---|---|
| `ce14e854` | MPC-transform + PARDISO phase 11/22/33 timings; per-field block extents logged |
| `455ea7a5` | matrixdump: scope dumps by solver instance (one is created **per analysis step**) |
| `19af29e8` | `scripts/benchmark_linsolve.py` — `pattern` / `reuse` / `threads` / `lagged` |
| `ac3dfed9` | `--unifyPattern`, to price what symbolic reuse would buy |
| `81d52193` | `PardisoSolver.factorize()` / `.solveFactorized()` — phase-separated entry points |
| `eb7af6bd` | time the lagged GMRES runs and record the thread count |
| `3fd4bc90` | per-tolerance iteration counts + the `‖A₂⁻¹ΔA‖` diagnostic — **this reversed the verdict** |
| `46cac2c6` | this handoff document |
| `dca36610` | surface AMGCL's iteration count/error (they were discarded); add the `amgcl` benchmark |
| `731bb2b3` | fix AMGCL's AMG smoother key (`relax`, not `relaxation`); sweep tolerance |
| `68f31121` | record the AMGCL result in this document |
| `b0ad20bf` | `pruneCondensedMatrixZeros` option; correct the AMGCL section's over-broad claim |

New/changed files:

- `edelweissfe/linsolve/matrixdump/` — dumps selected `(A, b)` pairs, delegates the solve. Registered
  in `config/registry.py`. JSON options: `directory`, `delegate`, `delegateOpts`, `dumpAt`,
  `skipFirst`, `maxDumps` (process-wide), `instances`.
- `edelweissfe/linsolve/pardiso/pardiso.pyx` — phase timings; `factorize()`/`solveFactorized()`.
  `__call__` deliberately untouched (production path, and the extension only builds on xeon).
- `edelweissfe/numerics/mpctransformation.py` — timings only; arithmetic verified bit-identical.
- `edelweissfe/solvers/nonlinearimplicitstatic.py` — per-field block extents next to the existing
  "total size of eq. system" message.
- `scripts/benchmark_linsolve.py`.

---

## 6. How to reproduce

Investigation directory on xeon — **not** the real project dir:
`~/constitutive_modeling/next_v2611/Pryout_profile_investigation/`

Inputs are symlinks to `../Pryout_anchorwashernut/`. **Never run in the real project dir**: it holds
existing `esExport` results and the input has `>>configuration, overwrite=yes`.

```bash
export PATH=/home/matthias/constitutive_modeling/mambaforge3/envs/next_v2611/bin:$PATH
cd ~/constitutive_modeling/next_v2611/Pryout_profile_investigation

# capture (~17 min): 3 step-2 increments, 44 solves, 9 dumps of ~0.46 GB
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 PYTHON_GIL=0 edelweissfe capture.inp > capture.log 2>&1

# offline, no FE run needed
B=~/constitutive_modeling/next_v2611/EdelweissFE/scripts/benchmark_linsolve.py
python $B pattern linsolveDumps                                    # seconds
python $B reuse   linsolveDumps --unifyPattern                     # ~7 min
python $B threads linsolveDumps --threads 1,4,8,16,36              # ~15 min
OMP_NUM_THREADS=16 python $B amgcl linsolveDumps                   # ~20 min
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python $B lagged linsolveDumps --backend pardiso  # ~14 min
```

Read `bench_lagged_tolerances.log` for the lagged result — the earlier `bench_lagged.log` (SuperLU,
never finished) and `bench_lagged_pardiso.log` / `bench_lagged_timed.log` (tight tolerance only)
are superseded. Other artefacts: `capture.log`, `bench_reuse_unified.log`, `bench_threads.log`,
`bench_drift.log`, `bench_amgcl_fixed.log` (the AMGCL result; `bench_amgcl.log` predates the
smoother-key fix), `baseline_omp16.log`, `linsolveDumps/` (4.2 GB, 9 systems + `manifest.jsonl`).

`capture.inp` = `profile.inp` + `linsolver=matrixdump, linsolverConfigFile=linsolveDumpConfig.json`
on step 2's `>>options` line. Step 2 `maxNumInc=3` yields **exactly 5 time-advancing increments** —
the agreed cap. Do not raise it without checking.

Sizing: ~5 min per step-2 increment, ~2 min setup+AMR+step 1. Three step-2 increments (~17 min) is
the right A/B unit — it already covers damage onset, two cutbacks and a 15-iteration increment.
Beyond 5 is waste. Prefer the offline bench over FE re-runs.

---

## 7. The open question

Blocks Lead 2 only. `eliminate_zeros()` is **deliberate** —
`nonlinearimplicitstatic.py:729-737`:

> Eliminating here is always safe, and PARDISO's reordering on these more poorly conditioned,
> path-dependent (contact/friction) condensed systems is sensitive enough to the extra explicit-zero
> structural entries to visibly drift from the converged reference path if they are kept.

There is also `stash@{0}`, *"fix(solvers): restore eliminate_zeros() on the MPC-transformed
matrix"* — so this has been fought over once, on **reproducibility** grounds, not correctness.

The pruning is now switchable via the `pruneCondensedMatrixZeros` solver option (`b0ad20bf`),
**defaulting to True, i.e. unchanged behaviour**. Measured on `TieHexa20Patch` with it off: 6303
explicit zeros retained, and structurally asymmetric entries fall from 8828 to **494** — the pruning
was also what destroyed the pattern's near-symmetry, which plausibly hurt the reordering it was meant
to protect. Convergence histories agree to round-off (residuals 2.05e-13 vs 1.85e-13, identical
iteration counts). That round-off difference *is* the drift mechanism in miniature: harmless on a patch
test, unproven over hundreds of softening increments.

**Argument for retrying:** the drift mechanism described is the reordering *changing* in response to
structural entries. A fixed pattern with a symbolic factorization computed once and frozen for the
step removes that variability rather than adding to it. §3.2 supports this indirectly (1e-13…1e-15
per solve on a real sequence).

**Why it is not settled:** 1e-13 per solve, in a softening + contact problem over hundreds of
increments, can amplify. The offline bench cannot test multi-increment path behaviour. Needs a
validation run against a reference path — and Matthias's judgement, since he hit the drift and I
have not.

Lead 1 is **not** blocked by this: a lagged factorization never refactorizes the changed matrix, so
it does not care whether the pattern is stable.

---

## 8. Gotchas worth not rediscovering

- **A linear solver instance is created per analysis step** (`nonlinearimplicitstatic.py:187`), so a
  stateful solver's counters restart each step. matrixdump handles this via `instances`.
- `edelweissfe/linsolve/klu/klu.c` is Cython-generated but **not gitignored** (`.gitignore` has
  `*.cpp`, not `*.c`; `klu.pyx` builds with `language="c"`). `git add -A` sweeps it in, `clang-format`
  then reformats it and aborts the commit. Consider gitignoring it.
- `pgrep -f "pattern"` over `ssh` **matches its own `bash -c` wrapper**, so `until ! ssh host 'pgrep
  -f ...'` loops forever — this silently prevented a benchmark from ever starting. Use the bracket
  trick (`benchmark_linsolve[.]py`).
- Python buffers stdout when not a tty: always `PYTHONUNBUFFERED=1`, and redirect to a log file on
  xeon rather than relying on captured `ssh` stdout, which came back empty every time.
- `panuapardiso` and `klu` fail to build on xeon (missing `pardiso.h` / `klu.h`). Pre-existing,
  optional, unrelated.
- SuperLU is unusable at this size: >10 min and 15.5 GB on a matrix PARDISO factorizes in 8 s. Use
  `--backend pardiso`.
- **Benchmark the tolerance the application actually needs.** §3.6.
- **AMGCL ignores unknown parameter keys** with only a warning on stderr, then silently uses its
  default. Check its stderr when a configuration behaves unexpectedly.

---

## 9. Not done

- **Phase 3, Lead 1: DONE and measured** — see [§10](#10-phase-3-lead-1-measured). Lead 2 is blocked on §7.
- **The goal is settled**: *both, faster-at-280k first then feasible-at-1M+* (Matthias, this session).
  With Lead 1 turning out break-even on this model (§10), the "faster" half now rests on Lead 2.
- **The iteration counts in Alkmim et al. (2026)**, which would gate the AMG work cheaply. Not read.
- **The nonlinear cost of loose linear tolerances: VERIFIED harmless** (§10). EW forcing reproduces the
  direct-solve Newton path exactly (15, 8, 5, 5 iterations, identical cutbacks).
- **AMG done properly.** §3.5's AMG rows tested the configuration the literature already calls
  ineffective (monolithic, block-unaware, no near-null-space). Rigid body modes + a block
  preconditioner, per Alkmim et al. (2026), are the real test and are untried. Requires extending the
  AMGCL wrapper to accept pointer-valued parameters, or using Trilinos as that paper does.
- **Docs.** `doc/source/documentation/linsolvers.rst` now documents `inexactnewton`, `amgcl` and
  `matrixdump` and the corrected config-file claim (`8c01a7d3`). Still no Doxygen-equivalent needed
  (Python).
- **Tests.** `inexactnewton` now has a registered edelweiss-only regression test (`CantileverBeamQuad8-
  NeoHookeInexactNewton`, SuperLU delegate, dependency-free). Still nothing covers `matrixdump`,
  `factorize`/`solveFactorized`, or `pruneCondensedMatrixZeros=False`.
- **Pre-existing, unrelated:** `run_tests_edelweissfe ./testfiles/edelweiss-only/` has one failure,
  `NodeToDeformableSurfaceContactPullOut`. Verified to fail identically with this branch's changes
  stashed, so it predates this work — but it is a *contact* test, and this branch's subject matter is
  adjacent enough that it is worth a look.
- **Whether this branch should merge as-is.** `matrixdump` and `benchmark_linsolve.py` are
  investigation tooling; the timings and the phase-separated PARDISO methods are worth keeping
  regardless. Decide deliberately.

---

## 10. Phase 3, Lead 1 — measured

Lead 1 is implemented as `linsolver=inexactnewton` (`edelweissfe/linsolve/inexactnewton/`, commits
`2e3ce198`/`8c01a7d3`/`3a833817`). It is a self-contained `(A, b) → x` solver — **no change to the
Newton loop** — that keeps an exact PARDISO LU of one iterate and reuses it as a GMRES preconditioner
under an Eisenstat–Walker forcing sequence, reconstructing the signals it needs (`‖b‖` = the condensed
Newton residual drives the forcing and flags new increments; GMRES iteration counts flag staleness).
The delegate factorizing backend is PARDISO in production, SuperLU for a dependency-free path.

**Verified on the same pryout model, step 2, OMP=MKL=16** (`inexact.inp` = `profile.inp` +
`linsolver=inexactnewton` on step 2's `>>options`). Three policy variants were run to completion:

| variant | policy | step-2 linear solve | vs baseline 695.3 s |
|---|---|---|---|
| v1 | naive reuse, no cap | slower (stale reuses ground to 31–65 GMRES iters) | worse |
| v2 | cap 20 + backoff (mis-derived break-even 15) | 695.6 s | wash |
| v3 | cap 25 + backoff (break-even ~22, skip GMRES on exact solves) | 719.4 s | ~wash |

**Two firm conclusions:**

1. **The nonlinear cost is not inflated — the §9 risk is closed.** Every variant reproduced the
   direct-solve Newton path *exactly*: 15, 8, 5, 5 iterations with identical cutbacks. Eisenstat–Walker
   forcing (clamped to `[1e-6, 1e-3]`) does not cost Newton iterations here.

2. **Lead 1 is break-even on this model, not the 2–3× the offline benchmark projected.** Only **4–7 of
   44 solves** reuse cheaply; the rest refactorize or bail. The factorization saved by a reuse (~15 s of
   phase 11+22) is cancelled by the GMRES apply overhead (0.66 s × the iterations needed, which on this
   model is often 15–30 because the preconditioner is poor). Root cause is physical: contact + damage
   **active-set thrashing** changes the Jacobian enough between Newton iterations that a lagged
   factorization preconditions badly. §3.4's "4–9 iterations" was measured on a *converging tail*
   (dumped ords 2–10 of one increment) — more favorable than the full nonlinear path, which is
   dominated by the thrashing heads of the 15- and 8-iteration increments.

**What this implies for the plan.** The break-even is
`(reorder + factorization) / (one back-substitution)` ≈ 22 iters — and **phase 11 (reorder) alone is
38 % of every solve** (272 s of the 719 s), a reordering recomputed every iteration because the
condensed pattern churns (§3.1). So the reliable lever on *this* model is **Lead 2 (freeze the symbolic
factorization)**, which cuts that 38 % on *every direct solve* — where the time actually goes — rather
than trying to avoid the factorization entirely. Lead 2 is blocked on the §7 drift question (needs
Matthias's judgement). Note Lead 2 also *lowers* Lead 1's ceiling further: with phase 11 frozen, a
direct solve is cheaper, so even fewer reuses clear the (now lower) break-even.

`inexactnewton` is worth keeping regardless: it is correct, safe (robustly ≈ the direct baseline, never
the runaway v1 was), and should pay on problems with a slowly varying Jacobian (smooth plasticity,
stable propagation stages) rather than this damage-onset + contact window. It is **not** the win for
this model.

Reproduce: on xeon, `~/constitutive_modeling/next_v2611/Pryout_profile_investigation/`,
`PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHON_GIL=0 edelweissfe inexact.inp >
inexact.log 2>&1`. Per-solve trace is on (`inexact.json` has `"verbose": true`); baseline is
`baseline_omp16.log`.

---

## 11. Phase 3, Lead 3 step 1 — measured

The cheap, no-geometry gate from §4: does supplying smoothed aggregation a hand-built near-null-space
improve monolithic AMG *at all*? If three translations do not help over AMGCL's default single
constant, the full rigid body modes will not either, and block-unaware AMG is a dead end here.

**Enabler (commit `4828673a`):** AMGCL takes near-null-space vectors as a raw pointer in its property
tree — they cannot travel through the JSON parameter string — so the wrapper gained a `set_nullspace(B)`
entry point (`.hpp`/`.pxd`/`.pyx`, additive; the default one-constant behaviour is unchanged when it is
not called). This is the §3.5 "highest-value single change", and it is reusable for the block-AMG work.
`benchmark_linsolve.py amgcl --nullspace translations` builds the three displacement-block translations
from the DOF layout alone (node-major, 3 components/node) and runs SA-AMG with 1 constant vs 3
translations vs 3 translations + a coupled-block constant.

**Measured** (first dumped system, 280,155 dof, 40.9M nnz, 16 threads, gmres(100), rtol 1e-4, maxiter
300; direct-solve target 13.1 s):

| coarsening / smoother | 1 constant | 3 translations | 3 transl + coupled const |
|---|---|---|---|
| SA-AMG, ilu0 | 2.25e-1 ✗ | 3.00e-1 ✗ | 2.95e-1 ✗ |
| SA-AMG, spai0 | 6.53e-1 ✗ | 6.42e-1 ✗ | 6.54e-1 ✗ |

✗ = hit maxiter (300) without converging; the number is the true relative residual reached.

**Verdict: the near-null-space is not the missing ingredient — block structure is.** Every variant
stalls at 0.2–0.65 residual, orders of magnitude from 1e-4, and none beats the direct solve. The three
translations are *no better* than (worse than, for ilu0) the default constant, and adding a constant on
the coupled block — which rules out the objection that the translations' zero on the damage dofs left
that block degenerate — does not change the picture. (The variants do give distinct residuals, so
`set_nullspace` is genuinely taking effect; this is a real result, not a no-op.)

This closes the §4 gate exactly as it anticipated: monolithic, block-unaware AMG is ineffective on this
coupled system *regardless of the near-null-space*, consistent with Alkmim et al. (2026) §3.2. The only
AMG route that removes the memory ceiling (the *feasible-at-1M+* half of the goal) is the **full block
preconditioner** — nested B-AMG (AMG per field inside block Gauss–Seidel) or monolithic AMG-B (block
transfer operators + block smoothers), Lead 3 steps 2–3. That needs the rigid body modes (nodal
coordinates, i.e. widening `createSolver`'s interface) *and* the field-split `[0, 214659) | [214659,
280155)` block structure — the substantial effort §4 describes, now the only remaining path to the
size goal. `set_nullspace` is the first piece of it and is in place.

Reproduce: on xeon, in the profile dir, `OMP_NUM_THREADS=16 python $B amgcl linsolveDumps --nullspace
translations --tolerances 1e-4 --maxiter 300` (`B` = `scripts/benchmark_linsolve.py`). Log:
`amg_nullspace.log`.

---

## 12. Phase 3, Lead 3 step 2 — block AMG is feasible

§11 showed monolithic AMG is dead here. The literature's fix is a *block* preconditioner (AMG per
field inside a block Gauss–Seidel sweep, Alkmim et al. 2026). This tests that offline with **pyamg**,
which is *not* a production candidate — it is pure-Python and serial (re-enables the GIL), so its wall
time is meaningless against a 16-thread PARDISO. The metric that transfers to a parallel AMG (AMGCL,
Trilinos/MueLu) is the **GMRES iteration count**, a property of preconditioner quality, not of the
backend. Script: `scripts/block_amg_prototype.py` (requires `pip install pyamg`).

Field split `[0, 214659) | [214659, 280155)`; block Gauss–Seidel with one AMG V-cycle per field;
displacement block given the 3 translational near-null-space vectors (DOF layout only, no geometry),
damage block the default constant. First dumped system, 280,155 dof; direct PARDISO target ~11.5 s.

| preconditioner | GMRES iters to 1e-4 | dev. from direct |
|---|---|---|
| monolithic AMG (§11) | ✗ stalls at 2e-1 (300 iters) | — |
| **block-GS + per-field AMG** | **93–117, converged** | 1e-4 … 3e-6 |

**Feasibility is proven: block structure is the missing ingredient.** Where monolithic AMG never gets
below 0.2, the block preconditioner drives the coupled system to 1e-11. This is the go/no-go the whole
AMG line hinged on, and it is a **go** for the *feasible-at-1M+* goal (AMG memory is O(n); it removes
PARDISO's fill-in wall).

Diagnostics on *why* the count is ~100 and not ~30:

- **Not the block coupling.** Extra block-GS sweeps and a symmetric variant give *identical* 117
  iterations — one sweep already resolves the coupling.
- **The per-field AMG hierarchies are weak.** Solved alone, the displacement block needs **136–175**
  iterations and the damage block **35–49** — poor for SA-AMG (a healthy elasticity AMG is ~15–30).
  Default pyamg SA builds only **3 levels** on the 214k-dof displacement block: aggressive, weak
  coarsening on this condensed *elasticity + contact-penalty + tie-condensation* operator.
- **Diagonal scaling helps a little but is not the fix.** Symmetric `D^-1/2 A D^-1/2` (with the near
  null-space correctly transformed to `D^1/2 B`) took the block count 117 → 93 and the damage block
  49 → 35, but left the displacement block stuck (still 3 levels). The 1e8 dynamic range is real but
  secondary.

So the ~100 count is a **hierarchy-quality** problem, and reducing it is exactly the step-3 production
work: a properly configured *parallel* AMG (MueLu/Trilinos as the paper uses, or a carefully tuned
AMGCL), the full **6 rigid body modes** (rotations need nodal coordinates → the `createSolver`
interface widening §4 flags), Chebyshev/ILU smoothers, and a real strength-of-connection measure.
None of that changes the feasibility verdict; it changes the constant.

**Where this leaves the size goal.** Block AMG works and is the route to 1M+ dof. The production build
is substantial: (1) a *parallel* AMG usable as an inner block solve — the current AMGCL wrapper
rebuilds its hierarchy on every `solve()` and would need a build-once/apply-many split like PARDISO's
`factorize`/`solveFactorized`, or Trilinos; (2) a block Gauss–Seidel / field-split driver; (3) the 6
RBMs, which need nodal coordinates threaded into the linsolver interface; (4) diagonal equilibration.
`set_nullspace` (§11) is the first piece and is in place. This is multi-day, interface-widening work —
a deliberate decision, not a quick add.

Reproduce: on xeon, in the profile dir, `OMP_NUM_THREADS=16 python
~/constitutive_modeling/next_v2611/EdelweissFE/scripts/block_amg_prototype.py linsolveDumps`. Log:
`block_amg.log`.
