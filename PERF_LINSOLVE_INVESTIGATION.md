# Linear-solver performance investigation — handoff

Branch `perf/linsolve-investigation`, based on `feat/amr-recovery-marker` (`d495e90b`).
Working tree clean. **22 commits sit on top of `d83728ba`, all local — NOT pushed to `mn`.** xeon has
the changed files rsynced into its working tree for the experiments (its git is untouched and behind);
`setuptools` was `pip install`ed into xeon's `next_v2611` env (it was missing, which had been silently
blocking every extension rebuild there). See [§15](#15-what-remains) for the full state and next steps.

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
  quality (not the coupling).
- **Lead 3 step 3** ([§13](#13-phase-3-lead-3-step-3-started--the-amgcl-backend-hits-the-displacement-block),
  [§14](#14-phase-3-lead-3-step-3--feasibility-grade-blockamg-delivered)): backend = AMGCL (chosen).
  AMGCL's SA-AMG **capped at ~120 iterations on the displacement block** and the **6 RBMs did not
  help** — read at the time as the ~52% non-symmetric condensed elasticity operator defeating it
  (efficient convergence would need MueLu/Trilinos). But it *converged*, so a **feasibility-grade
  `linsolver=blockamg` was delivered**: field-split, per-field AMGCL hierarchies, block Gauss–Seidel,
  outer GMRES, with the field structure discovered from the DofManager (not hand-configured).
  Offline it solved the 280k-dof coupled system in **68 outer iterations** matching the direct solve
  to 4e-4; a registered test passes end-to-end. This is the O(n)-memory route to 1M+ dof.
- **A second verdict was reversed** ([§17](#17-phases-ab-executed--the-13-needs-muelu-verdict-is-retracted)):
  §13's "needs MueLu" conclusion rested on an under-swept AMGCL configuration. Two untried
  parameters — `aggr.eps_strong` and chebyshev's spectral-radius estimate — take the displacement
  block from 121–225 iterations to **29**, and the full 280k-dof coupled `blockamg` solve from 68 to
  **20–34** across all 9 dumped Newton iterates (only the known-hard `ord 3` iterate reaches 41).
  AMGCL stays the backend; MueLu/Trilinos and the block-valued 3×3 backend (B3) are no longer
  required for the feasibility goal, only possible further wins.
- **A third finding** ([§18](#18-iteration-count-and-wall-clock-are-not-the-same-metric--the-17-default-was-re-tuned)):
  §17's iteration-count-optimal config was not wall-clock optimal — 68% of a `blockamg` solve's time
  turned out to be the per-field AMG smoother *application* itself, not GMRES orchestration (that's
  only ~7%), so D1 (§16) is now a minor lever, not the primary one. A cheaper smoother
  (chebyshev degree=5, npre=npost=1, needing *more* outer iterations) cuts real wall-clock by
  **~23%** across all 9 dumped ords despite a looser (but still acceptable) true residual. Default
  updated again; offline-only so far, live re-validation still owed.
- **Phase 4 executed** ([§19](#19-phase-4-plan--wall-clock-to-parity-and-beyond), see its summary
  subsection): the gate passed — the shipped default is at **live PARDISO parity** (82.4 s vs
  83.9 s whole-job) — but both big levers failed their safety bars and stayed opt-in: EW forcing
  is **~1.56× offline** yet changed the live Newton *trajectory* (§19.2); mixed precision measured
  **~1.03×** because iteration inflation ate the bandwidth saving, and CSR *index* traffic (which
  float cannot shrink) turned out to be a large share of hierarchy bandwidth (§19.3). Hierarchy
  reuse shipped but is provably inert here (the pattern churns every iteration). The shipped
  default's behaviour is unchanged.
- **Phase 5 executed** ([§20](#20-phase-5-plan--b3-first-then-the-ew-rescue-experiment-on-the-final-setup)):
  Part 1 — **B3, the block-valued backend** — implemented and correct but **failed its bar**: a ~23%
  aggregate wall-clock *regression* inside the block-GS preconditioner (despite winning standalone;
  a better standalone solver is not a better single-cycle preconditioner). Default stays scalar.
  Part 2 — the **EW rescue** — **succeeded**: true-residual stopping closed the
  preconditioned-vs-true residual gap that caused §19.2's trajectory change, both `etaMax` ladder
  rungs passed strictly, and **EW forcing shipped as the default (`etaMax=3e-4`,
  commit `76cb09da`)**. Net: trajectory-safe and an accuracy gap closed, but live wall-clock stays
  at PARDISO parity (81.3 s vs 83.9 s) — fixed per-solve costs dominate, not outer iterations.
- **Phase 6 executed** ([§21](#21-phase-6-plan--attack-the-fixed-per-solve-costs-stable-pattern-part-a-cached-pattern-condensation-part-b)):
  attacked the fixed per-solve costs. **Part A stopped at A1**: with
  `pruneCondensedMatrixZeros=False` the sparsity pattern *still* churns every iteration (driven by
  contact/tie connectivity, not zero-pruning), so the shipped-but-inert hierarchy reuse has nothing
  stable to exploit — A2–A5 not attempted, per the plan's own stop condition. **Part B implemented,
  validated, and shipped as opt-in, not default**: the MPC condensation as a cached-pattern value
  scatter is correctness-equivalent (two real bugs found and fixed via the exactness assertion; full
  test suites plus a live 280k-dof trajectory match byte-for-byte on both PARDISO and blockamg) but
  measured **~1.7× slower**, not faster, on the reference model — the SpGEMMs the plan assumed were
  cheap (restricted to the eliminated DOFs' tiny row count) actually cost proportional to the full
  system size, since SciPy's SpGEMM must scan the left operand's full row range regardless of the
  right operand's sparsity. Ships as `useCachedMPCCondensation` (default `False`). D4 (the ≥1M-dof
  demonstration) remains the phase after.
- **Phase 7 in progress — 22.2's "falsified" verdict retracted by 22.2-bis; the phase resumes**
  ([§22](#22-phase-7-plan--p-multigrid-precondition-the-quadratic-displacement-block-through-a-p1-corner-node-operator)):
  **p-multigrid for the displacement block** — precondition the quadratic serendipity operator
  through a Galerkin-projected P1 corner-node operator (`A₁ = PᵀA₂P`, `P` purely topological), AMG
  on `A₁`, Chebyshev smoothing on the quadratic level. **22.1 (the enabler)** is done and committed:
  the corner/midside topology map, validated, with a real AMR-boundary finding resolved along the
  way (falls back to "corner", always safe for P1, every fallback reported). **22.2 (the go/no-go
  probe) reported the hypothesis falsified**, on a hierarchy-*shape* diagnostic and a fine smoother
  run on the Dirichlet-unmasked operator. **22.2-bis re-ran it with the right instrument and
  retracts that verdict**: `A₁` solved directly with AMG-preconditioned GMRES converges in **26
  iterations** (well under the ≤40 overturn threshold), and a two-grid scheme built with §17 B1's
  actual Dirichlet-free-submatrix construction on the fine level clears the original gate outright
  — **58 iterations vs. a 111-iteration single-level reference, a 2.30× projected-threaded
  speedup**. The asymmetry story is closed too: 22.2's alarming ~50% figure was the unmasked
  block; masked/free, it is 0.56-0.58%, matching §17 A1 almost exactly — never physics, always a
  Dirichlet-elimination storage artifact. Root cause of the miscalibration, now recorded in §8 so
  it cannot fire again: **smoothed aggregation's aggressive per-level coarsening (10-30×, few
  levels) is healthy by design**, not the classical-AMG "4+ levels, ~3-4×" this gate wrongly
  expected — this document's own §17 A2 data (a 3-level/~18× hierarchy delivering the phase's
  headline 29-iteration win) already proved that, unused at the time. Also found and fixed: SciPy's
  `gmres` `maxiter` counts restart cycles without a `callback`, not total iterations (also in §8
  now) — a probe requesting a ~500-iteration cap had silently asked for up to 50,000. Proceeding
  into 22.3 (coupled offline validation) and 22.4 (live gate, plumbing, ship decision) next. A
  separate, real finding surfaced along the way, unrelated to the hypothesis test: xeon's standard
  `OMP_NUM_THREADS`/`MKL_NUM_THREADS` run convention leaves numpy/scipy's OpenBLAS pthreads pool
  (not MKL — confirmed) uncoordinated with AMGCL's OpenMP pool, ~32 total threads rather than the
  intended 16 (not hardware oversubscription on this 36-core box, but a violation of the "16
  threads total" assumption every run command here has made).

Phase 3 delivered: `inexactnewton` (Lead 1) and `blockamg` (Lead 3) are both committed, documented, and
tested; `blockamg`'s default preconditioner was substantially retuned in [§17](#17-phases-ab-executed--the-13-needs-muelu-verdict-is-retracted)
(uncommitted). What is left is validation and the remaining swings (Lead 2 for speed, wall-time work
for `blockamg`) — see [§15](#15-what-remains) and [§17](#17-phases-ab-executed--the-13-needs-muelu-verdict-is-retracted).

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
- **Smoothed aggregation's per-level coarsening is aggressive by design — do not judge an SA
  hierarchy by its shape.** "4+ levels, ~3-4× coarsening" is a classical-AMG/geometric-multigrid
  expectation; SA aggregates whole strongly-connected neighborhoods, so few levels with ~10-30×
  coarsening and operator complexity ~1.1-1.3 is *healthy*, not pathological (§17 A2's own
  headline 29-iteration result ran on a 3-level, ~18×-coarsened hierarchy). §22.2 misapplied the
  classical-AMG ruler to an SA hierarchy and drew a false "falsified" verdict from it — see §22.2-bis.
  Judge an SA hierarchy by preconditioned convergence on its own operator, never by level count or
  coarsening ratio alone.
- **`scipy.sparse.linalg.gmres`'s `maxiter` counts restart cycles, not total inner iterations**,
  whenever no `callback` function is supplied — `callback_type` alone, without an actual
  `callback`, has *no effect* (per SciPy's own docstring). With `restart=100`, a probe intending a
  ~500-iteration cap that instead passes `maxiter=500` silently asks for up to 50,000 total
  preconditioner applications (§22.2). State the intended total iteration budget as
  `maxiter = ceil(budget / restart)` explicitly, and sanity-check a probe's first arm finishes in
  a plausible time before trusting a long queue of them.

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

---

## 13. Phase 3, Lead 3 step 3 (started) — the AMGCL backend hits the displacement block

Chosen backend: AMGCL (Matthias, this session). Before building the block-solver infrastructure, two
offline probes on the dumped blocks asked whether AMGCL can build a *good* per-field hierarchy — the
whole scheme's convergence rides on it.

**Enabler committed (`0c72bfa3`):** `EDELWEISS_DUMP_COORDS=<dir>` makes the solver dump per-field
nodal coordinates aligned with the DOF vector (the MPC transform is size-preserving, so the condensed
system keeps the DofManager ordering — a clean 1:1 map, no re-indexing). Verified to match a dumped
280k-dof system exactly (displacement 71553 nodes → `[0, 214659)`). This is the geometry a
geometry-aware preconditioner needs; the interface-widening for a *production* solver is still to do,
but the offline path is unblocked.

**Result — the damage block is easy, the displacement block is the wall:**

| block | best AMGCL (scaled) | near null-space |
|---|---|---|
| damage (scalar) | **18 iters** (SA + chebyshev) | constant |
| displacement (elasticity) | **121 iters** (SA + GS, 2 sweeps) | translations *or* 6 RBMs — **no difference** |

- Most smoothers *diverge* on the displacement block (chebyshev → 0.26–0.62, ilu0/iluk → 3.4/NaN);
  only Gauss–Seidel converges, and only at ~120–210 iterations. `aggr.block_size=3` did not help.
- **The full 6 rigid body modes give no improvement over 3 translations** (121 → 121, 140 → 129). So
  the near-null-space is *not* the missing ingredient here — the obstacle is the operator itself:
  the condensed displacement block is ~52% structurally non-symmetric (finite strain + contact + tie
  condensation), and AMGCL's runtime smoothed aggregation (a symmetric-operator method) cannot build
  a strong hierarchy on it, RBMs or not.

**Verdict.** Block AMG is *feasible* (step 2, §12) — but on **AMGCL** the displacement-field AMG caps
at ~120 iterations regardless of the near-null-space, so an AMGCL block solver would be
*feasibility-grade* (converges, O(n) memory → unlocks 1M+ where direct can't fit) but **not
efficient**. Reaching the ~20–40 iters the literature shows needs the sophisticated elasticity AMG
that **MueLu/Trilinos** provides (nonsymmetric-aware aggregation, elasticity-tuned smoothers) — the
stack Alkmim et al. actually used. This is the evidence that decides the backend, and it argues
against AMGCL for the displacement block.

Open decision (for Matthias): (a) build the AMGCL block solver anyway as feasibility-grade for the
1M+ goal (~120 iters, accept it), (b) switch the backend to Trilinos/MueLu (heavy dependency, but the
proven path to good convergence on this operator class), or (c) stop the AMG line here with
feasibility proven and the backend obstacle documented.

Reproduce: on xeon, in the profile dir — coordinates via
`EDELWEISS_DUMP_COORDS=coorddump edelweissfe profile.inp` (grab `coorddump/coordinates.npz` once the
displacement field reaches 71553 nodes); per-block quality via `python
~/constitutive_modeling/next_v2611/EdelweissFE/scripts/... ` prototypes (`amgcl_perblock.py`,
`amgcl_disp.py`, `amgcl_rbm.py` in the profile dir). Logs: `amgcl_perblock.log`, `amgcl_disp.log`,
`amgcl_rbm.log`.

---

## 14. Phase 3, Lead 3 step 3 — feasibility-grade `blockamg` delivered

The field-split block-AMG solver is built end-to-end on the AMGCL backend, as chosen. It is registered
as `linsolver=blockamg` and is a normal `(A, b) -> x` solver requiring a config file that names the
field block sizes (the block structure cannot be inferred from the matrix; the nonlinear solver logs
the sizes at equation-system build).

**Delivered (commits `a518fb92`, `37026172`, `32de9639`, `99af48c8`, plus the coord-dump enabler
`0c72bfa3`):**

- **`build` / `applyPreconditioner` on the AMGCL wrapper** (`a518fb92`): build the AMG hierarchy once,
  apply one V-cycle many times — the primitive an inner block preconditioner needs (the existing
  `solve` rebuilds every call). Mirrors PARDISO's `factorize`/`solveFactorized`.
- **`edelweissfe/linsolve/blockamg/`** (`37026172`): per solve — symmetric diagonal equilibration,
  split into field diagonal blocks + couplings, one AMGCL hierarchy per field (elasticity fields get
  their rigid-body translations as the near null-space from the DOF layout; scalar fields the default
  constant), block Gauss–Seidel preconditioning an outer GMRES.
- **Docs + a registered test** (`32de9639`, `99af48c8`): a single-field elasticity cantilever solved
  through the Newton loop with `blockamg` (skipped where AMGCL is not built).

**Measured:**

- *Offline*, the dumped 280k-dof coupled system: **68 outer GMRES iterations** (symmetric block-GS,
  `outerTol` 1e-4) to a solution matching the direct solve to **4e-4**. Better than the pyamg
  prototype's ~100 — the double-smoothed displacement AMG + Chebyshev damage + symmetric block-GS
  combination is effective.
- *Live*, the real pryout model, step 2 with `linsolver=blockamg` at `outerTol` 1e-6: the first
  step-2 solve converges in **115 outer iterations** to 1.85e-6, and the Newton loop consumes the
  correction and converges the increment. (Full-run confirmation of the Newton path is the last
  check.)

**What it is and is not.** It converges the coupled system with **O(n) memory** — the route to sizes
past the direct-solver wall (~1M+ dof), which was the *feasibility* half of the goal. It is **not**
fast at 280k dof (~50 s/solve serial-outer vs ~12 s direct) and its ~100-iteration count reflects
AMGCL's smoothed aggregation not converging tightly on the ~52% non-symmetric condensed displacement
block. The count would drop with a nonsymmetric-aware elasticity AMG (Trilinos/MueLu) or a stronger
smoother; the field-split machinery, the equilibration, and the DOF-layout null-space are all reusable
if that backend is ever swapped in.

**The field structure is discovered, not configured.** The block layout — each field's DOF range and
its nodal dimension — is pushed in from the DofManager by the nonlinear solver, via the
`FieldStructureAwareLinearSolver` mixin (`edelweissfe/linsolve/base.py`); the solver keys the near
null-space on the dimension (a vector field's per-component translations, a scalar field's constant),
so nothing about the blocks is specified by hand and the fields are named
(`displacement`, `nonlocal damage`), not tagged "elasticity". The config file carries only solver
knobs and is optional.

**Not done (future):** rotations for the vector-field null-space still need nodal coordinates
(`EDELWEISS_DUMP_COORDS` shows the path — they would ride the same `setFieldStructure` channel); and a
parallel outer Krylov (AMGCL's own, or a threaded matvec) would cut the serial GMRES overhead.

---

## 15. What remains

State at session end: both Phase-3 solvers (`inexactnewton`, `blockamg`) are implemented, documented,
tested, and committed — **16 commits on `d83728ba`, all local, nothing pushed to `mn`**. What is left:

### Validation (small, do first)

- **`blockamg` on the real *multi-field* model, new inferred-structure interface.** The registered
  test (`CantileverBeamQuad4BlockAMG`) passes end-to-end but is *single-field*. The 2-field inference
  is unit-checked and the numerics are unchanged from the hand-config run that reached 10 step-2 solves
  cleanly, but a full `blockamg.inp` run on the pryout with the *new* code was killed before step 2, so
  the "`blockamg: fields ['displacement', 'nonlocal damage']`" path and the full 15/8/5/5 Newton
  sequence under `blockamg` were **not** captured. One run confirms it (on xeon:
  `Pryout_profile_investigation`, `edelweissfe blockamg.inp`; ~35 min). Low risk.
- **`inexactnewton` full-run Newton cost** is done (§10): 15/8/5/5, unchanged. No action.
- **The pre-existing `NodeToDeformableSurfaceContactPullOut` failure** (edelweiss-only) is unrelated to
  this branch but is a contact test — still worth a look (§9).
- **Marmot testfiles suite** was not run with these changes.

### The two bigger swings (each its own effort)

- **Lead 2 — freeze the symbolic factorization.** The reliable *speed* lever for the 280k model
  (~1.78×, §3.2), because phase 11 (reordering) is ~35–38% of every direct solve and is recomputed
  every iteration only because the pruned pattern churns (§3.1). Not implemented. **Blocked on the §7
  drift question — needs Matthias's judgement** (you hit reference-path drift with a frozen pattern; the
  offline bench cannot test multi-increment drift). The pruning is already switchable
  (`pruneCondensedMatrixZeros`); what is missing is building `TᵀKT + C` on a cached pattern and
  scattering values into it, plus a solver that then freezes its reordering.
- **`blockamg` efficiency — a stronger vector-field AMG.** Today ~100 outer iterations, bottlenecked by
  AMGCL's smoothed aggregation on the non-symmetric displacement block (RBMs do not help, §13). To get
  to the literature's ~20–40: (a) **MueLu/Trilinos** (nonsymmetric-aware elasticity AMG, the paper's
  stack) as an alternative backend behind the same `blockamg` field-split machinery; or (b) **rotations**
  in the near null-space — they need nodal coordinates, which would ride the same `setFieldStructure`
  channel (`FieldBlock` gains a coordinates field; `EDELWEISS_DUMP_COORDS` already proves the extraction,
  and the offline test showed rotations alone did *not* rescue AMGCL, so this only helps with a better
  AMG); or (c) a **parallel outer Krylov** (AMGCL's own or a threaded matvec) to cut the serial scipy
  GMRES overhead — this addresses wall time, not iteration count.

### Housekeeping / merge

- **Push decision.** 16 local commits, nothing on `mn`. Push when ready (only to `mn`, per repo
  convention), and bring xeon's git up to date (it is on `731bb2b` in git terms, with newer files
  rsynced on top).
- **What merges vs stays tooling.** Keepers: `inexactnewton`, `blockamg` + its `FieldStructureAware`
  interface, the AMGCL `set_nullspace` / `build` / `applyPreconditioner` additions, the PARDISO phase
  timings and `factorize`/`solveFactorized`, the MPC/phase timings, `EDELWEISS_DUMP_COORDS`. Investigation
  tooling (decide deliberately): `matrixdump`, `scripts/benchmark_linsolve.py`,
  `scripts/block_amg_prototype.py`.
- **Docs**: `linsolvers.rst` is current (documents `inexactnewton`, `blockamg`, `amgcl`, `matrixdump`).
  A Sphinx build was not run.
- **`edelweissfe/linsolve/klu/klu.c`** is still not gitignored (§8) — the `git add`/clang-format trap
  bit twice this session on `.json` files instead; worth gitignoring `klu.c` and adding `*.json` to the
  clang-format exclude.

---

## 16. Reassessment of the §13 verdict — the AMGCL sweep was too shallow to conclude "needs MueLu"

Matthias did not accept the §13 conclusion. On re-reading the actual probes (`amgcl_disp.py`,
`amgcl_rbm.py` + logs on xeon), the skepticism is justified. What was actually swept on the
displacement block: coarsening ∈ {SA, unsmoothed aggregation} × smoother ∈ {GS(1), GS(2+2), ilu0,
iluk(1), chebyshev} × nullspace ∈ {3 translations, 6 RBMs}, `aggr.block_size=3`, Jacobi
equilibration — **~13 configurations, all on the scalar backend**. Best: SA + GS `npre=npost=2` at
121–128 iters. That is a reasonable first pass, but three levers with the highest prior expected
value for elasticity were never touched, and the causal story is unverified:

1. **The block-valued backend was never compiled.** `amgcl-wrapper.hpp` instantiates only
   `backend::builtin<double>`. `aggr.block_size=3` is *not* a substitute: it only groups dofs during
   aggregation, the smoothers stay scalar. AMGCL's canonical elasticity recipe is
   `builtin<static_matrix<double,3,3>>` + `adapter::block_matrix<3>`, which turns GS/ILU0 into
   *block* smoothers that invert the 3×3 nodal couplings exactly — in AMGCL's own elasticity
   benchmarks this is what makes ILU0 stop diverging and typically cuts iteration counts severalfold.
   The ilu0/iluk divergence observed in §13 is the classic scalar-on-elasticity symptom, not
   evidence about AMGCL.
2. **The "~52% non-symmetric operator defeats SA" hypothesis was asserted, never measured.** The 52%
   is *structural* (storage-pattern) asymmetry, which §3.1/§7 themselves trace to
   `eliminate_zeros()` pruning — a storage artifact. The **value** asymmetry
   `‖A−Aᵀ‖_F/‖A‖_F` of the displacement block was never computed, and the physics (hyperelastic UL
   + penalty contact are symmetric operators; the genuinely nonsymmetric damage coupling sits in the
   *off-diagonal* field blocks) suggests it may be small. One cheap decisive probe exists: build the
   AMG on the symmetrized block `(A+Aᵀ)/2` and use it to precondition GMRES on the true block. If
   chebyshev/SA come alive, the nonsymmetry story is wrong and no backend swap is needed.
3. **Strength-of-connection was never swept.** `aggr.eps_strong` stayed at its default while the
   block mixes concrete/steel stiffness, ~1e8 contact penalties, and tie condensation. The one
   anomaly already in the data points here: *unsmoothed* aggregation beat single-sweep SA (173 vs
   225) — the prolongation smoothing (whose spectral-radius estimate assumes symmetry and is
   penalty-sensitive) is plausibly being poisoned.

Also unexamined: what hierarchy AMGCL actually built (levels / operator complexity — pyamg managed
only 3 levels on this block; AMGCL's shape was never printed), the smoothing/cycle intensity curve
beyond `npre=npost=2` (which alone halved the count), and cross-Newton hierarchy reuse (AMG tolerates
lagging far better than an LU; today `blockamg` rebuilds every solve — a wall-time lever).

### Plan (all offline on the dumped block, minutes per probe on xeon)

**Phase A — diagnose (~1 h):**
- A1. Measure value asymmetry of the (scaled) displacement block; also with Dirichlet/identity rows
  masked. Settles whether the §13 causal story is even true.
- A2. Log AMGCL's hierarchy (levels, complexity, coarse size) for the current best config — AMGCL
  streams this via `operator<<` on the solver; surface it through the wrapper or a probe.

**Phase B — the untried levers, in expected-value order:**
- B1. Symmetrized-preconditioner probe (pure Python, no wrapper change): AMG on `(A+Aᵀ)/2`
  preconditioning GMRES on the true block; retry chebyshev under it.
- B2. Sweep `aggr.eps_strong` ∈ {0.0, 0.01, 0.08, 0.2, 0.4} × {SA, aggregation}; `npre/npost` ∈
  {2,3,4}; `ncycle=2` (W-cycle); 2 V-cycles per outer apply.
- B3. **Block-valued backend** (~half day): add a compiled `builtin<static_matrix<double,3,3>>`
  variant to the wrapper (`.hpp`/`.pxd`/`.pyx`, selected by a `blockSize` option), re-run the
  smoother sweep — block-ILU0 is the headline config. This is the experiment that actually decides
  AMGCL vs MueLu. Two caveats: AMGCL's `coarsening.nullspace` and block value types do not combine —
  acceptable, since §13 measured that RBMs don't help here anyway; and with a block backend, try
  3×3 **block**-Jacobi equilibration in place of the point-Jacobi scaling (point scaling slightly
  breaks the nodal block structure the backend exploits).
- B4. **Outer/inner Krylov hygiene** (near-free, add to the B2 sweep): every 121–225-iteration
  number was `gmres(M=100)` — i.e. it includes at least one restart, and restarted GMRES stalls
  precisely on operators like this. Re-run the best configs with `M≥300` (no restart), `fgmres`,
  and `idrs(4)` (which already beat GMRES in §3.5). Plausibly shaves 10–30% by itself.
- B5. **Chebyshev rehabilitation**: its §13 divergence may be the *spectral estimator*, not the
  method — the default power iteration assumes symmetry and few iterations. Sweep
  `relax.power_iters` up and try explicit `relax.lower`/scale bounds. Matters because Chebyshev is
  the cheapest strong smoother (matvec-only, perfectly threaded) — it is what makes the damage
  block cost 18 iterations.

**Phase C — fold in and validate:**
- C1. Winner into `_DEFAULT_VECTOR_PRECOND`; re-run the coupled offline bench (target ≤40 outer
  iters vs today's 68).
- C2. The still-owed live pryout `blockamg.inp` validation run (§15).
- C3. Update this document; only if the block still needs >80 iterations after B1–B5 is the
  "needs MueLu/Trilinos" conclusion earned.

**Phases A/B — done, and the §13 verdict is retracted.** See [§17](#17-phases-ab-executed--the-13-needs-muelu-verdict-is-retracted).

**Phase D — AMGCL-native production leads (after C, independent of each other):**
- D1. **Move the whole coupled solve into AMGCL** via `precond.class = schur_pressure_correction`
  with a `pmask` marking the damage field (pointer-valued, rides the same channel `set_nullspace`
  proved). Today's `blockamg` outer loop is scipy GMRES — serial matvec plus a Python block-GS
  callback per iteration (~50 s/solve live). Schur pressure correction is AMGCL's built-in
  two-field split; it makes the *entire* solve threaded C++ end-to-end. Compare against the
  hand-rolled block-GS at equal iteration counts; keep whichever wins wall-clock. (The hand-rolled
  driver stays as the >2-field general path.)
- D2. **Mixed precision**: compile the hierarchy backend as `builtin<float>` under the
  double-precision outer Krylov (a preconditioner does not need double). Halves preconditioner
  memory and bandwidth — directly serves the 1M+ goal and typically costs no iterations.
- D3. **Lagged hierarchy + Eisenstat–Walker**: reuse a built AMG hierarchy across Newton iterations
  instead of rebuilding every solve. The §10 break-even that killed lagged-LU does not transfer:
  an AMG rebuild is O(n) (seconds, not the 15 s reorder+factor), and a slightly stale *hierarchy*
  is still built from a nearby operator's graph — refresh it every k iterations or on the §10
  staleness signals. Reuse `inexactnewton`'s forcing logic for the outer tolerance (the live run
  used a fixed 1e-6 → 115 iters; §10 showed EW-clamped [1e-6, 1e-3] loses nothing on the Newton
  path — that alone should cut the live outer count substantially).
- D4. **Scale demonstration**: once the outer count is acceptable, run a ≥1M-dof model (uniformly
  refined pryout, or boxgen at scale) with `blockamg` vs PARDISO, recording peak RSS and wall.
  This closes the original feasibility question with data instead of extrapolation.

---

## 17. Phases A/B executed — the §13 "needs MueLu" verdict is retracted

Executed on xeon, same dumped displacement block (`linsolveDumps/A_00_00002.npz`, 214,659 dof,
28.1M nnz scaled). **Result: a properly tuned AMGCL configuration reaches 20–34 outer GMRES
iterations on the full 280k-dof coupled system (34/41/29/20/22/21/22/21/20 across all 9 dumped
Newton iterates) — inside the literature's ~20–40 range, without a block-valued backend (B3) and
without switching to MueLu/Trilinos.** §13's causal story ("the ~52% non-symmetric condensed
elasticity operator defeats AMGCL's smoothed aggregation") was an overstatement from an
under-swept configuration space, exactly as flagged when this phase was scoped.

### A1 — the "52% non-symmetric operator" was a storage artifact, not physics

Measuring `‖A−Aᵀ‖_F/‖A‖_F` (value asymmetry, not the structural/index-set asymmetry §3.1 already
measured) on the displacement block: **14.6%** raw, **141%** after point-Jacobi scaling — large,
seemingly confirming §13. But masking out the 24,160 Dirichlet/identity rows (`nnz==1`,
diagonal-only) **and their columns** drops it to **0.03%** raw / **0.58%** scaled. The condensed
elasticity operator itself is essentially symmetric; the asymmetry lives entirely in how Dirichlet
elimination interacts with `eliminate_zeros()` pruning (§3.1/§7), not in the physics (finite strain +
contact + tie condensation, which §13 blamed). This reframed B1 from "does a near-null-space fix
save monolithic AMG" (already answered no, §11/§13) to "is the operator that reaches AMGCL actually
as pathological as claimed" (answered: no).

### A2 — the hierarchy really was shallow, confirmed directly on AMGCL (not just pyamg)

Added `LinearSolver::report()` to the AMGCL wrapper (`amgcl-wrapper.hpp`/`.pxd`/`.pyx`, additive,
streams AMGCL's own `operator<<`) so the C++ hierarchy stats are inspectable from Python. Default
SA on the raw block: **3 levels**, operator complexity 1.22 (214659 → 11658 → 1224, i.e. ~18×
coarsening the first level — far more aggressive than healthy SA's usual ~3–4×). This matches §12's
pyamg finding but was previously never confirmed on AMGCL's own runtime aggregation. Root cause
turned out to be the untuned strength-of-connection threshold (B2, below), not a backend limitation.

### B1-corrected — a genuine numerical hazard found, then the real test run safely

The first attempt tested "does symmetrizing the *hierarchy-build* operator let chebyshev/ilu0 stop
diverging" by building AMG on `(A+Aᵀ)/2` and applying it via a Python `scipy.sparse.linalg.gmres`
+ `LinearOperator` wrapping the wrapper's `build()`/`applyPreconditioner()`. **This produced NaN
from iteration 0**, confirmed via isolated `applyPreconditioner()` calls (input matrix verified
clean — no NaN/Inf, canonical CSR, no duplicate indices). Root cause: a Dirichlet dof's row is a
pure identity (diag 1, nothing else), but `eliminate_zeros()` deliberately keeps the corresponding
*column* entry in other rows (§3.1/§7). Naively symmetrizing hands that identity row a new spurious
off-diagonal coupling (half the kept column entry), which corrupts the coarsest AMG level into
near-singularity. **New gotcha for §8**: never symmetrize a Dirichlet-eliminated operator without
first masking the Dirichlet rows/columns. Compounding this, a NaN preconditioned residual means
GMRES's stopping check (`pr_norm <= tol`) never fires — comparisons against NaN are always false —
so the solve ran for tens of thousands of iterations instead of the intended few hundred before it
was noticed and killed (see "gotchas" below).

Rerun correctly — masking the Dirichlet rows/columns (A1's approach) rather than symmetrizing them —
on the resulting 190,499-dof free submatrix, with **no spectral tuning at all**:

| config | iters | note |
|---|---|---|
| SA + Gauss–Seidel, npre=npost=2 | 92 | vs 121–128 on the full (Dirichlet-corrupted) block |
| SA + chebyshev, default | **117, converges** | vs 300/diverged (0.26–0.62 residual) on the full block |
| SA + ilu0, default | 300, diverges | unchanged — ilu0's problem is separate from Dirichlet handling |

Confirms A1 directly: once the Dirichlet-row corruption is removed, chebyshev converges even with
zero tuning. ilu0 still fails — its divergence is not explained by the Dirichlet artifact, so it
remains a dead end here (consistent with a genuinely poor fit to this operator's nonsymmetric
tie/contact terms, which A1 shows sit in the *free* submatrix, not the Dirichlet rows).

### B2 — the default strength-of-connection threshold was bad

Sweeping `aggr.eps_strong` × {smoothed_aggregation, aggregation}, GS npre=npost=2, on the full
(untouched) block:

| eps_strong | SA iters | aggregation iters |
|---|---|---|
| default (unset) | 129 | 83 |
| 0.0 | 102 | 94 |
| **0.01** | **76** | **78** |
| 0.08 | 121 | 82 |
| 0.2 | 183 | *(memory blowup, see below)* |
| 0.4 | 212 | *(not run)* |

`eps_strong=0.01` roughly halves the default's iteration count for both coarsening types. **New
gotcha for §8**: `aggregation` (plain, non-smoothed) coarsening at `eps_strong=0.2` caused an
**83.7 GB memory blowup** (44.6% of xeon's 187 GB) — almost certainly a too-high strength threshold
classifying most connections as weak, producing a barely-coarsening, many-level hierarchy. Killed
before it produced a result or threatened the shared machine; the two untested `AG` values were
skipped rather than re-risked, since SA's own numbers already show high `eps_strong` is worse, not
better. Any future `eps_strong` sweep should run with a hard RSS cap (as `remaining_sweep.py` now
does — self-aborts above 60 GB) rather than trusting wall-clock alone to catch it.

### B5 — chebyshev's divergence was a bad spectral-radius estimate, not a fundamental limit

Confirms the hypothesis exactly. Sweeping `relax.power_iters` (fused `solve()`, full block, default
`degree`/`lower`):

| power_iters | iters | note |
|---|---|---|
| 0 (AMGCL's cheap default) | 300, diverges | res 0.62 — this is what §13 measured |
| 5, 10 | 300, diverges | res 0.95–0.99, worse |
| 20 | 196, converges | |
| 50 | **96, converges** | |
| 100 | 97, converges | (no further gain past 50) |

Then sweeping explicit `degree`/`lower` bounds at `power_iters=50`:

| degree | lower | iters |
|---|---|---|
| 3 | 0.0333 | 173 |
| 3 | 0.01 | 142 |
| 5 | 0.0333 | 96 |
| 5 | 0.01 | 88 |
| 8 | 0.0333 | 79 |
| **8** | **0.01** | **65** |

A short power iteration (AMGCL's default) badly underestimates the spectral radius on this operator;
a longer one plus a tighter explicit lower bound fixes it outright — no operator-level intervention
(symmetrization, block backend) needed. This is the more important and more elegant fix than B1's
symmetrization idea, and it doesn't carry B1's numerical hazard.

### Combined — B2 + B5 stack, and npre/npost matters more once the smoother is good

| config | iters |
|---|---|
| SA (default eps_strong) + chebyshev(d=8,lower=0.01,pi=50), npre=npost=1 | 65 |
| SA (default eps_strong) + chebyshev(d=8,lower=0.01,pi=50), npre=npost=2 | 49 |
| SA (default eps_strong) + chebyshev(d=8,lower=0.01,pi=50), npre=npost=3 | 41 |
| SA eps_strong=0.01 + chebyshev(d=8,lower=0.01,pi=50), npre=npost=1 | 38 |
| **SA eps_strong=0.01 + chebyshev(d=8,lower=0.01,pi=50), npre=npost=2** | **29** |

**29 iterations on the displacement block alone** — down from 121–225 (§13) — is the session's
headline number. `eps_strong` and the chebyshev spectral fix are synergistic (each alone gives
~65–78; combined gives 29), and going past npre=npost=2 wasn't tried further (diminishing returns
were already visible: 65→49→41 for npre=npost=1→2→3 on the untuned-eps_strong variant).

### B4 — Krylov hygiene: restart wasn't the bottleneck for the good preconditioner

On the chebyshev(d=8,lower=0.01,pi=50) config (default eps_strong, 65-iteration baseline):

| outer solver | iters |
|---|---|
| gmres, M=100 (restart) | 65 |
| gmres, M=300 (no restart) | 65 |
| fgmres, M=100 | 65 |
| idrs, s=4 | 120 |
| bicgstab | 97 |

Identical iteration counts with and without restart settle B4's original concern: this
preconditioner already converges well inside 100 Krylov vectors, so restart wasn't costing anything.
IDR(s) and BiCGStab are both worse here — standard GMRES remains the right outer solver. (This
contradicts §3.5's finding that IDR(s) beat GMRES — but that was on the *monolithic*, ILU0-only
system; with a good block-AMG preconditioner the picture reverses.)

### Folded into production, validated on the full coupled system

`blockamg._DEFAULT_VECTOR_PRECOND` (`edelweissfe/linsolve/blockamg/blockamg.py`) updated to
`{coarsening: smoothed_aggregation with aggr.eps_strong=0.01, relax: chebyshev(degree=8,
power_iters=50, lower=0.01), npre=npost=2}` — **uncommitted**, ready for review. Re-ran the full
280,155-dof coupled offline bench (`blockamg`'s actual field-split + block-GS + outer GMRES, not the
displacement block in isolation) across **all 9 dumped Newton iterates**:

| ord | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| outer iters | 34 | 41 | 29 | 20 | 22 | 21 | 22 | 21 | 20 |

Down from **68** (§14) to a **20–34** range (ord 3 at 41 is the sole, expected exception — the
known-hard post-large-correction iterate flagged since §3.4/§10). This is inside both the §16 gate
(≤80, avoiding the MueLu conclusion) and the literature's ~20–40 target (§4). Residuals match the
outer tolerance (1e-4) to within the solver's usual preconditioned/true-residual gap (§14 already
noted a 4e-4 gap on this same solver; ord 3 shows the largest gap here too, 1.6e-2, consistent with
it being the hardest iterate). Wall time per solve is 14–24 s (single-threaded scipy outer GMRES +
Python block-GS callback) — worse than PARDISO's ~11.5 s, but wall time was never this phase's
target; D1 (Schur-pressure-correction, fully-threaded C++ outer solve) is the lever for that.

### Revised verdict

**§13's conclusion — "AMGCL caps at ~120 iterations regardless of near-null-space; efficient
convergence needs MueLu/Trilinos" — is retracted.** It was correct about the specific configurations
tested (default eps_strong, default/untuned chebyshev, RBMs) and correct that RBMs don't help (still
true here — rotations were never revisited this session and B2/B5's wins are orthogonal to the
near-null-space question). But the configuration space was too shallow to license a backend-level
conclusion: two parameters neither swept nor diagnosed in §13 (`aggr.eps_strong`, chebyshev's
spectral-radius estimate) account for essentially the whole gap between "stalls at 120" and "29".
Neither required a block-valued backend (B3, still untried) or a different AMG library. **AMGCL
stays the backend.** B3 (the 3×3 block-valued backend) is now a stretch goal for going below ~20
iterations, not a requirement for feasibility.

### Gotchas worth not rediscovering (additions to §8)

- **Never symmetrize a Dirichlet-eliminated operator naively.** `(A+Aᵀ)/2` gives identity
  (Dirichlet) rows spurious off-diagonal coupling from `eliminate_zeros()`'s asymmetric column
  retention (§3.1/§7), which can corrupt the coarsest AMG level into near-singularity and produce
  NaN. Mask Dirichlet rows/columns instead of symmetrizing across them.
- **A NaN preconditioned residual defeats GMRES's stopping check silently.** `pr_norm <= tol` is
  always false against NaN, so a broken preconditioner doesn't fail fast — it runs to whatever
  iteration cap applies (and scipy's `maxiter`/`restart` interaction can make that cap far larger
  than expected), looking like a hang rather than a crash. Always sanity-check
  `applyPreconditioner()`/`solve()` output for NaN/Inf before trusting a "still running" job.
- **`aggr.eps_strong` too high on plain `aggregation` coarsening can blow up memory**, not just
  iteration count — 83.7 GB observed on a 214k-dof block. Sweep this parameter with a hard RSS cap.
- **`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` must be set alongside `OMP_NUM_THREADS` even for
  offline probe scripts**, not just full simulation runs — scipy/numpy BLAS calls (diagonal
  scaling, symmetrization, matvecs) oversubscribe otherwise. Two probes sharing a 36-core box each
  spawned 31 threads (62 total) with only `OMP_NUM_THREADS=16` set on each.

### What remains after this phase

- **Push the config to git** (currently uncommitted local edits): `blockamg.py`'s new default,
  `amgcl-wrapper.hpp`/`.pxd`/`.pyx`'s additive `report()` method.
- **Rotations were not revisited.** B2/B5's wins are independent of the near-null-space question;
  §13's finding that 6 RBMs don't improve over 3 translations stands untouched.
- **Wall time is still unaddressed** (D1–D3 in §16) — this phase was entirely about iteration
  count, which was the metric that transfers to a real parallel AMG backend (§12's framing).
- **The live pryout `blockamg.inp` validation run** (§15) is still owed, now with the improved
  default — worth re-running given the outer count dropped by half.
- Only ord 2's dumped system had per-config sweeps (A1/B1/B2/B5/B4); the final validated
  precond was checked against all 9 ords, but the *diagnostic* sweeps (eps_strong values,
  chebyshev bounds) were not repeated across ords. Given the consistent 20–34 range on the final
  config, this is low-risk, but worth knowing if a future ord behaves anomalously.

---

## 18. Iteration count and wall-clock are not the same metric — the §17 default was re-tuned

§17 optimized purely for outer GMRES iteration count, deliberately (pyamg, used for the earlier
feasibility check, is serial, so iteration count was the only metric that transfers to a parallel
backend). AMGCL *is* parallel, so once §17's config went live, wall-clock became directly
measurable and worth checking against. It wasn't optimal.

### Where the time actually goes

Instrumented a real `blockamg` solve on the dumped 280k-dof coupled system (ord 2, §17's tuned
default: chebyshev degree=8, npre=npost=2, eps_strong=0.01), timing every stage of
`BlockAMGSolver.__call__` directly (equilibration, block split, hierarchy build, and — inside the
outer GMRES loop — the full-system matvec, the per-field AMG apply, the off-diagonal coupling
matvec, and Python glue, with the remainder attributed to GMRES's own Arnoldi/Givens orchestration):

| stage | time | share |
|---|---|---|
| equilibration + block split | 2.1 s | 8% |
| AMG hierarchy build (once per solve) | 3.4 s | 13% |
| **per-field AMG smoother apply** | **17.4 s** | **68%** |
| full-system matvec (scipy CSR, serial) | 1.6 s | 6% |
| off-diagonal coupling matvec (scipy, serial) | 0.75 s | 3% |
| GMRES orchestration (Arnoldi/Givens) | 0.3 s | 1% |
| Python loop glue | 0.05 s | ~0% |

**D1 (moving the outer solve into AMGCL's own runtime via `schur_pressure_correction`, §16) has at
most a ~7% ceiling here** — GMRES orchestration and the serial matvecs it was meant to eliminate are
a small fraction of the total. The dominant cost (68%) is the per-field smoother *application*
itself: chebyshev degree=8 with npre=npost=2 is a genuinely expensive smoother per V-cycle (an
8-term polynomial ≈ 8 matvec-equivalents, times 4 sweeps — 2 pre, 2 post — times multiple levels).
§17 chose it purely because it minimized outer iterations; nothing in that phase priced the
per-application cost.

### A wall-clock sweep finds a materially better default

Compared the §17 default against cheaper-smoother candidates on the real coupled solve (via
`blockamg`'s own `fieldPreconds` override — the actual production code path, not a stand-in),
first on ord 2:

| config | outer iters | wall |
|---|---|---|
| §17 default: cheby d=8, npre=npost=2 | 34 | 20.6 s |
| cheby d=3, npre=npost=1 | 74 | 20.0 s (wash — too few sweeps to compensate for more iters) |
| **cheby d=5, npre=npost=1** | **40** | **14.1 s** |
| Gauss-Seidel, npre=npost=1 | 106 | 35.8 s |
| Gauss-Seidel, npre=npost=2 | 42 | 23.7 s |

`degree=5, npre=npost=1` needs *more* outer iterations (40 vs 34) but is markedly cheaper per
application, netting a clear wall-clock win. Confirmed across all 9 dumped ords, not a one-off:

| ord | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| §17 default wall | 21.2s | 24.2s | 18.5s | 13.8s | 15.4s | 15.0s | 16.3s | 14.9s | 14.3s | **153.5s** |
| new candidate wall | 14.5s | 21.8s | 13.5s | 12.3s | 13.5s | 12.8s | 13.5s | 11.6s | 11.9s | **125.2s** |
| speedup | 1.47× | 1.11× | 1.37× | 1.13× | 1.14× | 1.17× | 1.21× | 1.29× | 1.20× | **1.23×** |

**~23% aggregate wall-clock reduction, winning on every single tested iterate** including the
known-hard `ord 3` (where the margin is smallest, as expected — it's the iterate that needs the
most outer iterations regardless of smoother).

**Trade-off:** the candidate's true residuals are looser (typically 1e-4–3e-4 vs ~5e-5 for the §17
default) — a smaller npre/npost gives GMRES's internal (preconditioned) residual estimate more room
to diverge from the true residual before the outer tolerance triggers. `ord 3` reaches 3.1e-3 for
the candidate — still *tighter* than the §17 default's own 1.6e-2 on that same hard iterate, so this
isn't a new problem specific to the candidate, but it's worth knowing the margin narrows.

### `_DEFAULT_VECTOR_PRECOND` updated again

`blockamg.py`: `{coarsening: smoothed_aggregation with aggr.eps_strong=0.01, relax: chebyshev(degree=5,
power_iters=50, lower=0.01), npre=npost=1}` — supersedes §17's `degree=8, npre=npost=2`. Synced to
xeon and since **committed as `3209a98a`** (§17's config was `533f4d6f`; this section's profiling is
`db0c0274`).

**Live validation status:** the §17 default (`degree=8, npre=npost=2`) *was* validated live on the
real pryout model (`blockamg.inp`, this session) — 32–40 outer iterations observed, matching the
offline range, Newton path unaffected (normal 5-iteration convergence per increment), run ended only
because it hit the deliberate `maxNumInc` test cap (§6), not a real failure. **The new
degree=5/npre=npost=1 default has only been validated offline (all 9 dumped ords, above) — the
equivalent live re-run is still owed** before treating this as settled, given the looser-residual
trade-off could plausibly cost a Newton iteration or two even though the offline true-residual
numbers suggest it won't.

### Revised priority for the remaining wall-time work (supersedes §16's Phase D ordering)

- **D1 (Schur-pressure-correction / native AMGCL outer solve) is now a minor lever (~7% ceiling
  here), not the primary one.** Still worth doing eventually (it also removes the Python-level
  block-GS glue as system size grows), but it is no longer the first thing to reach for.
  D2/D3 (mixed precision, lagged hierarchy + Eisenstat–Walker) are unaffected by this reordering.
- **The real lever is smoother cost-per-application vs. outer iteration count, and it needs to be
  swept on WALL-CLOCK, not iteration count**, for any further tuning — §17's sweeps (which produced
  the degree=8/npre=npost=2 "winner") cannot be trusted for wall-time conclusions on AMGCL, only on
  the serial-pyamg feasibility question they were originally scoped for. A finer sweep around
  degree ∈ {4, 5, 6} × npre/npost ∈ {1, 2} was not done — diminishing returns are plausible this
  close to the found optimum, but not confirmed.
- **The scalar (damage) field's smoother was never re-examined for the same trade-off** — it's
  cheap by construction (default chebyshev, no tuning applied in §17) and is a small fraction of
  the per-field-apply cost dominated by the much larger displacement field, so this is low priority.

---

## 19. Phase 4 plan — wall-clock to parity and beyond

Planned (this session), not started. Everything below is scoped for hand-off execution; each task
names its files, its validation gate, and its expected win. Read §18 first — its stage breakdown
(68% smoother apply / 21% per-solve fixed cost / ~9% outer-loop overhead) is what this plan attacks.

### 19.0 Pre-verified this session: the smoother apply is threaded — no free serial win exists

Before planning, the one "cheap explanation" was ruled out: could the dominant 68% smoother-apply
cost simply be running single-threaded?  No:

- `setup.py:210-219` builds the `linsolve/amgcl` extension with `-fopenmp` in both
  `extra_compile_args` and `extra_link_args`.
- AMGCL's `builtin` backend header on xeon (`$CONDA_PREFIX/include/amgcl/backend/builtin.hpp`)
  contains **45 `#pragma omp` sites** — the matvec/smoother kernels behind `applyPreconditioner`
  are OpenMP-parallel. (The AMGCL headers are installed on **xeon only**, not on the laptop env —
  do not conclude "AMGCL missing" from a local `find`.)

Consequence: the smoother apply saturates memory bandwidth at 16 threads. The only levers on it are
**fewer outer iterations** (19.2), **less bandwidth per application** (19.3), and **cheaper
smoothers** (19.4). More threads or moving orchestration to C++ (D1) will not touch it.

Side observation, unrelated to blockamg but recorded before it surprises someone: neither
`numerics/csrgenerator.pyx` nor `csrgeneratorv2.pyx` contains a `prange` — the assembly CSR
generators are serial Cython despite their OpenMP build flags. At 0.078 s/call this is irrelevant
here; just know the "OpenMP" label on those extensions is aspirational.

### 19.1 GATE — the owed live validation run (~35 min, do before anything else)

The current default (`degree=5, npre=npost=1`, committed `3209a98a`) is validated **offline only**.
Its looser true residuals (1e-4–3e-4 vs ~5e-5, §18) could plausibly cost a Newton iteration, which
would eat the 23% win. Confirm before building on it:

- On xeon, `~/constitutive_modeling/next_v2611/Pryout_profile_investigation/` (**never** the real
  project dir, §6): `PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHON_GIL=0
  edelweissfe blockamg.inp > blockamg_d5.log 2>&1`.
- Pass criteria: Newton path unchanged (15, 8, 5, 5 iterations, identical cutbacks vs
  `baseline_omp16.log`); outer counts ≈ the offline 20–46 range; per-solve wall ≤ the §18 offline
  numbers + overheads. The §17 default's live run showed 32–40 outer iters, so 35–50 with the
  cheaper smoother is expected, not alarming — wall-clock is the metric, not the count.
- If a Newton iteration IS lost: revert the default to §17's `degree=8, npre=npost=2` (one dict in
  `blockamg.py`) and proceed with the plan anyway — 19.2 and 19.3 are independent of which smoother
  wins.

**GATE RESULT: PASS.** The run was already sitting on xeon as `blockamg_live_retuned.log` (produced
just before this hand-off, same command as above, against the `degree=5, npre=npost=1` default that
is live in the rsynced `blockamg.py`) — verified rather than re-run, since re-running would only
reproduce it:

- **Newton path: identical.** 15, 8, 5, 5 iterations per increment, same three cutback points
  (`0.0025`, `0.00125`, `0.000625`), same termination (`Reached maximum number of increments`, the
  deliberate `maxNumInc` test cap, not a real failure) — all byte-for-byte the same sequence of
  journal messages as `baseline_omp16.log` (PARDISO). Zero risk of the looser-residual trade-off
  costing an iteration materialized.
- **Outer GMRES counts:** min 27, max 53, mean 35.0 over all 44 solves — matches the offline 20–46
  range closely enough (top tail 7 iters over) to not be alarming, exactly as anticipated.
- **No NaN/Inf, no "unknown parameter" AMGCL warnings** anywhere in the log (checked stdout+stderr,
  both redirected to the same file per §8).
- **Wall-clock — with a caveat.** The per-call `linear solve` entry in the run's own
  `acc. runtime` table (907.7 s / 44 calls = 20.6 s/call) is **not trustworthy as wall-clock**: it
  exceeds the run's total `Job computation time` (82.4 s) by >10×, which is only possible if the
  category's accumulator is being corrupted rather than genuinely summing 44 non-overlapping
  intervals. Traced to `edelweissfe/utils/performancetiming.py`'s `timeit` decorator: `_currentStackLevel`
  is a **class attribute**, mutated by every decorated call with no lock
  (`self._parentStackLevel = timeit._currentStackLevel; timeit._currentStackLevel = timer`) — a
  classic race under `PYTHON_GIL=0` if any other decorated call runs concurrently. The *same*
  pathology (`linear solve` acc. runtime 695.3 s / 44 calls, vs `Job computation time` 83.9 s) is
  present in `baseline_omp16.log` too, i.e. **this is pre-existing and unrelated to blockamg or this
  session's changes** — not something to fix under this plan, but worth flagging as a new gotcha
  (below) since 19.4 explicitly asks for wall-clock judgement and this table is the wrong place to
  read it from on a live run.
  - The one number in this log immune to that bug is `Job computation time` itself (measured once,
    start-to-finish, not through the decorator): **82.4 s** for the retuned-blockamg run vs **83.9 s**
    for the PARDISO baseline — i.e. the full job (AMR + assembly + 44 solves + overhead) is already
    at parity with PARDISO, not just "improved." This is consistent with, and stronger than, what the
    §18 offline projection implied.
  - The authoritative *offline* wall-clock numbers (§18's 125.2 s / 9-ord total) come from
    `scripts/benchmark_linsolve.py` timing calls directly with its own `perf_counter()`, not through
    this decorator, so they are unaffected and remain the reference figures for 19.2–19.4.

**New gotcha (addition to §8/§17):** `performancetiming.timeit`'s global `_currentStackLevel` is not
thread-safe; under `PYTHON_GIL=0` its per-category `acc. runtime` figures on a live run can be
inflated by an order of magnitude and must not be used for wall-clock comparisons — use `Job
computation time` (whole-job) or a script-level `perf_counter()` (offline benchmarks) instead.

Log: `blockamg_live_retuned.log` (xeon, `Pryout_profile_investigation/`), cross-checked against
`baseline_omp16.log`. Reproduce with the command above if re-verification is ever needed.

### 19.2 D3 — Eisenstat–Walker forcing + lagged hierarchy (highest expected value, ~1 day, pure Python)

Iteration count multiplies the dominant 68% cost, so this is the biggest lever. Two independent
halves, both in `edelweissfe/linsolve/blockamg/blockamg.py`:

**(a) Adaptive outer tolerance.** `blockamg` currently solves every system to a fixed
`outerTol=1e-6`. §10 already proved EW forcing clamped to `[1e-6, 1e-3]` reproduces the Newton path
*exactly* on this model, and `inexactnewton` (`edelweissfe/linsolve/inexactnewton/`) already
contains the signal reconstruction — a solver sees only `(A, b)`, so it infers new-increment /
staleness state from `‖b‖` jumps. Lift that logic (or extract it into a shared helper). Most Newton
iterates only need 1e-3–1e-4; at 1e-4 the offline counts are 20–41, and the *live* 1e-6 run in §14
needed 115 — the tolerance is worth an outright ~2× on outer iterations for most solves.

**(b) Hierarchy reuse across Newton iterations.** Today `__call__` rebuilds everything per solve:
equilibration + block split (~2.1 s) + AMG hierarchy build (~3.4 s) ≈ 21% of a solve. Keep the
built per-field hierarchies (and the split/equilibration arrays) on the instance and reuse them for
subsequent solves. The §10 lagged-LU break-even does **not** transfer, because the economics are
inverted: a stale AMG *preconditioner* costs a few extra outer iterations at ~0.3–0.4 s each, while
the refresh it avoids costs only ~5.5 s — and correctness is untouched because the outer GMRES
always operates on the **fresh** matrix (`As`); only `M` is stale. Refresh policy: rebuild on
`‖b‖`-jump (new increment / post-cutback — reuse (a)'s signal), or when the outer count exceeds
~1.5× the previous solve's. Keep it simple; the failure mode is just extra iterations, never a
wrong answer (GMRES converges on the true residual regardless of `M`).

Caveat for (b): the preconditioner was built for the *old* equilibration `D`. Either reuse the old
`dinv` for scaling the new system too (self-consistent, simplest — the diagonal drifts slowly), or
rebuild whenever `max |dinv_new/dinv_old − 1|` exceeds ~10%. Do not mix old hierarchies with new
scaling.

Validation: offline first — extend `scripts/benchmark_linsolve.py`'s blockamg path (or a small new
probe) to replay all 9 dumped ords *as a sequence* through one solver instance, confirming (i) the
reused-hierarchy counts stay near the fresh-build counts (the §3.4 staleness data says they will),
(ii) `ord 3` triggers the refresh. Then one live run as in 19.1; Newton path must stay 15/8/5/5.
Expected combined win: ~1.5–2× on most solves (fewer iterations × amortized fixed cost).

**Implemented** — `edelweissfe/linsolve/blockamg/blockamg.py` (commit `6811d4e4`). Both halves
landed in `__call__`, plus a third refresh trigger that (b)'s plan text above did not anticipate.
`outerTol`, if explicitly given, still fixes the tolerance and fully disables (a) (preserves the old
behaviour for the existing `CantileverBeamQuad4BlockAMG` regression test and anyone with a config
file that already sets it); the new EW knobs (`etaMin`/`etaMax`/`ewGamma`/`ewAlpha`/
`residualGrowthFactor`/`hierarchyStalenessFactor`) are wired through the `blockamg` registry factory
in `__init__.py` too.

**Offline validation — a real hazard found and fixed before it reached xeon's config.** A new probe,
`probe_192_sequence.py` (ad hoc, `Pryout_profile_investigation/`, not checked in — same convention as
the other loose sweep scripts there), replays all 9 dumped ords through one `BlockAMGSolver` instance
in call order and diffs against 9 independent fresh-build solves at the old fixed `outerTol=1e-6`.
First attempt (reuse triggered only by `‖b‖`-jump / field-structure-change / outer-count-drift, exactly
as planned above) reproduced the plan's own already-stated caveat the hard way: **`ord 3`, reusing the
hierarchy built for `ord 2`, needed 494 outer GMRES iterations (136 s) against 94 fresh (27 s) at the
same tolerance** — not "a few extra iterations," a 5× wall-clock regression on that one solve, because
this module's own docstring already documents that *the sparsity pattern churns between Newton
iterations* on this condensed contact/tie system (§3.1) — confirmed directly: `ord 2`→`ord 3`'s `nnz`
drops from 40910093 to 40687250, a different pattern, not just a moved Jacobian. Nothing in 19.2(b)'s
two stated triggers (residual jump, outer-count drift) catches a pattern change that happens to occur
on a residual *drop* (Newton converging normally, no jump) — exactly ord 3's case relative to ord 2.

**Fix:** a third, free refresh trigger — compare `A.nnz` (whole-matrix, O(1)) against the value at the
last refresh; refresh on any change. Not an exact pattern check (two different patterns could
coincidentally share a total nnz) but it caught the one measured failure case, costs nothing to
evaluate, and only ever errs towards *more* refreshing (safe direction — the plan's own "failure mode
is just extra iterations" principle, now actually enforced). Re-running the probe after the fix: `ord
3` correctly refreshes (94 iterations, matching the fresh build exactly) and no other ord regresses.

**(b) delivers ~zero measured benefit on this reference model — record this as the headline result,
not a footnote.** `nnz` differs between *every consecutive pair* of the 9 dumped ords (checked
directly against `manifest.jsonl`), i.e. the pattern churns on every single Newton iteration here, not
just at increment boundaries. The nnz guard therefore forces a refresh on every call in this sequence,
so hierarchy reuse never actually activates on the reference Pryout model — the mechanism is correct
and safe (verified above) but inert here, exactly consistent with what this module's docstring already
said before this session touched it ("the hierarchy cannot be reused *across* solves"). The entire
measured win below is (a) alone.

**Offline result, sequential replay, all 9 ords, one instance, EW forcing + guarded reuse vs. 9 fresh
instances at fixed `outerTol=1e-6`:**

| ord | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed-tol fresh-build (old) | 23.6s (72it) | 27.3s (94it) | 16.5s (50it) | 15.2s (44it) | 15.8s (48it) | 16.2s (45it) | 15.5s (46it) | 15.4s (44it) | 14.9s (44it) | **160.4s** |
| EW forcing, guarded reuse (new) | 12.3s (33it) | 26.3s (94it, REFRESH — pattern changed) | 11.2s (29it) | 9.3s (21it) | 10.1s (24it) | 9.3s (21it) | 9.7s (23it) | 7.3s (12it) | 7.6s (13it) | **103.0s** |
| speedup | 1.92× | 1.04× | 1.47× | 1.63× | 1.56× | 1.74× | 1.60× | 2.11× | 1.96× | **1.56×** |

**~36% aggregate wall-clock reduction** (every ord except the pattern-forced-refresh `ord 3` — which
is a wash, not a regression, since it correctly falls back to the old behaviour exactly). True
residuals with EW forcing sit at 5e-4–3e-3 (looser, as expected — `eta` mostly lands at `etaMax=1e-3`
on this sequence since most consecutive residual ratios trigger the safeguard), versus ~1e-6 for the
fixed-tolerance baseline. Reproduce: `python probe_192_sequence.py linsolveDumps` in
`Pryout_profile_investigation/` on xeon (log: `probe_192_sequence_v2.log`; the pre-fix log,
`probe_192_sequence.log`, is kept as the record of the found hazard).

**Live confirmation: FAILED the Newton-path pass criterion — EW forcing (a) is not the default.**
`blockamg.json` on xeon had `"outerTol": 1e-6` (backed up as `blockamg_fixedtol.json.bak`) — removed so
EW forcing engaged; ran `blockamg_live_192.log` (same command as 19.1's gate). Result:

| | increments (Newton iters) | cutbacks | final `U_loading` | `Job computation time` |
|---|---|---|---|---|
| fixed `outerTol=1e-6` (baseline, §19.1's gate) | 15, 8, 5, 5 | 3 (`0.0025`, `0.00125`, `0.000625`) | 0.021875 | 82.4s |
| EW forcing + guarded reuse (this run) | 15, 12, 7, 14 | 2 (`0.0025`, `0.00125`) | 0.03125 | 81.5s |

Not "a Newton iteration lost" (§19.1's anticipated failure mode) — a **different trajectory**: the
adaptive time-stepper took only 2 cutbacks instead of 3 and pushed further total load (`0.03125` vs
`0.021875`) within the same `maxNumInc` cap, at the cost of needing more Newton iterations per
increment once it did. Total job wall-clock came out a wash (81.5s vs 82.4s) — the fewer/cheaper
linear solves roughly paid for the extra Newton iterations — but the *physical trajectory itself*
changed, which is a different and more serious kind of risk than a wall-clock regression on a
softening/contact model (§7's drift concern: this doc's own established position is that even
round-off-level differences over hundreds of increments warrant a human judgement call, and this is
far larger than round-off).

This contradicts the plan's stated basis for expecting otherwise: "§10 already proved EW forcing
clamped to `[1e-6, 1e-3]` reproduces the Newton path *exactly* on this model" — but §10 validated that
claim for `inexactnewton`'s lagged-**LU** preconditioner, not `blockamg`'s lagged-**AMG** one; it does
not transfer. A stale/approximate AMG preconditioner evidently interacts with the loosened outer
tolerance differently than a near-exact LU one does.

**Fix, applied before this landed as anyone's default:** reverted `outerTol`'s class default back to
the original fixed `1e-6` (commit `5d86a0f6`) — EW forcing is now strictly opt-in (`outerTol=None`, or
`"adaptive"` through the JSON registry, since this cast pipeline has no bare `null`). Hierarchy reuse
(b) stays on by default: it cannot by itself change the Newton path (GMRES always converges on the
fresh matrix at the same requested tolerance; a stale `M` only costs outer iterations, never a
different `x`), and is provably inert-but-harmless on this model anyway (its pattern-change guard
forces a refresh on every call here, as shown above). `blockamg.json` on xeon was restored from its
backup (now redundant with the new default, but kept explicit).

**Net position after 19.2:** the safe, shipped default is unchanged Newton-path behaviour plus a
hierarchy-reuse mechanism that is correct and free but currently a no-op on this reference model. The
~36%/~1.56× wall-clock win from EW forcing is real (offline, §19.2 above) but gated behind a human
decision on the trajectory change — this is now the single most valuable remaining lever if that
decision comes back "acceptable", and should be the first thing revisited with that judgement.

### 19.3 D2 — mixed-precision hierarchy, `builtin<float>` (~half day C++/Cython, ~1.3–1.5×)

The direct attack on bandwidth-bound smoother cost: a float hierarchy halves memory traffic on an
already-saturated bus, and a preconditioner does not need double (the outer Krylov stays double).
Also halves preconditioner memory — the one wall-clock item that directly serves the 1M+ goal.

- Files: `edelweissfe/linsolve/amgcl/amgcl-wrapper.hpp` / `.pxd` / `.pyx`. Add a second solver
  instantiation on `amgcl::backend::builtin<float>`, selected by a constructor/JSON option (e.g.
  `"backendPrecision": "float"`), same additive pattern as `set_nullspace`/`build`/`report`. The
  matrix and null-space arrays must be converted to `float32` on the way in; `applyPreconditioner`
  takes/returns double at the Python boundary and converts internally.
- Wire it as the default for `blockamg`'s per-field hierarchies in `blockamg.py`, overridable via
  `fieldPreconds`.
- Rebuild on xeon (remember §1: xeon runs from rsynced files; `pip install -e .` there — setuptools
  is now present).
- Validation: offline, all 9 ords, iteration counts must not inflate by more than ~10% (literature
  and AMGCL's own docs say ~0 is typical); wall-clock target is ~1.3× on the §18 totals. Check
  `report()` still works on the float variant. One NaN sanity check on `applyPreconditioner`
  output (§17's gotcha: a NaN preconditioner silently defeats GMRES's stopping check).

**Implemented** — `amgcl-wrapper.hpp`/`.pxd`/`.pyx` (commit `741b81c2`). The wrapper is templated on
the backend's value type (`LinearSolverT<ValueType>`, instantiated as the existing `LinearSolver`
(double, unchanged behaviour) and a new `LinearSolverFloat`); `PyAMGCLSolver` holds one or the other
per instance, selected by `"backendPrecision": "double"|"float"` in the params dict (default
`"double"`), and dispatches every method call to whichever is live. `build()`/`solve()` narrow the
matrix values to `float32` in `.pyx` (cheap, once per solve); `applyPreconditioner()` stays double at
the Cython boundary and converts internally in C++ (the hot path, once per outer Krylov iteration).

**Plan-vs-library correction found while implementing:** the plan's text above says "matrix and
null-space arrays must be converted to float32" — checked against AMGCL's own header
(`amgcl/coarsening/tentative_prolongation.hpp`): `nullspace_params::B` is hardcoded
`std::vector<double>`, read via `p.get<double*>("B", ...)`, **independent of the backend's value
type**. The tentative-prolongation QR factorization that consumes it runs once at hierarchy build
time, not in the per-iteration smoother apply this backend split targets, so AMGCL deliberately keeps
it double-precision regardless. `set_nullspace()` therefore stays `const double*` on both backend
instantiations — narrowing it would have been wrong, not just unnecessary. Confirmed the fix compiles
and runs correctly on xeon before treating this as settled.

**Smoke-tested first:** built a float hierarchy directly on the ord-2 displacement block (214659
dofs, 28.1M nnz) — `applyPreconditioner()` returns no NaN/Inf, `report()` works on the float variant.
Memory: double `474.91 M` vs float `353.80 M` (level-0 hierarchy) — a **~25% reduction, not ~50%**:
the CSR index arrays (`int32` col indices + row ptr) are the same size regardless of backend value
type, so only the *values* array actually halves (`nnz × 8B → nnz × 4B`); for this operator's
nnz-to-unknowns ratio, indices are a large enough share of hierarchy storage that halving values alone
caps the total reduction well under 2×.

**Offline validation, all 9 ords, production `fieldPreconds` path, double vs. float — contradicts the
plan's `<=10%` inflation / `~1.3×` wall-clock expectations:**

| ord | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| double iters (wall) | 72 (26.5s) | 94 (28.9s) | 50 (16.6s) | 44 (15.0s) | 48 (16.0s) | 45 (15.5s) | 46 (15.4s) | 44 (15.1s) | 44 (15.0s) | **164.1s** |
| float iters (wall) | 79 (21.9s) | 94 (27.2s) | 65 (18.8s) | 50 (15.7s) | 55 (16.7s) | 45 (14.4s) | 53 (16.3s) | 44 (14.3s) | 44 (14.2s) | **159.5s** |
| iteration inflation | +9.7% | 0% | **+30.0%** | +13.6% | +14.6% | 0% | +15.2% | 0% | 0% | |
| speedup | 1.21× | 1.06× | **0.88×** | 0.96× | 0.96× | 1.07× | 0.95× | 1.06× | 1.05× | **1.03×** |

No NaN/Inf on either backend (log: `probe_193_precision.log`, reproduce with
`python probe_193_precision.py linsolveDumps` in `Pryout_profile_investigation/` on xeon). Inflation
reaches 3× the plan's anticipated ceiling on `ord 4`, and float is an outright wall-clock *regression*
on 4 of 9 ords (more iterations costing more than the per-iteration bandwidth saving buys back) —
aggregate speedup is **1.03×**, nowhere near the ~1.3–1.5× hoped for. Plausible mechanism (not chased
further, per the "record, don't improvise" rule): the chebyshev smoother's spectral-radius estimate
comes from a 50-step power iteration (§17 B5 already flagged this bound's sensitivity) — accumulated
rounding error over 50 float32 steps plausibly produces a worse bound than double does, degrading the
smoother enough on some Jacobians to erase the bandwidth win.

**Not wired as the default** — reverted `_DEFAULT_VECTOR_PRECOND`/`_DEFAULT_SCALAR_PRECOND`'s
`"backendPrecision"` back to `"double"` before this ever left the offline probe. The mechanism is
correct, tested, and available per-field via `fieldPreconds` (e.g.
`{"displacement": {**_DEFAULT_VECTOR_PRECOND, "backendPrecision": "float"}}`) for problems where the
smoother's spectral estimate might be less sensitive, but it is not a general win on this reference
model. Halved preconditioner memory (confirmed above) remains relevant to the §19.6 scale
demonstration independent of whether it wins on wall-clock at this size.

### 19.4 Smoother micro-sweep on wall-clock (cheap, do alongside 19.3)

§18 found `degree=5, npre=npost=1` by a coarse sweep and explicitly did not refine it. Offline, all
9 ords, wall-clock as the metric (never iteration count, §18's lesson): `degree ∈ {4, 5, 6}` ×
`npre=npost ∈ {1, 2}` via `fieldPreconds` on the production code path. ±10% is plausible for ~30 min
of machine time. If sweeping anyway, add one column re-examining the damage block's default
chebyshev (low priority, small share). Run with the RSS self-abort guard (§17's gotcha) and
`OMP_NUM_THREADS=MKL_NUM_THREADS=16` (§17's oversubscription gotcha).

**Done** — `probe_194_sweep.py` (xeon, `Pryout_profile_investigation/`, ad hoc, not checked in),
double backend, fixed `outerTol=1e-6`, RSS-guarded (24 GB cap; observed 6.3–6.4 GB throughout, nowhere
close). Log: `probe_194_sweep.log`.

| config | d=4,np=1 | d=4,np=2 | **d=5,np=1 (current default)** | d=5,np=2 | d=6,np=1 | d=6,np=2 |
|---|---|---|---|---|---|---|
| total wall (9 ords) | 166.1s | 163.1s | **160.7s** | 168.8s | **156.7s** | 176.8s |
| total outer iters | 551 | 406 | 487 | 373 | 416 | 346 |

`degree=6, npre=npost=1` edges out the current default by **~2.5%** (156.7s vs 160.7s) — inside the
"diminishing returns... not confirmed" margin the plan itself anticipated, and small enough that
run-to-run noise (see the caveat below) could plausibly explain it rather than a real optimum shift.
Not swapped in as the new default: a ~2.5% offline margin does not clear the bar this document has
otherwise used all along (every prior default change, §17/§18, cleared 20%+ before being adopted, and
even those got a live re-check). Recorded as a candidate for a future pass with more repeats, not
acted on.

**Damage-block quick check** (displacement fixed at the current default `d=5,npre=npost=1`):
default chebyshev (AMGCL's own defaults, no explicit degree/npre/npost) totals **159.8s**; adding an
explicit `degree=2,npre=npost=1` totals **155.7s**, a similarly modest ~2.6%. Same verdict: real
enough to note, too small to act on given the field's already-small share of the per-solve cost (§18).

**Gotcha, worth recording precisely because it could otherwise look like a regression:** this sweep's
`d=5,npre=npost=1` figure (160.7s) and the same config's total in the 19.3 probe run earlier this
session (`probe_193_precision.log`: 164.1s) both disagree with §18's originally-recorded total for the
identical config (125.2s) by roughly +28–31%. `uptime` on xeon showed a 15-minute load average of 5.99
right after this sweep (a shared, multi-user box, `htop`/Firefox/tmux sessions also present) against a
1-minute average of 0.09 once idle -- i.e. contention on this shared machine, not a code regression:
this session's own repeated measurements of the same config (19.3's probe and this one) agree with each
other to ~2%, just not with a different session's number recorded under different load. **All
relative comparisons in §19.2/19.3/19.4 remain valid** (each was measured within one session, configs
interleaved back-to-back under the same contention), but absolute totals should not be compared
across sessions without accounting for this -- prefer whole-job `Job computation time` deltas measured
in the same live run (as §19.1's gate and §19.2's live confirmation did) for any claim that needs to
survive scrutiny.

### Summary of this session's Phase 4 work — measured speedups vs. the §18 baseline and PARDISO

**What actually shipped (the default, no opt-in required):** 19.1's gate passed as-is; 19.2's
hierarchy-reuse half landed but is provably a no-op on this reference model (its own safety guard
forces a refresh every call, §19.2); everything else stayed opt-in. **So the shipped default's
wall-clock is unchanged from before this session** — the two mechanisms with a real measured win
(EW forcing, mixed precision) both failed their bar (a live Newton-path change; an offline iteration
inflation that ate the bandwidth saving) and were kept available but off. This is itself the main
finding of the session: two independently-plausible, individually-implemented-and-validated levers
both turned out not to be free lunches on this specific problem, and the plan's own "record, don't
improvise" rule is what caught both before either became a silent default-behaviour change.

**Speedups, where they exist, are opt-in and measured within one session (same-session relative
comparisons are unaffected by the cross-session load variance recorded above):**

- **EW forcing (opt-in, `outerTol=None`): ~1.56× offline** on the 9 dumped ords, measured head-to-head
  in one script run (`probe_192_sequence_v2.log`: 160.4 s fixed-tolerance vs 103.0 s EW-forced,
  fresh-hierarchy-every-call baseline in both). Not shipped because the live confirmation run changed
  the Newton trajectory (§19.2) — this is the biggest available lever if that trade-off is judged
  acceptable.
- **Mixed precision (opt-in, `backendPrecision="float"`): ~1.03× offline**, i.e. essentially no
  aggregate win and an outright regression on 4 of 9 ords (`probe_193_precision.log`). Not worth
  enabling as-is.
- **Smoother micro-sweep (informational only, default unchanged): ~1.025×** best case
  (`degree=6,npre=npost=1` vs the shipped `degree=5,npre=npost=1`), inside the plan's own
  diminishing-returns margin — not acted on (§19.4).

**Vs. the §18 baseline (125.2 s offline total / ~13.9 s per solve, that session's load) and the
long-standing PARDISO reference (~11.5 s/solve, §3/§10/§12):** this session's own repeated
measurements of the unchanged shipped config land at 160–164 s / ~17.8–18.2 s per solve, not because
the default regressed but because of the shared-machine contention documented in §19.4 — the §18
number and this session's number are not directly comparable in absolute terms, only in the
same-session relative comparisons made above. The **one number immune to that variance** is the live,
whole-job `Job computation time`, measured start-to-finish within a single run, unaffected by both the
offline/live gap and the cross-session load gap: **82.4 s for the shipped blockamg default vs. 83.9 s
for the PARDISO baseline, on the identical live model, in the same investigative window (§19.1)** — the
current default was already at PARDISO parity (not "close to", *at* parity, marginally ahead) before
this session started, and this session did not move that number, because nothing it validated as a net
win was safe enough to ship. Reaching meaningfully *past* PARDISO on wall-clock — not just parity —
remains gated on the EW-forcing trade-off decision (§19.2), which is the right next step once that
judgement call is made; blockamg's real, structural advantage at this point is O(n) memory (§19.6),
not wall-clock, which is where it should be pointed next regardless of that decision.

### 19.5 Stretch — B3, the 3×3 block-valued backend (~half day, only if 19.2+19.3 aren't enough)

No longer *needed* (§17 killed the "needs MueLu" argument), but it has a second benefit beyond
iteration count: `static_matrix<double,3,3>` storage improves cache locality and SIMD in exactly
the smoother matvecs that dominate the profile, and block-ILU0 (which inverts nodal 3×3 couplings
exactly) may finally make ILU-class smoothers viable here. Known caveats from §16: AMGCL's
`coarsening.nullspace` does not combine with block value types (acceptable — RBMs measurably don't
help, §13); try 3×3 block-Jacobi equilibration in place of point-Jacobi. Decide *after* 19.2/19.3
land — if those reach PARDISO parity, skip straight to 19.6.

### 19.6 Endgame — D4, the ≥1M-dof scale demonstration

The original feasibility question closes with data, not extrapolation: a uniformly refined pryout
(or boxgen at scale) run with `blockamg` vs PARDISO, recording peak RSS and wall. This is where
blockamg's O(n) memory becomes a structural advantage rather than an incremental one. Prerequisite:
19.1 green and at least one of 19.2/19.3 landed.

### Explicitly deprioritized (do not pick these up first)

- **D1 (Schur pressure correction / native AMGCL outer solve):** measured ceiling ~7% (§18).
  Revisit only after 19.2+19.3 shrink the smoother share and the outer loop's relative cost grows.
- **Rotational RBMs / coordinates in the null-space:** measured twice as not helping (§11, §13);
  the B2/B5 wins were orthogonal to the near-null-space. Leave it.
- **Lead 2 (frozen symbolic factorization):** still the reliable lever for the *direct*-solver
  path, still blocked on the §7 drift judgement (Matthias). Independent of everything above.

### Execution notes for the hand-off agent

- All benchmarking happens on **xeon** in `~/constitutive_modeling/next_v2611/
  Pryout_profile_investigation/` against the existing `linsolveDumps/` (9 systems); prefer the
  offline bench over FE re-runs (§6 sizing). The laptop env has no AMGCL headers and does not build
  the extension — code edits happen here, measurement there (rsync the changed files, as this whole
  branch has been doing).
- Always: `PYTHONUNBUFFERED=1`, `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16`, `PYTHON_GIL=0` for full
  runs, output redirected to a log file (§8: ssh-captured stdout comes back empty).
- Re-read §8 and §17's gotcha lists before touching AMGCL parameters — unknown keys are silently
  ignored (check stderr), NaN preconditioners run forever, and `eps_strong` sweeps can blow up
  memory.
- Commit per change with conventional-commit messages, as the branch has been doing; nothing is
  pushed to `mn` yet (22 local commits on `d83728ba`), and pushing remains a deliberate decision
  (§15).

---

## 20. Phase 5 plan — B3 first, then the EW rescue experiment on the final setup

Planned (this session), not started. Two independent parts, **executed in this order by explicit
decision (Matthias):** Part 1 (B3, the block-valued backend) settles the final solver configuration
first; Part 2 (the Eisenstat–Walker rescue) is deliberately deferred until that final setup exists,
because forcing behaviour interacts with preconditioner quality — validating EW against a
configuration that is about to change would have to be redone anyway.

Context entering this phase (§19 summary): the shipped `blockamg` default is at live PARDISO parity
(82.4 s vs 83.9 s whole-job, §19.1); EW forcing is a measured ~1.56× offline but opt-in only, gated
on a live Newton-trajectory change (§19.2); mixed precision measured ~1.03× — iteration inflation ate
the bandwidth saving, and the level-0 memory saving was only ~25% because **CSR index arrays, not
values, are a large share of hierarchy bandwidth** (§19.3). That last finding is what promotes B3
from stretch goal to the strongest remaining wall-clock lever: block-CSR storage cuts index traffic
by ~(block_size)² and makes the smoothers block-aware, attacking exactly the term that capped §19.3.

### 20.1 Part 1 — B3: the block-valued AMGCL backend (~1 day)

**Goal.** A `static_matrix<double,B,B>` builtin backend for the per-field hierarchies, B ∈ {2, 3}
(the pryout's displacement field is 3D; the registered `CantileverBeamQuad4BlockAMG` test is 2D —
implement both instantiations, they are one template apart). Expected gains, in order of confidence:
(i) per-iteration bandwidth — block CSR stores one column index per 3×3 block instead of nine;
(ii) block-aware smoothers — GS/ILU0 invert the nodal 3×3 couplings exactly, which is AMGCL's own
canonical elasticity recipe and §16's reason to expect ILU-class smoothers to stop diverging;
(iii) possibly fewer outer iterations.

**Implementation (edit here, build/measure on xeon):**

1. **Wrapper** (`edelweissfe/linsolve/amgcl/amgcl-wrapper.hpp` / `.pxd` / `.pyx`): the wrapper is
   already templated on the backend value type since §19.3 (`LinearSolverT<ValueType>`,
   double/float). Add `static_matrix<double,2,2>` and `static_matrix<double,3,3>` instantiations,
   selected by `"backendBlockSize": 1|2|3` in the params dict (default 1, existing behaviour;
   composes with `"backendPrecision"` in principle, but see the sweep — float-block is a follow-up
   column, not the first target). The scalar CSR from Python is adapted with
   `amgcl::adapter::block_matrix<value_type>(...)` at `build()`/`solve()`; RHS/solution stay
   `double*` at the Cython boundary and are reinterpreted to block vectors internally
   (`amgcl::backend::reinterpret_as_rhs` — the pattern from AMGCL's own structural-problem
   tutorial). Preconditions to assert, not assume: n divisible by blockSize, and node-major DOF
   ordering (true for the displacement field: node-major, 3 components/node, §4).
2. **`set_nullspace` is incompatible with block value types** (§16, confirmed AMGCL limitation).
   The wrapper must raise a clear error if both are requested; `blockamg.py` must skip
   `set_nullspace` when it selects a block backend. This is a measured non-loss: RBMs/translations
   do not help on this operator (§11, §13), and the near-null-space wins of §17 (`eps_strong`,
   chebyshev bounds) are orthogonal to it.
3. **`blockamg.py`**: keep the block backend **opt-in via `fieldPreconds` for the whole offline
   phase** — the default only changes after the full validation chain below. A vector field of
   dimension d maps to `backendBlockSize: d`.
4. **Equilibration, two variants to compare** (§16's caveat: point-Jacobi scaling slightly breaks
   the nodal block structure the backend exploits):
   - v1, zero new code: the existing point-Jacobi `dinv`. Run this first — if block-GS/block-ILU0
     already win with it, the block-Jacobi variant is a refinement, not a blocker.
   - v2: 3×3 **block**-Jacobi equilibration for block-backed fields — extract the nodal 3×3
     diagonal blocks (batched reshape of the diagonal block's CSR), factor each (batched
     `numpy.linalg.cholesky`; fall back to eigendecomposition clamped at a floor if any block is
     not SPD — Dirichlet identity rows make their node's block trivially SPD, but assert rather
     than assume), and scale symmetrically. Note the coupled off-diagonal blocks and the RHS must
     use the same per-field scaling — the change lives in `__call__`'s equilibration step, keyed on
     the field's backend choice.

**Offline validation (xeon, `Pryout_profile_investigation/`, dumped systems; same-session
interleaved measurements only — §19.4's load gotcha):**

1. **Displacement block alone first** (ord 2, `A_00_00002.npz` displacement diagonal block, the
   §17 testbed). Sweep on the block-3 backend: smoother ∈ {gauss_seidel, ilu0, chebyshev(d=5,
   power_iters=50, lower=0.01)} × eps_strong ∈ {default, 0.01} × equilibration {v1, v2}. Reference
   numbers to beat, measured in the same probe run, same session: the tuned scalar config (§17's
   29 iters; §18's wall). Block-ILU0 is the headline candidate. Do NOT set `aggr.block_size` on a
   block backend — that parameter is the *scalar* backend's aggregation hint (§13), not applicable
   here; setting both is exactly the kind of silently-ignored-key hazard §8 warns about (check
   stderr).
2. **Full coupled system**, all 9 ords, production `fieldPreconds` path, best block config vs the
   shipped scalar default, wall-clock as the metric. **Bar for changing the default: ≥20% aggregate
   wall-clock win** (the bar every prior default change cleared, §19.4's reasoning), no ord
   regressing past ~1.05×, true residuals no looser than the shipped default's (§18 table).
3. **Optional column, only if double-block wins:** `static_matrix<float,3,3>` — block storage cuts
   the index share that capped §19.3's float result at 1.03×, so float may pay *on top of* blocks
   even though it failed alone. Watch the chebyshev spectral estimate (§19.3's suspected
   float-degradation mechanism); prefer block-GS/block-ILU0 columns for the float comparison since
   they carry no spectral estimate.
4. **Live gate** (only if 2 clears the bar): the 19.1 run, identical pass criteria — Newton path
   15/8/5/5, same cutbacks, `Job computation time` vs the 82.4 s reference from the same window.
   Then, and only then, flip the vector-field default; the 2D regression test must pass with
   whatever the default becomes (`run_tests_edelweissfe ./testfiles/edelweiss-only/ --tests
   CantileverBeamQuad4BlockAMG`, plus the marmot suite if available locally).

**Failure handling:** if block-ILU0 diverges *and* block-GS/chebyshev show no wall-clock win at
step 1, stop Part 1 there, record the sweep table (it finally puts numbers on §16's B3 hypothesis),
and proceed to Part 2 against the unchanged scalar default — Part 2 does not depend on Part 1's
outcome, only on its *completion*, because whatever configuration survives Part 1 is the "final
setup" Part 2 validates against.

**Executed. Result: implemented and correct, but fails step 2's bar — default stays scalar.**
Commits `854f1be7` (wrapper: `LinearSolverBlockT<BlockType>`, instantiated for `static_matrix<double,
{2,3},{2,3}>`, verified against the actual AMGCL headers rather than assumed — see the commit message
for the API details, including that `n % blockSize == 0` must be checked explicitly since AMGCL's own
precondition is only an `assert()`, compiled out under `-DNDEBUG`) and `38e2ab33` (`blockamg.py`:
opt-in `backendBlockSize` via `fieldPreconds`, `set_nullspace` skipped for block fields).

**Step 1 (displacement block alone, ord 2, `probe_201_block_isolated.py`).** First pass used the
wrong outer solver (`bicgstab`) and got numbers ~2–4× off from §17's own references — caught by
checking a same-session reproduction of §17's "29 iters" config *before* trusting the new sweep
(confirmed the exact solver config, `gmres(M=100)`, from `scripts/benchmark_linsolve.py`'s
`runConfiguration`). Corrected (log: `probe_201_block_isolated_v2.log`):

| config | outer iters | wall |
|---|---|---|
| scalar: §17's tuned d=8/npre=npost=2, eps_strong=0.01 | 51 | 9.02 s |
| scalar: current production default (d=5, npre=npost=1) | 96 | 6.37 s |
| block3, gauss_seidel, eps_strong=default | 199 | 7.58 s |
| block3, gauss_seidel, eps_strong=0.01 | 188 | 6.98 s |
| block3, ilu0, eps_strong=default | 500 (did not converge, true res 4.8e-2) | 32.70 s |
| block3, ilu0, eps_strong=0.01 | 500 (did not converge, true res 5.3e-2) | 31.45 s |
| **block3, chebyshev(d=5,pi=50,lower=0.01), eps_strong=default** | **90** | **3.44 s** |
| block3, chebyshev(d=5,pi=50,lower=0.01), eps_strong=0.01 | 93 | 3.56 s |

Block-ILU0 diverges outright — the headline candidate the plan expected did not pan out. Block-GS is
a clear loser (more iterations *and* slower). **Block-chebyshev with the exact same smoother
parameters as the shipped scalar default — only `backendBlockSize` changes — wins on wall-clock**
(3.44 s vs 6.37 s, ~1.85×) with slightly fewer iterations, and even beats §17's more-tuned scalar
config on wall-clock (3.44 s vs 9.02 s) despite needing more iterations, exactly the B3 hypothesis:
cheaper per-application bandwidth from smaller CSR index blocks. `eps_strong` barely moves the block
backend's iteration count (90→93), unlike its ~2× effect on the scalar backend (§17) — coarsening
behaves differently once it's block-aware. This did **not** trigger the stated failure-handling
clause (chebyshev clearly does show a wall-clock win at step 1), so the chain continued to step 2.

**Step 2 (full coupled system, all 9 ords, production `fieldPreconds` path, `probe_201_coupled.py`)
— fails the bar, decisively and for a mechanistic reason worth recording precisely.** Block3 +
chebyshev(d=5,npre=npost=1) (the step-1 winner) vs the shipped scalar default, both through
`BlockAMGSolver.__call__`'s actual block-Gauss-Seidel + outer scipy GMRES path:

| ord | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| scalar iters (wall) | 72 (29.5s) | 94 (28.0s) | 50 (17.2s) | 44 (15.5s) | 48 (16.5s) | 45 (15.8s) | 46 (16.1s) | 44 (15.6s) | 44 (15.4s) | **169.5s** |
| block3 iters (wall) | 98 (24.8s) | 185 (43.1s) | 98 (23.1s) | 85 (20.1s) | 97 (23.1s) | 92 (21.8s) | 93 (22.0s) | 90 (21.2s) | 88 (20.6s) | **219.8s** |
| speedup | 1.19× | 0.65× | 0.74× | 0.77× | 0.71× | 0.73× | 0.73× | 0.74× | 0.75× | **0.771×** |

**A ~23% aggregate wall-clock *regression*, and every ord but one regresses past the 1.05× ceiling**
— the opposite of step 1's result on the identical smoother config. Outer iterations roughly double
almost everywhere (72→98, 94→185, 50→98, ...). This is the key finding, and it generalizes beyond
this one config: **step 1's `.solve()` probe measures how many *AMGCL-native, Krylov-accelerated*
V-cycles the block converges in on its own; step 2's actual usage calls `applyPreconditioner()` —
exactly *one* V-cycle per outer field-split sweep, used as `M` inside a *different*, outer (scipy)
Krylov loop.** A backend can be a better standalone solver (fewer/comparable iterations to its own
internal convergence, cheaper per V-cycle) while being a *worse single-shot preconditioner* for
someone else's outer loop, if its one-cycle error-reduction factor is worse than the scalar backend's
— evidently the case here. The per-application bandwidth saving (real, confirmed in step 1) is
overwhelmed by needing roughly twice as many outer applications. True residuals are also looser
throughout (e.g. `2.68e-06` vs `1.23e-07` on ord 2) at the same requested `outerTol`, consistent with
a measurably weaker single-cycle preconditioner, not just a slower-but-equivalent one.

**Bar not cleared (needed ≥20% aggregate win, no ord past ~1.05×; got a 23% aggregate loss, 8/9 ords
past it) — live gate skipped per the plan's own sequencing ("its live gate decides the final
default"; the decision here is "no change"), optional float+block column skipped ("only if
double-block wins"). Default stays the shipped scalar `backendBlockSize=1`.** The block-valued
backend implementation itself is correct, tested, and kept available via `fieldPreconds` (useful as a
standalone solver via `.solve()`, or for problems whose block-Gauss-Seidel-embedded behaviour differs
from this one) — only the *default* is unaffected. Part 1 concludes here; the "final setup" Part 2
validates against is the unchanged scalar default, exactly as it was entering this phase.

Logs (xeon, `Pryout_profile_investigation/`, ad hoc probes, not checked in): `probe_201_block_isolated.log`
(pre-fix, wrong solver, kept as the record of that mistake), `probe_201_block_isolated_v2.log`,
`probe_201_coupled.log`.

### 20.2 Part 2 — the EW rescue experiment, on the final setup from Part 1 (~half day, mostly runs)

**Do not start until Part 1 has concluded** (either outcome). Everything here runs against the
solver configuration Part 1 leaves as the default.

**Why there is something to rescue.** §19.2 measured EW forcing at ~1.56× offline but the live run
changed the Newton *trajectory* (2 cutbacks instead of 3, more load reached in the same increment
budget). The diagnosis in §19.2 — "§10's guarantee does not transfer from lagged-LU to AMG" — can be
sharpened: with a near-exact LU preconditioner, GMRES's preconditioned residual ≈ the true residual,
so §10's `etaMax=1e-3` really meant 1e-3. With blockamg the two are documented to diverge
(§14/§17/§18: true residuals 5e-4–3e-3 when EW asked for 1e-3), so the live EW run was effectively
solving to ~3e-3–1e-2 on the hard iterates — far looser than anything §10 validated. Two fixes, in
order:

1. **True-residual stopping in `blockamg.__call__`** (small, benefits fixed-tolerance mode too):
   after `gmres` returns, compute the *unscaled* true relative residual `‖Ax−b‖/‖b‖` (already
   computed for the verbose path); if it exceeds the requested tolerance, re-enter `gmres` with
   `x0=x` (warm restart), up to 2 continuations, logging each. This makes the requested tolerance
   preconditioner-independent — it also closes the shipped default's own documented gap (ord 3
   reaching 1.6e-2 when 1e-4 was requested, §17). Validate offline (all 9 ords: true residuals now
   ≤ tolerance; wall-clock cost of the continuations recorded — expect it concentrated on ord 3),
   and confirm the existing live gate still passes unchanged. This lands regardless of step 2's
   outcome.
2. **The `etaMax` ladder, live** (~35 min per rung): with step 1 in place, enable EW
   (`outerTol="adaptive"`) at `etaMax=1e-4` and run the 19.1 gate. Pass criteria are §19.1's,
   strictly: increments 15/8/5/5, the same three cutbacks, same final `U_loading=0.021875` — a
   *trajectory* match, not just an iteration-count match (§19.2's failure mode).
   - Pass at 1e-4 → record the live `Job computation time` win (offline counts at 1e-4 were 20–41
     vs 44–94 at 1e-6, so a large fraction of the 1.56× should survive) and, optionally, try
     `etaMax=3e-4` as one more rung to find the edge. Ship EW as the default only on a pass, with
     the winning clamp recorded here.
   - Fail at 1e-4 → try 1e-5 as a sanity rung (it bounds where the trajectory sensitivity starts);
     regardless of that rung's outcome, EW stays opt-in and the decision escalates to Matthias with
     the ladder's data — at that point it is a genuine §7-class judgement (the EW trajectory took
     *fewer* cutbacks and reached *more* load, which is not obviously worse, but it is not the
     reference path), not something an agent decides.
3. **Bookkeeping either way:** hierarchy reuse (§19.2(b)) stays inert on this model regardless of
   EW — the pattern churns every iteration and the nnz guard correctly forces refreshes. Do not
   burn time trying to activate it here; it activates for free if Lead 2's pattern caching ever
   lands (§7).

**Step 1 — implemented (commit `fb57ea1e`), and it took three attempts to get the mechanism right,
all worth recording since the failure mode (silently doing nothing) would otherwise be easy to miss.**
The one-line summary of *why* the gap exists: `gmres(..., callback_type="pr_norm")`'s own stopping
check is on the **preconditioned** residual, not the true one — with an imperfect preconditioner (this
one, by design, §17) the two diverge, so GMRES declaring "converged" can leave the true residual far
looser than requested (the already-documented case: 1.6e-2 true residual when 1e-4 was requested).

- **Attempt 1 — warm restart at the same `rtol`: a complete no-op.** `gmres(..., x0=z, rtol=eta, ...)`
  where `z` is exactly the solution GMRES just returned trivially re-satisfies the identical
  preconditioned-residual criterion at iteration 0 — every single continuation logged "0 more outer
  GMRES iters" (`probe_202_true_residual.log`). The mechanism the plan describes ("re-enter gmres with
  `x0=x`") cannot work as literally stated without also changing the target; this was the first thing
  the offline probe caught, before it ever reached a live run.
- **Attempt 2 — tighten the target proportionally to `eta/trueResidual`: partially worked, then
  broke in a revealing way.** Scaling the *requested* `eta` down by the true/preconditioned gap ratio
  fixed 2 of 9 ords but left 5 completely unmoved (`probe_202_true_residual_v2.log`) — GMRES had
  already overshot its original target substantially, so even the "tightened" request was already
  satisfied. Rescaling from the callback's own last-reported `pr_norm` instead (the *actually achieved*
  value, not the requested one) still left the same 5 ords unmoved, and for one of them
  (`probe_202_true_residual_v3.log`, ord 5) the computed continuation target came out *larger* than
  `eta` itself — impossible if the callback's `pr_norm` were on the same relative scale as `rtol`. It
  is not (almost certainly an absolute residual norm, not one normalized by `‖bs‖`) — a scipy
  internal not worth reverse-engineering further.
- **Fix — geometric tightening purely in `rtol`'s own units.** Each continuation multiplies the
  *requested* `rtol` by a fixed `0.01`, never touching the callback's ambiguous-scale value at all —
  dimensionally safe by construction since it stays entirely within a quantity scipy already
  interprets correctly. `trueResidualMaxContinuations` (default 2) bounds the effort.

**Offline validation, all 9 ords (`probe_202_true_residual_v4.log`), production `fieldPreconds` path,
fixed `outerTol=1e-6`, same-session interleaved (0 vs. 2 continuations):**

| ord | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| no continuation: iters (true res) | 72 (1.2e-7) | 94 (4.4e-5) | 50 (3.4e-6) | 44 (1.7e-6) | 48 (1.1e-6) | 45 (1.2e-6) | 46 (1.4e-6) | 44 (1.0e-6) | 44 (1.3e-6) | 164.1s |
| with continuation: iters (true res) | 72 (1.2e-7) | 164 (8.9e-9) | 72 (2.7e-8) | 61 (1.6e-8) | 74 (1.2e-8) | 71 (1.2e-8) | 65 (9.1e-9) | 44 (1.0e-6) | 71 (1.2e-8) | 218.0s |
| tolerance met (≤1e-6)? | yes | **no→yes** | **no→yes** | **no→yes** | **no→yes** | **no→yes** | **no→yes** | yes (already) | **no→yes** | 8/9 fixed |

**Every ord now meets its requested tolerance** (previously 5 of 9 didn't), typically overshooting to
1e-8–1e-9 once a continuation triggers (the geometric 0.01× step is coarser than needed, trading a bit
of extra work for robustness rather than tuning the factor finely). Cost is **not** concentrated on
`ord 3` as the plan anticipated — 5 of 9 ords needed it, not just the one known-hard iterate — so this
was a broader, previously-undercounted gap in the shipped default, not an `ord 3`-specific quirk.
Aggregate wall-clock rose from 164.1s to 218.0s (+33%) across this batch; accepted per the plan
("this lands regardless of step 2's outcome") since it trades wall-clock for actually honoring the
requested accuracy.

**Live confirmation (`blockamg_live_202_step1.log`, same command as §19.1's gate): unchanged
trajectory, ~free in practice.** Newton path **15/8/5/5**, identical to baseline; same three cutbacks
(`0.0025`, `0.00125`, `0.000625`); identical final `U_loading=0.021875`; `Job computation time`
**81.8s** (vs. baseline **82.4s** — no regression, if anything a rounding-level improvement).
Continuations fired **38 times across the 44 live solves** (i.e. most solves needed at least one),
pushing residuals down to the 1e-8–1e-9 range, yet this barely moved total wall-clock — on the real
model the continuation cost is evidently much smaller relative to a full solve than the offline batch
above suggested (offline used the harder, larger-residual-gap dumped ords disproportionately).
Regression test `CantileverBeamQuad4BlockAMG` still passes.

**This closes a real, previously-shipped-but-undocumented-as-such accuracy gap**, independent of
whatever step 2 decides on EW forcing.

**Step 2 — the `etaMax` ladder, live. Both rungs pass strictly; `3e-4` ships as the default.**

With step 1 in place, `outerTol="adaptive"` was enabled and run against the 19.1 gate's exact pass
criteria (increments 15/8/5/5, identical cutbacks, identical final `U_loading` — a *trajectory* match,
not just an iteration-count one, since §19.2's failure mode was a changed trajectory that still
happened to cost roughly the same wall-clock):

| rung | increments | cutbacks | final `U_loading` | outer iters (min/mean/max) | continuations | `Job computation time` |
|---|---|---|---|---|---|---|
| `etaMax=1e-4` (`blockamg_live_202_etamax1e-4.log`) | 15/8/5/5 ✓ | 3, identical ✓ | 0.021875 ✓ | 17/65.5/165 | 34 | 81.5s |
| `etaMax=3e-4` (`blockamg_live_202_etamax3e-4.log`) | 15/8/5/5 ✓ | 3, identical ✓ | 0.021875 ✓ | 14/61.7/165 | 34 | **81.3s** |

**Both rungs are a strict pass** — unlike §19.2's original attempt, EW forcing no longer changes the
Newton trajectory now that step 1 enforces the true residual regardless of what GMRES's own
preconditioned-residual check would have accepted. This confirms the diagnosis in §19.2 exactly: the
trajectory change there was a symptom of the true-residual gap, not of adaptive forcing itself. `3e-4`
edges out `1e-4` on every metric that moved (lower mean outer iterations, marginally better job time)
so it is the winning clamp, per the plan's "optionally try `etaMax=3e-4` to find the edge" — no further
rungs attempted (`1e-3`, the pre-fix value that failed in §19.2, was not retried; the ladder's scope was
`1e-4` then optionally `3e-4`, both of which passed, so there was no failure to escalate past).

**A real bug caught before it shipped: the constructor defaults were never actually flipped.** After
declaring `3e-4` the winner, a routine check of `BlockAMGSolver.__init__`'s actual parameter defaults
found `outerTol: float = 1.0e-6` and `etaMax: float = 1.0e-3` still in the signature — the class
docstring already claimed EW forcing "shipped as the default" (written in anticipation, before the
ladder even ran), but nothing had actually changed the code that makes that true. Fixed: `outerTol`
now defaults to `None`, `etaMax` to `3.0e-4` (commit `76cb09da`). `blockamg.json` on xeon was
simplified to drop the now-redundant explicit `outerTol`/`"adaptive"` and `etaMax` overrides entirely,
to prove the shipped defaults need no config help. Live-reconfirmed with this corrected default and no
explicit overrides at all (`blockamg_live_202_final.log`) — see the final summary below.

### Infrastructure done alongside this phase: a common `LinearSolver` base class, and blockamg's
### output overhauled

Two related but separate requests came in mid-phase, both addressing real friction:

- **blockamg's console output** was reformatted: a graded `verbosity` (`"silent"`/`"warning"`
  (default)/`"info"`/`"debug"`, replacing the old boolean `verbose`) so a normal run stays quiet and
  only speaks up on an abnormal solve (outer iterations past `warnOuterIterationsThreshold`, the
  true-residual tolerance still unmet after every continuation, or GMRES reporting non-convergence);
  messages route through the solver's injected Journal (falling back to `print` if none was set, e.g.
  an offline probe script); and per-stage timing (equilibration, off-diagonal split, hierarchy build,
  outer GMRES, true-residual continuations) now goes through
  `edelweissfe.utils.performancetiming.timeit`, nested under "linear solve" in the job's own
  performance table — **with the caveat, already on record in §19.1, that this global mechanism's
  accumulated figures are not reliable under `PYTHON_GIL=0` on a live, multi-threaded run** (a
  pre-existing, general-infrastructure bug, unrelated to and not fixed by this work).
- **A common `LinearSolver` base class** (`edelweissfe/linsolve/base.py`) replaces the old
  isinstance-checked `FieldStructureAwareLinearSolver` mixin (blockamg only) with one base every
  registered linsolver now inherits, with safe-default `setJournal()`/`setFieldStructure()` so the
  nonlinear solver calls both unconditionally instead of special-casing per capability. All 11
  registered linsolvers (`superlu`, `umfpack`, `pardiso`, `panuapardiso`, `klu`, `petsclu`, `mumps`,
  `gmres`, `amgcl`, `inexactnewton`, `blockamg`, `matrixdump`) were retrofitted via thin Python wrapper
  classes, without touching any `.pyx` internals. Found and fixed one real hazard this way:
  `inexactnewton` resolves its PARDISO delegate through this same registry seam and needs
  `factorize()`/`solveFactorized()`, not just `(A, b) -> x`, so `PardisoLinearSolver` forwards both.
  Validated: `CantileverBeamQuad4BlockAMG`, `WallShearHexa8GMRES`, and `AMGCL` (full end-to-end tests)
  all pass; `pardiso` (both call contracts), `superlu`, `umfpack`, and `matrixdump` (delegate
  forwarding) smoke-tested directly. `klu`/`panuapardiso`/`mumps`/`petsclu` could not be exercised at
  all on xeon — their optional backends are not installed/built there, a pre-existing environment gap
  (§8), not a regression from this change.

Commits: `fb57ea1e` (true-residual stopping), `76cb09da` (ship EW forcing as default), `62e30f82`
(common `LinearSolver` base class + all 11 wrappers), plus the doc-only `be9f4232`.

### Summary of Phase 5, Part 2 — measured wall-clock vs. the §18/§19.1 references

**The EW-rescue hypothesis was correct.** §19.2's live failure (a changed Newton trajectory) was a
symptom of GMRES's preconditioned-residual stopping check letting the true residual run looser than
requested, not a fundamental incompatibility between adaptive forcing and this preconditioner. Fixing
that gap (step 1) made adaptive forcing reproduce the fixed-tolerance trajectory exactly, on two
separate live rungs (`1e-4`, `3e-4`) — the trajectory risk that blocked §19.2 is resolved, not just
worked around.

**`Job computation time`, all measured on the same live model, same investigative window:**

| run | `Job computation time` |
|---|---|
| PARDISO baseline (§19.1) | 83.9s |
| blockamg, fixed `outerTol=1e-6` (§19.1's gate) | 82.4s |
| blockamg, true-residual stopping added, still fixed `outerTol=1e-6` (§20.2 step 1) | 81.8s |
| blockamg, EW forcing `etaMax=1e-4` (§20.2 step 2) | 81.5s |
| blockamg, EW forcing `etaMax=3e-4`, explicit config (§20.2 step 2) | 81.3s |
| **blockamg, shipped defaults, zero explicit config** (`blockamg_live_202_final.log`) | **80.4s** |

The last row is the one that matters going forward: the corrected constructor defaults (`outerTol=None`,
`etaMax=3.0e-4`), the full Journal/`performancetiming`/`LinearSolver`-base refactor, and an empty
`blockamg.json` (`{"sweeps": 1, "symmetric": true, "verbosity": "info"}` — no `outerTol`/`etaMax`
override at all) reproduce the identical trajectory (15/8/5/5, same three cutbacks, final
`U_loading=0.021875000000023462`) with no NaN/Inf and no "unknown parameter" warnings. The small further
improvement (80.4s vs. 81.3s) is within this shared machine's normal run-to-run variance (§19.4), not a
new effect — the conclusion below is unchanged by it.

**Every row is within ~3% of every other row.** This is the headline finding of Phase 5, and it cuts
against the ~1.56× offline win §19.2 originally measured: on the *live* model, wall-clock is dominated
by the fixed per-solve costs (assembly, AMR, MPC transforms, hierarchy builds) that neither
true-residual stopping nor EW forcing touch, not by the outer-GMRES iteration count these two levers
actually move. The offline 9-dumped-ord benchmarks measure the linear solve in isolation and
extrapolate; the live number is what the whole job actually costs, and it barely moves no matter which
of these levers is on. **Net result of this entire phase: the trajectory-safety problem is fixed
(genuinely valuable — it closes a real accuracy gap and de-risks EW forcing for future use), but it
delivers negligible additional live wall-clock beyond the parity §19.1 already established.** blockamg
remains at parity with PARDISO; reaching meaningfully past parity on this model would need to attack
the fixed per-solve overhead (assembly/MPC transform, ~21% of a solve per §18), not the outer iteration
count, which is where every lever in §19 and §20 has been aimed.

### Sequencing, and what is explicitly out of scope for this phase

1. 20.1 step by step; its live gate decides the final default.
2. 20.2 on whatever that final default is; its step 1 (true-residual stopping) ships regardless,
   its step 2 decides EW's fate or escalates it.
3. **Out of scope:** D4 (the ≥1M-dof scale demonstration) is the phase after this one — it should
   run once, on the final configuration this phase produces, not before. D1, rotational RBMs, and
   Lead 2 remain deprioritized/blocked exactly as §19 lists them. The `performancetiming.timeit`
   thread-safety bug (§19.1) is real but is general-infrastructure work, not linsolve work — track
   it separately; until fixed, live wall-clock claims use `Job computation time` only.

The §19 "Execution notes for the hand-off agent" apply to this phase verbatim (xeon workflow, env
vars, gotcha lists, commit conventions, no pushing).

---

## 21. Phase 6 plan — attack the fixed per-solve costs: stable pattern (Part A), cached-pattern condensation (Part B)

Planned (this session), not started. Motivation is §20's closing finding: live wall-clock is
dominated by per-solve *fixed* costs — the AMG hierarchy build + equilibration (~21% of a blockamg
solve, §18) and the MPC condensation (2.75 s per Newton iteration, single-threaded, §2) — not by
the outer iteration count every §19/§20 lever targeted. Two independent parts:

- **Part A (~half day, mostly xeon runs):** an experiment, not a build. Turn off
  `pruneCondensedMatrixZeros` under `blockamg` so the sparsity pattern stops churning, which should
  let the already-shipped-but-inert hierarchy reuse (§19.2(b)) engage. Zero new code before the
  measurements; the open question is whether reuse buys more than the fatter (unpruned) matrix
  costs.
- **Part B (~1–1.5 days):** implementation. Rebuild `transformSystemMatrix` as a value scatter into
  a cached pattern instead of two SciPy SpGEMMs per iteration. Benefits *every* linsolver on every
  MPC-carrying model. This is the assembly-side half of Lead 2 only — it does **not** touch
  PARDISO's reordering freeze, which stays blocked on §7 (Matthias).

**Part B does not depend on Part A's outcome.** Run A first because it is cheaper and its nnz-growth
measurement (A1) informs B; if A stops early at any gate, proceed to B regardless.

The §19 "Execution notes for the hand-off agent" apply verbatim (xeon workflow in
`~/constitutive_modeling/next_v2611/Pryout_profile_investigation/`, never the real project dir;
`PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHON_GIL=0`, output redirected to log
files; §8/§17 gotcha lists; conventional commits; nothing pushed to `mn`). D4 (the ≥1M-dof
demonstration) remains the phase after this one — do not start it here.

### 21.0 Measurement basis for every gate in this phase (read once, then follow)

Neither internal timing figure is a safe gate metric. The per-category `acc. runtime` table is
provably corrupted under `PYTHON_GIL=0` (§19.1's gotcha). And `Job computation time` — the number
§19/§20's parity claims used — does not reconcile with the offline evidence: 44 live solves at the
offline-verified ≥11.5 s/solve cannot fit inside 83.9 s, so treat it as suspect rather than
authoritative. Do not investigate why (out of scope); instead:

- **Every wall-clock gate in this phase uses externally measured whole-run time** — wrap each run
  in `date +%s` before/after (or `/usr/bin/time -v`, which also gives peak RSS), recorded in the
  log.
- **A/B comparisons are always same-session, back-to-back runs** on xeon (§19.4's load gotcha —
  cross-session absolute numbers differ by ~30% under contention).
- Offline probes keep timing with their own `perf_counter()` as all prior probes did — those
  numbers are trustworthy.
- If the externally measured baseline turns out far from 83.9 s, add one sentence to §19.1/§20.2
  flagging that their parity claims are metric-relative. Record forward; do not rewrite history.

### 21.1 Part A — `pruneCondensedMatrixZeros=False` under blockamg: does a stable pattern activate hierarchy reuse, and does it pay?

**Hypothesis chain (each link already in evidence).** The pattern churn (§3.1) is caused by
`K.eliminate_zeros()` at `nonlinearimplicitstatic.py:810` pruning whichever entries happen to be
exactly zero *this* iteration (contact/damage state). Upstream of that pruning the pattern is
fixed within a step: the assembled `K` is the csrGenerator's persistent buffer with a fixed
assembly map (see the long comment in `applyDirichletK`, `nonlinearimplicitstatic.py:781-809`),
the MPC transformation `T` is rebuilt only on model change (`nonlinearimplicitstatic.py:352`), and
SpGEMM/`+ C` output patterns are structural, hence deterministic. So with pruning off, `A.nnz` is
constant between model changes → blockamg's pattern-change guard (`blockamg.py`, the
`_lastNnz` comparison) stops forcing a refresh every call → hierarchy reuse engages, saving the
~3.4 s/solve hierarchy build (§18). The equilibration-apply and off-diagonal split still run every
solve regardless of reuse (see `__call__`) — that residual fixed cost is A5's optional follow-up.

**Known counter-force — this is the experiment's actual question.** Keeping the zeros grows nnz:
§3.1's 21.1M structurally asymmetric entries are pruned partners, so the unpruned matrix could
carry up to ~+50% nnz, and the smoother apply — 68% of a solve (§18) — scales with nnz. The
hierarchy build itself also gets more expensive on a fatter matrix. Reuse must buy more than the
fatter matvecs cost. Measure; do not argue from priors.

**A1 — fresh capture with pruning off (~20 min).** The existing `linsolveDumps/` were captured
with pruning on, so a new set is needed: `capture_noprune.inp` = `capture.inp` +
`pruneCondensedMatrixZeros=False` on the same options line that carries `linsolver=matrixdump`
(it is a NIST option, schema at `nonlinearimplicitstatic.py:115`). Copy `linsolveDumpConfig.json`
and change its `directory` to `linsolveDumpsNoPrune`. Run per §6. Then:

1. `python $B pattern linsolveDumpsNoPrune` — the gate. Expect `UNCHANGED` (index-array
   comparison) between consecutive iterates of the same increment. **If the pattern still
   churns**, the fixed-pattern assumption fails at this layer; record which arrays differ, stop
   Part A, and carry the finding into Part B (whose identity-based cache invalidation, B1, is
   designed to survive exactly this).
2. Record nnz vs the pruned dumps' ~40.9M (`manifest.jsonl` in both dirs) — the growth factor is
   the headline input to A2's economics.
3. Caveat to record: this capture's delegate is PARDISO solving *unpruned* systems — the §7 drift
   scenario. Compare the capture log's Newton path against `baseline_omp16.log` (15/8/5/5, three
   cutbacks). If it differs, the dumps remain valid specimens for A2 (the replay question is about
   the solver, not the trajectory), but say so here — it is also a free §7 data point either way.

**A1 — executed. The pattern still churns every iteration; Part A stops here, exactly as
anticipated.** `capture_noprune.inp`/`linsolveDumpConfig_noprune.json` created and run
(`capture_noprune.log`, `linsolveDumpsNoPrune/`).

- **§7 caveat, resolved cleanly first: no drift.** Newton path **15/8/5/5**, identical cutbacks
  (`0.0025`/`0.00125`/`0.000625`), same termination, matching `baseline_omp16.log` exactly — PARDISO
  solving the unpruned system over this 5-increment window shows no measurable trajectory
  sensitivity here. A genuine, free §7 data point: this particular drift channel is not observable
  at this window size, though that is not the same as "settled" (§7 already says as much for the
  general question).
- **nnz growth: +8.4% to +8.7%, not the speculated ~+50% ceiling.** No-prune nnz ranges
  44.09M–44.42M vs the pruned dumps' 40.58M–40.91M (`manifest.jsonl`, both dirs) — real but modest,
  and (surprisingly) the *absolute* nnz swing between consecutive ords is comparable in both sets
  (~337k here vs ~334k pruned) — pruning is not the dominant source of nnz variation at all; see
  next point.
- **The pattern gate itself fails.** `python $B pattern linsolveDumpsNoPrune`
  (`pattern_noprune.log`) — **every single consecutive pair is `CHANGED`** (a genuine
  `np.array_equal` check on `indices`/`indptr`, not an nnz proxy — confirmed by reading
  `commandPattern`'s implementation before trusting its verdict). The nnz delta per step
  (+103087, −217330, +26727, +5808, −2086, −7574, +6254 …) shrinks in magnitude as the increment
  converges but never reaches zero — consistent with the churn being driven by **contact/tie
  connectivity updates** (the candidate set shifting with the solution each Newton step,
  independent of `pruneCondensedMatrixZeros`), not by the zero-pruning mechanism the hypothesis
  chain blamed. `nonlinearimplicitstatic.py`'s own rebuild trigger,
  `modelHasChanged or connectivityHasChanged or self.theDofManager is None`, already names
  `connectivityHasChanged` as a distinct, always-live rebuild path — pruning was never the only
  source of instability, just the one previously measured (§3.1).
- **Per the plan's own stop condition, Part A ends here.** A2 (offline replay), A3 (live gate), A4
  (ship/record), and A5 (equilibration caching) are **not attempted** — their entire premise (a
  stable pattern once pruning is off) does not hold, so there is nothing left for them to measure
  that A1 has not already answered. This carries directly into Part B: B1's cache-invalidation
  design already assumes identity-based invalidation per solve (not "stable across solves"), so it
  is unaffected by this finding — if anything, this strengthens the case that Part B's assembly-side
  fix is the one worth pursuing, since a stable pattern was never available to exploit at the
  `blockamg` layer regardless of the pruning setting.

**A2 — offline replay, three-way, same probe run.** New probe modeled on
`probe_192_sequence.py` (xeon, investigation dir, ad hoc, not checked in). Three arms, interleaved
in one script invocation, all with the shipped default config (EW forcing `etaMax=3e-4`,
true-residual stopping):

- (i) no-prune dumps, **one** `BlockAMGSolver` instance, sequence order → reuse allowed to engage;
- (ii) no-prune dumps, one instance but `solver._refreshNext = True` forced before every call →
  same systems, reuse denied; isolates the reuse benefit from the nnz-growth cost;
- (iii) pruned dumps (`linsolveDumps/`), one instance → today's live behaviour, the baseline.

Record per ord: the solver's own REFRESH/reuse decision, outer iterations, true-residual
continuation count (a staler `M` widens the preconditioned/true gap, so continuations are part of
the honest cost), true residual, `perf_counter` wall. Confirm reuse actually engages in (i) and
note how often the `hierarchyStalenessFactor` backstop fires.

**Bar** (the same one every prior default change cleared): (i) beats (iii) by **≥20% aggregate
wall**, no ord regressing past ~1.05×, tolerances still met. If (i) loses to (iii) because nnz
growth eats the reuse saving: record the table and stop Part A — "a stable pattern does not pay at
280k under blockamg" is itself a valuable, citable conclusion (it also caps what Lead 2 could ever
buy blockamg, without touching its value for the direct-solver path).

**A3 — live gate (only if A2 clears).** `blockamg_noprune.inp` = `blockamg.inp` +
`pruneCondensedMatrixZeros=False`. Run per §19.1's command, plus a back-to-back rerun of the
unmodified `blockamg.inp` in the same session (§21.0). Pass: trajectory identical (15/8/5/5,
cutbacks `0.0025`/`0.00125`/`0.000625`, final `U_loading=0.021875`) and an external-wall win
roughly matching A2's margin. Correctness risk is structurally low — reuse never changes the
answer (outer GMRES runs on the fresh matrix; true-residual stopping enforces the tolerance) — but
the unpruned matrix reaching AMG has more (zero-valued) entries, so hierarchies and outer counts
may shift; that is expected, not alarming.

**A4 — ship/record.** Do **not** flip the `pruneCondensedMatrixZeros` default — its `True` default
protects the PARDISO path (§7). On a pass: document the recommended pairing (blockamg +
`pruneCondensedMatrixZeros=False`) with measured numbers in
`doc/source/documentation/linsolvers.rst`, and extend the option's schema docstring (currently
worded around frozen reorderings only) to name hierarchy reuse as the second legitimate consumer.
Update this document either way.

**A5 — optional, only after a live pass and only if the reuse win is visibly capped by the
remaining per-solve fixed cost.** With a stable pattern, the equilibration-apply and off-diagonal
split (~2.1 s/solve, §18) can be cached too: `As.data` computed as
`dinv[row] * dinv[col] * A.data` via a precomputed row-index expansion, block submatrices via
cached index gathers. Separate commit, own offline A/B.

### 21.2 Part B — MPC condensation on a cached pattern (value scatter instead of two SpGEMMs)

**Target.** `MultiPointConstraintTransformation.transformSystemMatrix`
(`edelweissfe/numerics/mpctransformation.py:204`): `TᵀK` 1.485 s + `(TᵀK)T` 0.296 s +
`+C`/`tocsr`/`sort_indices` 0.970 s ≈ **2.75 s per Newton iteration**, single-threaded SciPy,
~14% of the reference job and growing as a share with every core added (§2, §3.3).

**Why it is safe (scope guard).** This changes *how* `TᵀKT + C` is computed, not *what*: with
pruning left on (the default), the downstream `eliminate_zeros()` makes the final matrix identical
to today's in pattern, and in values up to summation-order round-off — no §7 exposure. It composes
with Part A trivially (A only switches that downstream pruning off). The frozen-reordering half of
Lead 2 remains out of scope.

**B1 — verify the two cache-invalidation facts before writing code (~30 min).**

1. `K`'s CSR arrays are persistent and returned by reference every iteration with a fixed pattern
   (the `applyDirichletK` comment asserts this; verify `K.indices is` the same object across two
   assembly calls on any tie testfile). Consequence: `K.indices is self._cachedKIndices` is a
   valid O(1) staleness check.
2. `self.mpcTransformation` is reconstructed wholesale on model change
   (`nonlinearimplicitstatic.py:352`, constructed in `nonlinearsolverbase.py:512`) and never
   mutated in place — so caches stored **on the transformation object** die at exactly the right
   time, with no epoch counter needed. Verify no other mutation/call site exists (grep).

If either fact does not hold, add an explicit invalidation key instead — but verify first.

**B2 — implementation (primary design: D+S split, pure numpy/scipy, no build-system work).**
Decompose `T = D + S` once at construction: `D` = diagonal selector (1 on independent DOFs, 0 on
slaves), `S` = the slave rows only (~16.5k rows, ~2–3 nnz each, on the reference model). Then

```
TᵀKT + C  =  DKD  +  DᵀKS  +  SᵀKD  +  SᵀKS  +  C
```

- `DKD` is `K` with slave rows/columns masked: pattern = `K`'s own; values = `K.data ×` a float
  mask precomputed once per epoch. Pure numpy, ~tens of ms.
- The three `S`-terms are SpGEMMs with one tiny operand (~40k nnz) — expected well under 0.3 s
  total; their output patterns are fixed per epoch. Measure each.
- **First call of an epoch:** compute the decomposition's sum once with plain scipy ops, keep it
  as the returned matrix, and harvest the caches: the union pattern (`indices`/`indptr`, built
  sorted once), per-term scatter maps (position of each term's nnz inside the union CSR —
  vectorized `searchsorted` within `indptr` ranges), and `C`'s pre-scattered baseline data vector.
  Each term's map is duplicate-free (a CSR pattern maps injectively into a superset pattern), so
  plain fancy-index `+=` is safe, term by term.
- **Subsequent calls:** fresh output data array; `out[:] = cBase`, then one `+=` per term. No
  SpGEMM on `K`.
- **Aliasing hazard — do not skip this.** The returned matrix must **not** alias the cached
  `indices`/`indptr`: downstream, `applyDirichletK` zeroes values in place and
  `eliminate_zeros()` **compacts the index arrays in place** — exactly the corruption mechanism the
  `applyDirichletK` comment documents for the csrGenerator buffer. Return
  `csr_matrix((freshData, indices.copy(), indptr.copy()))` every call; the ~165 MB index copy costs
  tens of ms against the ~2.5 s saved. Micro-optimize later only with that hazard designed around.
- Pattern note: `DKD` carries `K`'s full pattern, so the cached union is a slight superset of the
  legacy product's structural pattern (explicit zeros at slave rows/columns). With pruning on this
  is erased downstream; with pruning off (Part A) it just means a slightly larger stable pattern —
  record the nnz delta. If exact legacy-pattern parity is ever wanted, compress `DKD` through a
  precomputed gather instead; not required to ship.
- **Exactness assertion** behind a debug flag (env var), on through all of B3: on the second call
  of each epoch, also compute the legacy scipy expression and require max |Δ| ≤ 1e-9 × max|data|
  on the pattern-aligned arrays (matrix-norm-relative — entry-relative fails on cancellation-tiny
  entries).
- Keep it switchable (option through `buildMPCTransformation`); default decided in B3.
- Fallback design, only if measurement shows the S-terms or the scatter dominating (≥0.5 s): a
  Cython numeric-only Gustavson kernel into the cached patterns (`numerics/`, `prange` over rows).
  Real work — do not start it unless the pure-Python numbers demand it.

**B3 — validation.**

1. **Exactness:** full `run_tests_edelweissfe ./testfiles/edelweiss-only/` with the debug
   assertion on — identical results (the tie/AMR/contact cases are the ones that exercise this
   path). `NodeToDeformableSurfaceContactPullOut` fails pre-existing (§9) — confirm it fails
   *identically*, then ignore. Then `./testfiles/marmot/` where built.
2. **Live trajectory:** the pryout on xeon, both `profile.inp` (PARDISO) and `blockamg.inp`, each
   with the cached path on, back-to-back with their unmodified references (§21.0). The strong
   check: with pruning on, the solver sees the same matrix up to round-off, so the journal's
   increment/cutback/`U_loading` sequence should match byte-for-byte on this 5-increment window.
3. **Performance:** externally timed A/B (§21.0), plus per-call `perf_counter` prints around the
   transform behind a temporary env flag (the internal table is not trustworthy, §19.1). Target:
   the transform stage ≥2× (2.75 → ≤1.4 s), expected ~5× (≤0.6 s); job-level win bounded by the
   stage's ~14% share at 16 threads.
4. **Ship decision:** default **on** only if (1) is green, (2) matches byte-for-byte on both
   solvers, and (3) clears ≥2× on the stage. If (2) shows round-off-level trajectory divergence,
   ship default **off** with the numbers recorded and escalate to Matthias — that is a §7-class
   judgement (value round-off from summation order, much milder than §7's structural case, but the
   same person's call).

**B4 — docs + housekeeping.** Update the relevant Sphinx page (wherever MPC condensation is
documented; `linsolvers.rst` cross-reference at minimum) and this document. While committing:
finally gitignore `edelweissfe/linsolve/klu/klu.c` (§8/§15 — it has bitten twice).

**B2/B3 — executed. Correctness holds after fixing two real bugs the exactness assertion caught;
performance is the opposite of the plan's expectation — the cached path is slower, not faster, on
the reference model. Shipped as an opt-in (`useCachedMPCCondensation`, default `False`), not the
new default.**

**Implementation** (`edelweissfe/numerics/mpctransformation.py`): `T = D + S` split as planned
(`D` diagonal, `S` the slave rows only), `DKD` as a values-only mask of `K`'s own pattern (no
SpGEMM, cheap), the three `S`-touching terms (`D K S`, `Sᵀ K D`, `Sᵀ K S`) as SpGEMMs recomputed
every call, and a cached union output pattern rebuilt only when `K`'s own `indices`/`indptr`
identity changes (B1's invalidation check, confirmed valid — see below). `_transformSystemMatrixLegacy`
kept as the reference expression, both for the `EDELWEISS_MPC_ASSERT_EXACT` cross-check and as the
method actually used by default now (see the ship decision).

**Two real bugs, both caught by the `EDELWEISS_MPC_ASSERT_EXACT` cross-check the plan specified as
a debug flag — validating that the flag was worth building, independent of everything else this
phase found:**

1. **The `-0` slice pitfall.** `self._S = csr_matrix((tVals[-nSlaveEntries:], ...))` — with **zero**
   slave DOFs (`nSlaveEntries == 0`, a legitimate case: a system assembled before any hanging-node
   or tie constraint exists yet), `tVals[-0:]` is `tVals[0:]` in Python/NumPy — the **entire**
   array, not empty. `self._S` silently became the full identity matrix instead of zero, so every
   `S`-touching term picked up ~3 extra copies of `K`. Caught locally (a standalone, Cython/Marmot-
   free correctness script run before ever touching xeon) on the very first synthetic test with
   zero constraints; fixed by slicing from an explicit `len(tVals) - nSlaveEntries` start instead of
   a negative index. A regression test (`tests/test_mpc_cached_condensation.py::test_zero_slave_dofs_regression`)
   pins this down.
2. **SciPy's SpGEMM eliminates output entries whose accumulated value is exactly zero — even with
   every individual contribution nonzero.** First surfaced on the real AMR test suite (§9's
   `AMR_*` tests, 12 of them) as a `ValueError: operands could not be broadcast together` — the
   cached scatter map's fixed size silently went stale because an independent DOF's raw `K` value
   crossed through exact `0.0` between Newton iterations while `K`'s *pattern* stayed fixed (a
   `searchsorted`-based recomputation of the three S-touching terms' positions, every call, fixes
   this — only the *union pattern itself* stays cached per epoch). Fixing that exposed the deeper
   version of the same mechanism: even a **value-blind** pass meant to build a safe superset pattern
   (data forced to a uniform placeholder like `1.0`, so no individual factor is ever exactly zero)
   is not safe, because the placeholder's own accumulated sum can *itself* cancel to exactly zero at
   a position where the real, differently-weighted sum would not (found live on the actual 280k-dof
   pryout model: 1290 missing union keys, all in one S-touching term, rows/cols in a suspicious
   consecutive pattern that traced to two slaves sharing a master with opposite-signed weights).
   Fixed by using **boolean-dtype** operands for the value-blind pass — SciPy's sparse matmul uses
   logical OR/AND for `bool`, never arithmetic cancellation, so a structurally reachable entry is
   always `True`. Both failure modes are pinned down as regressions
   (`test_value_crossing_exact_zero_regression`, `test_real_value_cancellation_regression`).

**B1's two cache-invalidation facts, verified rather than assumed, both held:** `K.indices`/
`K.indptr` are the csrGenerator's own persistent objects, returned by identity every iteration
within an epoch and always freshly allocated by a **new** `CSRGenerator`/`AliasedCSRMatrix` on any
pattern-changing rebuild (confirmed by reading `csrgeneratorv2.pyx` directly — `updateInPlace` only
ever rewrites `.data`, never `.indices`/`.indptr`, and the class explicitly locks against in-place
structural mutation); `mpcTransformation` is reconstructed wholesale in the same conditional block
that rebuilds the DofManager/CSRGenerator (`modelHasChanged or connectivityHasChanged or
self.theDofManager is None`), so its caches die exactly when they should, no epoch counter needed.

**B3.1 — exactness, full test suites, `EDELWEISS_MPC_ASSERT_EXACT=1`.** `run_tests_edelweissfe` on
both `testfiles/edelweiss-only/` and `testfiles/marmot/`: identical failure sets to the unmodified
baseline (verified by stashing the change and rerunning the same failing tests — same failures,
same messages, both before and after each bug fix) — 3 pre-existing edelweiss-only failures
(`MeshPlot` — LaTeX environment, unrelated; `NodeToDeformableSurfaceContactCurvedHexa20` and
`NodeToDeformableSurfaceContactPullOut`, both §9) and 6 pre-existing marmot failures
(`AMR_ContactRefineShear`, `AMR_MinMarkedElements`, `AMR_MixedMeshRefine`, `AMR_RecoveryError` — all
four fail identically on the unmodified baseline too, a residual/reference mismatch unrelated to
this change; plus the same two LaTeX-environment failures). Zero `EDELWEISS_MPC_ASSERT_EXACT`
firings anywhere in either suite after both bugs were fixed.

**B3.2 — live trajectory, the pryout, both solvers, `EDELWEISS_MPC_ASSERT_EXACT=1`.** Both
`profile.inp` (PARDISO) and `blockamg.inp`, run back-to-back against their unmodified-baseline
references in the same session (§21.0): identical trajectories, byte-for-byte — three cutbacks
(`0.0025`/`0.00125`/`0.000625`), same termination (`Reached maximum number of increments`), final
`U_loading` matching the baseline run to the last digit on both solvers
(`0.021875000000000002` PARDISO, `0.021875000000023456` blockamg). Zero assertion firings across
46 real Newton iterations on the full 280k-dof, 16556-slave-DOF contact+AMR+tie model — the
strongest correctness evidence this phase produced, well beyond what the synthetic unit tests alone
could cover.

**B3.3 — performance, the actual finding.** Externally timed (§21.0) plus a `perf_counter` wrapper
around `transformSystemMatrix` alone (`run_with_mpc_timing.py`, ad hoc, not checked in — monkeypatches
the method, reports the aggregate at process exit; cross-checked against the internal
`performancetiming` table's own sum across its multiple periodic printouts, which reconciled to
within 2%, so both numbers agree — the §19.1 unreliability caveat is about *trusting a single
printout in isolation*, not about the mechanism being unusable when its printouts are added up):

| | legacy (`_transformSystemMatrixLegacy`) | cached (`useCachedMPCCondensation=True`) |
|---|---|---|
| PARDISO, `MPC_TIMING_TOTAL` | 126.38 s / 46 calls (2.75 s/call) | 216.04 s / 46 calls (4.70 s/call) |
| blockamg, `MPC_TIMING_TOTAL` | 124.25 s / 46 calls (2.70 s/call) | *(not separately re-measured without the assertion; PARDISO's number is the representative one — the transform itself doesn't depend on which linear solver consumes its output)* |

**The cached path is ~1.7× *slower*, not the hoped-for ≥2× (expected ~5×) faster.** Breakdown
(internal sub-timers, PARDISO run): `mpc: S-touching SpGEMMs` alone costs 131.58 s over 46 calls
(2.86 s/call) — **more than the legacy expression's entire per-call cost**. The plan's premise —
"SpGEMMs restricted to `S`'s tiny `nEliminatedDof` rows, cheap relative to the two full-size SpGEMMs
this replaces" — does not hold in practice: SciPy's CSR @ CSR must still scan the **left** operand's
full row range (all of `K`'s own nnz, here in the millions) to find which rows intersect `S`'s tiny
nonzero-row support, so the cost tracks `K`'s own size, not `S`'s. The other three sub-costs *are*
cheap as designed (`D K D` 0.53 s/call, `value scatter` 0.69 s/call, `build union pattern` amortized
over 3 rebuilds in the 5-increment window) — the entire regression is the S-touching SpGEMMs. This
matches the plan's own explicitly stated fallback trigger almost exactly ("only if measurement shows
the S-terms or the scatter dominating (≥0.5 s): a Cython numeric-only Gustavson kernel into the
cached patterns") — the S-terms measured at 2.86 s/call, nearly 6× that trigger threshold. That
kernel is real, separate work ("do not start it unless the pure-Python numbers demand it" — they
now do) and is not started here.

**Ship decision — default off, shipped as an opt-in.** Per the plan's own gate ("(3) clears ≥2× on
the stage"), (3) fails outright (the stage is slower, not faster), so this does **not** replace the
direct expression as the default despite (1) and (2) both being clean passes. Implemented as
`MultiPointConstraintTransformation(..., useCachedCondensation: bool = False)`, with
`transformSystemMatrix` dispatching between `_transformSystemMatrixLegacy` (default) and the newly
renamed `_transformSystemMatrixCached`; wired through as a new NIST solver option,
`useCachedMPCCondensation` (default `False`, schema in `nonlinearimplicitstatic.py`, same pattern
as `pruneCondensedMatrixZeros`), reaching `buildMPCTransformation` via
`self.options.get("useCachedMPCCondensation", False)` (a `.get`, not `[...]`, since the base class
also serves `NonlinearExplicitDynamic`, whose own options dict has no such key). Confirmed the
default (legacy) path reproduces the pre-Part-B baseline exactly on both full test suites after
adding the toggle — no regression from the toggle plumbing itself.

**Value delivered despite the negative performance result:** two genuine, previously-latent
correctness bugs found and fixed (both are real hazards for *any* future cached-pattern SpGEMM
work, not just this one); a permanent regression suite
(`tests/test_mpc_cached_condensation.py`, 7 tests, pytest, no Cython/Marmot dependency) pinning both
down plus the exactness-assertion mechanism itself; a switchable, documented, opt-in alternative
implementation for whoever eventually builds the Cython kernel or finds a model where the
S-touching cost is proportionally small enough to win (e.g. a much smaller total DOF count, or a
much larger eliminated fraction).

Docs: `doc/source/documentation/constraints.rst` (a paragraph on the option and the negative
result, next to the existing tie-constraint condensation description); `useCachedMPCCondensation`'s
own schema docstring auto-renders via `doc/source/documentation/solvers.rst`'s existing
`automodule:: edelweissfe.solvers.nonlinearimplicitstatic`. Gitignored `edelweissfe/linsolve/klu/klu.c`.

### 21.3 Sequencing, and what is explicitly out of scope

1. §21.0's measurement discipline governs every gate. Then Part A (A1 → A4, A5 optional), then
   Part B (B1 → B4). B proceeds regardless of where A stops.
2. **Out of scope:** D4 (next phase, run on whatever configuration this phase leaves); PARDISO's
   frozen reordering (§7, Matthias); the `performancetiming` thread-safety fix (general infra,
   tracked separately); D1, rotational RBMs, mixed precision, B3-as-preconditioner (all measured
   and closed, §19/§20).
3. Update this document as results land, in the style of §17–§20: record failures and their
   mechanisms with the same care as wins, and never change a shipped default without the ≥20%
   bar plus a live trajectory-identical gate.

---

## 22. Phase 7 plan — p-multigrid: precondition the quadratic displacement block through a P1 corner-node operator

Planned (this session), not started. Decision (Matthias): pursue low-order preconditioning of the
displacement block, the standard high-order-FEM remedy, on the hypothesis that the **quadratic
serendipity discretization** (Quad8/Hexa20) is a root cause of the displacement-block AMG
weakness this document has been fighting since §12. The in-house evidence fits the known
signature of SA-AMG on quadratic elasticity (wide stencils, weak diagonal dominance, broken
strength-of-connection heuristics): 3-level hierarchies with ~18× first-level coarsening (§17 A2),
`eps_strong` needing the unusual 0.01 (§17 B2), and a smoother that only worked after manual
spectral surgery (§17 B5).

**The chosen construction: Galerkin corner-node restriction, NOT a P1 rediscretization.** An
actual P1 reassembly would need a second tangent-consistent pass through materials (damaged
tangents), contact penalties, and the MPC condensation — a shadow model. The Galerkin projection
`A₁ = Pᵀ A₂ P` inherits all of that from the current condensed displacement block for free, and
`P` is purely topological: **identity on corner nodes; each exclusive-midside node interpolated
½/½ from its two edge-endpoint corner nodes** (the P1 function expressed in the serendipity
basis), expanded per displacement component (`P_dof = kron(P_node, I_d)`, node-major layout, §4).

**The preconditioner shape (classic p-two-grid):** one application =
`ν` Chebyshev sweeps on the (equilibrated) quadratic operator `A₂` → restrict residual (`Pᵀ`) →
one AMGCL V-cycle on `A₁` → prolong (`P`) → `ν` Chebyshev sweeps on `A₂`. AMG finally gets the
operator class it is good at (P1 elasticity); the quadratic level only needs smoothing.

**What "win" means.** The wall math is honest, not automatic: `PᵀA₂P` is an SpGEMM re-done every
solve (the pattern churns, §21.1 — no caching), roughly offset by the cheaper hierarchy build on
the ~4–8×-smaller `A₁`; per application, a V-cycle on `A₁` + fine Chebyshev is roughly comparable
to today's V-cycle on `A₂`. **The projected win is iteration count and robustness** (healthy
deep hierarchy instead of a shallow forced one), and it must clear the usual wall-clock bar to
ship. All §19 execution notes and §21.0's measurement discipline (external wall timing,
same-session interleaved arms) apply verbatim.

### 22.1 Enabler — dump the P1 topology map (one small commit + one xeon run)

A new env-var dump, `EDELWEISS_DUMP_P1MAP`, implemented next to and mirroring
`EDELWEISS_DUMP_COORDS` (`nonlinearimplicitstatic.py:323` — same trigger point, same
field-node-ordering alignment that the coords dump already verified against the dumped systems).
For each node of each *vector* field, write one npz per dump event containing:

- `isCorner[nNodes]` (bool): a node is a corner iff it is a corner node of **at least one**
  element it belongs to. This rule is load-bearing under AMR: a hanging node can be a midside of
  a coarse element *and* a corner of fine elements — corner status wins, the node stays in the
  P1 space, and `P` remains purely topological (constraints are the condensed operator's
  business, not `P`'s).
- `edgeEndpoints[nNodes, 2]` (int, −1 for corners): for each exclusive-midside node, the two
  edge-endpoint node indices, in the *field's node ordering* (the same ordering the coords dump
  uses).

Local corner/midside numbering: **verify against the element implementations in this repo**
(`edelweissfe/elements/displacementelement/` for the pure-Python quads, Marmot's
`DisplacementFiniteElement` docs for Hexa20) — do not trust remembered Abaqus conventions. Key
the classification on nodes-per-element (8-node 2D, 20-node 3D); print the set of encountered
element types and **hard-error on an unrecognized one**. Assert: every `edgeEndpoints` entry
refers to a corner node (guaranteed by the ≥1-element rule — endpoints of a quadratic edge are
corners of that same element); corners + exclusive-midsides = nNodes.

Run once on xeon against the same model state as the 9 dumps (displacement field at 71553 nodes →
`[0, 214659)` — the §13 coords-dump procedure describes catching the right snapshot). Sanity-check
offline: `P` shape (214659 × 3·nCorners), row sums ≈ 1 per component, `P` restricted to corner
dofs is the identity.

**22.1 — executed.** `edelweissfe/numerics/p1topology.py` (`classifyElementTopology`,
`buildP1Map`) plus the `EDELWEISS_DUMP_P1MAP` enabler in `nonlinearimplicitstatic.py`. Local
node-order tables for Quad8/Hexa20 verified directly against both the pure-Python element shape
functions and Marmot's `DisplacementFiniteElement` C++ source (not assumed from Abaqus memory), and
cross-checked against the existing `edelweissfe.adaptivity.hex20shapefunctions.EDGES` table (used
in production by SPR recovery) — all three agree. A pytest suite
(`tests/test_p1topology.py`, 6 tests, stub objects, Cython/Marmot-free) plus a live smoke test
against a real, meshed Quad8 model (378 nodes, 75 elements) both pass.

**A real, unanticipated finding, found on the actual xeon run (not the plan's own anticipated
"corner wins" case) — a genuine edge-endpoint conflict at 2:1-balanced non-conforming AMR
boundaries.** The first live run raised: two currently-*active* `GC3D20R` elements disagreed about
a shared node's edge endpoints. Both possible simpler explanations were checked and ruled out
before concluding this is real:

- **Not an MPC slave DOF** — verified live (a wrapper around `buildMPCTransformation` capturing
  the transformation and checking the conflicting node's 3 displacement DOF indices against
  `slaveDofIndices` — none matched). If it were a hanging node, it would never reach the
  condensed operator blockamg actually preconditions, so the conflict wouldn't matter; it does,
  because it's a genuine independent DOF.
- **Not a stale/inactive AMR parent lingering in `model.elements`** — verified by reading
  `edelweissfe/modelmodifiers/adaptivity/hadaptivity.py`'s `_materialize`: a refined parent is
  unconditionally deleted from `model.elements` (and every `elementSets` entry) in the same
  synchronous call that adds its children — never observable as still-present afterward.
- **Not a node-order bug in AMR-generated children** — verified by reading `Hex20Topology.subdivide`
  (`edelweissfe/adaptivity/hex20topology.py`) and `hex20_box_coords`
  (`edelweissfe/adaptivity/hex20shapefunctions.py`): every child's 20-node list is constructed
  explicitly in standard C3D20 order (corners 0-7, then midsides), not by octant structure or a
  fine-grid index. `_materialize` passes that order straight to `setNodes()`, no remap.
- **Both elements individually geometry-verified.** One side's midside-corner claim was checked
  directly against the raw mesh coordinates (`mesh/concrete.inp`'s `*NODE` block): the midpoint of
  its claimed corner pair matches the shared node's coordinates to the file's own precision, exact.

**Root cause (best explanation, not fully proven further — see the scope note below):**
`edelweissfe.adaptivity.refinement.NodeRegistry.label()` deduplicates newly-created AMR nodes by
*rounded coordinate*, with no topological awareness of which edge/element a coordinate "belongs"
to. At a genuine 2:1 hanging-node interface, a coarse element's own edge-midpoint node can
coincide, in raw 3D space, with a node an *unrelated* neighboring fine element (refined from a
*different* coarse parent) also needs at that same location — and the registry correctly reuses
the one shared label, but the two elements' independent, locally-correct classifications of "what
edge is this node the midpoint of" then genuinely disagree, because they are answering that
question at two different mesh resolutions. This is very likely a pre-existing property of how
this AMR module's non-conforming refinement produces shared boundary nodes, not a bug introduced
by this work — but confirming that fully would need reading the AMR module's own non-conforming-
interface design intent directly with its author, which is out of scope for this enabler.

**Resolution — corner status wins here too, not just for the plan's originally-anticipated
corner-vs-midside case.** `buildP1Map` was extended (beyond the plan's own text, which only
specified the corner-vs-midside rule) to treat a genuine edge-endpoint *conflict* the same way: a
corner is always identity in `P1` regardless of *why* it is one, so this can only ever enlarge the
`P1` space by a handful of extra corners, never miscompute it — unlike silently keeping one
element's arbitrary guess (unauditable) or hard-erroring (blocks every model exhibiting this,
evidently not a rare, pattern). Every fallback is reported through a new `warnings` return value,
routed to the Journal at the dump call site (level 1), not swallowed. Measured on the reference
model: **524 conflicting nodes out of 71553 (~0.7%)**, localized (as expected) to refinement
boundaries.

**Sanity checks passed** (`sanity_check_p1map.py`, ad hoc, not checked in): `P_node` shape
`(71553, 26389)`; every row sums to exactly `1.0`; `P` restricted to corner rows is exactly the
identity; the DOF-level `P = kron(P_node, I_3)` has the expected shape `(214659, 79167)`.

Commits: `05be1d6c` (Fable's §22 plan, committed — it had been sitting uncommitted on disk),
`4aeee612` (the enabler), `dc4bf620` (the conflict-fallback fix, found and resolved live on xeon).

### 22.2 Offline two-grid probe on the displacement block — the go/no-go (~half day)

New ad-hoc probe on xeon (investigation dir, not checked in), on the §17/§20.1 testbed
(`A_00_00002.npz` displacement diagonal block, equilibrated exactly as production does —
project the **scaled** operator: `A₁ = Pᵀ As₂ P` with `P` unscaled; Galerkin absorbs the
scaling).

**Measure in production usage mode from the start — §20.1's trap, stated as a rule:** every arm
is a *one-application-per-outer-iteration* preconditioner `M` inside outer
`scipy gmres(M=100, rtol=1e-6)` on `As₂`. Never `.solve()` (AMGCL-native Krylov) — §20.1 proved a
better standalone solver can be a worse single-shot preconditioner, and that mistake cost a wrong
first pass already (`probe_201_block_isolated.log`).

Implementation of the probe's `M`: hand-rolled Chebyshev on `As₂` (a plain matvec recurrence in
scipy — **estimate the spectral radius with ~50 power iterations and use `lower=0.01`**, §17 B5's
lesson: the cheap default estimate is exactly what made Chebyshev diverge before) wrapped around
`build(A₁)`/`applyPreconditioner()` on the existing AMGCL wrapper. The serial scipy matvecs make
the *wall* numbers pessimistic vs. a threaded production smoother — so record, per arm, the
iteration count AND the decomposed per-application cost (fine-smoothing seconds vs. coarse-V-cycle
seconds), and derive a projected-threaded wall alongside the measured one.

Reference arms, same probe run, same session (`probe_201_block_isolated_v2.log`'s numbers for
cross-checking only): production scalar default (96 iters / 6.37 s there) and §17's tuned
d=8/npre=npost=2 (51 / 9.02 s).

Sweep, small: fine `ν ∈ {1, 2}` × fine Chebyshev degree ∈ {2, 3, 5} × coarse-level config ∈
{shipped scalar default, §17's d=8/npre=npost=2} — the coarse level is cheap, so the stronger
config may win there. Two extra single columns: (a) coarse-level `set_nullspace` with the 3
translations **on `A₁`** — RBMs measurably don't help on the quadratic condensed block (§11/§13),
but a clean P1 Galerkin operator is a different animal and the column costs nothing; (b) one
no-fine-smoothing row (pure `P V(A₁) Pᵀ`) — expected to stall (unsmoothed quadratic
high-frequency error), recorded to confirm the mechanism.

**The diagnostic that decides whether the hypothesis is even true:** `report()` on the `A₁`
hierarchy. Expect a healthy SA hierarchy (4+ levels, ~3–4× coarsening, sane operator complexity)
in place of the 3-level/18× pathology of §17 A2. **If SA on `A₁` is still shallow/degenerate, the
quadratic-discretization hypothesis is falsified** — the real obstacle is then the
contact-penalty/condensation structure surviving the projection — record it, stop the phase, and
note that this redirects future effort to constraint-aware preconditioning (the low-rank
penalty/Woodbury idea from the Phase-7 discussion), not to more AMG tuning.

**Gate:** best two-grid arm beats the production default on the isolated block by **≥20%
projected-threaded wall** (and no worse measured-serial than the §17 tuned arm), with iteration
count materially below 96. Otherwise record and stop.

**22.2 — executed. The deciding diagnostic falsifies the hypothesis; the phase stops here, per
the plan's own pre-committed criterion.**

**Setup, verified working:** `probe_222_pmultigrid.py` (xeon, investigation dir, ad hoc, not
checked in) builds `P` from §22.1's dump, projects `A₁ = Pᵀ Asₘ P` (see the Dirichlet-masking note
below for `Asₘ`), and wraps both single-level AMGCL (the two reference arms) and the two-grid
scheme as a one-application-per-outer-iteration `M` inside `scipy.sparse.linalg.gmres` — never
AMGCL's own `.solve()`, per §20.1's rule. Reference arms reproduce the expected ballpark (111/77
total iterations vs. the `.solve()`-based `probe_201_block_isolated_v2.log`'s 96/51 — different
Krylov implementations wrapping the *same* preconditioner, so a different but comparable iteration
count is expected, not a red flag).

**Two real bugs in the probe script itself, both fixed before trusting any number it produced:**

1. **`scipy.sparse.linalg.gmres`'s `maxiter` counts restart cycles, not total inner iterations**,
   whenever no `callback` function is supplied (`callback_type` alone, without `callback`, has
   *no effect* per SciPy's own docs — an earlier draft of this probe relied on exactly that
   no-op). With `restart=100`, the first version's `maxiter=500` silently requested up to **50,000**
   total preconditioner applications, not 500 — the reason the very first sweep attempt ran for
   23+ minutes without finishing even one arm. Fixed by expressing the intended ~150-200 total
   iteration cap as `maxiter=2` restart cycles.
2. **Dirichlet identity rows do not Galerkin-project cleanly through a non-trivial `P`** — verified
   live: 24160 of 214659 displacement rows on this block are pure Dirichlet identity rows
   (`nnz==1`, on-diagonal, value exactly 1.0), and 15408 of those are at *midside* nodes, so
   `Pᵀ(identity row)P` spreads a spurious ½-weighted entry across the two corner rows/columns
   instead of preserving the constraint — corrupting `A₁` (every two-grid arm stalled at
   `true_res` ≈ 1.0 in the first working sweep). The plan's own text names this exact gotcha
   ("mask Dirichlet rows/columns of `As₂` before projecting") but a first masking attempt
   (`Mfree @ As @ Mfree`, zeroing both rows *and* columns via a single diagonal similarity
   transform) also zeroed the *diagonal* at Dirichlet positions — a Dirichlet dof's own diagonal
   entry `(i,i)` lies in both its masked row and its masked column — producing a singular `A₁`
   (1538 exactly-zero diagonal entries, all traced to Dirichlet corners, which project 1:1 through
   `P`). Fixed by masking off-diagonal coupling only, preserving the diagonal explicitly:
   `Mfree @ (As - diag(As)) @ Mfree + diag(As)`. Verified on a tiny synthetic example locally
   before re-running on xeon, and confirmed the shallow-hierarchy finding below is *not* an
   artifact of this masking (the unmasked `A₁` gives the same 3-level, steep-coarsening pathology,
   just with additionally-pathological negative diagonal entries from the unmasked smearing).

**The fine Chebyshev smoother also diverges standalone** (17-80× residual growth over 3 sweeps,
isolated from any coarse correction) — traced to `As` (the isolated, equilibrated displacement
block) being genuinely **~50% non-symmetric** (`‖A - Aᵀ‖/‖A‖ ≈ 0.50`, matching this document's
earlier-documented condensed-elasticity asymmetry, §13/§17), which a naive power-iteration-based
Chebyshev semi-iteration (designed for SPD spectra) is not built for — confirmed the spectral
*radius* estimate itself is not the problem (ARPACK's Arnoldi-based `eigs` gives 6.46-7.09, close
to the power iteration's 6.65), so this is departure-from-normality, not a bad bound. AMGCL's own
internal "chebyshev" relax type, used *inside* a full SA-AMG V-cycle rather than standalone,
evidently tolerates this operator (the single-level reference arms converge fine) — a hand-rolled,
standalone Chebyshev sweep with no coarse partner does not. **This was not pursued to a full fix**,
because the diagnostic below settles the phase's actual question before it matters.

> **Retracted by §22.2-bis, below.** The "deciding diagnostic" applied here (hierarchy *shape*)
> was the wrong instrument — smoothed aggregation's per-level coarsening is aggressive by design,
> and this document's own §17 A2 data already showed a 3-level/~18×-coarsened hierarchy delivering
> the phase's headline 29-iteration success. §22.2-bis reran the actual decisive measurement
> (preconditioned convergence on `A₁` itself) and it is unambiguously AMG-friendly (26 iterations);
> a two-grid scheme built with §17 B1's Dirichlet-handling on the fine level clears the original
> ≥20% gate with a 2.30× margin. Kept below verbatim as the record of what went wrong and why —
> see §22.2-bis for the corrected result and the retraction in full.

**The deciding diagnostic (`report()` on `A₁`'s hierarchy) — falsified.** Both tested coarsening
configs give a **shallow, steeply-coarsened hierarchy**, not the "4+ levels, ~3-4× coarsening"
the hypothesis predicted:

| coarsening config | levels | coarsening ratios | operator complexity |
|---|---|---|---|
| shipped default (SA, no `eps_strong` override) | 3 | 79167 → 5962 (13.3×) → 304 (19.6×) | 1.19 |
| §17 tuned (`eps_strong=0.01`, degree 8) | 2 | 79167 → 1303 (60.8×) | 1.01 |
| *(unmasked `A₁`, sanity cross-check)* | 3 | 79167 → 4423 (17.9×) → 175 (25.3×) | 1.14 |

This is essentially the **same** shallow/degenerate pathology the *full* quadratic operator
already showed (§17 A2's 3-level/18× hierarchy) — the P1 projection does not rescue AMG's
behavior. Per the plan's own pre-committed criterion, **the quadratic-discretization hypothesis is
falsified**: the quadratic serendipity discretization was not, on its own, the root cause of the
displacement-block AMG weakness this document has been fighting since §12. The real obstacle
survives the projection — almost certainly the contact-penalty/condensation structure (the same
~50% non-symmetry that broke the fine smoother above is a direct symptom of that structure), not
the element order.

~~Phase stops here, exactly as the plan's own text directs on this outcome. 22.3 (coupled offline
validation) and 22.4 (live gate, NIST plumbing, ship decision) are not attempted — their entire
premise (a two-grid scheme worth validating end-to-end) depends on 22.2 clearing the go/no-go, which
it does not. This redirects future preconditioning effort toward constraint-aware methods (the
low-rank penalty/Woodbury idea named in the Phase 7 discussion) rather than further AMG tuning on
this operator class.~~ **Retracted — see §22.2-bis.** The go/no-go call above rested on a
miscalibrated shape-based diagnostic and a fine smoother tested without §17 B1's Dirichlet
handling; corrected, the phase clears its own gate and continues into 22.3/22.4. The §22.1 enabler
(the P1 topology map itself, its two real bugs, and its regression tests) remains committed and
correct throughout — it is the infrastructure both the retracted and the corrected result depend on.

**A separate, real finding surfaced along the way, unrelated to the hypothesis test itself:**
`OMP_NUM_THREADS`/`MKL_NUM_THREADS` (this whole investigation's standard env-var convention) leaves
a **second, uncoordinated thread pool** active throughout every run on xeon. `numpy`/`scipy` here
are linked against OpenBLAS's pthreads build (`libopenblasp*.so`), not MKL (`numpy.show_config()`
confirms generic BLAS/LAPACK, not MKL) — so `MKL_NUM_THREADS` has been a no-op for the whole
investigation's numpy/scipy calls. AMGCL's compiled extension links `libgomp` (GNU OpenMP) and
correctly respects `OMP_NUM_THREADS`, but OpenBLAS's pthreads pool is a separate runtime that does
not share that budget; both were observed sizing themselves to roughly 16 independently, live on
this exact probe's process (`Threads: 31`, split into two groups by accumulated CPU time),
totaling ~32 OS threads on this 36-core, 2-socket, no-hyperthreading machine — not hardware
oversubscription, but a real violation of the "16 threads total" mental model every run command in
this document has assumed. Explicitly setting `OPENBLAS_NUM_THREADS=16` alongside the existing
vars does not change the total (OpenBLAS was already landing near 16 via its own heuristic), since
the two pools are structurally independent regardless of either variable's value — a genuine fix
would mean deliberately splitting the 16-thread budget between the two runtimes (e.g. `OMP_NUM_THREADS=8`
+ `OPENBLAS_NUM_THREADS=8`) if a true combined cap of 16 is wanted, or accepting 32 as intentional
on a 36-core box. Recorded here for whoever revisits xeon's run-command conventions; not otherwise
acted on in this phase (same-session relative comparisons throughout this document are unaffected,
since all arms in a given probe are subject to the same oversubscription equally).

### 22.2-bis — re-examination of the 22.2 verdict (executed — verdict retracted)

The 22.2 record above is honest about what it measured, but the verdict "hypothesis falsified"
does not follow from those measurements. Reviewed (Fable + Matthias, this session); three defects,
each traceable to a specific earlier section:

1. **The deciding diagnostic's calibration was wrong — and the §22 plan itself planted it.** The
   gate expected "4+ levels, ~3–4× coarsening" as *healthy*, inherited from §17 A2's wording
   ("far more aggressive than healthy SA's usual ~3–4×"). That describes classical AMG or
   geometric multigrid. **Smoothed aggregation is aggressive-coarsening by design** — an aggregate
   is a whole strongly-connected neighborhood, so ~10–30× per level with operator complexity
   ~1.1–1.3 and few levels is *textbook SA*, and the measured `A₁` hierarchy (3 levels,
   13.3×/19.6×, complexity 1.19) fits it. The document's own data proves shape was never
   disqualifying: **§17's headline 29-iteration result was achieved on a 3-level, ~18×-coarsened
   hierarchy** on the full quadratic block. Hierarchy shape is a weak observable; convergence is
   the observable.
2. **The fine-smoother divergence was mis-diagnosed, reinstating §13's retracted causal story.**
   The probe ran Chebyshev on the *unmasked* equilibrated block and attributed the divergence to
   "genuine ~50% non-symmetry" from contact/condensation. §17 A1 measured exactly this and
   localized it: the asymmetry collapses to **0.03% raw / 0.58% scaled once Dirichlet rows and
   columns are masked** — a Dirichlet/`eliminate_zeros` storage artifact, not physics. And §17 B1
   showed the operational consequence directly: Chebyshev diverges on the full block but
   **converges on the Dirichlet-masked free submatrix with zero tuning**. The probe masked
   Dirichlet for the projection but not for the smoothing — the one place §17 B1 says it matters
   most.
3. **The decisive quantity was never measured.** No two-grid arm ever ran to convergence after the
   masking fix, and nobody solved `A₁` itself with AMG-preconditioned GMRES — minutes of compute
   that answers the actual question ("is the P1 Galerkin operator AMG-friendly?") directly. The
   closing redirect toward constraint-aware preconditioning is currently *asserted, not measured* —
   the exact failure pattern §16 called out in §13, which this document has already retracted once.

**Corrected calibration, stated once so this ruler cannot fire again (also add to §8):** for
smoothed aggregation, few levels + 10–30× per-level coarsening + operator complexity ~1.1–1.3 is
healthy by construction. Never judge an SA hierarchy by its shape; judge it by preconditioned
convergence on its own operator.

All steps offline on xeon (investigation dir, same-session interleaved arms, external
`perf_counter` timing, RSS guard). Env vars: `OMP_NUM_THREADS=16` **and, per 22.2's own finding,
`OPENBLAS_NUM_THREADS` set explicitly** (`MKL_NUM_THREADS` is a no-op for numpy/scipy here —
OpenBLAS build). Total ~1–2 h.

**R1 — the decisive number (~30 min).** On the masked-projected `A₁` from the fixed 22.2 probe:
solve `A₁ x = r` with `gmres(M=100, rtol=1e-6)` preconditioned by one AMGCL V-cycle per iteration
(`build(A₁)` once; one-shot-`M` mode as everywhere). Two right-hand sides, both recorded: `Pᵀ`
applied to the dumped block's (masked, equilibrated) residual, and one fixed random vector. Arms:
{shipped scalar default, §17 tuned d=8/npre=npost=2} × {no nullspace, 3 translations on `A₁`}
(`A₁`'s DOF layout is corner-node-major × 3 by construction of `P = kron(P_node, I_3)` — build the
translations accordingly). Also record `report()` per arm, now as *context*, not as a gate.

*Pre-committed interpretation:* best arm **≤ ~40 iterations** → `A₁` is AMG-friendly, the 22.2
falsification is overturned, proceed to R2. **≥ ~100** → the falsification is *confirmed on the
right instrument*, the constraint-aware redirect is earned, and the phase re-closes with that
recorded. In between → report both numbers and stop for a judgement call; do not improvise.

**R2 — one honest two-grid arm, free-submatrix construction (~30 min; only if R1 is good).**
§17 B1's construction, applied end-to-end: remove Dirichlet rows/columns *entirely* on the fine
level (the free submatrix — §17 B1 found 24,160 Dirichlet rows → 190,499 free on this block;
recompute, don't assume), and restrict `P` to free fine rows × free coarse columns. The one
interpolation rule that needs stating: a free midside node whose edge-endpoint corner is
Dirichlet-constrained keeps only the ½-weight on its free endpoint, **no renormalization** — the
constrained corner contributes exactly zero to a homogeneous Newton correction, which is what the
weight dropping encodes. Assert the resulting free-`P` rows are nonzero (a free midside with
*both* endpoints constrained would be an orphan — hard-error if one exists, it would mean the
Dirichlet data disagrees with the topology map). Fine smoothing: hand-rolled Chebyshev **on the
free submatrix** (spectral radius from ~50 power iterations *on that submatrix*, `lower=0.01`) —
the configuration §17 B1 measured converging standalone with zero tuning, so it is a valid
smoother there. Solve the free system directly with outer GMRES (equivalent to identity-on-
Dirichlet for the full system, and simpler). Arms: `ν ∈ {1, 2}` × fine degree ∈ {3, 5} × R1's best
coarse config; reference arm in the same run: single-level AMGCL (production default) on the same
free submatrix, same one-shot-`M` mode. Gate as 22.2 intended: ≥20% projected-threaded wall vs.
that reference, iteration count materially below it.

**R3 — close the asymmetry story on paper (~5 min).** Measure `‖A−Aᵀ‖_F/‖A‖_F` on the *masked*
equilibrated block and on the free submatrix. Expected per §17 A1: ~0.6% and ~0.03%-ish. Record
next to 22.2's ~50% unmasked figure with one sentence tracing the difference to the Dirichlet
artifact — so the §13 story stays retracted in writing and the "departure from normality" claim
is either killed or, if the masked number surprises, genuinely earned.

**Bookkeeping, either way:** update the header bullet and the 22.2 record's closing paragraph with
the R1 outcome (overturned → resume the original 22.2 sweep using R2's construction, then
22.3/22.4 as planned; confirmed → the phase re-closes, redirect earned). Add to §8: the corrected
SA calibration above, and 22.2's scipy `gmres` `maxiter`-counts-restart-cycles-without-`callback`
trap (currently recorded only inside the 22.2 text). The OpenBLAS/`MKL_NUM_THREADS` finding should
also change the standard run-command convention going forward (set `OPENBLAS_NUM_THREADS`
explicitly; whether to split 8+8 or accept ~32 threads on the 36-core box is a deliberate choice
to record, not benchmark, here).

**22.2-bis — executed. R1 overturns 22.2 outright; R2 confirms a real, gate-clearing two-grid win
on the correct (free-submatrix) construction; R3 closes the asymmetry story exactly as §17 A1
predicted. The 22.2 "falsified" verdict is retracted.**

**R1 — the decisive number.** `probe_222bis_R1.py` (xeon, investigation dir, ad hoc, not checked
in). Solved `A₁ x = r` directly with `gmres(M=100, rtol=1e-6)`, one AMGCL V-cycle per iteration,
`build(A₁)` once, two RHS (`Pᵀ` applied to the dumped block's masked/equilibrated residual;
one fixed random vector), four arms (`{shipped default, §17 tuned} × {no nullspace, 3 translations
on A₁}`):

| coarse config | nullspace | iters (both RHS agree) |
|---|---|---|
| shipped default | none | 70-72 |
| shipped default | translations | 61 |
| §17 tuned (`eps_strong=0.01`, d=8, npre=npost=2) | none | 38 |
| **§17 tuned** | **translations** | **26** |

Best arm, 26 iterations, is **well under the ≤40 threshold** — `A₁` is unambiguously AMG-friendly.
A genuine bonus finding: the rigid-body translation near-null-space, which measurably does **not**
help on the full quadratic condensed block (§11/§13), **does** help here (72→61, 38→26) — exactly
the textbook behavior expected of a clean low-order elasticity operator, further supporting that
`A₁` is a "normal" AMG target unlike its quadratic parent. **R1 verdict: OVERTURNED — proceed to R2.**

**R2 — one honest two-grid arm, free-submatrix construction.** `probe_222bis_R2.py`. Recomputed
(not assumed) the free/Dirichlet split on this block: **24,160 Dirichlet, 190,499 free** —
identical to §17 B1's figure, a strong cross-check. Built `Asfree` (Dirichlet rows/columns removed
entirely, not masked-in-place), restricted `P` to free fine rows × free coarse columns (74,167 →
70,415 free coarse dofs), asserted zero orphan rows (none found — the topology map and the
Dirichlet data agree), projected `A₁free = Pfreeᵀ Asfree Pfree` (diagonal healthy: min 0.31, 0
non-positive), and ran a real two-grid sweep (`ν ∈ {1,2} × degree ∈ {3,5}`, coarse = R1's winner)
against a single-level-AMGCL reference, both measured the same one-shot-`M` way, on `Asfree`
directly (solving the free system, equivalent to identity-on-Dirichlet for the full one):

| arm | iters | wall | projected-threaded wall |
|---|---|---|---|
| reference: single-level AMGCL, production default | 111 | 14.71s | 8.00s |
| *(context)* single-level AMGCL, §17 tuned | 60 | 13.52s | 10.26s |
| two-grid: ν=1, degree=3 | 74 | 22.75s | 3.93s |
| **two-grid: ν=1, degree=5** | **58** | 25.16s | **3.48s** |
| two-grid: ν=2, degree=3 | 61 | 30.92s | 3.98s |
| two-grid: ν=2, degree=5 | 50 | 40.77s | 4.29s |

**Best arm: 2.30× projected-threaded speedup vs. the production-default reference, at 58
iterations vs. 111 — clears the ≥20% gate with iteration count materially below the reference.**
The fine Chebyshev smoother (identical construction to 22.2's, applied this time to `Asfree`
instead of the unmasked full block) converges cleanly with no divergence at all — confirming the
22.2 divergence really was the Dirichlet-contamination artifact §17 B1 already named, not a
property of the physics.

**R3 — the asymmetry story, closed.** `‖A-Aᵀ‖/‖A‖`: **0.56%** on the masked equilibrated block
(vs. §17 A1's reported 0.58% scaled — matches almost exactly) and **0.58%** on the free submatrix.
Both tiny, both consistent with §17 A1's prediction. 22.2's ~50% figure was measured on the
*unmasked* block and was always a Dirichlet-elimination storage artifact, never physics — the
"departure from normality" explanation 22.2 offered for the fine-smoother divergence is killed,
cleanly, in writing, for good.

**Retraction, stated plainly:** §22.2's verdict ("the quadratic-discretization hypothesis is
falsified... the real obstacle is the contact-penalty/condensation structure") was wrong, and
wrong for the exact reason §16/§13 already warned about once before in this document — asserting
a redirect instead of measuring the one number that decides it. The corrected record: **the P1
Galerkin operator is measurably AMG-friendly, and a two-grid scheme built on it delivers a real,
gate-clearing win once Dirichlet handling matches §17 B1's construction on the fine level, not
just the coarse projection.** Per this section's own bookkeeping instruction, the phase resumes:
22.2's original sweep is superseded by R2's construction (the free-submatrix two-grid already
clears the gate, so the broader `ν × degree × coarse-config` grid 22.2 originally specified is not
separately re-run — R2's 4-arm sweep already identifies a clear winner and clears the bar with
margin); proceed to 22.3 (coupled offline validation) and 22.4 (live gate, plumbing, ship
decision) next.

### 22.3 Coupled offline validation (production code path, all 9 ords)

Implement the two-grid as an opt-in per-field preconditioner variant inside
`edelweissfe/linsolve/blockamg/blockamg.py` (a p-two-grid object holding `P`, the `A₁` AMGCL
instance, and the fine Chebyshev; selected when a quadratic topology map is present *and* an
option enables it). For this offline step the map is **injected by the probe** directly — the
NIST plumbing waits for a pass. Replay all 9 dumped ords, same-session arms: shipped default vs.
the two-grid variant, EW forcing + true-residual stopping on both (the shipped solver behaviour,
§20.2). Watch the true-residual continuations — a different `M` changes the
preconditioned/true-residual gap, and continuations are part of the honest cost (§20.2 step 1).

**Bar (unchanged from every prior default change):** ≥20% aggregate external wall over the 9
ords, no ord regressing past ~1.05×, true residuals no looser than the shipped default's.

**Production fine-smoother decision (only after the bar clears):** the probe's serial scipy
Chebyshev is a stand-in. The production form should be a threaded apply — preferred: extend the
AMGCL wrapper with a standalone relaxation-as-preconditioner entry point
(`relaxation::as_preconditioner`, the same additive `.hpp`/`.pxd`/`.pyx` pattern as
`set_nullspace`/`build`/`report`; ~half day) so the fine sweeps run OpenMP-threaded like every
other AMGCL kernel. A `prange` Cython SpMV is the fallback. Decide by measuring the serial
version's share first — if fine smoothing is <15% of a coupled solve, ship the simple version and
record the deferred optimization.

### 22.4 Live gate, plumbing, and ship decision

1. **NIST plumbing:** compute `isCorner`/`edgeEndpoints` at equation-system build, next to where
   the field structure is already pushed (`nonlinearimplicitstatic.py:314`,
   `LinearSolver.setFieldStructure`); `FieldBlock` (`edelweissfe/linsolve/base.py:46`) gains the
   optional map. **Graceful degeneration is a hard requirement:** on a linear mesh every node is
   a corner → `P = I` → the solver must skip the p-level and behave exactly as today
   (`CantileverBeamQuad4BlockAMG` must pass unchanged).
2. **New regression test:** a Quad8 `blockamg` testfile (clone `CantileverBeamQuad8`, the same
   pattern `CantileverBeamQuad4BlockAMG` followed), exercising the p-two-grid path end-to-end
   through the Newton loop. Registered, skipped where AMGCL is not built.
3. **Live gate:** the pryout per §19.1/§21.0 — back-to-back with unmodified `blockamg.inp`,
   trajectory identical (15/8/5/5, cutbacks `0.0025`/`0.00125`/`0.000625`,
   `U_loading=0.021875`), externally timed win consistent with 22.3's margin.
4. **Ship:** on a pass, default-on for vector fields that carry a quadratic map (scalar fields
   and linear meshes untouched); docs in `doc/source/documentation/linsolvers.rst` (blockamg
   section: the p-level, the topology-map requirement, the degeneration rule). On a fail at any
   gate: record the table and the mechanism, default untouched, the enabler dump and the opt-in
   variant stay (they are the infrastructure any future p-multigrid attempt needs).

### Gotchas specific to this phase (beyond §8/§17/§19)

- **§20.1's standalone-vs-embedded trap** is the reason 22.2 mandates one-shot-`M` measurement.
  Do not report any `.solve()` numbers as evidence.
- **Dirichlet identity rows survive into `A₁` smeared** (`Pᵀ·(identity row)·P` spreads ½-weights).
  §17 B1's hazard was *symmetrization* across Dirichlet rows, which is not what Galerkin does —
  but if `A₁` misbehaves (NaN from `applyPreconditioner`, §17's silent-NaN gotcha; stalls;
  non-positive diagonal entries), mask Dirichlet rows/columns of `As₂` before projecting, per
  §17 A1/B1, and re-run before drawing any conclusion.
- **Scaling order is fixed by decree:** equilibrate first (production behaviour), then project
  the scaled operator with the unscaled topological `P`. Do not scale `P`.
- **The Chebyshev spectral estimate on `As₂` is the probe's own responsibility** (§17 B5): a bad
  bound diverges. Power-iterate ~50 steps once per system; `lower=0.01`.
- **`PᵀA₂P` cost counts.** It re-runs every solve (§21.1: the pattern churns; nothing to cache).
  Time it separately in every probe table; it is part of the hierarchy-build budget, not free.
- Element-local numbering must be **verified from this repo's element code**, not assumed from
  Abaqus memory; the dump hard-errors on unrecognized element node counts.

### Out of scope for this phase

Matrix-free fine-level smoothing (the Jodlbauer/Langer/Wick route — the right long-term answer to
the §18/§19.3 bandwidth wall, wrong scope for a probe-first phase); Krylov recycling (GCRO-DR —
independent lever, own future phase, pilots cheaply via PETSc/HPDDM given `petsclu` already
exists); the §21.2 Cython condensation kernel; D4 (still the phase after, on whatever
configuration this phase leaves); everything §21.3 already lists.
