# Linear-solver performance investigation — handoff

Branch `perf/linsolve-investigation`, based on `feat/amr-recovery-marker` (`d495e90b`).
Local, remote `mn` and xeon in sync. Working tree clean.

**Status: Phases 1 (instrument + capture) and 2 (offline benchmark) are complete. Phase 3
(implement) has not started.** Two leads survive measurement; see [§4](#4-recommendation). One open
question blocks part of it, see [§7](#7-the-open-question).

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

Neither is the thing originally proposed. Plain "GMRES + AMG/ILU" is not what pays off — the payoff
comes from reusing exact factorizations across Newton iterations.

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

### 3.5 `amgcl` — fully parallel, and still loses

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

**AMG does not work on this system at all** — not at any tolerance, and *worse* with the correct
ILU0 smoother than with the default weak one, because ILU0 smoothing is expensive per iteration and
still does not fix the convergence. This is the expected outcome for an operator that is 52%
structurally asymmetric, couples two fields with no constant nodal block size, carries a ~1e8
dynamic range, and has constraint-equation rows that are not an elliptic operator.

**Single-level ILU0 works but is not competitive.** The best configuration, IDR(s=4) + ILU0, is
slower than the direct solve at every usable tolerance: 20.6 s at 1e-4 versus 11.46 s, i.e. **1.8×
slower**. The 1e-2 rows look close on wall time but are useless as Newton corrections — a 1e-2
residual leaves a **15–19% error** in the solution, another consequence of the conditioning
(residual is not error here).

**This refutes the "it just needs to be properly parallel" hypothesis directly.** AMGCL *is*
properly parallel and it still loses. Compare at the same 1e-4 tolerance:

| | iterations to 1e-4 | wall |
|---|---|---|
| lagged exact LU + SciPy GMRES (partly serial) | **4–9** | 3–7 s |
| AMGCL ILU0 + IDR(s), fully threaded | **298** | 20.6 s |

~35× more iterations. Preconditioner *quality* dominates, and parallelism cannot make up the
difference. On this system only an exact factorization — even a stale one — is a good enough
preconditioner.

Two defects found and fixed along the way (`dca36610`, `731bb2b3`): AMGCL's iteration count and
error were computed and discarded, so an unconverged solve was indistinguishable from a converged
one (a finite wrong answer passes the nonlinear solvers' NaN check); and the AMG smoother key is
`relax`, not `relaxation`, so the wrapper's shipped "BiCGStab + ILU0" default had never actually used
ILU0 — AMGCL warns on stderr about the unknown key and silently substitutes its default.

Still untried: AMGCL's `schur_pressure_correction` / field-split preconditioners against the
`[0, 214659) | [214659, 280155)` block structure, and a scaling/equilibration pass to attack the 1e8
dynamic range. Given the margin above, neither looks likely to close a 35× iteration gap.

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

**Lead 2 — deterministic pattern + frozen symbolic factorization.** 1.78× on the direct solves that
remain. Requires building `TᵀKT + C` on a cached pattern and scattering values into it instead of
running two SciPy SpGEMMs per iteration, which also recovers most of the 2.75 s condensation cost.
Blocked on [§7](#7-the-open-question).

**Not a lead — AMGCL.** Tested (§3.5) and rejected. AMG does not converge on this system at any
tolerance; single-level ILU0 does, but at 1.8× the direct solve. It is fully OpenMP-parallel and
still loses, because it needs ~35× more iterations than a lagged exact LU. Parallelism was not the
missing ingredient; preconditioner quality is.

**Also:** pin threads to one socket (§3.3).

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

- **Phase 3.** Lead 1 is unblocked and is the place to start; Lead 2 is blocked on §7.
- **The nonlinear cost of loose linear tolerances.** Lead 1's headline assumes Newton is unharmed by
  a 1e-4 linear solve. Unverified, and it could eat the gain.
- **AMGCL field-split / equilibration.** §3.5 rejected AMGCL as configured; `schur_pressure_correction`
  against the two-field block structure and a scaling pass against the 1e8 dynamic range remain
  untried, but face a 35× iteration gap.
- **Docs.** `doc/source/documentation/linsolvers.rst` is stale — omits `amgcl` and now `matrixdump`,
  and still claims only `gmres` accepts a config file. Per `CLAUDE.md`, docs gate merging.
- **Tests.** Nothing covers `matrixdump` or `factorize`/`solveFactorized`.
- **Whether this branch should merge as-is.** `matrixdump` and `benchmark_linsolve.py` are
  investigation tooling; the timings and the phase-separated PARDISO methods are worth keeping
  regardless. Decide deliberately.
