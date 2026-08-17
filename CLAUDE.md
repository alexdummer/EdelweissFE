# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

EdelweissFE is a light-weight, platform-independent, parallel finite element framework (Unit of Strength of
Materials and Structural Analysis, University of Innsbruck). Python handles non performance-critical code;
Cython/C/C++ handle performance-critical routines (element loops, CSR matrix assembly, Dirichlet BC
application, linear solver interfaces). It optionally links against the sister C++ library
[Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot/) for element/material formulations, but also
ships pure-Python/Cython element and material implementations that work standalone.

Requires Python >= 3.14. The pip-only environment (`pip_requirements.txt` / `conda_requirements.txt`
"without Marmot") targets the free-threaded ("nogil") CPython build (`python-freethreading`); see the
`freethreading_compatible` Cython directive in `setup.py` — importing a non-freethreading-safe extension
would silently re-enable the GIL process-wide and disable the thread-parallel element loops.

## Build & install

There is no editable/incremental dev-install shortcut documented beyond `pip install .` — Cython extensions
are compiled at install time via `setup.py` (`cythonize`, `-O3 -march=native`, several extensions are
`optional=True` and just get skipped with a `[FAIL]` log line if their native library isn't found — see
`edelweissfe/built_extensions.log` after a build).

```console
# Without Marmot (pure Python/Cython elements & materials only)
mamba install --file conda_requirements.txt
pip install -r pip_requirements.txt
pip install .

# With Marmot (adds Marmot-backed elements/materials): requires Eigen, autodiff, Fastor, AMGCL,
# and Marmot itself installed into $CONDA_PREFIX first — see README.md for the exact steps.
pip install -v .
```

Optional native linear-solver backends compiled as Cython extensions: MKL Pardiso, Panua Pardiso
(`optional=True`), AMGCL, KLU (SuiteSparse). Missing libraries for these do not fail the whole build.

Environment overrides for the build: `MARMOT_INSTALL_DIR`, `MKL_INCLUDE_DIR`, `EIGEN_INCLUDE_DIR`.

## Running tests

Tests are full finite-element input-deck regression tests (not primarily pytest), driven by the
`run_tests_edelweissfe` console script (`edelweissfe/_cli/_run_tests_edelweissfe.py`), which walks a
directory of test cases, each a subdirectory containing `test.inp` (+ a `U.ref` reference solution).

```console
run_tests_edelweissfe ./testfiles/edelweiss-only/      # tests that don't need Marmot
run_tests_edelweissfe ./testfiles/marmot/              # tests that need a Marmot install

run_tests_edelweissfe ./testfiles/marmot/ --tests ADVonMises,ADLinearElastic   # run only named cases
run_tests_edelweissfe ./testfiles/marmot/ --create     # (re)generate U.ref reference solutions
```

A test passes if the final DOF vector matches `U.ref` within `1e-6` max abs residual. A test whose Cython
extension isn't built, or that depends on another (itself-skipped) test's generated input, raises
`NotImplementedError`/`ModuleNotFoundError` and is reported as SKIPPED rather than FAILED.

There are also a handful of plain `unittest`-based unit tests colocated with source (e.g.
`edelweissfe/numerics/test_numerics.py`); run those with `pytest edelweissfe/numerics/test_numerics.py` or
plain `pytest` from repo root.

CI (`.github/workflows/run_tests_with_marmot.yml`, `run_tests_without_marmot.yml`) builds both
configurations from scratch on every push/PR. The Marmot-enabled CI job checks out the Marmot branch
matching `github.base_ref` (falling back to `next_v26.11` for non-PR pushes) — it deliberately does **not**
fall back to Marmot's `master`, since a stale/API-incompatible Marmot revision would build "successfully"
but be wrong. Keep this in mind when EdelweissFE and Marmot changes need to land in lockstep: give the
Marmot-side branch the same name as the EdelweissFE branch/target.

## Linting / formatting

Enforced by `.pre-commit-config.yaml` and mirrored in `.gitlab-ci.yml` (`linting` stage): `autoflake`
(remove unused imports), `black --line-length 120`, `isort --profile black`, `flake8` (ignores `E203,E501`,
extends ignore `W503`), `cython-lint --max-line-length=120 --ignore=E741` on `.pyx` files, and
`clang-format` on C/C++ sources. Run `pre-commit run --all-files` before submitting changes, or apply the
individual tools with the flags above.

## Architecture

### The `*keyword`-based input file DSL

Simulations are defined in a custom text format (`*.inp`), parsed by `edelweissfe/utils/inputfileparser.py`
against a schema in `edelweissfe/utils/inputlanguage.py`. Every `*someKeyword, option=value, ...` block is
parsed into a dict entry; `>>subKeyword` lines nest sub-definitions (e.g. `>>dirichlet`, `>>nodeforces`
inside `*step`; `>>perNode` inside `*fieldOutput`); indented, unmarked lines below a keyword are "datalines"
(e.g. node/element connectivity tables). `edelweissfe -k` / `--keywords` prints the full keyword reference;
`--doc=<module>` prints a module's docstring-based docs.

A minimal job looks like:

```
*material, name=linearelastic, id=linearelastic, provider=edelweiss
30000.0, 0.15

*section, name=section1, material=linearelastic, type=solid
all

*job, name=myjob, domain=3d
*solver, solver=NIST, name=theSolver

*node
      1,    0., 0., 0.
      ...
*element, type=C3D8, provider=edelweiss
      1,    1, 2, 3, 4, 5, 6, 7, 8

*nSet, nSet=left
      1, 2, 3, ...

*fieldOutput
>>perNode, elSet=all, field=displacement, result=U, name=displacement

*output, type=ensight, name=esExport
>>perNode, fieldOutput=displacement

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=1000, maxIter=100
>>dirichlet, name=left, nSet=left, field=displacement, 1=0, 2=0, 3=0
```

Full keyword/syntax reference: `doc/source/documentation/syntax.rst` and `doc/source/documentation/keywords.rst`.

### Plugin-registry pattern for extensibility

Almost every pluggable concept (elements, materials, solvers, generators, step-actions, output managers,
constraints, sections, analytical fields) follows the same pattern: a small `edelweissfe/config/<thing>.py`
module maps a user-facing name (from the input file, e.g. `type=C3D8` or `solver=NISTParallel`) to a
submodule path, then `importlib`-imports and returns the class lazily (so unused optional deps — e.g.
Marmot-backed elements — are never imported unless actually requested). Example: `getElementClass()` in
`edelweissfe/config/elementlibrary.py` dispatches on `provider=` (`edelweiss` vs `marmot` vs
`marmotsingleqpelement`); `edelweissfe/config/solvers.py::solverLibrary` maps `NIST`/`NISTParallel`/etc. to
solver modules. When adding a new element/material/solver/etc., register it in the corresponding
`edelweissfe/config/*.py` file rather than importing it eagerly elsewhere.

### Simulation flow

Entry point: `edelweissfe/drivers/inputfiledrivensimulation.py::finiteElementSimulation()` (called by both
the `edelweissfe` CLI and `run_tests_edelweissfe`):

1. Parse the input file → build an `FEModel` (`edelweissfe/models/femodel.py`), the model tree holding
   nodes, elements, node/element sets, sections, surfaces, constraints, materials, analytical fields,
   scalar variables, rigid bodies (`edelweissfe/helpers/inputfilehelpers.py::fillFEModelFromInputFile`).
2. Build the field output controller, plotter, output managers, and named solvers from the input file.
3. A `StepManager` (from `*step` blocks) yields `Step` objects one at a time; each step collects its
   step-actions (`>>dirichlet`, `>>nodeforces`, `>>options`, ...) and calls `step.solve()` against its
   assigned solver.
4. Field outputs and output managers (Ensight, Paraview, CSV, matplotlib, ...) are finalized after all
   steps, or on failure (`StepFailed`) / `KeyboardInterrupt`.

Solvers (`edelweissfe/solvers/`) come in implicit/explicit static/dynamic flavors, each with a serial and a
`...Parallel` variant (thread-parallel element loops, relying on free-threaded CPython — see above), plus an
arc-length variant (`NISTPArcLength`). Dirichlet BC application is itself a Cython extension
(`edelweissfe/solvers/base/dirichlet.pyx`) for performance.

Linear solvers (`edelweissfe/linsolve/`) are separate from the nonlinear solvers above and are selected
independently; interfaces exist for MKL Pardiso, Panua Pardiso, AMGCL, and KLU, each an optional Cython
extension around a native library.

### Documentation

Sphinx docs live in `doc/source/documentation/*.rst`, one file per subsystem (elements, materials, solvers,
steps, stepactions, generators, fields, dofmanager, parallelization, etc.) — check the relevant `.rst` file
first when working on an unfamiliar subsystem, since these describe intended design/usage, not just API.
