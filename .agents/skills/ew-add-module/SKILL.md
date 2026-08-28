---
name: ew-add-module
description: >-
  Universal workflow and architectural lifecycle for adding or extending any kind of functionality in EdelweissFE.
  Use when implementing new solvers, linear solvers, step actions, constraints, output managers, generators, analytical fields, or when routing to specialized skills.
---

# Universal Module & Feature Development Lifecycle in EdelweissFE

This skill is the general entry point and orchestrator for implementing, registering, testing, and documenting any new subsystem module or feature in EdelweissFE.

---

## 1. Routing to Specialized Skills

If you are implementing a specific subsystem covered by a dedicated skill, activate and follow that skill:

| Subsystem | Dedicated Skill |
| :--- | :--- |
| **Material Models** (elastic, hyperelastic, plastic, damage) | [`ew-add-material`](../ew-add-material/SKILL.md) |
| **Finite Element Formulations** (continuum, structural, mixed) | [`ew-add-element`](../ew-add-element/SKILL.md) |
| **Regression Test Decks** (`test.inp` + `U.ref`) | [`ew-create-regression-test`](../ew-create-regression-test/SKILL.md) |
| **Sphinx Documentation & Keywords** | [`ew-documentation`](../ew-documentation/SKILL.md) |
| **Pre-commit & PR Code Review** | [`ew-code-review`](../ew-code-review/SKILL.md) |

For all other modules (solvers, linear solvers, step-actions, constraints, AMR markers, output managers, generators, analytical fields), follow the universal lifecycle below.

---

## 2. Universal Subsystem Directory & Registry Map

EdelweissFE uses the **Plugin-Registry Pattern**. Every pluggable feature lives in a dedicated module directory and is mapped in `edelweissfe/config/`:

| Feature Type | Implementation Directory | Plugin Registry (`edelweissfe/config/`) |
| :--- | :--- | :--- |
| **Nonlinear Solvers** | `edelweissfe/solvers/` | `solvers.py::solverLibrary` |
| **Linear Solvers** | `edelweissfe/linsolve/` | `linsolve.py::getLinSolverByName` |
| **Step Actions** (BCs, loads, controllers) | `edelweissfe/stepactions/` | `stepactions.py` |
| **Constraints & MPCs** | `edelweissfe/constraints/` | `constraints.py` |
| **AMR & Error Markers** | `edelweissfe/adaptivity/` | `amrmarkers.py` |
| **Output Managers** | `edelweissfe/outputmanagers/` | `outputmanagers.py` |
| **Mesh / Geometry Generators** | `edelweissfe/generators/` | `generators.py` |
| **Analytical Fields & Variables** | `edelweissfe/analyticalfields/` | `analyticalfields.py` |
| **Sections** | `edelweissfe/sections/` | `sections.py` |

---

## 3. The 6-Step Feature Implementation Lifecycle

### Step 1: Code Reuse & Base Class Audit
- **Audit Existing Implementations**: Check existing classes in the target subsystem.
- **Inherit from Subsystem Base Class**: Ensure your class implements the standard contract (e.g. `LinearSolver` in `linsolve`, `StepActionBase` in `stepactions`, `ConstraintBase` in `constraints`).
- **Extract Shared Helpers**: If introducing reusable math, tensor, or solver routines, place them in a common module (e.g. `edelweissfe/numerics/` or subsystem base) to prevent duplication across the codebase.

### Step 2: Implementation & Free-Threading Safety
- **Free-Threading / GIL Safety**: Never import third-party C-extensions lacking `Py_MOD_GIL_NOT_USED` (such as `gstools`) at module top-level; always import them lazily inside the specific method/class where they are used.
- **CSR Matrix Assembly**: If the module modifies equations or matrix sparsity, preserve explicit off-diagonal zeros and avoid per-iteration sparsity rebuilds.

### Step 3: Lazy Plugin Registry Registration
Register the user-facing name in the appropriate `edelweissfe/config/<subsystem>.py`:
```python
# Always import lazily inside the conditional branch:
if strCaseCmp(name, "my_new_feature"):
    from edelweissfe.subsystem.myfeature import MyFeatureClass

    return MyFeatureClass
```

### Step 4: Input Language & Keyword Integration
Register the keyword, options, and dataline parsing schema in `edelweissfe/utils/inputlanguage.py` using `InputSystemRegistry`:
```python
@InputSystemRegistry.registerKeyword(keyword="*myFeature")
def parseMyFeature(parser, options, dataLines):
    # Parse options and datalines into model structure
    ...
```

### Step 5: Regression Testing (`ew-create-regression-test`)
- Create a minimal, deterministic test deck under `testfiles/edelweiss-only/<TestName>/test.inp` (or `testfiles/marmot/` if Marmot is required).
- Generate the reference solution:
  ```bash
  run_tests_edelweissfe ./testfiles/edelweiss-only/ --tests <TestName> --create
  ```
- Verify the test passes with residual $< 10^{-6}$.

### Step 6: Sphinx Documentation (`ew-documentation`)
- Add user-facing documentation and input deck snippets to the corresponding `.rst` file under `doc/source/documentation/`.
- Update the keyword catalog in `doc/source/documentation/keywords.rst`.
- Verify the documentation build compiles cleanly:
  ```bash
  sphinx-build ./doc/source/ ./docs -b html
  ```

---

## 4. Final Review Checklist (`ew-code-review`)
Before submitting changes, run the review checklist:
- [ ] Run `pre-commit run --all-files` and `bash scripts/format_cython_files.sh`.
- [ ] Ensure all tests pass (`run_tests_edelweissfe ./testfiles/edelweiss-only/`).
- [ ] Confirm no stray agent comments or temporary conversational artifacts remain.
- [ ] Ensure commit messages follow Conventional Commits format (`feat(...)`, `fix(...)`).
- [ ] Target the correct PR branch (`master` for bugfixes, `next_v<YY>.<MM>` for new features).
