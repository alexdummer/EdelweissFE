---
name: ew-code-review
description: >-
  Quality assurance, static check, and architectural review checklist for EdelweissFE changes and pull requests.
  Use when reviewing code, checking pre-commit compliance, inspecting free-threading safety, or preparing PR submissions.
---

# Code Review & PR Quality Checklist for EdelweissFE

This skill provides a systematic review rubric for EdelweissFE pull requests and codebase modifications.

## 1. Free-Threading & Performance Safety

- [ ] **No GIL Re-enabling Imports**: Ensure third-party C-extensions without free-threading tags (e.g. `gstools`, optional dependencies) are imported lazily inside specific classes/functions, never at module top-level.
- [ ] **Cython Directives**: Verify `freethreading_compatible: True` is present in `setup.py` compiler directives for any newly added Cython extensions.
- [ ] **Element Loop Thread Safety**: Check that thread-parallel element loops (in `...Parallel` solvers) access element data without race conditions or shared state mutation.
- [ ] **Persistent Thread Pools**: Ensure parallel operations use `parallelizationutilities.getThreadPool()` rather than spinning up/joining ad-hoc threads per iteration.
- [ ] **CSR Matrix Sparsity Preservation**: Ensure stiffness matrix assembly uses in-place operations (`updateInPlace`), and Dirichlet BC zeroing retains explicit off-diagonal zeros without rebuilding sparsity patterns.

## 2. Architecture, Extensibility & Code Reuse

- [ ] **Code Reuse & Anti-Duplication**: Verify that new code reuses existing base classes, math/tensor utilities, quadrature rules, and shape functions rather than duplicating logic. Ensure shared components are extracted into common modules.
- [ ] **Lazy Plugin Registrations**: Check that new elements, materials, linear solvers, and step-actions are registered in `edelweissfe/config/<subsystem>.py` and loaded dynamically via `importlib`.
- [ ] **Input System Registry**: Ensure new input deck keywords and options are declared via `InputSystemRegistry` with clear validation.
- [ ] **Linear Solver Base Contract**: Verify new linear solver interfaces adhere to `LinearSolverBase` and handle singular matrix conditions by returning NaNs rather than hanging or continuing with corrupted values.

## 3. Formatting, Static Linting & Code Hygiene

Run automated checks locally:
```bash
# 1. Run all pre-commit hooks
pre-commit run --all-files

# 2. If Cython linting issues are flagged, run the automated fixer:
bash scripts/format_cython_files.sh
```

Inspect specific tools:
- `black --line-length 120`
- `isort --profile black`
- `flake8 --max-line-length 120 --ignore E203,E501 --extend-ignore W503`
- `cython-lint --max-line-length=120 --ignore=E741`
- `clang-format` on C/C++ sources

**Cleanliness & Artifact Checks**:
- [ ] **No Stray Agent Comments or Conversational Artifacts**: Scan modified files to ensure no temporary comments relating to AI agent plans, development conversations, or prompt instructions remain (e.g. `# Step 1 of plan...`, `# Agent note...`, `# As discussed in chat...`, `# Co-authored-by comments in source files`). Code comments must describe the math, algorithm, or codebase design—not the conversational context.
- [ ] **No Debug Print Statements or Temp Files**: Remove leftover `print()`, debug outputs, scratch dump files, or temporary testing artifacts before committing.

## 4. Test Deck Verification

- [ ] Every bug fix or new feature must have an associated regression test in `testfiles/edelweiss-only/` or `testfiles/marmot/`.
- [ ] Test cases must be deterministic, small, and fast-running.
- [ ] Run the regression suite before submitting:
  ```bash
  run_tests_edelweissfe ./testfiles/edelweiss-only/
  ```

## 5. Documentation & Sphinx

- [ ] Subsystem docs under `doc/source/documentation/*.rst` updated.
- [ ] Keyword reference in `keywords.rst` updated.
- [ ] Sphinx build compiles without warnings:
  ```bash
  sphinx-build ./doc/source/ ./docs -b html
  ```

## 6. Agent Instructions & Custom Skills Maintenance

- [ ] **Agent Guidance (`AGENTS.md` & `CONTRIBUTING.md`)**: Check if new subsystems, build flags, or workflow changes require updating `AGENTS.md` and `CONTRIBUTING.md`.
- [ ] **Workspace Skills (`.agents/skills/`)**: Check if new development workflows, refactored procedures, or newly added patterns require creating or updating tailored skills in `.agents/skills/` (e.g. `ew-add-module`, `ew-add-material`, `ew-add-element`, `ew-create-regression-test`, `ew-documentation`, etc.).

## 7. Conventional Commits & Target Branch

- [ ] Commit messages follow `<type>(<scope>): <summary>` (e.g. `feat(solvers): ...`, `fix(materials): ...`).
- [ ] PR Target Branch:
  - **Bug fixes**: Target `master`.
  - **Features & Enhancements**: Target `next_v<YY>.<MM>` (e.g. `next_v26.11`).
