---
name: ew-add-element
description: >-
  Procedure for adding, registering, testing, and documenting a new finite element formulation in EdelweissFE.
  Use when implementing continuum elements, structural/beam elements, mixed/EAS elements, or Marmot element wrappers.
---

# Implementing a New Finite Element in EdelweissFE

## Code Reuse & Anti-Duplication Principles

Before writing any new element code, adhere strictly to these principles:
- **Reuse Existing Base Classes**:
  - Always inherit from base classes such as `DisplacementElement`, `DisplacementTLElement`, or relevant continuum/structural element base classes.
- **Reuse Quadrature Rules & Shape Functions**:
  - Do not hardcode Gauss quadrature points, weights, or shape function derivatives inside individual element classes. Use or extend shared quadrature and shape function utilities.
- **Extract Shared Matrix & Transformation Routines**:
  - Extract common $B$-matrix routines, Jacobian matrix inversion, strain transformation matrices, geometric stiffness contributions, and lumped mass accumulation into shared helpers in `edelweissfe/elements/` or `edelweissfe/numerics/`.
- **Parametrize Formulations**:
  - Avoid creating duplicate element classes for minor variations (e.g. reduced integration vs full integration, EAS mode counts, plane stress vs plane strain). Consolidate formulations with flags/options or shared base logic.

## Node Numbering & Conventions

EdelweissFE follows standard Abaqus/Marmot node ordering conventions:
- **Quad4**: Counter-clockwise 1, 2, 3, 4
- **Quad8 / CPS8R**: Corners 1, 2, 3, 4; mid-side nodes 5, 6, 7, 8
- **Hexa8 (C3D8)**: Bottom face (1, 2, 3, 4) counter-clockwise, top face (5, 6, 7, 8) counter-clockwise
- **Hexa20 (C3D20)**: Bottom face corners (1-4), top face corners (5-8), bottom mid-sides (9-12), top mid-sides (13-16), vertical mid-sides (17-20)

## Implementation Steps

### 1. Element Class Implementation
Create the element implementation in `edelweissfe/elements/<element_family>/`:
- Inherit from `DisplacementElement`, `DisplacementTLElement`, or relevant base element class.
- Reuse existing shape functions, Gauss point quadratures, and Jacobian calculators.
- Implement key routines:
  - Shape function and derivative evaluation at integration points (Gauss points).
  - Jacobian matrix and determinant computation ($J = \det(\boldsymbol{J})$).
  - $B$-matrix / strain-displacement operator formulation.
  - Internal force vector evaluation:
    $$\boldsymbol{f}^{\text{int}} = \int_{\Omega} \boldsymbol{B}^T \boldsymbol{\sigma} \, d\Omega$$
  - Element tangent stiffness matrix evaluation:
    $$\boldsymbol{K}^e = \int_{\Omega} \boldsymbol{B}^T \mathbb{C} \boldsymbol{B} \, d\Omega + \boldsymbol{K}_{\text{geom}}$$
  - Mass matrix computation (for explicit/dynamic solvers, lumped or consistent).
  - Element property interface support (via `*elementproperty` string-keyed attributes).

### 2. Register in Element Library
1. Add element metadata to `edelweissfe/elements/library.py` (specifying node count, spatial dimensions, DOF names per node, and integration points).
2. Register the loader in `edelweissfe/config/elementlibrary.py`:
   - Under `provider="edelweiss"` for native elements.
   - Or configure Marmot wrappers under `provider="marmot"` or `"marmotsingleqpelement"`.

### 3. Define Input File Syntax
Ensure `*element` keyword syntax parses correctly in `edelweissfe/utils/inputlanguage.py`:
```
*element, type=<ElementType>, provider=edelweiss
<el_id>, <node1>, <node2>, ...
```

### 4. Regression & Verification Tests
- Create a test deck in `testfiles/edelweiss-only/<TestName>/` with known boundary conditions (e.g. patch test, pure bending, or cantilever tension).
- Generate reference solution:
  ```bash
  run_tests_edelweissfe ./testfiles/edelweiss-only/ --tests <TestName> --create
  ```
- Verify parallel element loop execution with `...Parallel` solvers (e.g., `NISTParallel`, `NEDParallel`).

### 5. Sphinx Documentation
- Document the element in `doc/source/documentation/elements.rst` (degrees of freedom, topology, integration order, supported materials, formulation type).
- Verify doc build:
  ```bash
  sphinx-build ./doc/source/ ./docs -b html
  ```
