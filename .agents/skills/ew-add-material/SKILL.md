---
name: ew-add-material
description: >-
  Step-by-step procedure to implement, register, test, and document a new material model in EdelweissFE.
  Use when creating or refactoring material formulations (elastic, hyperelastic, hypoelastic, plastic, or damage models).
---

# Implementing a New Material Model in EdelweissFE

This skill guides the implementation, registration, testing, and documentation of a new material model.

## Code Reuse & Anti-Duplication Principles

Before writing any new material code, adhere strictly to these principles:
- **Reuse Existing Base Classes & Helpers**:
  - Always inherit from established base classes (`BaseMaterial`, `BaseHypoElasticMaterial`, `BaseHyperElasticMaterial`) rather than re-implementing common lifecycle methods.
  - Reuse shared tensor operations, matrix inversion, Voigt notation mappings, and eigenvalue/eigenvector routines from `edelweissfe/numerics/` and `edelweissfe/materials/`.
- **Extract Shared Constitutive Logic**:
  - If multiple materials share common components (e.g. isotropic elasticity baseline, yield functions, plastic potential derivatives, return-mapping integration algorithms, or damage degradation functions), extract them into shared helper modules under `edelweissfe/materials/` or a common base class.
- **Parametrize Rather Than Duplicate**:
  - Prefer parametrizing an existing material model over creating new copy-pasted variants (e.g. like the consolidated neo-Hookean and Pence-Gou models). Avoid parallel classes that differ only by a single parameter or yield condition.
- **Performance & Invariant Caching**:
  - Cache invariant elasticity matrices ($\mathbb{C}$) and pre-allocate local arrays to avoid runtime allocations in constitutive evaluation loops.

## Implementation Steps

### 1. Create the Material Module
Create a new package or module under `edelweissfe/materials/<material_name>/`:
- Inherit from the appropriate base class (e.g. `BaseMaterial` or `BaseHypoElasticMaterial`).
- Reuse existing parameter parsers and tensor helpers.
- Implement the constitutive update method:
  - Input: Strain / deformation measure (e.g., small strain tensor $\boldsymbol{\varepsilon}$, deformation gradient $\boldsymbol{F}$, or rate of deformation $\boldsymbol{D}$).
  - State variables: Accept previous state variables and return updated state variables.
  - Output: Stress tensor (Cauchy $\boldsymbol{\sigma}$ or 2nd Piola-Kirchhoff $\boldsymbol{S}$) and consistent tangent operator ($\mathbb{C}$ or $\mathbb{D}$).
- Cache invariant elasticity matrices or common tensor operations where possible for performance.

### 2. Register in the Material Library
Register the new material in `edelweissfe/config/materiallibrary.py` under the `provider="edelweiss"` block:
```python
elif strCaseCmp(materialName, "my_new_material"):
    from edelweissfe.materials.mymaterial.mymaterial import MyMaterialClass

    material = MyMaterialClass
```
*Note*: Always import lazily inside the condition to keep startup overhead low and prevent unintentional GIL re-enabling.

### 3. Define Input File Syntax
Ensure `*material` block syntax is supported in `edelweissfe/utils/inputlanguage.py`:
```
*material, name=MatName, id=my_new_material, provider=edelweiss
<param1>, <param2>, <param3>, ...
```

### 4. Regression & Verification Tests
Follow [`ew-create-regression-test`](../ew-create-regression-test/SKILL.md) to add a minimal verification test deck under `testfiles/edelweiss-only/<TestName>/` (e.g. single-element Quad4/Hexa8 tension or shear test) and generate its `U.ref` reference solution.

### 5. Documentation
Follow [`ew-documentation`](../ew-documentation/SKILL.md) to document the material formulation, governing equations, parameter definitions, and an input deck example in `doc/source/documentation/materials.rst`.
