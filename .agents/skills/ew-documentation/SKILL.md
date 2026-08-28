---
name: ew-documentation
description: >-
  Procedure for writing, updating, structuring, and building documentation in EdelweissFE.
  Use when adding or modifying Sphinx docs (.rst files), input keyword references, docstrings, or theory manuals.
---

# Writing & Building Documentation in EdelweissFE

This skill guides the creation and maintenance of Sphinx documentation, API docstrings, and input keyword references in EdelweissFE.

## Documentation Organization

All documentation source files live under `doc/source/`:
- **`doc/source/index.rst`**: Main documentation landing page and table of contents.
- **`doc/source/installation.rst`**: Prerequisites and installation guide.
- **`doc/source/documentation/*.rst`**: Dedicated subsystem reference pages:
  - `elements.rst`: Finite element formulations, topology, DOF layouts.
  - `materials.rst`: Constitutive models, hyperelasticity, plasticity, damage.
  - `solvers.rst`: Implicit/explicit, static/dynamic, arc-length, and parallel solvers.
  - `linsolvers.rst`: Direct/iterative linear solvers (Pardiso, AMGCL, KLU, Block-AMG).
  - `constraints.rst`: Multipoint constraints, surface ties, equal-value, penalty constraints.
  - `contacttheory.rst`: Penalty contact formulations (node-to-surface, discrete rigid body).
  - `rigidbodies.rst`: Linearized and discrete rigid body mechanics.
  - `steps.rst` & `stepactions.rst`: Step types, boundary conditions, loads, field outputs.
  - `keywords.rst` & `syntax.rst`: Input deck DSL syntax and full keyword catalog.
  - `parallelization.rst`, `dofmanager.rst`, `generators.rst`, `analyticalfields.rst`.

---

## Building the Documentation Locally

Run the following command from the repository root:

```bash
# Build HTML documentation into ./docs
sphinx-build ./doc/source/ ./docs -b html
```

### Checking for Errors
Inspect the build output to ensure there are no Sphinx warnings or broken cross-references. CI enforces clean builds.

---

## Documenting Subsystems & Features

### 1. Adding a New Material, Element, or Subsystem Feature
When introducing a new model or formulation:
1. Open the relevant `.rst` file under `doc/source/documentation/` (e.g. `materials.rst` or `elements.rst`).
2. Include:
   - **Theoretical formulation & governing equations** (using `:math:` and `.. math::`).
   - **Constitutive / element parameters** table or list with physical units and descriptions.
   - **Input deck example snippet** using the custom `edelweiss` code-block lexer.

```rst
My New Material Model
---------------------

Governing strain energy density function:

.. math::

   W(\boldsymbol{C}) = \frac{\mu}{2} (\bar{I}_1 - 3) + \frac{K}{2} (J - 1)^2

Parameters:
- ``C10``: Initial shear modulus parameter (:math:`\text{MPa}`).
- ``K``: Bulk modulus (:math:`\text{MPa}`).

Example input deck definition:

.. code-block:: edelweiss

   *material, name=myMat, id=myNewMaterial, provider=edelweiss
   150.0, 1000.0
```

### 2. Updating Input Keyword Catalog
1. If introducing or modifying a keyword, update `doc/source/documentation/keywords.rst`.
2. Document all required and optional parameters, data line formats, and subkeywords (`>>subKeyword`).

---

## Docstring Conventions

EdelweissFE follows **NumPy-style docstrings** parsed by Napoleon/Autodoc:

```python
class MyFeature:
    """Brief one-line summary of the class or function.

    Extended multi-line description explaining theoretical assumptions,
    input expectations, and behavior.

    Parameters
    ----------
    stiffness : float
        Penalty stiffness parameter in N/m.
    tolerance : float, optional
        Convergence tolerance for the local Newton loop (default is 1e-6).

    Returns
    -------
    numpy.ndarray
        Computed residual force vector of shape (n_dofs,).
    """
```

### CLI Module Docs
Docstrings registered with `InputSystemRegistry` can be queried interactively by users:
```bash
edelweissfe --doc=<ModuleName>
edelweissfe --keywords
```
Ensure module and handler docstrings are comprehensive and clean.
