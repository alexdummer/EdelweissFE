---
name: ew-documentation
description: >-
  Procedure for writing, updating, structuring, and building documentation in EdelweissFE.
  Use when adding or modifying Sphinx docs (.rst files), input keyword references, docstrings, or theory manuals.
---

# Sphinx Documentation Guide

All docs live under `doc/source/`. Subsystem docs live in `doc/source/documentation/*.rst`.

## 1. Local Build Command
```bash
sphinx-build ./doc/source/ ./docs -b html
```
CI fails on Sphinx warnings or broken references.

## 2. Subsystem Documentation (`doc/source/documentation/`)
- `materials.rst`, `elements.rst`, `solvers.rst`, `linsolvers.rst`, `constraints.rst`, `contacttheory.rst`, `keywords.rst`.

### Template for Models / Elements
```rst
My Feature Name
---------------

Governing equations:

.. math::
   \boldsymbol{\sigma} = (1 - d) \mathbb{C}^0 : \boldsymbol{\varepsilon}

Parameters:
- ``E``: Young's modulus (:math:`\text{MPa}`).
- ``nu``: Poisson's ratio.

Input deck snippet:

.. code-block:: edelweiss

   *material, name=myMat, id=myMaterial, provider=edelweiss
   210000.0, 0.3
```

## 3. Docstrings & Keyword Catalog
- **Style**: NumPy-style docstrings (`Parameters`, `Returns`).
- **Keywords**: Add new keywords to `doc/source/documentation/keywords.rst`.
- **CLI query**: `edelweissfe --doc=<Module>` and `edelweissfe --keywords`.
