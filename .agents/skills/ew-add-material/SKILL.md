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
  - Always inherit from established base classes (`BaseMaterial`, `BaseHypoElasticMaterial`, `BaseHyperElasticMaterial`).
  - Reuse tensor utilities rather than writing custom matrix/Voigt math:
    ```python
    # Common imports to reuse directly:
    from edelweissfe.materials.base.basehypoelasticmaterial import (
        BaseHypoElasticMaterial,
    )
    from edelweissfe.materials.base.basematerial import BaseMaterial
    from edelweissfe.utils.exceptions import CutbackRequest
    from edelweissfe.utils.voigtnotation import (
        undoVoigtStrain,
        undoVoigtStress,
        voigtStrain,
        voigtStress,
    )
```
- **Extract Shared Constitutive Logic**:
  - If multiple materials share components (isotropic baseline, yield surfaces, flow rules, return mapping, damage degradation), extract them into shared helper functions or common base classes.
- **Parametrize Rather Than Duplicate**:
  - Prefer parametrizing existing materials over creating copy-pasted variants.
- **Performance & Invariant Caching**:
  - Cache invariant elasticity matrices ($\mathbb{C}$) during initialization to avoid per-iteration allocations.

## Implementation Steps

### 1. Create the Material Module
Create a new package under `edelweissfe/materials/<material_name>/<material_name>.py`:
```python
import numpy as np

from edelweissfe.materials.base.basehypoelasticmaterial import (
    BaseHypoElasticMaterial,
)
from edelweissfe.utils.exceptions import CutbackRequest
from edelweissfe.utils.voigtnotation import undoVoigtStrain


class MyMaterial(BaseHypoElasticMaterial):
    def __init__(self, E, nu, *args, **kwargs):
        super().__init__()
        self.E = float(E)
        self.nu = float(nu)
        # Precompute & cache invariant elasticity tensor
        self._computeElasticMatrix()

    def getInitialInternalStateVariables(self, totalNumberOfIntegrationPoints):
        # Return initial state array (e.g. zeros of shape (num_points, num_state_vars))
        return np.zeros((totalNumberOfIntegrationPoints, self.numStateVars))

    def evaluate(self, strain, stateVarsOld, **kwargs):
        # Return: stress (Voigt), tangent_C (6x6), stateVarsNew
        ...
```

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
