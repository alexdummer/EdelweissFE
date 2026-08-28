---
name: ew-create-regression-test
description: >-
  Step-by-step instructions for constructing, validating, and registering regression test decks (test.inp + U.ref) in EdelweissFE.
  Use when adding new test cases for bug fixes, new features, or solver benchmarking.
---

# Creating Regression Tests in EdelweissFE

EdelweissFE tests are full finite-element input-deck regression tests executed via `run_tests_edelweissfe`.

## Test Directory Structure

```text
testfiles/
├── edelweiss-only/     # Tests runnable with pure Python/Cython (standalone)
│   └── <TestName>/
│       ├── test.inp    # The simulation input deck
│       └── U.ref       # Reference DOF displacement vector
└── marmot/             # Tests requiring native Marmot C++ libraries
    └── <TestName>/
        ├── test.inp
        └── U.ref
```

## Available 2D / 3D Mesh Generators (`*modelGenerator`)

Always use built-in procedural generators (`*modelGenerator`) to generate fast, structured test meshes without manual `*node` or `*element` tables:

| Dimensionality | Generator | Supported Elements | Pre-Generated Node Sets |
| :--- | :--- | :--- | :--- |
| **2D (Plane / Axisymmetric)** | `planeRectQuad` | `CPS4`, `CPS8R`, `CPE4`, `CPE8`, `CAX4`, etc. | `bottom`, `top`, `left`, `right`, `all` |
| **3D (Solid / Continuum)** | `boxGen` | `C3D8`, `C3D20`, `C3D20R`, etc. | `<name>_bottom`, `<name>_top`, `<name>_left`, `<name>_right`, `<name>_front`, `<name>_back`, `<name>_all` |

---

## Guidelines for Input Decks (`test.inp`)

1. **Keep Meshes Coarse & Fast**: Use 1 to 4 elements (`nX=1..2, nY=1..2, nZ=1..2`) so test decks execute in seconds.
2. **Determinism**: Avoid unseeded stochastic parameters or unstable load paths.
3. **Boundary Conditions & Outputs**: Apply Dirichlet BCs on pre-generated generator sets (`>>dirichlet`) and request displacement field output (`>>perNode, field=displacement, result=U`).

---

## Workflow

### 1. Create the Test Case Directory
```bash
# Choose folder based on dependency:
mkdir -p testfiles/edelweiss-only/<TestName>
```

### 2. Write `test.inp`

#### Template A: 2D Simulation (`planeRectQuad`)
```
*material, name=mat1, id=myMaterial, provider=edelweiss
30000.0, 0.2, ...

*section, name=sec1, thickness=1.0, material=mat1, type=plane
all

*job, name=test2d, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, elSet=all, field=displacement, result=U, name=displacement

*modelGenerator, generator=planeRectQuad, name=gen
x0=0, l=10.0
y0=0, h=10.0
elType=CPS4
elProvider=edelweiss
nX=2
nY=2

*step, solver=theSolver
maxInc=0.5, minInc=1e-5, maxNumInc=10, maxIter=25
>>dirichlet, name=fix_y, nSet=bottom, field=displacement, 2=0.0
>>dirichlet, name=fix_x, nSet=bottom, field=displacement, 1=0.0
>>dirichlet, name=pull,  nSet=top,    field=displacement, 2=0.05
```

#### Template B: 3D Simulation (`boxGen`)
```
*material, name=mat1, id=myMaterial, provider=edelweiss
210000.0, 0.3, ...

*section, name=sec1, material=mat1, type=solid
all

*job, name=test3d, domain=3d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, elSet=all, field=displacement, result=U, name=displacement

*modelGenerator, generator=boxGen, name=gen
x0=0, lX=10.0
y0=0, lY=10.0
z0=0, lZ=10.0
elType=C3D8
elProvider=edelweiss
nX=2
nY=2
nZ=2

*step, solver=theSolver
maxInc=0.5, minInc=1e-5, maxNumInc=10, maxIter=25
>>dirichlet, name=fix_z, nSet=gen_bottom, field=displacement, 3=0.0
>>dirichlet, name=fix_y, nSet=gen_bottom, field=displacement, 2=0.0
>>dirichlet, name=fix_x, nSet=gen_bottom, field=displacement, 1=0.0
>>dirichlet, name=pull,  nSet=gen_top,    field=displacement, 3=0.05
```

### 3. Generate Reference Solution (`U.ref`)
Run the test runner with `--create`:
```bash
run_tests_edelweissfe ./testfiles/edelweiss-only/ --tests MyNewTest --create
```
This runs the simulation and dumps the final displacement state vector to `U.ref`.

### 4. Verify Pass
Run without `--create` to ensure the residual matches within the `1e-6` tolerance:
```bash
run_tests_edelweissfe ./testfiles/edelweiss-only/ --tests MyNewTest
```

### 5. Run Full Suite
Check that the addition does not break existing test cases:
```bash
run_tests_edelweissfe ./testfiles/edelweiss-only/
```
