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

## Guidelines for Input Decks (`test.inp`)

1. **Keep Meshes Small & Fast**: Use coarse meshes (e.g. 1–20 elements) with minimal degrees of freedom so the test suite completes quickly.
2. **Determinism**: Avoid unseeded random numbers or unstable configurations that might oscillate near bifurcation points.
3. **Solver & Time Stepping**: Set robust convergence criteria:
   ```
   *step, solver=theSolver
   maxInc=1.0, minInc=1e-8, maxNumInc=100, maxIter=50
   ```
4. **Boundary Conditions & Outputs**: Apply Dirichlet BCs on defined node sets (`>>dirichlet`) and record field output (`>>perNode, field=displacement, result=U`).

## Workflow

### 1. Create the Test Case Directory
```bash
# Choose appropriate folder based on Marmot dependency:
mkdir -p testfiles/edelweiss-only/MyNewTest
```

### 2. Write `test.inp`
Create `testfiles/edelweiss-only/MyNewTest/test.inp`:
```
*material, name=elastic, id=linearelastic, provider=edelweiss
210000.0, 0.3

*section, name=sec1, material=elastic, type=solid
all

*job, name=mytest, domain=3d
*solver, solver=NIST, name=theSolver

*node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
...

*element, type=C3D8, provider=edelweiss
1, 1, 2, 3, 4, 5, 6, 7, 8

*nSet, nSet=fixed
1, 2, 3, 4

*nSet, nSet=loaded
5, 6, 7, 8

*fieldOutput
>>perNode, elSet=all, field=displacement, result=U, name=displacement

*step, solver=theSolver
maxInc=1.0, minInc=1e-5, maxNumInc=10, maxIter=25
>>dirichlet, name=fix, nSet=fixed, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=load, nSet=loaded, field=displacement, 1=0.05
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
