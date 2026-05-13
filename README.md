[![documentation](https://github.com/EdelweissFE/EdelweissFE/actions/workflows/sphinx.yml/badge.svg)](https://edelweiss-numerics.github.io/EdelweissFE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/1095513352.svg)](https://doi.org/10.5281/zenodo.17603044)

# EdelweissFE: A light-weight, platform-independent, parallel finite element framework.

<p align="center">
  <img width="512" height="512" src="./doc/source/borehole_damage_lowdilation.gif">
</p>

See the [documentation](https://edelweiss-numerics.github.io/EdelweissFE).

EdelweissFE aims at an easy to understand, yet efficient implementation of the finite element method.
Some features are:

 * Python for non performance-critical routines
 * Cython for performance-critical routines
 * Parallelization
 * Modular system, which is easy to extend
  * Output to Paraview, Ensight, CSV, matplotlib
  * Interfaces to powerful direct and iterative linear solvers

EdelweissFE makes use of the [Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot/) library for finite element and constitutive model formulations.

## Installation

The project workflows currently validate two installation paths from the repository root and an active conda environment.

### Working installation without Marmot

This setup builds EdelweissFE with the non-Marmot functionality covered by the `run_tests_without_marmot.yml` workflow:

```console
mamba install --file conda_requirements.txt
pip install -r pip_requirements.txt
pip install .
run_tests_edelweissfe ./testfiles/edelweiss-only/
```

### Working installation with Marmot

This setup follows the `run_tests_with_marmot.yml` workflow and enables the Marmot-backed functionality:

```console
mamba install --file conda_requirements.txt
pip install -r pip_requirements.txt

cd ..
git clone --branch 3.4.0 https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir build
cd build
cmake -DBUILD_TESTING=OFF -DINCLUDE_INSTALL_DIR=$CONDA_PREFIX/include -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../..

git clone --branch v1.1.0 https://github.com/autodiff/autodiff.git
cd autodiff
mkdir build
cd build
cmake -DAUTODIFF_BUILD_TESTS=OFF -DAUTODIFF_BUILD_PYTHON=OFF -DAUTODIFF_BUILD_EXAMPLES=OFF -DAUTODIFF_BUILD_DOCS=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../..

git clone https://github.com/romeric/Fastor.git
cd Fastor
mkdir build
cd build
cmake -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../..

git clone --branch 1.4.7 --depth 1 https://github.com/ddemidov/amgcl.git
cd amgcl
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../..

git clone --recurse-submodules https://github.com/MAteRialMOdelingToolbox/Marmot/
cd Marmot
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../../EdelweissFE

pip install -v .
run_tests_edelweissfe ./testfiles/marmot/
run_tests_edelweissfe ./testfiles/edelweiss-only/
```

When you need Marmot and EdelweissFE branches to match, use the same Marmot branch-selection logic as the workflow before configuring Marmot:

```console
TARGET_BRANCH=<target EdelweissFE branch>
if git show-ref --verify --quiet refs/remotes/origin/$TARGET_BRANCH; then
    git checkout $TARGET_BRANCH
fi
```

The full installation recipe in the documentation mirrors these workflow steps in more detail.
