Installation
============

The currently maintained installation recipes are the ones exercised by the project workflows.
They assume that you are in the EdelweissFE repository root and that a conda environment is active,
so that ``$CONDA_PREFIX`` points to the installation prefix used by the build steps.

Common base setup
*****************

Both supported installation paths start with the same package installation steps:

.. code-block:: console

    mamba install --file conda_requirements.txt
    pip install -r pip_requirements.txt

Working installation without Marmot
**********************************

The workflow ``run_tests_without_marmot.yml`` builds a working EdelweissFE installation without Marmot support as follows:

.. code-block:: console

    mamba install --file conda_requirements.txt
    pip install -r pip_requirements.txt
    pip install .

Validate that installation with the same command used in CI:

.. code-block:: console

    run_tests_edelweissfe ./testfiles/edelweiss-only/

This installation path is sufficient for the EdelweissFE-only examples and tests. Marmot-backed elements and material models
require the additional dependencies described below.

Working installation with Marmot
********************************

The workflow ``run_tests_with_marmot.yml`` extends the base setup with the external libraries needed for Marmot-enabled builds.

Install Eigen:

.. code-block:: console

    cd ..
    git clone --branch 3.4.0 https://gitlab.com/libeigen/eigen.git
    cd eigen
    mkdir build
    cd build
    cmake -DBUILD_TESTING=OFF -DINCLUDE_INSTALL_DIR=$CONDA_PREFIX/include -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
    make install
    cd ../..

Install autodiff:

.. code-block:: console

    git clone --branch v1.1.0 https://github.com/autodiff/autodiff.git
    cd autodiff
    mkdir build
    cd build
    cmake -DAUTODIFF_BUILD_TESTS=OFF \
      -DAUTODIFF_BUILD_PYTHON=OFF \
      -DAUTODIFF_BUILD_EXAMPLES=OFF \
      -DAUTODIFF_BUILD_DOCS=OFF \
      -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
      ..
    make install
    cd ../..

Install Fastor:

.. code-block:: console

    git clone https://github.com/romeric/Fastor.git
    cd Fastor
    mkdir build
    cd build
    cmake -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
    make install
    cd ../..

Install AMGCL:

.. code-block:: console

    git clone --branch 1.4.7 --depth 1 https://github.com/ddemidov/amgcl.git
    cd amgcl
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
    make install
    cd ../..

Install Marmot:

.. code-block:: console

    git clone --recurse-submodules https://github.com/MAteRialMOdelingToolbox/Marmot/
    cd Marmot

Select the Marmot branch with the same logic used in the workflow:

.. code-block:: console

    TARGET_BRANCH=<your EdelweissFE branch, for example master>
    echo "Pull Request Target/Current Branch is: $TARGET_BRANCH"
    if git show-ref --verify --quiet refs/remotes/origin/$TARGET_BRANCH; then
        echo "Matching branch found. Checking out $TARGET_BRANCH..."
        git checkout $TARGET_BRANCH
    else
        echo "Branch $TARGET_BRANCH not found in Marmot, staying on default branch."
    fi

Then build and install Marmot:

.. code-block:: console

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
    make install
    cd ../../EdelweissFE

Build EdelweissFE with Marmot available:

.. code-block:: console

    pip install -v .

Validate that installation with the same CI commands:

.. code-block:: console

    run_tests_edelweissfe ./testfiles/marmot/
    run_tests_edelweissfe ./testfiles/edelweiss-only/

Alternative local build
***********************

If you want to compile the extensions in place instead of installing the package with pip, you can still use:

.. code-block:: console

    python setup.py build_ext -i

Force a recompilation with:

.. code-block:: console

    python setup.py build_ext -i --force

Build the documentation
***********************

The documentation workflow builds the HTML output with:

.. code-block:: console

    sphinx-build ./doc/source/ ./docs -b html
