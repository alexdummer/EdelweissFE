#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""§21.2 B2/B3 correctness tests for the cached-pattern value scatter
(``MultiPointConstraintTransformation(..., useCachedCondensation=True)``) against the direct
``T^T K T + C`` expression it is an opt-in alternative to (default ``False`` -- §21.2 B3 found it
measured *slower*, not faster, on the real reference model, so it ships as an experiment, not the
default).

Two real bugs were caught this way and are pinned down as regressions here:

- ``tVals[-nSlaveEntries:]`` with zero slave DOFs is ``tVals[-0:]`` == ``tVals[0:]``, the *entire*
  array, not empty (test 6).
- SciPy's CSR @ CSR product eliminates an output entry whenever its accumulated contributions sum
  to exactly zero, even with every individual contribution nonzero -- both when an independent
  DOF's raw K-value happens to be exactly 0 on one Newton iteration and nonzero on the next with
  the same K pattern (test 7), and when two distinct contributions to the same union position
  cancel under a naive uniform-placeholder value-blind pass, which a real MPC weighted sum would
  not (test 8) -- fixed by using boolean-dtype operands for that pass, since SciPy's sparse matmul
  uses logical OR/AND for bool, never arithmetic cancellation.
"""

import os

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from edelweissfe.numerics.mpctransformation import MultiPointConstraintTransformation


def _randomSparseK(n, density=0.02, seed=0):
    r = np.random.default_rng(seed)
    A = sp.random(n, n, density=density, format="csr", random_state=r, data_rvs=lambda k: r.standard_normal(k))
    A = (A + A.T) / 2.0 + sp.eye(n) * n  # diagonally dominant-ish, symmetric
    return A.tocsr()


def _maxRelDiff(A, B):
    diff = (A - B).tocsr()
    maxDiff = np.max(np.abs(diff.data)) if diff.nnz else 0.0
    scale = max(np.max(np.abs(A.data)) if A.nnz else 0.0, np.max(np.abs(B.data)) if B.nnz else 0.0, 1e-300)
    return maxDiff / scale


def test_cache_build_and_reuse_match_legacy():
    n = 200
    # slaves 5,6,7 depend on masters 0,1,2 (weights); slave 8 depends on slave 6 (chained) and master 3.
    records = [
        (5, [(0, 0.5), (1, 0.5)]),
        (6, [(1, 0.3), (2, 0.7)]),
        (7, [(0, 1.0)]),
        (8, [(6, 0.4), (3, 0.6)]),  # chained: master 6 is itself a slave
    ]
    mpc = MultiPointConstraintTransformation(records, n, useCachedCondensation=True)

    K1 = _randomSparseK(n, seed=1)
    result1a = mpc.transformSystemMatrix(K1)
    legacy1a = mpc._transformSystemMatrixLegacy(K1)
    assert _maxRelDiff(result1a, legacy1a) < 1e-9

    # mutate K1's VALUES in place (same object, same pattern) -- simulates a new Newton iteration's
    # fresh values on the same csrGenerator buffer.
    K1.data[:] = K1.data * 1.7 + 0.01
    result1b = mpc.transformSystemMatrix(K1)
    legacy1b = mpc._transformSystemMatrixLegacy(K1)
    assert _maxRelDiff(result1b, legacy1b) < 1e-9
    assert mpc._cachedKIndices is K1.indices, "cache should have been reused, not rebuilt"

    # a genuinely different K (new pattern) triggers a cache rebuild
    K2 = _randomSparseK(n, density=0.03, seed=2)
    result2 = mpc.transformSystemMatrix(K2)
    legacy2 = mpc._transformSystemMatrixLegacy(K2)
    assert _maxRelDiff(result2, legacy2) < 1e-9
    assert mpc._cachedKIndices is K2.indices, "cache should now point at K2"

    # back to K1 (still the same object/pattern as before) -- rebuilds again (different identity)
    result3 = mpc.transformSystemMatrix(K1)
    legacy3 = mpc._transformSystemMatrixLegacy(K1)
    assert _maxRelDiff(result3, legacy3) < 1e-9


def test_assert_exact_env_var_catches_a_real_corruption():
    n = 200
    records = [(5, [(0, 0.5), (1, 0.5)])]
    mpc = MultiPointConstraintTransformation(records, n, useCachedCondensation=True)
    K = _randomSparseK(n, seed=3)

    os.environ["EDELWEISS_MPC_ASSERT_EXACT"] = "1"
    try:
        mpc.transformSystemMatrix(K)  # correct implementation -- must not raise

        mpc._cBaseData = mpc._cBaseData + 12345.0  # deliberately corrupt the cache
        with pytest.raises(AssertionError):
            mpc.transformSystemMatrix(K)
    finally:
        del os.environ["EDELWEISS_MPC_ASSERT_EXACT"]


def test_chained_records_are_flattened_to_independent_dofs():
    n = 200
    records = [
        (5, [(0, 0.5), (1, 0.5)]),
        (6, [(1, 0.3), (2, 0.7)]),
        (8, [(6, 0.4), (3, 0.6)]),  # chained: master 6 is itself a slave
    ]
    mpc = MultiPointConstraintTransformation(records, n, useCachedCondensation=True)

    idx8 = list(mpc.slaveDofIndices).index(8)
    wRow8 = mpc._W[idx8, :].toarray().flatten()
    nonzeroCols = np.nonzero(wRow8)[0]
    assert 6 not in nonzeroCols, "chained slave 6 should have been substituted, not left as a column"
    assert set(nonzeroCols) <= {1, 2, 3}, "unexpected columns after flattening"


def test_zero_slave_dofs_regression():
    """`tVals[-nSlaveEntries:]` with `nSlaveEntries == 0` is `tVals[-0:]` == `tVals[0:]`, the entire
    array, not empty -- a system assembled before any hanging-node/tie constraint exists yet must
    still condense to K unchanged (T = identity, C = 0)."""
    n = 200
    mpc = MultiPointConstraintTransformation([], n, useCachedCondensation=True)
    K = _randomSparseK(n, seed=6)

    result = mpc.transformSystemMatrix(K)
    assert _maxRelDiff(result, mpc._transformSystemMatrixLegacy(K)) < 1e-9
    assert _maxRelDiff(result, K) < 1e-9

    K.data[:] = K.data * 2.3  # cache-reuse path must also hold
    resultReused = mpc.transformSystemMatrix(K)
    assert _maxRelDiff(resultReused, K) < 1e-9


def test_value_crossing_exact_zero_regression():
    """Reproduces the real AMR failure: SciPy's CSR @ CSR product drops an output position whenever
    every contributing factor is exactly zero. An independent DOF's raw K-value can be exactly 0.0
    on one Newton iteration and nonzero on the next, with K's *pattern* unchanged -- this broke a
    fixed-size scatter-map design that assumed the S-touching terms' own nnz was epoch-stable."""
    records = [(2, [(0, 0.5), (1, 0.5)])]
    mpc = MultiPointConstraintTransformation(records, 4, useCachedCondensation=True)
    indices = np.array([0, 2, 1, 2, 0, 1, 2], dtype=np.int32)  # rows: 0->{0,2}, 1->{1,2}, 2->{0,1,2}, 3->{}
    indptr = np.array([0, 2, 4, 7, 7], dtype=np.int32)

    K = csr_matrix((np.array([5.0, 0.0, 6.0, 0.0, 1.0, 1.0, 9.0]), indices, indptr), shape=(4, 4))
    result1 = mpc.transformSystemMatrix(K)
    assert _maxRelDiff(result1, mpc._transformSystemMatrixLegacy(K)) < 1e-9

    # mutate K's OWN .data in place (same object identity, mirroring the real csrGenerator) -- the
    # two entries that were exactly 0.0 above are now nonzero.
    K.data[:] = np.array([5.0, 3.0, 6.0, 4.0, 1.0, 1.0, 9.0])
    result2 = mpc.transformSystemMatrix(K)
    assert _maxRelDiff(result2, mpc._transformSystemMatrixLegacy(K)) < 1e-9


def test_real_value_cancellation_regression():
    """Two slaves map to the same master with opposite-signed weights. The real K-weighted sum at
    that union position is nonzero, but a value-blind pass using a *uniform* placeholder (e.g. 1.0
    for every entry -- the first attempt at building a safe superset pattern) sums to exactly
    0.5 + (-0.5) == 0 there and gets silently dropped by SciPy's SpGEMM -- reproduces the real
    failure found on the actual pryout model (1290 missing union keys, all in one S-touching term).
    Fixed by using boolean-dtype operands for the value-blind pass instead."""
    records = [(2, [(0, 0.5), (1, 0.5)]), (3, [(0, -0.5), (1, 1.5)])]
    mpc = MultiPointConstraintTransformation(records, 4, useCachedCondensation=True)
    indices = np.array([2, 3, 2, 3], dtype=np.int32)  # row0: cols {2,3}; row1: cols {2,3}
    indptr = np.array([0, 2, 4, 4, 4], dtype=np.int32)
    K = csr_matrix((np.array([1.0, 1.0, 2.0, 3.0]), indices, indptr), shape=(4, 4))

    result = mpc.transformSystemMatrix(K)
    assert _maxRelDiff(result, mpc._transformSystemMatrixLegacy(K)) < 1e-9


def test_default_uses_legacy_expression():
    mpc = MultiPointConstraintTransformation([], 10)
    assert mpc._useCachedCondensation is False
