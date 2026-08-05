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
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------

import os

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from edelweissfe.utils import performancetiming

#: Set to enable a per-epoch cross-check of :meth:`MultiPointConstraintTransformation.transformSystemMatrix`'s
#: cached-pattern value scatter against the legacy ``T^T @ K @ T + C`` expression it replaces (§21.2 B2).
#: Expensive (recomputes the legacy expression every call it fires on) -- development/CI use only.
_ASSERT_EXACT_ENV_VAR = "EDELWEISS_MPC_ASSERT_EXACT"

"""
Master-slave condensation (Abaqus-style DOF elimination) of linear multi-point constraints

.. math::
    u_s = \\sum_a N_a \\, u_{m_a}

expressed as a full-size square transformation, so the equation system keeps its size ``nDof`` and
every consumer of the DOF vector layout (Dirichlet indices, convergence checks, node-field
write-back, field outputs) remains untouched.

With the transformation matrix :math:`T` (identity on independent DOFs, row :math:`s` carrying the
weights :math:`N_a` in the master columns, slave columns entirely zero) and the constraint-row
matrix :math:`C` (rows :math:`s` only: :math:`C_{ss} = 1`, :math:`C_{s m_a} = -N_a`), the
condensed implicit system reads

.. math::
    (T^T K \\, T + C) \\; \\delta U = T^T R,

where the slave rows of :math:`T^T K T` are structurally zero and :math:`C` re-inserts the
constraint equations, so the solution satisfies
:math:`\\delta U_s = \\sum_a N_a \\, \\delta U_{m_a}` exactly. The row replacement breaks symmetry
in exactly the same benign way the existing Dirichlet row treatment does.

For explicit dynamics with a lumped (diagonal) mass vector, the consistent row-sum lumping of
:math:`T^T M T` stays diagonal: each master receives its own mass plus the :math:`N_a`-weighted
mass of the slaves glued to it (total mass is conserved when :math:`\\sum_a N_a = 1`, which holds
for interpolation-type constraints). Forces are folded as :math:`\\tilde{P} = T^T P`, and the slave
kinematics are assigned directly from the masters, adding zero stiffness and hence leaving the
critical time step untouched.
"""


def _flattenChainedRecords(
    records: list[tuple[int, list[tuple[int, float]]]],
) -> list[tuple[int, list[tuple[int, float]]]]:
    """Resolve a slave DOF's masters that are themselves slave DOFs of another (or the same) record,
    substituting them recursively until every master is an independent DOF.

    Distinct MPC instances are free to compose this way -- e.g. a tie constraint's projected facet
    can legitimately reference a hanging-node MPC's slave node as one of its own interpolation
    nodes. :class:`~edelweissfe.adaptivity.refinement.AdaptiveMesh` already flattens chains *within*
    the hanging-node MPC's own records; this generalizes the same substitution *across* all of a
    model's multi-point constraints, in whatever order they were collected.

    Parameters
    ----------
    records
        The raw per-constraint records, one per slave DOF.

    Returns
    -------
    list of (int, list of (int, float))
        The same slave DOFs, with every master substituted down to independent DOFs and duplicate
        ultimate masters (reached via more than one path) coalesced by summing their coefficients.
    """
    recordOf = dict(records)
    resolved = {}

    def resolve(slaveDof, visiting):
        if slaveDof in resolved:
            return resolved[slaveDof]
        if slaveDof in visiting:
            raise ValueError(
                "Multi-point constraints: circular master/slave dependency detected at DOF {:}.".format(slaveDof)
            )
        visiting = visiting | {slaveDof}
        flat = {}
        for masterDof, coefficient in recordOf[slaveDof]:
            if masterDof in recordOf:
                for mm, cc in resolve(masterDof, visiting).items():
                    flat[mm] = flat.get(mm, 0.0) + coefficient * cc
            else:
                flat[masterDof] = flat.get(masterDof, 0.0) + coefficient
        resolved[slaveDof] = flat
        return flat

    return [(slaveDof, list(resolve(slaveDof, frozenset()).items())) for slaveDof in recordOf]


class MultiPointConstraintTransformation:
    """The assembled master-slave condensation operator for all linear multi-point constraints of
    an equation system.

    Parameters
    ----------
    records
        The linear dependency records, one per slave DOF:
        ``(slaveDofIndex, [(masterDofIndex, coefficient), ...])``, as collected from all
        :class:`~edelweissfe.constraints.base.multipointconstraintbase.MultiPointConstraintBase`
        instances of a model.
    nDof
        The total size of the equation system.
    useCachedCondensation
        Compute ``transformSystemMatrix`` as a cached-pattern value scatter (§21.2 B2) instead of
        the direct ``Tᵀ K T + C`` expression. Correctness-equivalent (validated: full
        edelweiss-only/marmot suites plus a live 280k-dof contact+AMR+tie model, byte-for-byte
        trajectory match, both PARDISO and blockamg) but measured **slower** on that same model
        (§21.2 B3: ~4.6 s/call vs ~2.75 s/call) -- the "S-touching" SpGEMMs it restricts to the
        eliminated DOFs' rows still cost proportional to `K`'s own size, not to the (tiny) number of
        eliminated rows, since SciPy's SpGEMM must scan the full left operand regardless of the
        right operand's sparsity. Default ``False`` (the direct expression) until that asymmetry is
        addressed (e.g. a Cython numeric-only kernel) or a model is found where it actually wins.
    """

    def __init__(
        self,
        records: list[tuple[int, list[tuple[int, float]]]],
        nDof: int,
        useCachedCondensation: bool = False,
    ):
        slaveDofs = [slaveDof for slaveDof, _ in records]

        if len(set(slaveDofs)) != len(slaveDofs):
            raise ValueError("Multi-point constraints: a DOF is claimed as slave by more than one constraint record.")

        for slaveDof, masters in records:
            if not masters:
                raise ValueError("Multi-point constraints: slave DOF {:} has no master DOFs.".format(slaveDof))

        # a master referenced by one constraint may itself be a slave DOF of another (or the same)
        # constraint -- e.g. a tie facet referencing a hanging-node MPC's slave node. Substitute those
        # down to independent DOFs rather than rejecting the composition.
        records = _flattenChainedRecords(records)

        self.nDof = nDof
        self.slaveDofIndices = np.array(sorted(slaveDofs), dtype=int)

        recordOfSlaveDof = {slaveDof: masters for slaveDof, masters in records}

        # W: (nSlaves x nDof) weight matrix, row k carrying the master weights of the k-th
        # (sorted) slave DOF. Serves the drift correction, the slave-kinematics assignment,
        # and the lumped-mass folding.
        wRows, wCols, wVals = [], [], []
        for k, slaveDof in enumerate(self.slaveDofIndices):
            for masterDof, coefficient in recordOfSlaveDof[slaveDof]:
                wRows.append(k)
                wCols.append(masterDof)
                wVals.append(coefficient)
        self._W = csr_matrix((wVals, (wRows, wCols)), shape=(len(self.slaveDofIndices), nDof))

        # T: identity on independent DOFs, slave rows carrying the master weights, slave columns
        # entirely zero.
        independentDofs = np.setdiff1d(np.arange(nDof, dtype=int), self.slaveDofIndices, assume_unique=True)
        tRows = np.concatenate(
            [independentDofs, np.repeat(self.slaveDofIndices, [len(recordOfSlaveDof[s]) for s in self.slaveDofIndices])]
        )
        tCols = np.concatenate([independentDofs, np.array(wCols, dtype=int)])
        tVals = np.concatenate([np.ones(len(independentDofs)), np.array(wVals)])
        self._T = csr_matrix((tVals, (tRows, tCols)), shape=(nDof, nDof))

        # C: the constraint equations themselves, re-inserted into the (structurally zero) slave
        # rows of T^T K T.
        cRows = np.concatenate(
            [
                self.slaveDofIndices,
                np.repeat(self.slaveDofIndices, [len(recordOfSlaveDof[s]) for s in self.slaveDofIndices]),
            ]
        )
        cCols = np.concatenate([self.slaveDofIndices, np.array(wCols, dtype=int)])
        cVals = np.concatenate([np.ones(len(self.slaveDofIndices)), -np.array(wVals)])
        self._C = csr_matrix((cVals, (cRows, cCols)), shape=(nDof, nDof))

        # §21.2 B2: T = D + S, D diagonal (1 on independent DOFs, 0 on slaves), S the slave rows only
        # (T's own entries, minus the independent-DOF identity part -- reuses tRows/tCols/tVals's
        # already-computed slave-row tail rather than rebuilding it). This lets
        # transformSystemMatrix() compute the dominant, identity-heavy term (D^T K D) as a cheap
        # values-only elementwise scale instead of a full nDof x nDof SpGEMM, and restricts the
        # genuine SpGEMMs to the tiny S operand (~nEliminatedDof rows).
        self._D = np.ones(nDof)
        self._D[self.slaveDofIndices] = 0.0
        # slice from an explicit non-negative start, not `-nSlaveEntries:` -- with zero slave DOFs
        # (nSlaveEntries == 0, a legitimate case: a system assembled before any hanging-node/tie
        # constraint exists yet) `arr[-0:]` is `arr[0:]`, the *entire* array, not empty.
        nSlaveEntries = len(wVals)
        sliceStart = len(tVals) - nSlaveEntries
        self._S = csr_matrix((tVals[sliceStart:], (tRows[sliceStart:], tCols[sliceStart:])), shape=(nDof, nDof))
        # a boolean-dtype twin of S, used only for the value-blind structural pass that builds the
        # cached union pattern (§21.2 B2/B3): SciPy's *numeric* SpGEMM eliminates an output entry
        # whenever the several contributions accumulating into it happen to sum to exactly zero --
        # true even when every contributing factor is individually nonzero, and forcing all values
        # to a uniform placeholder (e.g. 1.0) does not avoid this, since the placeholder sum can
        # itself cancel at positions where the real weighted sum would not (or vice versa) -- verified
        # directly with a 2-contribution toy SpGEMM. Boolean dtype sidesteps this entirely: SciPy's
        # sparse matmul uses logical OR/AND for bool operands, so an entry that is structurally
        # reachable through *any* contribution is always True, never cancelled.
        self._SBool = csr_matrix(
            (np.ones(self._S.nnz, dtype=bool), self._S.indices.copy(), self._S.indptr.copy()),
            shape=(nDof, nDof),
            dtype=bool,
        )

        self._useCachedCondensation = useCachedCondensation

        # transformSystemMatrix()'s cache: identity markers for the K pattern the cache was built for
        # (K.indices/K.indptr are the csrGenerator's own persistent, in-place-updated arrays when no
        # further pruning intervenes upstream -- verified directly, not assumed, §21.2 B1), the union
        # output pattern (a value-blind structural superset, safe for the whole epoch -- §21.2 B2/B3),
        # DKD's own scatter position into it (stable, no SpGEMM involved), and C's contribution
        # pre-scattered (constant for this object's entire lifetime, since C never changes after
        # __init__). The three S-touching terms' scatter positions are deliberately not cached here
        # -- see `_buildCacheAndScatter`'s docstring.
        self._cachedKIndices = None
        self._cachedKIndptr = None
        self._cachedKRowOfNnz = None
        self._unionIndices = None
        self._unionIndptr = None
        self._unionKeys = None
        self._term1ScatterMap = None
        self._cBaseData = None

    @property
    def nEliminatedDof(self) -> int:
        """The number of slave DOFs eliminated from the equation system."""

        return len(self.slaveDofIndices)

    def checkDirichletConflicts(self, dirichletDofIndices: np.ndarray):
        """Raise if any Dirichlet-constrained DOF is a slave DOF of a multi-point constraint --
        a slave DOF's motion is fully determined by its masters and cannot be prescribed.

        Parameters
        ----------
        dirichletDofIndices
            The global DOF indices constrained by Dirichlet boundary conditions.
        """

        conflicts = np.intersect1d(dirichletDofIndices, self.slaveDofIndices)
        if len(conflicts):
            raise ValueError(
                "{:} Dirichlet-constrained DOF(s) are slave DOFs of a multi-point constraint. "
                "Prescribe the masters instead.".format(len(conflicts))
            )

    @performancetiming.timeit("mpc transform system matrix")
    def _transformSystemMatrixLegacy(self, K: csr_matrix) -> csr_matrix:
        """The original, un-cached ``T^T @ K @ T + C`` expression -- kept only as the reference for
        :envvar:`EDELWEISS_MPC_ASSERT_EXACT` (§21.2 B2); not on the production hot path."""
        KT = self._T.T @ K
        Kt = KT @ self._T
        Kt = (Kt + self._C).tocsr()
        Kt.sort_indices()
        return Kt

    def _buildCacheAndScatter(self, K, kRowOfNnz, dkdData) -> None:
        """Build (and cache) the union output pattern -- a safe structural superset valid for
        *every* call within this `K`-pattern epoch, not just this one -- plus term1's (DKD's) and
        C's scatter positions into it.

        The three S-touching terms (DKS, S^T KD, S^T K S) are deliberately **not** scattered here,
        even though this method only runs once per epoch: their own output nnz can legitimately
        grow or shrink call to call within the *same* K pattern, because SciPy's CSR @ CSR product
        eliminates an output entry whenever its accumulated contributions happen to sum to exactly
        zero -- true even with every individual contribution nonzero, and this includes an
        independent DOF's raw K-value crossing through exact zero between Newton iterations, which
        is routine, not exotic (verified directly: recomputing a failing AMR case's `term2` with the
        cache-build call's own frozen values reproduced the cache-build nnz exactly; only
        substituting the later call's actual values changed it). A positionally-fixed scatter map
        for those terms is therefore unsound. What *is* epoch-stable is the union pattern itself,
        provided it is built from a value-blind structural pass -- which must use **boolean**-dtype
        operands, not floats forced to a uniform placeholder: a uniform placeholder's own
        contributions can still cancel to exactly zero at positions where the real weighted sum
        would not (verified directly with a toy 2-contribution SpGEMM), so it is not actually a safe
        superset. Boolean dtype sidesteps this: SciPy's sparse matmul uses logical OR/AND for bool
        operands, so a structurally reachable entry is always True, never cancelled -- so callers
        scatter terms 2-4 fresh, every call, via a cheap `searchsorted` into this cached union (see
        `transformSystemMatrix`). term1 (DKD) has no such issue: its positions are exactly `K`'s own
        (no SpGEMM involved), so its scatter map is safe to cache and reuse unconditionally.
        """
        n = self.nDof

        dRowBool = self._D[kRowOfNnz] != 0
        dColBool = self._D[K.indices] != 0
        DKBool = csr_matrix((dRowBool, K.indices, K.indptr), shape=(n, n), dtype=bool)
        KDBool = csr_matrix((dColBool, K.indices, K.indptr), shape=(n, n), dtype=bool)
        KBool = csr_matrix((np.ones(len(K.data), dtype=bool), K.indices, K.indptr), shape=(n, n), dtype=bool)
        term2Bool = (DKBool @ self._SBool).tocsr()
        term3Bool = (self._SBool.T @ KDBool).tocsr()
        term4Bool = (self._SBool.T @ KBool @ self._SBool).tocsr()

        termRows = [kRowOfNnz]
        termCols = [K.indices]
        for term in (term2Bool, term3Bool, term4Bool):
            termRows.append(np.repeat(np.arange(n), np.diff(term.indptr)))
            termCols.append(term.indices)

        cRow = np.repeat(np.arange(n), np.diff(self._C.indptr))

        allRows = np.concatenate(termRows + [cRow])
        allCols = np.concatenate(termCols + [self._C.indices])
        allData = np.zeros(len(allRows))  # positions only -- this pass never contributes values

        # coo_matrix -> tocsr() sums duplicate (row, col) entries and sorts indices canonically in
        # one pass -- exactly the pattern-construction work the legacy path paid every call via
        # "+ C, tocsr, sort indices"; here it is paid once per epoch.
        union = coo_matrix((allData, (allRows, allCols)), shape=(n, n)).tocsr()
        union.sum_duplicates()
        union.sort_indices()

        self._unionIndices = union.indices.copy()
        self._unionIndptr = union.indptr.copy()

        # CSR with sorted-within-row indices, rows visited in increasing order, means row*n + col is
        # monotonically increasing across the *entire* flattened array -- so a single vectorized
        # searchsorted (not a per-row loop) locates any term's entries within the union.
        unionRowOfNnz = np.repeat(np.arange(n), np.diff(union.indptr))
        self._unionKeys = unionRowOfNnz.astype(np.int64) * n + union.indices.astype(np.int64)

        term1Keys = kRowOfNnz.astype(np.int64) * n + K.indices.astype(np.int64)
        self._term1ScatterMap = self._scatterPositions(term1Keys)

        cKeys = cRow.astype(np.int64) * n + self._C.indices.astype(np.int64)
        cPositions = self._scatterPositions(cKeys)
        self._cBaseData = np.zeros(len(self._unionKeys), dtype=np.float64)
        self._cBaseData[cPositions] += self._C.data

    def _scatterPositions(self, keys: np.ndarray) -> np.ndarray:
        """Locate `keys` (``row * nDof + col``) within the cached union pattern via a vectorized
        `searchsorted`, asserting every key is an *exact* match -- by construction (the union is a
        structural superset of any term's actual pattern within the epoch, §21.2 B2/B3), a miss
        means the scatter mechanism itself is broken, not a numerical tolerance issue -- fail
        loudly, always, not just once per epoch."""
        positions = np.searchsorted(self._unionKeys, keys)
        if not np.array_equal(self._unionKeys[positions], keys):
            raise AssertionError(
                "mpctransformation: a term's (row, col) key was not found in the union pattern -- "
                "the cached-pattern value scatter's own bookkeeping is broken, not a numerical "
                "tolerance issue."
            )
        return positions

    def _assertExact(self, K: csr_matrix, result: csr_matrix) -> None:
        """:envvar:`EDELWEISS_MPC_ASSERT_EXACT` cross-check: the cached value-scatter result must
        match the legacy ``T^T K T + C`` expression to within a matrix-norm-relative tolerance
        (entry-relative fails on cancellation-tiny entries) -- §21.2 B2/B3."""
        legacy = self._transformSystemMatrixLegacy(K)
        diff = (result - legacy).tocsr()
        maxDiff = np.max(np.abs(diff.data)) if diff.nnz else 0.0
        scale = max(
            np.max(np.abs(result.data)) if result.nnz else 0.0,
            np.max(np.abs(legacy.data)) if legacy.nnz else 0.0,
            1e-300,
        )
        if maxDiff > 1e-9 * scale:
            raise AssertionError(
                "mpctransformation: EDELWEISS_MPC_ASSERT_EXACT caught a real mismatch between the "
                "cached value-scatter result and the legacy T^T K T + C expression: max|delta|={:.3e}, "
                "tolerance={:.3e} (1e-9 x max|data|={:.3e}).".format(maxDiff, 1e-9 * scale, scale)
            )

    def transformSystemMatrix(self, K: csr_matrix) -> csr_matrix:
        """Condense the system matrix: :math:`\\tilde{K} = T^T K \\, T + C`.

        Dispatches to the cached-pattern value scatter (§21.2 B2) or the direct expression
        depending on ``useCachedCondensation`` (constructor argument) -- see the class docstring
        for why the cached path is not the default despite being correctness-equivalent.
        """
        if self._useCachedCondensation:
            return self._transformSystemMatrixCached(K)
        return self._transformSystemMatrixLegacy(K)

    @performancetiming.timeit("mpc transform system matrix")
    def _transformSystemMatrixCached(self, K: csr_matrix) -> csr_matrix:
        """Condense the system matrix via a cached-pattern value scatter (§21.2 B2), instead of the
        direct two-SpGEMM expression :meth:`_transformSystemMatrixLegacy` computes.
        With :math:`T = D + S` (:math:`D` diagonal, 1 on independent DOFs, 0 on slaves; :math:`S` the
        slave rows only, structurally zero elsewhere):

        .. math::
            T^T K T + C \\;=\\; D K D \\;+\\; D K S \\;+\\; S^T K D \\;+\\; S^T K S \\;+\\; C

        :math:`DKD` is :math:`K`'s own pattern with values masked -- a cheap elementwise operation,
        no SpGEMM. The three :math:`S`-touching terms are SpGEMMs restricted to :math:`S`'s
        ``nEliminatedDof`` *output* rows, but **not** cheap in practice (§21.2 B3, measured on a real
        280k-dof model): SciPy's CSR @ CSR must still scan the *left* operand's full row range (all
        of `K`'s own nnz) to find which rows intersect `S`'s tiny nonzero-row support, so their cost
        tracks `K`'s own size, not `S`'s -- on that model they alone cost more than
        :meth:`_transformSystemMatrixLegacy`'s entire per-call cost, which is why this path is not
        the default (see the class docstring). The expensive union-pattern
        construction is cached, built once per epoch whenever `K`'s own pattern changes (checked by
        array *identity*, not equality -- §21.2 B1 verified the csrGenerator returns the same
        `indices`/`indptr` objects by reference every iteration when no pruning intervenes
        upstream). `DKD`'s own scatter position into that union is likewise cached -- it has no
        SpGEMM involved, so its positions are always exactly `K`'s own. The three S-touching terms'
        positions are *not* cached, even though the union is: SciPy's CSR @ CSR product drops a
        candidate output entry whenever every contributing factor is exactly zero, and an
        independent DOF's raw `K` value crossing through exact zero between Newton iterations is
        routine (not exotic) even while `K`'s *pattern* stays fixed -- so those three terms'
        positions are recomputed fresh every call via a cheap `searchsorted` into the cached union
        (itself built from a value-blind structural pass, so it is a safe superset for the whole
        epoch regardless of any one call's incidental zeros).

        Parameters
        ----------
        K
            The assembled system matrix.

        Returns
        -------
        csr_matrix
            The condensed system matrix, same size, with the constraint equations in the slave rows.
            A fresh, independent array triple every call -- never aliases the cached union pattern,
            since downstream code (Dirichlet zeroing, ``eliminate_zeros()``) mutates its result in
            place, and the cache must survive that untouched.
        """
        n = self.nDof
        patternChanged = (
            self._cachedKIndices is None or K.indices is not self._cachedKIndices or K.indptr is not self._cachedKIndptr
        )

        with performancetiming.timeit("mpc: D K D (values only)"):
            if patternChanged or self._cachedKRowOfNnz is None:
                kRowOfNnz = np.repeat(np.arange(n), np.diff(K.indptr))
            else:
                kRowOfNnz = self._cachedKRowOfNnz
            dkdData = K.data * self._D[kRowOfNnz] * self._D[K.indices]

        with performancetiming.timeit("mpc: S-touching SpGEMMs"):
            dkData = K.data * self._D[kRowOfNnz]
            DK = csr_matrix((dkData, K.indices, K.indptr), shape=(n, n))
            kdData = K.data * self._D[K.indices]
            KD = csr_matrix((kdData, K.indices, K.indptr), shape=(n, n))
            term2 = (DK @ self._S).tocsr()
            term3 = (self._S.T @ KD).tocsr()
            term4 = (self._S.T @ K @ self._S).tocsr()

        if patternChanged:
            with performancetiming.timeit("mpc: build union pattern"):
                self._buildCacheAndScatter(K, kRowOfNnz, dkdData)
                self._cachedKIndices = K.indices
                self._cachedKIndptr = K.indptr
                self._cachedKRowOfNnz = kRowOfNnz

        with performancetiming.timeit("mpc: value scatter"):
            freshData = self._cBaseData.copy()
            freshData[self._term1ScatterMap] += dkdData
            for term in (term2, term3, term4):
                rowOfNnz = np.repeat(np.arange(n), np.diff(term.indptr))
                keys = rowOfNnz.astype(np.int64) * n + term.indices.astype(np.int64)
                positions = self._scatterPositions(keys)
                freshData[positions] += term.data

        result = csr_matrix((freshData, self._unionIndices.copy(), self._unionIndptr.copy()), shape=(n, n))

        if os.environ.get(_ASSERT_EXACT_ENV_VAR):
            self._assertExact(K, result)

        return result

    @performancetiming.timeit("mpc transform residual")
    def transformResidual(self, R: np.ndarray, dU: np.ndarray) -> np.ndarray:
        """Condense the residual: :math:`\\tilde{R} = T^T R`, with the slave rows replaced by the
        current constraint violation :math:`-(dU_s - \\sum_a N_a \\, dU_{m_a})` (exactly zero for
        any consistently accumulated increment; written explicitly as a drift correction).

        Parameters
        ----------
        R
            The assembled residual.
        dU
            The current displacement increment.

        Returns
        -------
        np.ndarray
            The condensed residual.
        """

        Rt = self._T.T @ R
        Rt[self.slaveDofIndices] = -(dU[self.slaveDofIndices] - self._W @ dU)
        return Rt

    def foldLumpedMass(self, M: np.ndarray):
        """Fold the slave masses onto their masters (in place): :math:`M_{m_a} \\mathrel{+}= N_a
        M_s`, then :math:`M_s = 0` -- the row-sum lumping of :math:`T^T M T`, which keeps the mass
        vector diagonal and conserves total mass for interpolation-type constraints
        (:math:`\\sum_a N_a = 1`).

        Parameters
        ----------
        M
            The lumped (diagonal) mass vector, modified in place.
        """

        foldedSlaveMasses = self._W.T @ M[self.slaveDofIndices]
        M[self.slaveDofIndices] = 0.0
        M += foldedSlaveMasses

    def foldExplicitForce(self, P: np.ndarray) -> np.ndarray:
        """Fold the nodal forces acting on slave DOFs onto their masters:
        :math:`\\tilde{P} = T^T P` (slave rows zero) -- the action-reaction transfer through the
        rigid interpolation link.

        Parameters
        ----------
        P
            The assembled force vector.

        Returns
        -------
        np.ndarray
            The folded force vector.
        """

        return self._T.T @ P

    def applySlaveKinematics(self, V: np.ndarray):
        """Assign the slave DOFs their master-interpolated values (in place):
        :math:`V_s = \\sum_a N_a \\, V_{m_a}`. Used on the velocity vector in explicit dynamics;
        displacements then follow automatically from the time integration.

        Parameters
        ----------
        V
            The vector to slave, modified in place.
        """

        V[self.slaveDofIndices] = self._W @ V
