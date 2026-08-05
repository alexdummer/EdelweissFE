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
"""p-multigrid (Galerkin P1 corner-node) preconditioner for one field's diagonal block inside
:class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`'s per-field sweep (§22).

Precondition the quadratic serendipity operator through a low-order P1 operator: :math:`\\nu`
Chebyshev sweeps on the field's own (equilibrated) block, restrict the residual through :math:`P^T`,
one AMGCL V-cycle on the Galerkin-projected :math:`A_1 = P^T A P`, prolong through :math:`P`,
:math:`\\nu` more Chebyshev sweeps. :func:`build` and :meth:`PTwoGridPreconditioner.applyPreconditioner`
match :class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLSolver`'s ``build``/``applyPreconditioner``
call shape closely enough to slot into the same per-field sweep, so
:class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver` needs no change beyond choosing
which class to build (see ``blockamg.py``'s ``p1Maps`` option).

**Dirichlet handling (§22.2-bis/§17 B1, load-bearing, do not simplify away).** A field's diagonal
block still carries its Dirichlet identity rows (production applies Dirichlet elimination upstream
of blockamg). Both the fine smoothing and the Galerkin projection must operate on the *free*
submatrix with Dirichlet rows/columns removed entirely, not merely masked in place -- §22.2 first
tried masking-in-place for the fine smoother alone and it diverged outright (17-80x residual
growth), reinstating a "genuine non-symmetry" explanation §22.2-bis then retracted: the asymmetry
is a Dirichlet/`eliminate_zeros` storage artifact (~50% raw, ~0.6% once removed), not physics. A
free midside node whose edge-endpoint corner is Dirichlet-constrained keeps only the surviving
½-weight on its free endpoint -- no renormalization (the constrained corner contributes exactly
zero to a homogeneous Newton correction, which the dropped weight already encodes).
"""

import numpy as np
import scipy.sparse as sp

#: R1/R2's winning coarse-level AMGCL configuration (§22.2-bis): 26 (R1, on A1 alone) / 58 (R2,
#: full two-grid on the free submatrix) iterations, both comfortably clearing their gates.
_DEFAULT_COARSE_PRECOND = {
    "coarsening": {"type": "smoothed_aggregation", "aggr": {"eps_strong": 0.01}},
    "relax": {"type": "chebyshev", "degree": 8, "power_iters": 50, "lower": 0.01},
    "npre": 2,
    "npost": 2,
}
#: R2's winning fine-smoother configuration: nu=1 sweeps, Chebyshev degree 5.
_DEFAULT_NU = 1
_DEFAULT_FINE_DEGREE = 5


def _buildChebyshevSmoother(A, degree, powerIters=50, lower=0.01, higher=1.1):
    """The fine smoother, backed by AMGCL's own OpenMP-threaded ``runtime::relaxation::wrapper``
    (§22.4) instead of a serial scipy/numpy polynomial -- §22.3 measured the latter at 81%+ of the
    preconditioner's own apply time. The spectral radius (power iteration, §17 B5: a short/default
    estimate is what made Chebyshev diverge before) is now computed inside AMGCL's own chebyshev
    constructor via the identical algorithm, so it no longer needs a separate Python-side pass.
    ``higher`` defaults to 1.1, matching the old hand-rolled smoother's ``upper=1.1`` safety margin
    above the estimated spectral radius -- AMGCL's own chebyshev defaults ``higher`` to 1.0, which
    would silently retune the fine smoother relative to what §22.3 validated."""
    from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLRelaxationSmoother

    smoother = PyAMGCLRelaxationSmoother(
        {"type": "chebyshev", "degree": degree, "power_iters": powerIters, "lower": lower, "higher": higher}
    )
    smoother.build(A)

    def smooth(x, rhs, sweeps):
        for _ in range(sweeps):
            smoother.applyStep(x, rhs)
        return x

    return smooth


def buildNodeLevelP(isCorner: np.ndarray, edgeEndpoints: np.ndarray) -> sp.csr_matrix:
    """The node-level P1 restriction operator: identity on corners, ½/½ on each exclusive midside
    from its two edge-endpoint corners (§22.1's ``buildP1Map`` output, in its own node order)."""
    nNodes = len(isCorner)
    cornerNodeRows = np.nonzero(isCorner)[0]
    nCorners = len(cornerNodeRows)
    cornerLocalIdx = -np.ones(nNodes, dtype=int)
    cornerLocalIdx[cornerNodeRows] = np.arange(nCorners)

    rows, cols, vals = [], [], []
    for node in range(nNodes):
        if isCorner[node]:
            rows.append(node)
            cols.append(cornerLocalIdx[node])
            vals.append(1.0)
        else:
            a, b = edgeEndpoints[node]
            rows += [node, node]
            cols += [cornerLocalIdx[a], cornerLocalIdx[b]]
            vals += [0.5, 0.5]
    return sp.csr_matrix((vals, (rows, cols)), shape=(nNodes, nCorners))


class PTwoGridPreconditioner:
    """A p-two-grid preconditioner for one vector field's diagonal block, built once per solve
    (mirroring :class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLSolver`'s ``build`` /
    ``applyPreconditioner`` life cycle) and applied many times within the outer GMRES.

    Parameters
    ----------
    isCorner, edgeEndpoints
        This field's P1 topology map (§22.1, :func:`edelweissfe.numerics.p1topology.buildP1Map`),
        in the field's own node order.
    nu
        Fine Chebyshev sweeps before *and* after the coarse-grid correction.
    fineDegree
        Fine Chebyshev polynomial degree.
    coarsePrecond
        AMGCL parameter tree for the coarse-level (:math:`A_1`) solve.
    useCoarseNullspace
        Whether to give the coarse AMGCL solver the rigid-body translations on the free-corner
        space as its near null-space (§22.2-bis R1: measurably helps here, unlike on the full
        quadratic block).
    """

    def __init__(
        self,
        isCorner: np.ndarray,
        edgeEndpoints: np.ndarray,
        nu: int = _DEFAULT_NU,
        fineDegree: int = _DEFAULT_FINE_DEGREE,
        coarsePrecond: dict = None,
        useCoarseNullspace: bool = True,
    ):
        self._isCorner = isCorner
        self._P_node = buildNodeLevelP(isCorner, edgeEndpoints)
        self._nu = nu
        self._fineDegree = fineDegree
        self._coarsePrecond = dict(coarsePrecond) if coarsePrecond is not None else dict(_DEFAULT_COARSE_PRECOND)
        self._useCoarseNullspace = useCoarseNullspace
        # cumulative wall time, fine (serial scipy Chebyshev) vs. coarse (AMGCL, OpenMP-threaded) --
        # §22.3's "production fine-smoother decision" needs this split measured, not assumed.
        self.fineSeconds = 0.0
        self.coarseSeconds = 0.0
        self.applyCalls = 0

    def build(self, A: sp.csr_matrix, dinv: np.ndarray) -> None:
        """Build the free submatrix, the restricted ``P``, the Galerkin coarse operator, its AMGCL
        hierarchy, and the fine Chebyshev smoother -- everything :meth:`applyPreconditioner` needs.

        Parameters
        ----------
        A
            This field's equilibrated diagonal block (``As[slice, slice]``), still carrying its
            Dirichlet identity rows.
        dinv
            This field's own slice of the global equilibration vector (``dinv[block.start:block.stop]``),
            needed to scale the coarse near-null-space consistently with ``A``'s own scaling
            (:meth:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver._translationNullspace`'s
            convention).
        """
        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver

        A = A.tocsr()
        n = A.shape[0]
        nNodes = len(self._isCorner)
        if n % nNodes != 0:
            raise ValueError(
                "ptwogrid: block size {:} is not a multiple of the topology map's node count {:} -- "
                "the map does not match this field.".format(n, nNodes)
            )
        nDim = n // nNodes
        P_dof = sp.kron(self._P_node, sp.identity(nDim), format="csr")

        dirichletMaskBool = np.diff(A.indptr) == 1
        self._dirichletRows = np.nonzero(dirichletMaskBool)[0]
        self._freeRows = np.nonzero(~dirichletMaskBool)[0]

        A_free = A[self._freeRows, :][:, self._freeRows].tocsr()

        cornerNodeRows = np.nonzero(self._isCorner)[0]
        nCorners = len(cornerNodeRows)
        fullDofRowsForCorners = np.repeat(cornerNodeRows, nDim) * nDim + np.tile(np.arange(nDim), nCorners)
        coarseColIsFree = ~dirichletMaskBool[fullDofRowsForCorners]
        freeCoarseCols = np.nonzero(coarseColIsFree)[0]

        # restrict P to free fine rows x free coarse columns -- slicing alone implements the "no
        # renormalization" interpolation rule (a free midside's 1/2 weight to a Dirichlet corner is
        # simply dropped, not redistributed).
        P_free = P_dof[self._freeRows, :][:, freeCoarseCols].tocsr()
        rowSums = np.asarray(np.abs(P_free).sum(axis=1)).flatten()
        orphanRows = np.nonzero(rowSums == 0)[0]
        if len(orphanRows):
            raise AssertionError(
                "ptwogrid: {:} free midside row(s) of the restricted P are entirely zero -- both "
                "edge-endpoint corners are Dirichlet-constrained, an orphan the topology map "
                "disagrees with the Dirichlet data on. First bad row: {:}.".format(len(orphanRows), orphanRows[0])
            )

        self._As_free = A_free
        self._P_free = P_free

        A1_free = (P_free.T @ A_free @ P_free).tocsr()

        coarseSolver = PyAMGCLSolver({"precond": self._coarsePrecond, "backendBlockSize": 1})
        if self._useCoarseNullspace:
            freeCornerFullDofRows = fullDofRowsForCorners[freeCoarseCols]
            nFreeCoarse = len(freeCoarseCols)
            B = np.zeros((nFreeCoarse, nDim))
            localRows = np.arange(nFreeCoarse)
            B[localRows, freeCoarseCols % nDim] = 1.0
            B = B / dinv[freeCornerFullDofRows][:, None]
            coarseSolver.set_nullspace(B)
        coarseSolver.build(A1_free)
        self._coarseSolver = coarseSolver

        self._smooth = _buildChebyshevSmoother(A_free, self._fineDegree)

    def applyPreconditioner(self, r: np.ndarray) -> np.ndarray:
        """One two-grid V-cycle: pre-smooth, coarse-grid correction, post-smooth, on the free
        submatrix; Dirichlet rows pass through unchanged (``A[i, i] = 1`` there by construction, so
        the exact local solve is ``x[i] = r[i]``)."""
        import time

        self.applyCalls += 1
        rFree = r[self._freeRows]
        xFree = np.zeros_like(rFree)

        t0 = time.perf_counter()
        self._smooth(xFree, rFree, self._nu)
        self.fineSeconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        res = rFree - self._As_free @ xFree
        resCoarse = self._P_free.T @ res
        corrCoarse = self._coarseSolver.applyPreconditioner(resCoarse)
        xFree = xFree + self._P_free @ corrCoarse
        self.coarseSeconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        self._smooth(xFree, rFree, self._nu)
        self.fineSeconds += time.perf_counter() - t0

        x = np.empty_like(r)
        x[self._freeRows] = xFree
        x[self._dirichletRows] = r[self._dirichletRows]
        return x

    def report(self) -> str:
        """The coarse level's AMGCL hierarchy report (context only, per §22.2-bis -- not a gate)."""
        return self._coarseSolver.report()
