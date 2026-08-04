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

"""A field-split block-AMG linear solver for large coupled multi-field systems.

Why this exists
---------------

On the large coupled fracture models (displacement + gradient-enhanced damage, penalty contact,
adaptive refinement), a direct factorization dominates the run and -- more importantly -- hits a
memory wall past ~1M dof, because its fill-in grows superlinearly. Algebraic multigrid has O(n)
memory and is the route to those sizes, but applied *monolithically* to the coupled system it is
ineffective: a single AMG hierarchy cannot represent the disparate physics and scales of the fields
at once (measured -- it stalls at a 0.2 residual, see PERF_LINSOLVE_INVESTIGATION.md).

The remedy, following Alkmim et al. (IJNME 2026), is a *block* preconditioner: an AMG hierarchy per
field, combined by a block Gauss-Seidel sweep, used to precondition an outer GMRES over the full
coupled system. Each field's operator (elasticity for displacement, a Helmholtz-like operator for
damage) is individually AMG-friendly even though their monolithic coupling is not.

What it does per solve
----------------------

#. **Equilibrate.** Symmetric diagonal (Jacobi) scaling :math:`\\hat A = D^{-1/2} A D^{-1/2}` removes
   the ~1e8 dynamic range (Dirichlet penalties + stiffness) that otherwise wrecks AMG's
   strength-of-connection. The solve is done on :math:`\\hat A` and unscaled at the end.
#. **Split** :math:`\\hat A` into the field diagonal blocks and their couplings, from the field sizes
   (which are contiguous, field-major, in the DOF vector).
#. **Build one AMG hierarchy per field** (AMGCL, built once per solve via ``build`` and applied many
   times via ``applyPreconditioner`` -- the pattern churns between Newton iterations, so the hierarchy
   cannot be reused *across* solves, but it is reused across the outer GMRES iterations *within* a
   solve). An elasticity field is given its rigid-body *translations* as the near null-space,
   constructible from the DOF layout alone (node-major, ``components`` per node); a scalar field takes
   the default constant.
#. **Precondition GMRES** with a block Gauss-Seidel sweep over the fields, each field's correction
   coming from one AMG V-cycle on its block, the couplings folded in between fields.

This is a *feasibility-grade* solver: on the reference model AMGCL's smoothed aggregation converges
but not tightly on the (non-symmetric, contact + tie condensed) displacement block, so the outer
GMRES needs O(100) iterations. That is acceptable where the point is to fit in memory at sizes a
direct solver cannot reach; the iteration count would come down with a stronger elasticity AMG
(rotations, a nonsymmetric-aware library such as MueLu). See the handoff document, section 13.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres

_DEFAULT_ELASTICITY_PRECOND = {
    "coarsening": {"type": "smoothed_aggregation"},
    "relax": {"type": "gauss_seidel"},
    "npre": 2,
    "npost": 2,
}
_DEFAULT_SCALAR_PRECOND = {
    "coarsening": {"type": "smoothed_aggregation"},
    "relax": {"type": "chebyshev"},
}


class BlockAMGSolver:
    """Field-split block-AMG preconditioned GMRES. Callable as ``(A, b) -> x``.

    Parameters
    ----------
    fields
        The ordered list of field blocks, one dict per field, each with:

        ``size``
            The number of DOFs in the block (they are contiguous and field-major in the DOF vector).
        ``elasticity``
            If True, the block gets its rigid-body translations as the AMG near null-space; else the
            default constant. Default False.
        ``components``
            For an elasticity block, the number of displacement components per node (node-major).
            Default 3.
        ``precond``
            An optional AMGCL preconditioner parameter tree overriding the default for this block.
    outerTol, outerRestart, outerMaxiter
        The outer GMRES relative tolerance, restart length, and maximum restart cycles.
    sweeps
        Block Gauss-Seidel sweeps per preconditioner application.
    symmetric
        If True, each sweep is followed by a reverse-order sweep (symmetric block Gauss-Seidel).
    verbose
        If True, print the outer iteration count and residual per solve.
    """

    def __init__(
        self,
        fields: list,
        *,
        outerTol: float = 1.0e-4,
        outerRestart: int = 100,
        outerMaxiter: int = 500,
        sweeps: int = 1,
        symmetric: bool = False,
        verbose: bool = False,
    ):
        if not fields:
            raise ValueError("blockamg: 'fields' must list at least one field block")
        self._fields = fields
        self._outerTol = outerTol
        self._outerRestart = outerRestart
        self._outerMaxiter = outerMaxiter
        self._sweeps = sweeps
        self._symmetric = symmetric
        self._verbose = verbose

    def _translationNullspace(self, size: int, components: int, blockDinv: np.ndarray) -> np.ndarray:
        """The rigid-body translations of an elasticity block, transformed for the scaled operator.

        Translations are 1 on each component (node-major). The near null-space of the scaled block
        :math:`D^{-1/2} A D^{-1/2}` is :math:`D^{1/2}` times that of :math:`A`, i.e. the raw
        translations divided by ``blockDinv`` (``= sqrt(|diag A|)`` times them).
        """
        B = np.zeros((size, components))
        rows = np.arange(size)
        B[rows, rows % components] = 1.0
        return B / blockDinv[:, None]

    def __call__(self, A, b):
        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver

        A = A.tocsr()
        n = A.shape[0]
        sizes = [int(field["size"]) for field in self._fields]
        if sum(sizes) != n:
            raise ValueError(
                "blockamg: field sizes {:} sum to {:}, but the matrix is {:}x{:}".format(sizes, sum(sizes), n, n)
            )
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        slices = [slice(int(bounds[i]), int(bounds[i + 1])) for i in range(len(sizes))]

        # Symmetric diagonal equilibration. Solve A x = b as (D A D)(D^-1 x) = D b, i.e. As z = bs
        # with x = D z; D = diag(dinv), dinv = 1/sqrt(|diag A|).
        dinv = 1.0 / np.sqrt(np.abs(A.diagonal()))
        As = (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr()
        bs = dinv * np.asarray(b).reshape(n)

        # Field diagonal blocks (for the per-field AMG) and off-diagonal couplings (for the sweep).
        diagBlocks = [As[sl, :][:, sl].tocsr() for sl in slices]
        offBlocks = {}
        for i in range(len(slices)):
            rowBlock = As[slices[i], :]
            for j in range(len(slices)):
                if i != j:
                    offBlocks[(i, j)] = rowBlock[:, slices[j]].tocsr()

        # One AMG hierarchy per field, built once for this solve.
        preconditioners = []
        for i, field in enumerate(self._fields):
            precondParams = field.get(
                "precond",
                _DEFAULT_ELASTICITY_PRECOND if field.get("elasticity", False) else _DEFAULT_SCALAR_PRECOND,
            )
            solver = PyAMGCLSolver({"precond": precondParams})
            if field.get("elasticity", False):
                nullspace = self._translationNullspace(sizes[i], int(field.get("components", 3)), dinv[slices[i]])
                solver.set_nullspace(nullspace)
            solver.build(diagBlocks[i])
            preconditioners.append(solver)

        nFields = len(slices)

        def sweepOnce(order, residual, x):
            for i in order:
                localResidual = residual[slices[i]].copy()
                for j in range(nFields):
                    if j != i:
                        localResidual -= offBlocks[(i, j)] @ x[j]
                x[i] = preconditioners[i].applyPreconditioner(localResidual)

        def blockGaussSeidel(residual):
            x = [np.zeros(sizes[i]) for i in range(nFields)]
            for _ in range(self._sweeps):
                sweepOnce(range(nFields), residual, x)
                if self._symmetric:
                    sweepOnce(range(nFields - 1, -1, -1), residual, x)
            return np.concatenate(x)

        preconditioner = LinearOperator((n, n), matvec=blockGaussSeidel, dtype=As.dtype)

        history = []
        z, info = gmres(
            As,
            bs,
            M=preconditioner,
            rtol=self._outerTol,
            atol=0.0,
            restart=self._outerRestart,
            maxiter=self._outerMaxiter,
            callback=lambda residualNorm: history.append(residualNorm),
            callback_type="pr_norm",
        )
        x = dinv * z

        if self._verbose:
            trueResidual = np.linalg.norm(A @ x - np.asarray(b).reshape(n)) / max(np.linalg.norm(b), 1e-300)
            print(
                "blockamg: {:} outer GMRES iters, info={:}, true rel.res={:.2e}".format(
                    len(history), info, trueResidual
                ),
                flush=True,
            )

        return x
