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
coupled system. Each field's operator (elasticity for a displacement field, a Helmholtz-like operator
for a damage field) is individually AMG-friendly even though their monolithic coupling is not.

The field structure -- which DOFs belong to which field, and each field's nodal dimension -- is not
carried by the matrix; it is pushed in from the DofManager by the nonlinear solver (via
:class:`~edelweissfe.linsolve.base.LinearSolver`), so nothing about the block layout
has to be specified by hand.

What it does per solve
----------------------

#. **Equilibrate.** Symmetric diagonal (Jacobi) scaling :math:`\\hat A = D^{-1/2} A D^{-1/2}` removes
   the large dynamic range (Dirichlet penalties + stiffness) that otherwise wrecks AMG's
   strength-of-connection. The solve is done on :math:`\\hat A` and unscaled at the end.
#. **Split** :math:`\\hat A` into the field diagonal blocks and their couplings, from the field ranges.
#. **Build one AMG hierarchy per field** (AMGCL, built once per solve via ``build`` and applied many
   times via ``applyPreconditioner`` -- the pattern churns between Newton iterations, so the hierarchy
   cannot be reused *across* solves, but it is reused across the outer GMRES iterations *within* a
   solve). A vector field (nodal dimension > 1, e.g. displacement) is given its rigid-body
   *translations* as the near null-space -- one per component, from the DOF layout alone; a scalar
   field takes the default constant.
#. **Precondition GMRES** with a block Gauss-Seidel sweep over the fields, each field's correction
   coming from one AMG V-cycle on its block, the couplings folded in between fields.

This is a *feasibility-grade* solver: on the reference model AMGCL's smoothed aggregation converges
but not tightly on the (non-symmetric, contact + tie condensed) displacement block, so the outer
GMRES needs O(100) iterations. That is acceptable where the point is to fit in memory at sizes a
direct solver cannot reach; the iteration count would come down with a stronger vector-field AMG
(a nonsymmetric-aware library such as MueLu). See the handoff document, section 13.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.linsolve.base import FieldBlock, LinearSolver

# Ordered low-to-high; index comparison decides whether a message at a given level should print.
# "warning": only abnormal conditions (excessive outer iterations, unmet true-residual tolerance,
# non-convergence) -- the default, so a normal run stays quiet. "info": one compact line per solve.
# "debug": full detail, including every true-residual continuation attempt. Independent of, and does
# not affect, the Journal instance's own message-level suppression (see _log()) -- this gate decides
# whether blockamg attempts to log at all; Journal's own level decides whether the attempt is shown.
_VERBOSITY_LEVELS = ("silent", "warning", "info", "debug")
_JOURNAL_LEVEL = {"warning": 0, "info": 1, "debug": 2}
_IDENTIFICATION = "BlockAMGSolver"

# "backendPrecision" and "backendBlockSize" are not AMGCL parameters -- they select the AMGCL
# wrapper's own backend value type (§19.3: "double" (default) or "float"; §20.1: 1 (default, scalar),
# 2, or 3 for a block-valued backend operating on B x B nodal blocks). __call__ pops both out of
# whichever precond dict applies (default or a fieldPreconds override) before forwarding the rest
# verbatim as the AMGCL JSON parameter tree.
#
# backendPrecision defaults to "double": measured on all 9 dumped ords, "float" inflated outer GMRES
# iterations by up to 30% (plan expected <=10%) and only won on wall-clock ~half the time, netting ~3%
# aggregate, not the ~1.3-1.5x hoped for -- plausibly the chebyshev smoother's power-iteration
# spectral-radius estimate (§17 B5 already flagged this as sensitive) losing accuracy in float32.
#
# backendBlockSize defaults to 1 (scalar, unchanged): kept opt-in via fieldPreconds through the whole
# §20.1 offline validation phase -- the default only changes if that validation clears its bar. A
# vector field of nodal dimension d maps to backendBlockSize: d (3 for the pryout displacement field,
# 2 for the registered 2D CantileverBeamQuad4BlockAMG test). Setting it > 1 skips set_nullspace (see
# __call__) -- AMGCL's near-null-space path is unimplemented for block value types.
#
# Both are available as an opt-in via fieldPreconds, e.g.
# {"displacement": {**_DEFAULT_VECTOR_PRECOND, "backendPrecision": "float"}} or
# {"displacement": {**_DEFAULT_VECTOR_PRECOND, "backendBlockSize": 3}}.
_DEFAULT_VECTOR_PRECOND = {
    "backendPrecision": "double",
    "backendBlockSize": 1,
    "coarsening": {"type": "smoothed_aggregation", "aggr": {"eps_strong": 0.01}},
    "relax": {"type": "chebyshev", "degree": 5, "power_iters": 50, "lower": 0.01},
    "npre": 1,
    "npost": 1,
}
_DEFAULT_SCALAR_PRECOND = {
    "backendPrecision": "double",
    "backendBlockSize": 1,
    "coarsening": {"type": "smoothed_aggregation"},
    "relax": {"type": "chebyshev"},
}


class BlockAMGSolver(LinearSolver):
    """Field-split block-AMG preconditioned GMRES. Callable as ``(A, b) -> x``.

    The block structure is not configured here -- it is supplied by the nonlinear solver through
    :meth:`~edelweissfe.linsolve.base.LinearSolver.setFieldStructure`. A field's
    near null-space is decided from its nodal dimension: a vector field (dimension > 1) gets its
    per-component rigid-body translations, a scalar field the default constant.

    Stateful across calls in two independent ways (both driven by ``||b||`` alone -- this solver, like
    :mod:`~edelweissfe.linsolve.inexactnewton.inexactnewton`, sees only ``(A, b)`` per call and
    reconstructs Newton-loop state from residual jumps rather than being told about them):

    #. **Adaptive outer tolerance** (Eisenstat--Walker forcing, "choice 2"), the default since §20.2.
       Most Newton iterates do not need the solve tight, and loosening it cuts outer iterations
       substantially. §19.2 first tried this and it changed the live Newton path (15/12/7/14 iterations
       and 2 cutbacks, vs the fixed-tolerance 15/8/5/5 and 3 cutbacks) -- root-caused to GMRES's
       preconditioned-residual stopping check letting the *true* residual run looser than `eta`
       actually asked for, so `etaMax=1e-3` was never really `1e-3` on this preconditioner. Once the
       true-residual enforcement above closed that gap, a live `etaMax` ladder (`1e-4`, then `3e-4`)
       both reproduced the fixed-tolerance trajectory *exactly* (identical 15/8/5/5, identical cutbacks,
       identical final load), so `etaMax=3e-4` (the looser, marginally faster rung) shipped as the
       default. Pass ``outerTol=<a float>`` to pin a fixed tolerance instead (the solver's original,
       pre-§20.2 behaviour) if a specific model needs that guarantee.
    #. **Per-field AMG hierarchy reuse across Newton iterations**, on by default. Building a hierarchy
       (~3.4 s on the reference model) is a large, avoidable fraction of a solve when the Jacobian has
       moved only a little since the last one. The outer GMRES always operates on the current, fresh
       matrix; only the block preconditioner ``M`` may be stale, so the *converged solution* is
       unaffected regardless of ``M`` -- a stale ``M`` only costs a few extra outer iterations at the
       *same* requested tolerance, so (unlike (a)) it cannot by itself change the Newton path. Refreshed
       on a residual jump (new increment / cutback), on a field-structure change (e.g. AMR), on any
       sparsity-pattern change (this class of problem's pattern churns every Newton iteration -- see
       §3.1 and §19.2 -- so in practice this keeps the hierarchy fresh every call on the reference
       model, a safe no-op there, not a win; problems without contact/tie condensation may see one), or
       when the previous solve's outer count grew past ``hierarchyStalenessFactor`` times the one
       before it.

    Every solve also enforces the requested tolerance on the *true* (unpreconditioned) residual, not
    just GMRES's own preconditioned stopping check -- see ``trueResidualMaxContinuations`` and
    PERF_LINSOLVE_INVESTIGATION.md §20.2. GMRES's convergence test here is on the preconditioned
    residual (``callback_type="pr_norm"``), which can be considerably smaller than the true one under
    an imperfect preconditioner (this one, by design), so "converged" could previously mean a true
    residual well above ``eta`` (measured: 1.6e-2 true residual when 1e-4 was requested, §17/§18).

    Parameters
    ----------
    outerTol
        A fixed outer GMRES relative tolerance, overriding Eisenstat--Walker forcing. ``None`` (the
        default since §20.2) uses adaptive forcing instead -- see ``etaMin``/``etaMax`` below.
    outerRestart, outerMaxiter
        The outer GMRES restart length and maximum restart cycles.
    sweeps
        Block Gauss-Seidel sweeps per preconditioner application.
    symmetric
        If True, each sweep is followed by a reverse-order sweep (symmetric block Gauss-Seidel).
    fieldPreconds
        Optional mapping of field name to an AMGCL preconditioner parameter tree, overriding the
        dimension-based default for that field.
    etaMin, etaMax
        Clamp on the Eisenstat--Walker forcing tolerance (ignored if ``outerTol`` is given). ``etaMax``
        is also the tolerance used whenever there is no residual history to base a ratio on (the first
        solve, or the one right after a detected new increment/cutback).
    ewGamma, ewAlpha
        The Eisenstat--Walker "choice 2" parameters: ``eta_k = gamma * (||b_k|| / ||b_{k-1}||) **
        alpha``. Defaults ``gamma = 0.9``, ``alpha = (1 + sqrt 5) / 2`` are the classic values.
    residualGrowthFactor
        A solve whose ``||b||`` exceeds this multiple of the previous solve's ``||b||`` is treated as
        the first solve of a new increment (or a cutback): the forcing tolerance resets to ``etaMax``
        and the AMG hierarchies are refreshed rather than reused.
    hierarchyStalenessFactor
        Refresh the AMG hierarchies before the *next* solve if this solve's outer GMRES count exceeded
        this factor times the previous solve's -- a growing count is the signal that the reused
        hierarchies are drifting away from the current Jacobian.
    trueResidualMaxContinuations
        How many times to warm-restart GMRES (``x0`` from the previous attempt, same ``As``/``bs``/``M``
        /``eta``) if the true relative residual still exceeds ``eta`` after GMRES itself reports
        convergence. ``0`` disables this and restores the original preconditioned-residual-only
        behaviour.
    verbosity
        One of ``"silent"``, ``"warning"`` (default), ``"info"``, ``"debug"``. Messages go through the
        injected Journal (:meth:`~edelweissfe.linsolve.base.LinearSolver.setJournal`), falling back to
        ``print`` if none was set. ``"warning"`` emits nothing on a normal solve and only speaks up on
        an abnormal one: outer iterations past ``warnOuterIterationsThreshold``, the true-residual
        tolerance still unmet after every continuation, or GMRES itself reporting non-convergence.
        ``"info"`` adds one compact line per solve; ``"debug"`` adds one line per true-residual
        continuation attempt too. Per-stage wall-clock (equilibration, off-diagonal split, hierarchy
        build, outer GMRES, continuations) is recorded via
        :mod:`edelweissfe.utils.performancetiming` regardless of verbosity, nested under "linear solve"
        in the job's own performance table -- note PERF_LINSOLVE_INVESTIGATION.md §19.1's caveat that
        this global mechanism's accumulated figures are not reliable under ``PYTHON_GIL=0`` on a live,
        multi-threaded run (a pre-existing, general-infrastructure bug, not specific to blockamg).
    warnOuterIterationsThreshold
        A solve needing more outer GMRES iterations than this triggers a ``"warning"``-level message
        (a preconditioner-quality red flag), even when ``verbosity="silent"`` is not set to suppress it.
    """

    def __init__(
        self,
        *,
        outerTol: float = None,
        outerRestart: int = 100,
        outerMaxiter: int = 8,
        sweeps: int = 1,
        symmetric: bool = True,
        fieldPreconds: dict = None,
        etaMin: float = 1.0e-6,
        etaMax: float = 3.0e-4,
        ewGamma: float = 0.9,
        ewAlpha: float = 1.618033988749895,
        residualGrowthFactor: float = 4.0,
        hierarchyStalenessFactor: float = 1.5,
        trueResidualMaxContinuations: int = 2,
        verbosity: str = "warning",
        warnOuterIterationsThreshold: int = 100,
    ):
        self._outerTol = outerTol
        self._outerRestart = outerRestart
        self._outerMaxiter = outerMaxiter
        self._sweeps = sweeps
        self._symmetric = symmetric
        self._fieldPreconds = fieldPreconds or {}
        self._etaMin = etaMin
        self._etaMax = etaMax
        self._ewGamma = ewGamma
        self._ewAlpha = ewAlpha
        self._residualGrowthFactor = residualGrowthFactor
        self._hierarchyStalenessFactor = hierarchyStalenessFactor
        self._trueResidualMaxContinuations = trueResidualMaxContinuations
        if verbosity not in _VERBOSITY_LEVELS:
            raise ValueError("verbosity must be one of {:}, got {:!r}".format(_VERBOSITY_LEVELS, verbosity))
        self._verbosityIndex = _VERBOSITY_LEVELS.index(verbosity)
        self._warnOuterIterationsThreshold = warnOuterIterationsThreshold

        self._solveCount = 0
        self._fieldsAnnounced = None

        # Eisenstat-Walker forcing state.
        self._lastResidualNorm = None
        self._lastEta = etaMax

        # Reused hierarchy state: the built per-field AMG hierarchies, the equilibration they were
        # built for, and the field-block layout they assume -- all None until the first solve.
        self._preconditioners = None
        self._dinv = None
        self._blocks = None
        self._n = None
        self._lastNnz = None
        self._lastOuterIters = None
        self._refreshNext = False

    def _log(self, level: str, message: str) -> None:
        """Emit ``message`` through the injected Journal (see ``setJournal``, inherited from
        :class:`~edelweissfe.linsolve.base.LinearSolver`) if this instance's verbosity is at least
        ``level`` -- falls back to a plain ``print`` if no Journal has been set (e.g. an offline probe
        script driving this solver directly, outside the full nonlinear-solver/driver stack).
        """
        if self._verbosityIndex < _VERBOSITY_LEVELS.index(level):
            return
        if self._journal is not None:
            self._journal.message(message, _IDENTIFICATION, level=_JOURNAL_LEVEL[level])
        else:
            print(message, flush=True)

    def _forcingTolerance(self, residualNorm: float, newIncrement: bool) -> float:
        """The Eisenstat--Walker "choice 2" forcing tolerance for this solve, clamped and safeguarded.

        ``eta_k = gamma (||b_k|| / ||b_{k-1}||) ** alpha``, with the classic safeguard against
        over-solving (a large tightening step is only trusted if the previous tolerance was already
        small), clamped to ``[etaMin, etaMax]``. Falls back to ``etaMax`` with no history to compare
        against (the first solve, or the one right after a new increment / cutback jump -- the ratio
        across that jump does not reflect Newton convergence and is not meaningful).
        """
        if self._lastResidualNorm is None or newIncrement:
            return self._etaMax

        ratio = residualNorm / self._lastResidualNorm
        eta = self._ewGamma * ratio**self._ewAlpha

        safeguard = self._ewGamma * self._lastEta**self._ewAlpha
        if safeguard > 0.1:
            eta = max(eta, safeguard)

        return min(self._etaMax, max(self._etaMin, eta))

    def setFieldStructure(self, fields: list) -> None:
        """Receive the ordered field blocks of the DOF vector (in DOF order).

        Overrides :meth:`~edelweissfe.linsolve.base.LinearSolver.setFieldStructure`'s no-op default --
        this is the one solver in the registry that actually needs the field-block structure, to
        split the equation system per field.
        """
        self._fieldStructure = list(fields)

    def _resolveBlocks(self, n: int) -> list:
        """The field blocks tiling ``[0, n)``, in DOF order, with any trailing DOFs not covered by a
        node field (e.g. scalar variables) folded into a final scalar block."""

        if self._fieldStructure is None:
            raise RuntimeError(
                "blockamg: field structure not set. It is pushed in by the nonlinear solver via "
                "setFieldStructure(); this solver must be driven by one that does so."
            )
        blocks = sorted(self._fieldStructure, key=lambda field: field.start)
        cursor = 0
        for block in blocks:
            if block.start != cursor:
                raise ValueError(
                    "blockamg: field '{:}' starts at {:}, expected {:} -- fields must tile the DOF "
                    "vector contiguously".format(block.name, block.start, cursor)
                )
            cursor = block.stop
        if cursor < n:
            # DOFs past the last node field: scalar variables. One scalar block.
            blocks = blocks + [FieldBlock("scalar variables", cursor, n, 1)]
        elif cursor != n:
            raise ValueError("blockamg: field blocks cover {:} dofs, but the matrix is {:}x{:}".format(cursor, n, n))
        return blocks

    def _translationNullspace(self, block: FieldBlock, blockDinv: np.ndarray) -> np.ndarray:
        """The rigid-body translations of a vector field, transformed for the scaled operator.

        Translations are 1 on each of the ``dimension`` components (node-major). The near null-space of
        the scaled block :math:`D^{-1/2} A D^{-1/2}` is :math:`D^{1/2}` times that of :math:`A`, i.e.
        the raw translations divided by ``blockDinv``.
        """
        size = block.stop - block.start
        components = block.dimension
        B = np.zeros((size, components))
        rows = np.arange(size)
        B[rows, rows % components] = 1.0
        return B / blockDinv[:, None]

    def __call__(self, A, b):
        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver

        self._solveCount += 1
        A = A.tocsr()
        n = A.shape[0]
        blocks = self._resolveBlocks(n)
        slices = [slice(block.start, block.stop) for block in blocks]
        b = np.asarray(b).reshape(n)

        fieldNames = [block.name for block in blocks]
        if fieldNames != self._fieldsAnnounced:
            self._log("info", "blockamg: fields = {:}".format(fieldNames))
            self._fieldsAnnounced = fieldNames

        residualNorm = float(np.linalg.norm(b))
        newIncrement = (
            self._lastResidualNorm is not None and residualNorm > self._residualGrowthFactor * self._lastResidualNorm
        )
        # A's sparsity pattern churns between Newton iterations on this class of problem (condensed
        # contact/tie systems, see the module docstring and PERF_LINSOLVE_INVESTIGATION.md §3.1) -- a
        # hierarchy built for a different pattern is not just "a bit stale", it can be a drastically
        # worse preconditioner (measured: 494 vs. 94 outer iterations on one such transition, an
        # outright wall-clock regression, not a graceful few-extra-iterations degradation). ``nnz`` is a
        # cheap, free (O(1)) proxy for "the pattern changed" -- not exact (two different patterns could
        # coincidentally share a total nnz), but it caught the one measured failure case and errs in the
        # safe direction (an unnecessary refresh costs time, never correctness).
        patternChanged = self._lastNnz is not None and A.nnz != self._lastNnz

        # Refresh the per-field AMG hierarchies (rather than reuse the standing ones) when there is
        # nothing to reuse yet, the field-block layout changed (e.g. an AMR event resized the DOF
        # vector), the sparsity pattern changed, a residual jump marks a new increment / cutback, or the
        # previous solve's own outer count asked for it (drifted too far from the one before it).
        mustRefresh = (
            self._preconditioners is None
            or blocks != self._blocks
            or n != self._n
            or patternChanged
            or newIncrement
            or self._refreshNext
        )
        self._refreshNext = False

        with performancetiming.timeit("blockamg: equilibration"):
            if mustRefresh:
                # Symmetric diagonal equilibration. Solve A x = b as (D A D)(D^-1 x) = D b, i.e.
                # As z = bs with x = D z; D = diag(dinv), dinv = 1/sqrt(|diag A|).
                dinv = 1.0 / np.sqrt(np.abs(A.diagonal()))
            else:
                # Reuse the equilibration the standing hierarchies were built for. This stays a valid
                # diagonal similarity scaling of the *current* A x = b regardless of how it was chosen,
                # so correctness (the outer GMRES converges on the true, fresh As/bs) is unaffected;
                # only the preconditioner's quality can drift, which costs outer iterations, never a
                # wrong answer.
                dinv = self._dinv

            As = (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr()
            bs = dinv * b

        # Off-diagonal couplings (for the sweep) are needed every solve regardless of refresh/reuse.
        with performancetiming.timeit("blockamg: off-diagonal split"):
            offBlocks = {}
            for i in range(len(slices)):
                rowBlock = As[slices[i], :]
                for j in range(len(slices)):
                    if i != j:
                        offBlocks[(i, j)] = rowBlock[:, slices[j]].tocsr()

        if mustRefresh:
            with performancetiming.timeit("blockamg: hierarchy build"):
                # One AMG hierarchy per field, built fresh. A vector field gets its translations as the
                # near null-space; a scalar field the default constant.
                diagBlocks = [As[sl, :][:, sl].tocsr() for sl in slices]
                preconditioners = []
                for i, block in enumerate(blocks):
                    isVectorField = block.dimension > 1
                    precondParams = dict(
                        self._fieldPreconds.get(
                            block.name, _DEFAULT_VECTOR_PRECOND if isVectorField else _DEFAULT_SCALAR_PRECOND
                        )
                    )
                    backendPrecision = precondParams.pop("backendPrecision", "double")
                    backendBlockSize = precondParams.pop("backendBlockSize", 1)
                    solver = PyAMGCLSolver(
                        {
                            "precond": precondParams,
                            "backendPrecision": backendPrecision,
                            "backendBlockSize": backendBlockSize,
                        }
                    )
                    # set_nullspace() is unsupported (and always raises) on a block backend -- AMGCL's
                    # own near-null-space path is unimplemented for block value types (§20.1). This is
                    # an accepted, measured non-loss: rigid-body translations do not help on this
                    # operator anyway (§11, §13).
                    if isVectorField and backendBlockSize == 1:
                        solver.set_nullspace(self._translationNullspace(block, dinv[slices[i]]))
                    solver.build(diagBlocks[i])
                    preconditioners.append(solver)
                self._preconditioners = preconditioners
                self._dinv = dinv
                self._blocks = blocks
                self._n = n
                self._lastNnz = A.nnz
        preconditioners = self._preconditioners

        nFields = len(slices)
        sizes = [block.stop - block.start for block in blocks]

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

        if self._outerTol is not None:
            eta = self._outerTol
        else:
            eta = self._forcingTolerance(residualNorm, newIncrement)

        with performancetiming.timeit("blockamg: outer GMRES"):
            history = []
            z, info = gmres(
                As,
                bs,
                M=preconditioner,
                rtol=eta,
                atol=0.0,
                restart=self._outerRestart,
                maxiter=self._outerMaxiter,
                callback=lambda residualNorm: history.append(residualNorm),
                callback_type="pr_norm",
            )
            x = dinv * z
            trueResidual = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)

        # GMRES's own stopping check (callback_type="pr_norm") is on the *preconditioned* residual,
        # not the true one -- with an imperfect preconditioner (this one, by design, §17) the two can
        # diverge substantially, so "converged" per GMRES can still leave a true residual well above
        # the requested eta (documented gap: ord 3 reaching 1.6e-2 true residual when 1e-4 was
        # requested, §17/§18).
        #
        # Two failed attempts before this one, both found empirically (every continuation logged "0
        # more outer GMRES iters" until fixed):
        #   1. A warm restart with the *same* rtol is a no-op -- x0=z already satisfies that exact
        #      preconditioned-residual criterion, so GMRES re-declares convergence immediately.
        #   2. Tightening the *requested* eta proportionally (eta * (eta/trueResidual)), or relative to
        #      the callback's last-reported "pr_norm" -- both still occasionally produced a target
        #      *larger* than eta itself (impossible if the callback's pr_norm were on the same relative
        #      scale as rtol; it evidently is not -- likely an absolute residual, not one normalized by
        #      ||bs||). Rather than reverse-engineer scipy's internal residual bookkeeping, avoid it
        #      entirely: tighten purely within rtol's own units, which scipy already interprets
        #      correctly by construction.
        # Fix: geometrically tighten the *requested* rtol itself by a fixed factor each continuation
        # (0.01x), guaranteeing a strictly smaller, dimensionally-consistent target with no dependency
        # on the callback's residual scale.
        with performancetiming.timeit("blockamg: true-residual continuations"):
            continuationEta = eta
            continuations = 0
            while trueResidual > eta and continuations < self._trueResidualMaxContinuations:
                continuations += 1
                continuationEta *= 0.01
                continuationHistory = []
                z, info = gmres(
                    As,
                    bs,
                    x0=z,
                    M=preconditioner,
                    rtol=continuationEta,
                    atol=0.0,
                    restart=self._outerRestart,
                    maxiter=self._outerMaxiter,
                    callback=lambda residualNorm: continuationHistory.append(residualNorm),
                    callback_type="pr_norm",
                )
                x = dinv * z
                trueResidual = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)
                history.extend(continuationHistory)
                self._log(
                    "debug",
                    "blockamg:   continuation {:}/{:} eta={:.1e} +{:}it res={:.1e}".format(
                        continuations,
                        self._trueResidualMaxContinuations,
                        continuationEta,
                        len(continuationHistory),
                        trueResidual,
                    ),
                )

        outerIters = len(history)
        if self._lastOuterIters is not None and outerIters > self._hierarchyStalenessFactor * self._lastOuterIters:
            self._refreshNext = True
        self._lastOuterIters = outerIters
        self._lastResidualNorm = residualNorm
        self._lastEta = eta

        self._log(
            "info",
            "blockamg: solve #{:<4d} {:7s} {:3d}it eta={:.1e} res={:.1e} cont={:}".format(
                self._solveCount,
                "REFRESH" if mustRefresh else "reuse",
                outerIters,
                eta,
                trueResidual,
                continuations,
            ),
        )
        if outerIters > self._warnOuterIterationsThreshold:
            self._log(
                "warning",
                "blockamg: WARNING solve #{:} needed {:} outer GMRES iterations (> threshold {:}) -- "
                "possible preconditioner degradation".format(
                    self._solveCount, outerIters, self._warnOuterIterationsThreshold
                ),
            )
        if trueResidual > eta:
            self._log(
                "warning",
                "blockamg: WARNING solve #{:} true residual {:.2e} still exceeds requested eta={:.2e} "
                "after {:} continuation(s) -- did not fully converge".format(
                    self._solveCount, trueResidual, eta, continuations
                ),
            )
        if info != 0:
            self._log(
                "warning",
                "blockamg: WARNING solve #{:} GMRES reported info={:} (did not converge within "
                "maxiter on its own preconditioned-residual criterion)".format(self._solveCount, info),
            )

        return x
