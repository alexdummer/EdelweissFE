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

"""An inexact-Newton linear solver: preconditioned GMRES whose preconditioner is a *lagged* exact LU
factorization, reused across several Newton iterations, tightened by an Eisenstat--Walker forcing
sequence.

Why this exists
---------------

On a large coupled fracture model (penalty contact + adaptive refinement + gradient-enhanced damage,
no saddle-point structure) the linear solve dominates: on the reference 280k-dof anchor pry-out
model, 82% of the run is spent in PARDISO, and *of that*, 42% is the numeric factorization and 35% is
the reordering. A direct solve redoes both on every Newton iteration even though the Jacobian barely
moves from one iterate to the next.

The measured way out (``scripts/benchmark_linsolve.py lagged``) is not "GMRES + ILU" -- single-level
ILU0 loses to the direct solve at every usable tolerance on this problem. It is to keep an *exact*
PARDISO factorization of one Newton iterate and reuse it as a preconditioner for the next few
iterates' matrices. The preconditioned operator :math:`A_{\\mathrm{base}}^{-1} A_k` is not near the
identity (its distance is O(1), because :math:`A_{\\mathrm{base}}^{-1}` amplifies the small
Jacobian drift by 180--850x on the stiff directions) -- so this does *not* converge in one iteration
-- but its spectrum decays fast, which is exactly the regime where GMRES annihilates most of the
error in a handful of iterations. Measured: iterates a few steps stale reach a relative residual of
1e-4 in **4--9 GMRES iterations** (~3--7 s) against **~11.5 s** for a direct solve, and the count is
flat in staleness -- one factorization serves at least eight subsequent Newton iterations. Break-even
against the direct solve is around 15 iterations.

Two caveats the benchmark also surfaced, both handled by the policy below:

- The *tolerance matters*. A Newton step does not need 1e-8 (which needs 30--60 iterations and would
  lose); it needs an inexact-Newton forcing tolerance. But 1e-2 is too loose -- a 1e-2 *residual*
  leaves a 15--19% *error* in the correction on this ill-conditioned system, because residual is not
  error here. The Eisenstat--Walker sequence lands in the productive 1e-4..1e-6 band automatically.
- The *first solve after a large state change* (a new increment's first correction, an AMR event) is
  the one iterate that preconditions badly. The policy keeps those direct.

Design
------

This is a self-contained ``(A, b) -> x`` linear solver -- it plugs into the nonlinear solvers exactly
like PARDISO, requiring **no** change to the Newton loop. It reconstructs the two signals it needs
from what it is already handed:

- ``b`` *is* the (condensed, Dirichlet-eliminated) Newton residual, so ``||b||`` drives the
  Eisenstat--Walker forcing sequence directly, and a *jump* in ``||b||`` marks a new increment (or a
  cutback) without the Newton loop having to announce one.
- the GMRES iteration count reports when a factorization has gone stale.

The refactorization policy, all knobs configurable (see :func:`createSolver`):

#. Refactorize (exact PARDISO LU) and solve near-exactly when there is no factorization yet, when the
   residual grows past ``residualGrowthFactor`` x the previous one (new increment / cutback), on the
   iterate immediately after such a growth (the large first correction is itself hard to
   precondition), once ``maxReuse`` reuses have accumulated (bounded staleness), or when the previous
   reuse failed to reach its forcing tolerance within the Krylov budget.
#. Otherwise reuse the standing factorization as a GMRES preconditioner, with the Eisenstat--Walker
   forcing tolerance for this iterate.

A fresh factorization is a perfect preconditioner in exact arithmetic, so the solve right after one
converges in ~1 GMRES iteration regardless of the forcing tolerance; the forcing sequence therefore
only governs the reuse solves, which is where it belongs.
"""

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, gmres, splu


class _SuperLUFactorization:
    """A ``factorize``/``solveFactorized`` factorizing delegate backed by SciPy's SuperLU.

    :class:`~edelweissfe.linsolve.pardiso.pardiso.PardisoSolver` already exposes this two-method
    contract; this adapter gives the same contract to SciPy's ``splu`` so the wrapper has a
    dependency-free delegate for testing and for installs without the optional PARDISO extension.
    SuperLU is far slower than PARDISO on the large condensed systems this wrapper targets and is not
    the intended production delegate -- see the class docstring of the module.
    """

    def __init__(self):
        self._lu = None

    def factorize(self, A):
        """Compute and store an LU factorization of ``A`` (expected CSR; converted to CSC)."""
        self._lu = splu(csc_matrix(A))

    def solveFactorized(self, b):
        """Solve against the stored factorization; raises if :meth:`factorize` has not run."""
        if self._lu is None:
            raise RuntimeError("no factorization available; call factorize() first")
        return self._lu.solve(np.asarray(b))

    def __call__(self, A, b):
        """Factorize ``A`` and solve for ``b`` in one call (the plain direct-solve contract)."""
        self.factorize(A)
        return self.solveFactorized(b)


class InexactNewtonSolver:
    """Preconditioned GMRES with a lagged exact-LU preconditioner and Eisenstat--Walker forcing.

    See the module docstring for the rationale and the measured numbers. Callable as ``(A, b) -> x``,
    stateful across calls: it holds the standing factorization and the data needed to decide when to
    refresh it.

    Parameters
    ----------
    delegate
        A factorizing solver exposing ``factorize(A)`` and ``solveFactorized(b)`` -- a
        :class:`~edelweissfe.linsolve.pardiso.pardiso.PardisoSolver` in production, or
        :class:`_SuperLUFactorization` for testing.
    maxReuse
        How many consecutive reuse solves one factorization may serve before it is refreshed
        regardless of GMRES's opinion. Bounds staleness; the measured plateau extends to at least 8.
    residualGrowthFactor
        A solve whose ``||b||`` exceeds this multiple of the previous solve's ``||b||`` is treated as
        the first solve of a new increment (or a cutback) and refactorized. Newton residuals fall
        monotonically *within* an increment and jump by orders of magnitude *between* increments, so a
        generous factor cleanly separates the two.
    etaMin, etaMax
        Clamp on the Eisenstat--Walker forcing tolerance. ``etaMax`` is the loosest tolerance a reuse
        solve is allowed (1e-2 is too loose -- see the module docstring -- so the default sits at
        1e-3); ``etaMin`` the tightest, near Newton convergence.
    ewGamma, ewAlpha
        The Eisenstat--Walker "choice 2" parameters: ``eta_k = gamma * (||b_k|| / ||b_{k-1}||) **
        alpha``. Defaults ``gamma = 0.9``, ``alpha = (1 + sqrt 5) / 2`` are the classic values.
    gmresRestart
        The GMRES Krylov subspace dimension between restarts.
    gmresMaxOuter
        The maximum number of GMRES restart cycles, so at most ``gmresRestart * gmresMaxOuter`` matrix
        applies before a solve is declared not converged (and the factorization refreshed).
    staleIterationThreshold
        If a reuse solve needs more GMRES iterations than this to converge, the factorization is
        refreshed on the next solve. Defaults to the measured break-even against the direct solve.
    verbose
        If True, print one line per solve (refactorize?, forcing tolerance, iteration count) -- useful
        for reading the nonlinear cost of the forcing sequence off a real run.
    """

    def __init__(
        self,
        delegate,
        *,
        maxReuse: int = 8,
        residualGrowthFactor: float = 4.0,
        etaMin: float = 1.0e-6,
        etaMax: float = 1.0e-3,
        ewGamma: float = 0.9,
        ewAlpha: float = 1.618033988749895,
        gmresRestart: int = 50,
        gmresMaxOuter: int = 4,
        staleIterationThreshold: int = 15,
        verbose: bool = False,
    ):
        self._delegate = delegate
        self._maxReuse = maxReuse
        self._residualGrowthFactor = residualGrowthFactor
        self._etaMin = etaMin
        self._etaMax = etaMax
        self._ewGamma = ewGamma
        self._ewAlpha = ewAlpha
        self._gmresRestart = gmresRestart
        self._gmresMaxOuter = gmresMaxOuter
        self._staleIterationThreshold = staleIterationThreshold
        self._verbose = verbose

        self._hasFactorization = False
        self._reuseCount = 0
        self._lastResidualNorm = None
        self._lastEta = etaMax
        # Set when the *next* solve must refactorize: after a new-increment refactorization (its first
        # correction is the largest and preconditions worst), or after a reuse solve that under-performed.
        self._refactorizeNext = False

    def _forcingTolerance(self, residualNorm: float) -> float:
        """The Eisenstat--Walker "choice 2" forcing tolerance for a reuse solve, clamped and safeguarded.

        ``eta_k = gamma (||b_k|| / ||b_{k-1}||) ** alpha``, with the standard safeguard that prevents
        the tolerance from tightening faster than the nonlinear residual actually warrants, then
        clamped to ``[etaMin, etaMax]``.
        """

        ratio = residualNorm / self._lastResidualNorm
        eta = self._ewGamma * ratio**self._ewAlpha

        # Classic Eisenstat--Walker safeguard against over-solving: only allow a large tightening step
        # if the previous forcing tolerance was already small.
        safeguard = self._ewGamma * self._lastEta**self._ewAlpha
        if safeguard > 0.1:
            eta = max(eta, safeguard)

        return min(self._etaMax, max(self._etaMin, eta))

    def __call__(self, A, b):
        """Solve ``A x = b`` with a lagged-LU-preconditioned inexact-Newton GMRES.

        Parameters
        ----------
        A
            The system matrix (CSR).
        b
            The right hand side, i.e. the current (condensed) Newton residual.

        Returns
        -------
        ndarray
            The solution, reshaped to ``b``'s shape. NaNs propagate from the delegate on a
            factorization/substitution failure, so the nonlinear solvers' existing NaN check turns
            that into a cutback.
        """

        residualNorm = float(np.linalg.norm(b))

        newIncrement = (
            self._lastResidualNorm is not None and residualNorm > self._residualGrowthFactor * self._lastResidualNorm
        )
        refactorize = (
            not self._hasFactorization or self._reuseCount >= self._maxReuse or newIncrement or self._refactorizeNext
        )
        self._refactorizeNext = False

        if refactorize:
            self._delegate.factorize(A)
            self._hasFactorization = True
            self._reuseCount = 0
            # Keep the large first correction after a state change direct as well: that iterate
            # preconditions worst of all (measured), so reusing this fresh factorization for it would
            # cost a long GMRES burst.
            if newIncrement:
                self._refactorizeNext = True
            # A fresh factorization is a near-perfect preconditioner, so any tolerance converges in ~1
            # iteration; ask for the loosest so nothing is wasted overshooting it.
            eta = self._etaMax
        else:
            eta = self._forcingTolerance(residualNorm)

        preconditioner = LinearOperator(A.shape, matvec=self._delegate.solveFactorized, dtype=A.dtype)

        history = []
        x, info = gmres(
            A,
            b,
            M=preconditioner,
            rtol=eta,
            atol=0.0,
            restart=self._gmresRestart,
            maxiter=self._gmresMaxOuter,
            callback=history.append,
            callback_type="pr_norm",
        )
        iterations = len(history)

        if info != 0:
            # The standing factorization was too stale to reach the forcing tolerance within the
            # Krylov budget: refresh it and solve exactly. Costs one wasted GMRES burst, but this is
            # the safety net, not the common path (the proactive rules above pre-empt most staleness).
            self._delegate.factorize(A)
            self._hasFactorization = True
            self._reuseCount = 1
            x = self._delegate.solveFactorized(b)
            iterations = 0
            eta = 0.0
        else:
            self._reuseCount += 1
            self._lastEta = eta
            if not refactorize and iterations > self._staleIterationThreshold:
                # Converged, but only just: refresh next time before it stops converging.
                self._refactorizeNext = True

        self._lastResidualNorm = residualNorm

        if self._verbose:
            print(
                "inexactnewton: |b|={:.3e} {:} eta={:.1e} gmres_iters={:} reuse={:}".format(
                    residualNorm,
                    "REFACTORIZE" if refactorize else "reuse     ",
                    eta,
                    iterations,
                    self._reuseCount,
                ),
                flush=True,
            )

        return np.asarray(x).reshape(np.asarray(b).shape)
