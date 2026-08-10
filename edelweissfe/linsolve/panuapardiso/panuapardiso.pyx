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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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

"""
This module provides an interface to the commercial Panua PARDISO solver.
It is only available, if the PARDISO solver is installed on the system.
You can get the binaries from https://panua.ch/. A license is required to use the solver.

The interface mirrors the MKL one in :mod:`edelweissfe.linsolve.pardiso.pardiso`: a stateful
:class:`PanuaPardisoSolver` holding the solver handle across solves (with optional reuse of the
symbolic factorization, phase-separated :meth:`~PanuaPardisoSolver.factorize` /
:meth:`~PanuaPardisoSolver.solveFactorized`, and per-phase timings), plus the one-shot
:func:`panuaPardisoSolve` convenience wrapper this module used to consist of.
"""

import os

import numpy as np

from edelweissfe.utils import performancetiming

cimport numpy as np


cdef extern from "pardiso.h" nogil:
    void pardiso(void*, int*, int *, int*, int *, int *,
                 void*, int*, int *, int*, int *, int *,
                 int*, void*, void*, int*, double*)

    void pardisoinit(void*, int*, int*, int*, double*, int*)


cdef class PanuaPardisoSolver:
    """
    Stateful interface to the Panua PARDISO solver.

    Mirrors :class:`~edelweissfe.linsolve.pardiso.pardiso.PardisoSolver`; the differences are
    Panua's, not this class's: an additional ``dparm`` array and ``solver`` selector threaded through
    every call, a ``pardisoinit`` that also performs the license check (and can therefore fail), and
    a thread count that must be handed over explicitly through ``iparm[2]`` because -- unlike MKL --
    Panua PARDISO does not read ``OMP_NUM_THREADS`` itself.

    The reordering and symbolic factorization (PARDISO phase 11) depend only on the sparsity pattern
    of the system matrix, so it is in principle safe to compute it once and reuse it for every
    subsequent solve with the same pattern (each solve then only performs the numerical
    factorization, phase 22, and back substitution, phase 33). A change of the sparsity pattern is
    detected automatically and triggers a re-analysis.

    However, this reuse has been observed -- with the MKL backend, on the very same call sequence --
    to silently produce numerically wrong results (with PARDISO reporting ``error == 0``, so the
    usual NaN-based failure check does not catch it) for some coupled-DOF problems where the
    numerically relevant pivot structure shifts substantially between solves even though the
    sparsity pattern itself does not change. Nothing about that failure mode is MKL-specific, and it
    has not been ruled out here, so reuse is **disabled by default** for this backend too; pass
    ``reuseSymbolicFactorization=True`` to opt in once this has been verified safe for the problem at
    hand (e.g. by comparing against one-shot solves on the actual matrix sequence). With reuse
    disabled, this class behaves like the free function :func:`panuaPardisoSolve`, just as a reusable
    object.

    The number of threads is taken from ``OMP_NUM_THREADS`` once, at construction time, and passed on
    as ``iparm[2]``.
    """

    cdef void *pt[64]     # internal solver memory pointer
    cdef int iparm[64]    # integer parameters for pardiso
    cdef double dparm[64] # float parameters for pardiso (Panua-specific)
    cdef int mtype        # real and unsymmetric matrix
    cdef int solver       # sparse direct solver
    cdef int maxfct
    cdef int mnum
    cdef int msglvl
    cdef int rows
    cdef int numThreads
    cdef bint ptIsActive  # pt may hold PARDISO-internal allocations (phase -1 required)
    cdef bint hasSymbolicFactorization
    cdef bint hasNumericFactorization
    cdef bint reuseSymbolicFactorization

    # the pattern arrays of the currently analyzed matrix (0-based, for change detection)
    cdef object currentIndices
    cdef object currentIndptr
    # the value array the stored numeric factorization belongs to, kept alive because phase 33
    # performs iterative refinement against it -- see solveFactorized
    cdef object currentData
    # persistent 1-based copies handed to pardiso (fortran indexing)
    cdef int[::1] indicesFortran
    cdef int[::1] indptrFortran

    def __cinit__(self, reuseSymbolicFactorization=False):
        cdef int i

        self.mtype = 11   # real and unsymmetric matrix
        self.solver = 0   # use the sparse direct solver
        self.maxfct = 1   # maximum number of numerical factorizations kept
        self.mnum = 1     # which factorization to use
        self.msglvl = 0   # do not print statistical information
        self.ptIsActive = False
        self.hasSymbolicFactorization = False
        self.hasNumericFactorization = False
        self.reuseSymbolicFactorization = reuseSymbolicFactorization
        self.currentIndices = None
        self.currentIndptr = None
        self.currentData = None

        for i in range(64):
            self.pt[i] = NULL
            self.iparm[i] = 0
            self.dparm[i] = 0.0

        # Panua PARDISO does not consult OMP_NUM_THREADS itself -- the thread count reaches it only
        # through iparm[2] (see _initializeSolver). Read once here rather than per solve: the
        # environment does not change over an instance's lifetime, while iparm does (pardisoinit
        # overwrites the whole array), so the value has to be re-applied, not re-read.
        ompNumThreads = os.environ.get("OMP_NUM_THREADS")
        self.numThreads = int(ompNumThreads) if ompNumThreads is not None else -1

        # keep the license banner out of the solver output
        os.environ["PARDISOLICMESSAGE"] = "1"

    def __dealloc__(self):
        self._releaseMemory()

    cdef int _initializeSolver(self) except -1:
        """Set up a fresh solver handle: license check, then parameter defaults (pardisoinit)."""
        cdef int error = 0
        cdef int i

        for i in range(64):
            self.pt[i] = NULL
            self.iparm[i] = 0
            self.dparm[i] = 0.0

        pardisoinit(self.pt, &self.mtype, &self.solver, &self.iparm[0], &self.dparm[0], &error)

        if error != 0:
            # Panua reports the license state here, unlike MKL's pardisoinit which cannot fail;
            # -10/-11/-12 are "no license found" / "license expired" / "wrong username or hostname".
            raise RuntimeError(
                "Panua PARDISO initialization failed with error code {:}".format(error)
            )

        self.ptIsActive = True

        # iparm[0] == 0 asks PARDISO to fill in its own defaults for all remaining entries during
        # phase 11. iparm[2] is exempt from that and always has to be supplied by the caller, so it
        # is set after -- and has to be re-set after every pardisoinit, which resets iparm.
        self.iparm[0] = 0
        self.iparm[2] = self.numThreads

        return 0

    cdef void _releaseMemory(self):
        """Release all internal PARDISO memory (phase -1)."""
        cdef int phase = -1
        cdef int error = 0
        cdef int nRhs = 1
        cdef int idum = 0
        cdef double ddum = 0
        cdef int *indptr = &idum
        cdef int *indices = &idum

        if not self.ptIsActive:
            return

        # Panua's own examples hand the real index arrays to phase -1 (only the value array is a
        # dummy there), so do the same whenever they are still around -- MKL's phase -1 takes
        # dummies for all three, but there is no reason to make Panua deviate from its own examples.
        if (
            self.indptrFortran is not None and self.indptrFortran.shape[0] > 0
            and self.indicesFortran is not None and self.indicesFortran.shape[0] > 0
        ):
            indptr = &self.indptrFortran[0]
            indices = &self.indicesFortran[0]

        pardiso(self.pt, &self.maxfct, &self.mnum, &self.mtype, &phase,
                &self.rows, &ddum, indptr, indices, &idum, &nRhs,
                &self.iparm[0], &self.msglvl, &ddum, &ddum, &error, &self.dparm[0])

        self.ptIsActive = False
        self.hasSymbolicFactorization = False
        self.hasNumericFactorization = False
        self.currentIndices = None
        self.currentIndptr = None
        self.currentData = None

    cdef bint _hasSamePattern(self, A):
        """Check if the sparsity pattern of A matches the analyzed one."""
        if not self.hasSymbolicFactorization:
            return False

        indices = A.indices
        indptr = A.indptr

        # fast path: in-place assembly reuses the identical pattern arrays
        if indices is self.currentIndices and indptr is self.currentIndptr:
            return True

        if (
            A.shape[0] == self.rows
            and np.array_equal(indptr, self.currentIndptr)
            and np.array_equal(indices, self.currentIndices)
        ):
            # same pattern in new arrays; adopt them for future identity checks
            self.currentIndices = indices
            self.currentIndptr = indptr
            return True

        return False

    cdef int _analyze(self, A) except -1:
        """Run reordering and symbolic factorization (phase 11) for the pattern of A."""
        cdef int phase = 11
        cdef int error = 0
        cdef int nRhs = 1
        cdef int idum = 0
        cdef double ddum = 0

        self._releaseMemory()

        if A.nnz > np.iinfo(np.intc).max:
            raise ValueError(
                "matrix has {:} nonzeros, exceeding the 32-bit PARDISO interface".format(A.nnz)
            )
        if A.shape[0] > np.iinfo(np.intc).max:
            raise ValueError(
                "matrix has {:} rows, exceeding the 32-bit PARDISO interface".format(A.shape[0])
            )

        self.rows = A.shape[0]
        # pardiso uses fortran 1-based indexing; scipy may use int64 index arrays for
        # large matrices, so cast explicitly to the 32-bit interface type.
        #
        # Timed separately from phase 11 itself because this is a pure O(nnz) allocate-and-copy that
        # is paid on *every* solve whenever symbolic reuse is off -- an avoidable cost that has
        # nothing to do with the reordering it precedes, and would otherwise hide inside it.
        with performancetiming.timeit("panua pardiso index preparation"):
            self.indicesFortran = np.ascontiguousarray(A.indices + 1, dtype=np.intc)
            self.indptrFortran = np.ascontiguousarray(A.indptr + 1, dtype=np.intc)

        self._initializeSolver()

        cdef double[::1] data = A.data

        with performancetiming.timeit("panua pardiso phase 11 (reorder + symbolic factorization)"):
            pardiso(self.pt, &self.maxfct, &self.mnum, &self.mtype, &phase,
                    &self.rows, &data[0], &self.indptrFortran[0], &self.indicesFortran[0], &idum, &nRhs,
                    &self.iparm[0], &self.msglvl, &ddum, &ddum, &error, &self.dparm[0])

        if error != 0:
            self._releaseMemory()
            raise RuntimeError("Panua PARDISO analysis failed with error code {:}".format(error))

        self.hasSymbolicFactorization = True
        self.currentIndices = A.indices
        self.currentIndptr = A.indptr

        return 0

    def __call__(self, A, b):
        """
        Solve a linear system of equations.

        Parameters
        ----------
        A : csr_matrix
            The system matrix.
        b : ndarray
            The right-hand side vector (or matrix for multiple right-hand sides).

        Returns
        -------
        ndarray
            The solution vector.
        """

        # if reuse is disabled, always re-analyze; _hasSamePattern is not even
        # evaluated in that case (see the class docstring for why reuse is opt-in)
        if not self.reuseSymbolicFactorization or not self._hasSamePattern(A):
            self._analyze(A)

        cdef double[::1] data = A.data

        # prepare rhs and solution
        cdef double[::1, :] b_ = np.asfortranarray(b.reshape((self.rows, -1)))
        cdef int nRhs = b_.shape[1]
        cdef double[::1, :] x = np.zeros_like(b_, order="F")

        cdef int phase
        cdef int error = 0
        cdef int idum = 0
        cdef double ddum = 0

        # numerical factorization
        phase = 22
        with performancetiming.timeit("panua pardiso phase 22 (numeric factorization)"):
            pardiso(self.pt, &self.maxfct, &self.mnum, &self.mtype, &phase,
                    &self.rows, &data[0], &self.indptrFortran[0], &self.indicesFortran[0], &idum, &nRhs,
                    &self.iparm[0], &self.msglvl, &ddum, &ddum, &error, &self.dparm[0])

        if error == 0:
            # back substitution and iterative refinement
            phase = 33
            with performancetiming.timeit("panua pardiso phase 33 (back substitution)"):
                pardiso(self.pt, &self.maxfct, &self.mnum, &self.mtype, &phase,
                        &self.rows, &data[0], &self.indptrFortran[0], &self.indicesFortran[0], &idum, &nRhs,
                        &self.iparm[0], &self.msglvl, &b_[0, 0], &x[0, 0], &error, &self.dparm[0])

        if error != 0:
            # signal failure via NaNs; the nonlinear solvers translate this into a cutback
            np.asarray(x).fill(np.nan)

        return np.reshape(x, b.shape)

    def factorize(self, A):
        """
        Analyze (if needed) and numerically factorize A, without solving anything.

        Together with :meth:`solveFactorized` this splits what :meth:`__call__` fuses, so that one
        factorization can serve several right hand sides -- which :meth:`__call__` cannot do, because
        it re-runs the numeric factorization (phase 22) on every call even for an unchanged matrix.

        The motivating use is a lagged (modified Newton) preconditioner: factorize one Newton
        iterate's matrix, then apply that factorization repeatedly inside a Krylov solve on a later,
        slightly different iterate. Whether that pays off is an empirical question about how fast the
        Jacobian drifts; see ``scripts/benchmark_linsolve.py lagged``.

        Parameters
        ----------
        A : csr_matrix
            The system matrix to factorize.

        Raises
        ------
        RuntimeError
            If PARDISO reports a factorization error. Unlike :meth:`__call__`, which signals failure
            by returning NaNs for the nonlinear solvers to turn into a cutback, this raises: there is
            no solution vector to poison, and a caller about to reuse this factorization many times
            needs to know immediately.
        """

        cdef int phase = 22
        cdef int error = 0
        cdef int nRhs = 1
        cdef int idum = 0
        cdef double ddum = 0

        if not self.reuseSymbolicFactorization or not self._hasSamePattern(A):
            self._analyze(A)

        cdef double[::1] data = A.data

        with performancetiming.timeit("panua pardiso phase 22 (numeric factorization)"):
            pardiso(self.pt, &self.maxfct, &self.mnum, &self.mtype, &phase,
                    &self.rows, &data[0], &self.indptrFortran[0], &self.indicesFortran[0], &idum, &nRhs,
                    &self.iparm[0], &self.msglvl, &ddum, &ddum, &error, &self.dparm[0])

        if error != 0:
            self.hasNumericFactorization = False
            self.currentData = None
            raise RuntimeError(
                "Panua PARDISO numeric factorization failed with error code {:}".format(error)
            )

        # Held onto deliberately: phase 33 runs iterative refinement against the matrix values, and
        # for a lagged preconditioner those must stay the values this factorization was built from,
        # not whatever matrix is currently being solved.
        self.currentData = A.data
        self.hasNumericFactorization = True

    def solveFactorized(self, b):
        """
        Apply the factorization stored by :meth:`factorize` to a right hand side (phase 33 only).

        Cheap relative to a factorization -- back substitution is a small fraction of a solve -- which
        is the whole point of separating the two.

        Parameters
        ----------
        b : ndarray
            The right hand side vector (or matrix, for multiple right hand sides).

        Returns
        -------
        ndarray
            The solution, or NaNs if PARDISO reports a back-substitution error.

        Raises
        ------
        RuntimeError
            If no factorization is currently stored.
        """

        if not self.hasNumericFactorization:
            raise RuntimeError("no numeric factorization available; call factorize() first")

        cdef double[::1] data = self.currentData

        cdef double[::1, :] b_ = np.asfortranarray(b.reshape((self.rows, -1)))
        cdef int nRhs = b_.shape[1]
        cdef double[::1, :] x = np.zeros_like(b_, order="F")

        cdef int phase = 33
        cdef int error = 0
        cdef int idum = 0

        with performancetiming.timeit("panua pardiso phase 33 (back substitution)"):
            pardiso(self.pt, &self.maxfct, &self.mnum, &self.mtype, &phase,
                    &self.rows, &data[0], &self.indptrFortran[0], &self.indicesFortran[0], &idum, &nRhs,
                    &self.iparm[0], &self.msglvl, &b_[0, 0], &x[0, 0], &error, &self.dparm[0])

        if error != 0:
            np.asarray(x).fill(np.nan)

        return np.reshape(x, b.shape)

    def invalidate(self):
        """
        Force a fresh reordering and symbolic factorization (PARDISO phase 11) on the
        next solve, regardless of what the array-identity / ``array_equal`` pattern
        check in :meth:`_hasSamePattern` would otherwise conclude.

        Call this whenever the caller knows the sparsity pattern may have changed
        through a channel the automatic detection might not reliably catch — e.g. a
        solver that rebuilds its CSR generator whenever the active domain changes,
        which can happen more often than once per analysis step (unlike EdelweissFE's
        own static-mesh usage, where a fresh instance is constructed once per step and
        the pattern is never actually re-checked against a real change).

        No-op when ``reuseSymbolicFactorization`` is False, since every solve already
        re-analyzes unconditionally in that case.

        Any factorization stored by :meth:`factorize` is dropped as well: it was built on the pattern
        being invalidated, so :meth:`solveFactorized` must not keep applying it.
        """
        self.hasSymbolicFactorization = False
        self.hasNumericFactorization = False
        self.currentData = None


def panuaPardisoSolve(A, b):
    """
    Solve the linear system Ax = b using the Panua PARDISO solver.

    One-shot convenience wrapper around :class:`PanuaPardisoSolver`; for repeated solves
    with an identical sparsity pattern, use a persistent :class:`PanuaPardisoSolver`
    instance to reuse the symbolic factorization.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The matrix A of the linear system.
    b : numpy.ndarray
        The right-hand side of the linear system.

    Returns
    -------
    numpy.ndarray
        The solution x of the linear system.
    """

    return PanuaPardisoSolver()(A, b)
