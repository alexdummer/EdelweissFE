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

import json

import numpy as np
import scipy.sparse

cimport numpy as np


cdef class PyAMGCLSolver:
    cdef LinearSolver* solver
    cdef LinearSolverFloat* solverFloat
    cdef LinearSolverBlock2* solverBlock2
    cdef LinearSolverBlock3* solverBlock3
    cdef readonly bint isFloat
    """True if this instance runs the mixed-precision (builtin<float>) hierarchy, see §19.3."""
    cdef readonly int blockSize
    """1 (scalar, default), 2, or 3 -- the AMGCL backend's block size, see §20.1."""
    cdef readonly int lastIterations
    """The iteration count AMGCL reported for the most recent solve, -1 before the first."""
    cdef readonly double lastError
    """The relative residual AMGCL reported for the most recent solve, NaN before the first."""

    def __cinit__(self, dict params=None):
        """
        Initialize with parameters only.

        The AMG smoother key is ``relax``, not ``relaxation``; AMGCL ignores unknown keys with only a
        warning on stderr, so a misspelling here silently runs the default smoother instead of the
        requested one.

        ``backendPrecision``: ``"double"`` (default) or ``"float"``. Selects the AMGCL backend's value
        type -- ``"float"`` halves the memory traffic of the hierarchy build and, more importantly, of
        every :meth:`applyPreconditioner` call (the dominant cost on large coupled solves, §18). Not an
        AMGCL parameter itself, so it is popped out before the rest of ``params`` is forwarded as JSON.
        The near null-space (:meth:`set_nullspace`) always stays double regardless -- AMGCL's own
        ``coarsening::nullspace_params::B`` is hardcoded double, independent of the backend.

        ``backendBlockSize``: ``1`` (default, scalar), ``2``, or ``3``. Selects a block-valued backend
        (§20.1, B3) operating on node-major B x B nodal blocks instead of scalar entries -- shrinks the
        CSR index traffic by ~B² and lets block-aware smoothers (block-ILU0, block-GS) invert each
        node's coupling exactly. The matrix still arrives as a plain scalar CSR; the wrapper adapts it
        internally. Not yet combinable with ``backendPrecision == "float"`` (raises) -- float-block is
        a follow-up, not implemented in this pass. :meth:`set_nullspace` always raises on a block
        backend -- AMGCL's own near-null-space path is unimplemented for block value types.

        Example params:
        {
            "solver": {"type": "bicgstab", "tol": 1e-6},
            "precond": {"coarsening": {"type": "smoothed_aggregation"}, "relax": {"type": "ilu0"}}
        }
        """
        if params is None:
            # Default to Robust Blackbox (BiCGStab + ILU0)
            params = {
                "solver": {"type": "bicgstab", "tol": 1e-6},
                "precond": {
                    "coarsening": {"type": "smoothed_aggregation"},
                    "relax": {"type": "ilu0"}
                }
            }
        else:
            params = dict(params)

        backendPrecision = params.pop("backendPrecision", "double")
        if backendPrecision not in ("double", "float"):
            raise ValueError(f"backendPrecision must be 'double' or 'float', got {backendPrecision!r}")
        self.isFloat = backendPrecision == "float"

        blockSize = params.pop("backendBlockSize", 1)
        if blockSize not in (1, 2, 3):
            raise ValueError(f"backendBlockSize must be 1, 2, or 3, got {blockSize!r}")
        self.blockSize = blockSize
        if self.blockSize > 1 and self.isFloat:
            raise NotImplementedError(
                "backendBlockSize > 1 combined with backendPrecision='float' is not implemented "
                "(§20.1: float-block is a follow-up column, not this pass's target)."
            )

        self.lastIterations = -1
        self.lastError = float("nan")

        # Convert dict to JSON string for C++
        cdef bytes json_bytes = json.dumps(params).encode("utf-8")
        cdef const char* c_json = json_bytes
        self.solver = NULL
        self.solverFloat = NULL
        self.solverBlock2 = NULL
        self.solverBlock3 = NULL
        if self.blockSize == 2:
            self.solverBlock2 = new LinearSolverBlock2(c_json)
        elif self.blockSize == 3:
            self.solverBlock3 = new LinearSolverBlock3(c_json)
        elif self.isFloat:
            self.solverFloat = new LinearSolverFloat(c_json)
        else:
            self.solver = new LinearSolver(c_json)

    def __dealloc__(self):
        if self.solver != NULL:
            del self.solver
        if self.solverFloat != NULL:
            del self.solverFloat
        if self.solverBlock2 != NULL:
            del self.solverBlock2
        if self.solverBlock3 != NULL:
            del self.solverBlock3

    def set_nullspace(self, object B):
        """
        Supply near null-space vectors for smoothed-aggregation coarsening.

        AMGCL's smoothed aggregation defaults to a single constant near null-space vector, which is
        wrong for an elasticity operator -- there the near null-space is the rigid body modes. AMGCL
        takes these vectors as a raw pointer in its property tree, so they cannot be passed through the
        JSON parameter string like every other option; this is the separate entry point for them.

        B: an (n, cols) array-like; column j is the j-th near null-space vector (float64). Copied into
        the C++ solver, so the array need not be kept alive. Must be called before the first solve().
        Passing an (n, 0) array clears any previously set null-space. Always double, regardless of
        ``backendPrecision`` -- see the class docstring. Raises on a block backend (``blockSize > 1``)
        -- AMGCL's own near-null-space path is unimplemented for block value types (§20.1).
        """
        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] B_arr = np.ascontiguousarray(B, dtype=np.float64)
        cdef int rows = B_arr.shape[0]
        cdef int cols = B_arr.shape[1]
        cdef double[:, ::1] B_view
        if cols == 0:
            if self.blockSize == 2:
                self.solverBlock2.set_nullspace(<const double*> 0, rows, 0)
            elif self.blockSize == 3:
                self.solverBlock3.set_nullspace(<const double*> 0, rows, 0)
            elif self.isFloat:
                self.solverFloat.set_nullspace(<const double*> 0, rows, 0)
            else:
                self.solver.set_nullspace(<const double*> 0, rows, 0)
            return
        B_view = B_arr
        if self.blockSize == 2:
            self.solverBlock2.set_nullspace(&B_view[0, 0], rows, cols)
        elif self.blockSize == 3:
            self.solverBlock3.set_nullspace(&B_view[0, 0], rows, cols)
        elif self.isFloat:
            self.solverFloat.set_nullspace(&B_view[0, 0], rows, cols)
        else:
            self.solver.set_nullspace(&B_view[0, 0], rows, cols)

    def build(self, object A):
        """
        Build the AMG hierarchy for A once, for repeated preconditioner application via
        :meth:`applyPreconditioner` -- the build-once / apply-many split :meth:`solve` fuses.

        A: scipy.sparse.csr_matrix. Its values narrow to float32 first if ``backendPrecision ==
        "float"`` -- cheap here (build() runs once per solve, not once per outer Krylov iteration). A
        block backend (``blockSize > 1``) instead takes the plain scalar CSR unchanged and adapts it
        to node-major blocks inside the C++ wrapper.
        """
        if not scipy.sparse.isspmatrix_csr(A):
            A = A.tocsr()

        cdef int n = A.shape[0]
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        cdef int[::1] indptr_ = indptr
        cdef int[::1] indices_ = indices

        cdef np.ndarray[np.float32_t, ndim=1, mode="c"] dataFloat
        cdef float[::1] dataFloat_
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] data
        cdef double[::1] data_

        if self.blockSize == 2 or self.blockSize == 3:
            data = np.ascontiguousarray(A.data, dtype=np.float64)
            data_ = data
            if self.blockSize == 2:
                self.solverBlock2.build(n, &indptr_[0], &indices_[0], &data_[0])
            else:
                self.solverBlock3.build(n, &indptr_[0], &indices_[0], &data_[0])
        elif self.isFloat:
            dataFloat = np.ascontiguousarray(A.data, dtype=np.float32)
            dataFloat_ = dataFloat
            self.solverFloat.build(n, &indptr_[0], &indices_[0], &dataFloat_[0])
        else:
            data = np.ascontiguousarray(A.data, dtype=np.float64)
            data_ = data
            self.solver.build(n, &indptr_[0], &indices_[0], &data_[0])

    def applyPreconditioner(self, object rhs):
        """
        Apply one AMG cycle of the hierarchy built by :meth:`build` to rhs: returns M^-1 rhs.

        rhs: array-like, converted to 1D float64 (C-contiguous). Always double at this boundary --
        any backend-specific narrowing/widening or block reinterpretation happens inside the C++
        wrapper, not here, since this is called once per outer Krylov iteration.
        """
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] rhs_arr = np.ascontiguousarray(rhs, dtype=np.float64)
        cdef int n = rhs_arr.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x = np.zeros(n, dtype=np.float64)

        cdef double[::1] rhs_ = rhs_arr
        cdef double[::1] x_ = x
        if self.blockSize == 2:
            self.solverBlock2.applyPreconditioner(n, &rhs_[0], &x_[0])
        elif self.blockSize == 3:
            self.solverBlock3.applyPreconditioner(n, &rhs_[0], &x_[0])
        elif self.isFloat:
            self.solverFloat.applyPreconditioner(n, &rhs_[0], &x_[0])
        else:
            self.solver.applyPreconditioner(n, &rhs_[0], &x_[0])
        return x

    def report(self):
        """
        Human-readable AMGCL hierarchy report (levels, operator complexity, coarse size) for the most
        recently built() or solve()'d hierarchy.

        Raises RuntimeError if nothing has been built yet.
        """
        if self.blockSize == 2:
            return self.solverBlock2.report().decode("utf-8")
        if self.blockSize == 3:
            return self.solverBlock3.report().decode("utf-8")
        if self.isFloat:
            return self.solverFloat.report().decode("utf-8")
        return self.solver.report().decode("utf-8")

    def solve(self, object A, object rhs):
        """
        A: scipy.sparse.csr_matrix
        rhs: array-like, will be converted to 1D float64 (C-contiguous)
        """
        if not scipy.sparse.isspmatrix_csr(A):
            A = A.tocsr()

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] rhs_arr = np.asarray(rhs, dtype=np.float64, order="C")
        if rhs_arr.ndim != 1:
            raise ValueError("rhs must be a 1D array-like")

        cdef int n = A.shape[0]
        if rhs_arr.shape[0] != n:
            raise ValueError(f"Dimension mismatch: Matrix {n}x{n}, RHS {rhs_arr.shape[0]}")
        if n > np.iinfo(np.int32).max:
            raise OverflowError(f"Matrix dimension {n} exceeds supported int32 range for AMGCL wrapper")

        cdef long long int32_max = np.iinfo(np.int32).max
        cdef long long nnz = A.nnz
        if nnz > int32_max:
            raise OverflowError(f"CSR nnz value {nnz} exceeds supported int32 range for AMGCL wrapper")
        if int(A.indptr.min()) < 0:
            raise OverflowError("CSR indptr contains negative values, which are unsupported")
        if A.indices.size > 0:
            if int(A.indices.max()) > int32_max:
                raise OverflowError("CSR indices contain values that exceed int32 range for AMGCL wrapper")
            if int(A.indices.min()) < 0:
                raise OverflowError("CSR indices contain negative values, which are unsupported")

        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indptr = A.indptr.astype(np.int32, copy=False)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indices = A.indices.astype(np.int32, copy=False)

        # Ensure contiguous (astype might return non-contiguous if copy=False was possible)
        if not indptr.flags["C_CONTIGUOUS"]:
            indptr = np.ascontiguousarray(indptr)
        if not indices.flags["C_CONTIGUOUS"]:
            indices = np.ascontiguousarray(indices)

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x = np.zeros(n, dtype=np.float64)

        cdef int iters = 0
        cdef double error = 0.0

        cdef int[::1] indptr_ = indptr
        cdef int[::1] indices_ = indices
        cdef double[::1] rhs_ = rhs_arr
        cdef double[::1] x_ = x

        cdef np.ndarray[np.float32_t, ndim=1, mode="c"] dataFloat
        cdef float[::1] dataFloat_
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] data
        cdef double[::1] data_

        if self.blockSize == 2 or self.blockSize == 3:
            data = np.ascontiguousarray(A.data, dtype=np.float64)
            data_ = data
            if self.blockSize == 2:
                self.solverBlock2.solve(
                        n, &indptr_[0], &indices_[0], &data_[0], &rhs_[0], &x_[0], iters, error
                    )
            else:
                self.solverBlock3.solve(
                        n, &indptr_[0], &indices_[0], &data_[0], &rhs_[0], &x_[0], iters, error
                    )
        elif self.isFloat:
            dataFloat = np.ascontiguousarray(A.data, dtype=np.float32)
            dataFloat_ = dataFloat
            self.solverFloat.solve(
                    n,
                    &indptr_[0],
                    &indices_[0],
                    &dataFloat_[0],
                    &rhs_[0],
                    &x_[0],
                    iters,
                    error
                )
        else:
            data = np.ascontiguousarray(A.data, dtype=np.float64)
            data_ = data
            self.solver.solve(
                    n,
                    &indptr_[0],
                    &indices_[0],
                    &data_[0],
                    &rhs_[0],
                    &x_[0],
                    iters,
                    error
                )

        # AMGCL reports these by reference and they were being dropped on the floor, which left an
        # unconverged solve indistinguishable from a converged one: an iterative solver that hits
        # maxiter still returns a finite x, so the nonlinear solvers' NaN check does not catch it and
        # the Newton loop silently consumes a wrong correction. Surfaced here so callers can check.
        self.lastIterations = iters
        self.lastError = error

        return x


cdef class PyAMGCLRelaxationSmoother:
    """A standalone, OpenMP-threaded relaxation smoother (§22.4) -- e.g. the p-two-grid
    preconditioner's fine sweep (:mod:`edelweissfe.linsolve.blockamg.ptwogrid`), one level below a
    full AMG hierarchy. Wraps ``amgcl::runtime::relaxation::wrapper``, which is itself
    runtime-selectable via the ``type`` key (``chebyshev``, ``gauss_seidel``, ``ilu0``, ``spai0``,
    ...) -- the same smoother catalogue as a hierarchy's own ``relax`` sub-tree, just applied directly
    as a preconditioner-shaped object instead of on an AMG level.

    Unlike :class:`PyAMGCLSolver`, this exposes a single in-place smoothing step
    (:meth:`applyStep`, ``amgcl``'s ``apply_pre``) rather than a full solve: callers decide "start
    from zero" (zero ``x`` themselves) and "how many sweeps" (call :meth:`applyStep` repeatedly) --
    see :mod:`ptwogrid`'s ``smooth(x, rhs, sweeps)`` closure.
    """
    cdef RelaxationSmoother* smoother

    def __cinit__(self, dict params):
        """
        params: the relaxation's own flat parameter tree, e.g.
        {"type": "chebyshev", "degree": 5, "power_iters": 50, "lower": 0.01}
        -- not nested under "precond.relax" like :class:`PyAMGCLSolver`, since there is no hierarchy
        here.
        """
        cdef bytes json_bytes = json.dumps(params).encode("utf-8")
        cdef const char* c_json = json_bytes
        self.smoother = new RelaxationSmoother(c_json)

    def __dealloc__(self):
        if self.smoother != NULL:
            del self.smoother

    def build(self, object A):
        """Build the smoother for A once, for repeated :meth:`applyStep` calls. A: scipy.sparse
        csr_matrix."""
        if not scipy.sparse.isspmatrix_csr(A):
            A = A.tocsr()

        cdef int n = A.shape[0]
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] data = np.ascontiguousarray(A.data, dtype=np.float64)
        cdef int[::1] indptr_ = indptr
        cdef int[::1] indices_ = indices
        cdef double[::1] data_ = data
        self.smoother.build(n, &indptr_[0], &indices_[0], &data_[0])

    def applyStep(self, object x, object rhs):
        """One in-place smoothing step: ``x`` is updated in place, continuing the relaxation's own
        recursion from its current value (zero it first for a from-zero application). ``x`` must be a
        1D float64 (C-contiguous) numpy array; ``rhs`` is converted to the same."""
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x_arr = x
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] rhs_arr = np.ascontiguousarray(rhs, dtype=np.float64)
        cdef int n = x_arr.shape[0]
        cdef double[::1] x_ = x_arr
        cdef double[::1] rhs_ = rhs_arr
        self.smoother.applyStep(n, &rhs_[0], &x_[0])
