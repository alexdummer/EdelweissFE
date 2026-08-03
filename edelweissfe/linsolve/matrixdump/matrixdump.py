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

"""A diagnostic linear solver that writes the equation systems it is handed to disk and then
delegates the actual solve to a real linear solver, so the simulation proceeds unchanged.

The purpose is to lift linear-solver experiments out of the finite element run. Comparing solver
variants -- direct vs. iterative, reordering options, preconditioners, thread counts -- by rerunning
the simulation is slow and confounded: every run also repeats mesh generation, adaptive refinement
and element assembly, and a variant that changes the Newton iterates changes the *sequence* of
matrices being compared, so the comparison is no longer like-for-like. Dumping one authentic
sequence of :math:`(A, b)` pairs once and replaying it offline makes the comparison both fast and
controlled: every variant sees byte-identical input.

A *sequence* rather than a single system, because the properties that matter most for a Newton solve
are sequential ones: whether the sparsity pattern is stable enough for a symbolic factorization to
be reused across iterations, and how well a factorization of an earlier iterate works as a
preconditioner for a later one. Neither question can be asked of a single matrix.

Dumps are deliberately capped (see :class:`MatrixDumpSolver`): a coupled 3D system of a few hundred
thousand DOFs runs to over a gigabyte per matrix, so an unbounded dump of a full analysis would fill
a disk long before it finished.
"""

import json
import os

import numpy as np
from scipy.sparse import csr_matrix, save_npz


class MatrixDumpSolver:
    """Write selected equation systems to disk, then solve them with a delegate solver.

    Parameters
    ----------
    directory
        The directory to write the dumps into. Created if missing.
    delegate
        A ``(A, b) -> x`` callable performing the actual solve. The dump is a side effect; the
        simulation's own numerics are entirely the delegate's.
    dumpAt
        The zero-based ordinals of the solves to dump, counted over the lifetime of this solver
        instance. An explicit list is the useful form for capturing one contiguous Newton sequence
        (e.g. ``[10, 11, 12, 13, 14, 15]``); when empty, ``skipFirst`` and ``maxDumps`` select a
        prefix instead.
    skipFirst
        How many initial solves to pass through undumped when ``dumpAt`` is empty. The first solves
        of an analysis are usually the least representative -- a linear-elastic first increment
        rather than a converging nonlinear one.
    maxDumps
        A hard ceiling on the number of dumps written, applied even when ``dumpAt`` asks for more.
        This is a disk-space guard, not a preference: it is the one option that cannot be
        accidentally overridden by a generous ``dumpAt``.
    """

    def __init__(
        self,
        directory: str,
        delegate,
        dumpAt: list[int],
        skipFirst: int,
        maxDumps: int,
    ):
        self._directory = directory
        self._delegate = delegate
        self._dumpAt = set(dumpAt)
        self._skipFirst = skipFirst
        self._maxDumps = maxDumps

        self._solveCounter = 0
        self._dumpCounter = 0

        os.makedirs(self._directory, exist_ok=True)
        self._manifestPath = os.path.join(self._directory, "manifest.jsonl")

    def _shouldDump(self, ordinal: int) -> bool:
        """Decide whether the solve with this ordinal gets dumped."""

        if self._dumpCounter >= self._maxDumps:
            return False

        if self._dumpAt:
            return ordinal in self._dumpAt

        return ordinal >= self._skipFirst

    def _dump(self, ordinal: int, A: csr_matrix, b: np.ndarray):
        """Write one equation system plus a manifest record describing it."""

        matrixPath = os.path.join(self._directory, "A_{:05d}.npz".format(ordinal))
        rhsPath = os.path.join(self._directory, "b_{:05d}.npy".format(ordinal))

        # Uncompressed: these matrices are large, and compression would spend more time than the
        # solve being investigated -- distorting the very run whose timings are being measured.
        save_npz(matrixPath, A, compressed=False)
        np.save(rhsPath, np.asarray(b))

        record = {
            "ordinal": ordinal,
            "rows": int(A.shape[0]),
            "cols": int(A.shape[1]),
            "nnz": int(A.nnz),
            "matrixFile": os.path.basename(matrixPath),
            "rhsFile": os.path.basename(rhsPath),
            "matrixBytes": os.path.getsize(matrixPath),
            # Cheap fingerprints, so an offline replay can assert it is reading back the system that
            # was actually written -- and so the sequence can be recognised: successive Newton
            # iterates of one increment show a monotonically falling residual norm, which is how a
            # dumped range is identified as a converging sequence rather than a set of unrelated
            # systems.
            "rhsNorm": float(np.linalg.norm(np.asarray(b))),
            "diagonalNorm": float(np.linalg.norm(A.diagonal())),
        }

        with open(self._manifestPath, "a") as manifestFile:
            manifestFile.write(json.dumps(record) + "\n")

        self._dumpCounter += 1

        print(
            "matrixdump: wrote solve {:} to {:} ({:} rows, {:} nnz, {:.2f} GB); {:} of {:} dumps used".format(
                ordinal,
                os.path.basename(matrixPath),
                A.shape[0],
                A.nnz,
                record["matrixBytes"] / 1024**3,
                self._dumpCounter,
                self._maxDumps,
            ),
            flush=True,
        )

    def __call__(self, A: csr_matrix, b: np.ndarray) -> np.ndarray:
        """Dump the system if selected, then return the delegate's solution.

        Parameters
        ----------
        A
            The system matrix.
        b
            The right hand side.

        Returns
        -------
        ndarray
            The delegate's solution, unmodified.
        """

        ordinal = self._solveCounter
        self._solveCounter += 1

        if self._shouldDump(ordinal):
            self._dump(ordinal, A, b)

        return self._delegate(A, b)
