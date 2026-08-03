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
#  ---------------------------------------------------------------------

"""Offline benchmark of linear-solver strategies on a captured sequence of equation systems.

Reads what the ``matrixdump`` linear solver wrote (see
:mod:`edelweissfe.linsolve.matrixdump.matrixdump`) and replays it, so solver variants are compared
on byte-identical input instead of by rerunning the simulation. Each check is a separate subcommand
because they cost wildly different amounts of time: ``pattern`` is seconds, ``reuse`` and ``threads``
are minutes, and ``lagged`` involves a SuperLU factorization of the whole system and can run much
longer. Nothing here mutates the dumps.

Usage::

    python benchmark_linsolve.py pattern  <dumpDir>
    python benchmark_linsolve.py reuse    <dumpDir>
    python benchmark_linsolve.py threads  <dumpDir> [--threads 1,4,8,16,36]
    python benchmark_linsolve.py lagged   <dumpDir> [--maxiter 200] [--rtol 1e-8]

``threads`` must be run one thread count per process -- MKL latches its thread count at first use, so
sweeping inside a single process would silently measure the first setting repeatedly. The subcommand
therefore re-executes itself once per value.
"""

import argparse
import json
import os
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def loadManifest(dumpDir: str) -> list[dict]:
    """Read the dump manifest, in capture order."""

    manifestPath = os.path.join(dumpDir, "manifest.jsonl")
    with open(manifestPath) as manifestFile:
        records = [json.loads(line) for line in manifestFile if line.strip()]

    records.sort(key=lambda record: (record["instance"], record["ordinal"]))
    return records


def loadSystem(dumpDir: str, record: dict):
    """Load one dumped system, checking it against its manifest fingerprints."""

    A = sp.load_npz(os.path.join(dumpDir, record["matrixFile"])).tocsr()
    b = np.load(os.path.join(dumpDir, record["rhsFile"]))

    if A.nnz != record["nnz"]:
        raise ValueError(
            "{:}: nnz {:} does not match the manifest's {:}".format(record["matrixFile"], A.nnz, record["nnz"])
        )

    return A, b


def commandPattern(args):
    """Report how the sparsity pattern evolves along the captured sequence.

    This decides whether PARDISO's symbolic factorization can be reused at all: reuse is only
    possible if the pattern is genuinely identical from one solve to the next. The nnz column is the
    cheap tell, and the explicit index comparison is the real one -- two matrices can share an nnz
    count with different patterns.
    """

    records = loadManifest(args.dumpDir)
    print("{:} dumped systems in {:}\n".format(len(records), args.dumpDir))

    print(
        "{:>4} {:>4} {:>12} {:>14} {:>14} {:>10} {:>12}".format(
            "inst", "ord", "nnz", "||b||", "||diag(A)||", "nnz delta", "pattern"
        )
    )

    previousIndices = None
    previousIndptr = None

    for record in records:
        A, b = loadSystem(args.dumpDir, record)

        if previousIndices is None:
            patternNote = "-"
            nnzDelta = "-"
        else:
            samePattern = np.array_equal(A.indices, previousIndices) and np.array_equal(A.indptr, previousIndptr)
            patternNote = "identical" if samePattern else "CHANGED"
            nnzDelta = "{:+}".format(A.nnz - len(previousIndices))

        print(
            "{:>4} {:>4} {:>12} {:>14.6e} {:>14.6e} {:>10} {:>12}".format(
                record["instance"],
                record["ordinal"],
                A.nnz,
                record["rhsNorm"],
                record["diagonalNorm"],
                nnzDelta,
                patternNote,
            )
        )

        previousIndices = A.indices.copy()
        previousIndptr = A.indptr.copy()

    # Structural symmetry decides whether the cheaper structurally-symmetric PARDISO matrix types are
    # even applicable, and how much fill-in the reordering has to guess at. Measured on the last
    # system loaded.
    pattern = sp.csr_matrix((np.ones(A.nnz), A.indices, A.indptr), shape=A.shape)
    asymmetric = (pattern - pattern.T).nnz
    print(
        "\nlast system: {:} x {:}, {:} nnz ({:.1f} per row), {:} structurally asymmetric entries".format(
            A.shape[0], A.shape[1], A.nnz, A.nnz / A.shape[0], asymmetric
        )
    )
    print("explicitly stored zeros: {:}".format(int(np.count_nonzero(A.data == 0.0))))


def commandReuse(args):
    """Time and cross-check PARDISO with symbolic-factorization reuse on versus off.

    Reuse is opt-in precisely because it has been observed to return wrong-but-finite results on
    coupled problems, so this does not just time the two: it compares their solutions against the
    no-reuse result on every system of the sequence. A speedup is only worth anything if the answers
    agree, and this is the cheapest place to find out -- the sequence is real, and no simulation has
    to be rerun to test it.
    """

    from edelweissfe.linsolve.pardiso.pardiso import PardisoSolver

    records = loadManifest(args.dumpDir)
    systems = [loadSystem(args.dumpDir, record) for record in records]

    print("MKL threads: {:}\n".format(os.environ.get("OMP_NUM_THREADS", "(unset)")))

    referenceSolutions = []
    print("--- reuse OFF (current default) ---")
    solverOff = PardisoSolver(reuseSymbolicFactorization=False)
    for record, (A, b) in zip(records, systems):
        start = perf_counter()
        x = solverOff(A, b)
        elapsed = perf_counter() - start
        residual = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1.0e-300)
        referenceSolutions.append(x)
        print("  ord {:>3}: {:8.3f} s   rel. residual {:9.3e}".format(record["ordinal"], elapsed, residual))

    print("\n--- reuse ON (one persistent solver across the sequence) ---")
    solverOn = PardisoSolver(reuseSymbolicFactorization=True)
    for record, (A, b), reference in zip(records, systems, referenceSolutions):
        start = perf_counter()
        x = solverOn(A, b)
        elapsed = perf_counter() - start
        residual = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1.0e-300)
        deviation = np.linalg.norm(x - reference) / max(np.linalg.norm(reference), 1.0e-300)
        flag = "" if deviation < 1.0e-10 else "   <-- DIVERGES FROM NO-REUSE RESULT"
        print(
            "  ord {:>3}: {:8.3f} s   rel. residual {:9.3e}   rel. dev. {:9.3e}{:}".format(
                record["ordinal"], elapsed, residual, deviation, flag
            )
        )


def commandThreads(args):
    """Measure how a single PARDISO solve scales with MKL thread count.

    Re-executes itself once per thread count: MKL fixes its thread count on first use, so a sweep
    inside one process would report the first setting for every entry. Answers what the production
    ``OMP_NUM_THREADS`` should be, and -- read against the single-threaded SciPy SpGEMMs of the MPC
    condensation -- how the serial fraction of an increment grows as cores are added.
    """

    if args.child is None:
        threadCounts = [int(value) for value in args.threads.split(",")]
        for threadCount in threadCounts:
            environment = dict(os.environ)
            environment["OMP_NUM_THREADS"] = str(threadCount)
            environment["MKL_NUM_THREADS"] = str(threadCount)
            subprocess.run(
                [sys.executable, __file__, "threads", args.dumpDir, "--child", str(threadCount)],
                env=environment,
                check=True,
            )
        return

    from edelweissfe.linsolve.pardiso.pardiso import PardisoSolver

    records = loadManifest(args.dumpDir)
    A, b = loadSystem(args.dumpDir, records[0])

    solver = PardisoSolver(reuseSymbolicFactorization=False)
    solver(A, b)  # warm up: first call also pays page faults on the freshly loaded arrays

    timings = []
    for _ in range(args.repeats):
        start = perf_counter()
        solver(A, b)
        timings.append(perf_counter() - start)

    print(
        "threads {:>3}: best {:8.3f} s   median {:8.3f} s   (of {:} repeats)".format(
            args.child, min(timings), sorted(timings)[len(timings) // 2], args.repeats
        )
    )


def commandLagged(args):
    """Count GMRES iterations when an earlier iterate's LU is reused as a preconditioner.

    This is the make-or-break number for a modified Newton-Krylov scheme: if the factorization of
    system k preconditions system k+j well enough that GMRES converges in a handful of matrix-vector
    products, one expensive factorization can serve several Newton iterations, and the cost per
    iteration collapses. If the count instead climbs steeply with staleness j, the whole route is
    dead and the effort belongs on the direct solve.

    SuperLU rather than PARDISO does the factorization here on purpose. It is the slower factorizer,
    but the iteration count depends only on how well the operator ``A_k^-1 A_(k+j)`` is conditioned,
    which is a property of the matrices and not of whichever LU implementation inverted the first
    one. So this measures the decisive quantity without needing a phase-33-only entry point in the
    PARDISO wrapper -- that is an optimization worth adding only if this result is favourable.
    """

    records = loadManifest(args.dumpDir)
    systems = [loadSystem(args.dumpDir, record) for record in records]

    baseRecord, (baseMatrix, _) = records[0], systems[0]
    print(
        "factorizing the base system (instance {:}, ordinal {:}) with SuperLU ...".format(
            baseRecord["instance"], baseRecord["ordinal"]
        )
    )
    start = perf_counter()
    baseFactorization = spla.splu(baseMatrix.tocsc())
    print("  done in {:.1f} s\n".format(perf_counter() - start))

    preconditioner = spla.LinearOperator(baseMatrix.shape, matvec=baseFactorization.solve, dtype=baseMatrix.dtype)

    print("GMRES preconditioned by the base factorization, rtol={:}, maxiter={:}".format(args.rtol, args.maxiter))
    print("{:>4} {:>10} {:>8} {:>12} {:>14}".format("ord", "staleness", "iters", "converged", "rel. residual"))

    for offset, (record, (A, b)) in enumerate(zip(records, systems)):
        iterationCount = 0

        def countIteration(_):
            nonlocal iterationCount
            iterationCount += 1

        x, info = spla.gmres(
            A,
            b,
            M=preconditioner,
            rtol=args.rtol,
            atol=0.0,
            restart=args.restart,
            maxiter=args.maxiter,
            callback=countIteration,
            callback_type="pr_norm",
        )

        residual = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1.0e-300)
        print(
            "{:>4} {:>10} {:>8} {:>12} {:>14.3e}".format(
                record["ordinal"],
                offset,
                iterationCount,
                "yes" if info == 0 else "NO (info={:})".format(info),
                residual,
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    patternParser = subparsers.add_parser("pattern", help="report sparsity-pattern evolution")
    patternParser.add_argument("dumpDir")
    patternParser.set_defaults(function=commandPattern)

    reuseParser = subparsers.add_parser("reuse", help="time and cross-check symbolic-factorization reuse")
    reuseParser.add_argument("dumpDir")
    reuseParser.set_defaults(function=commandReuse)

    threadsParser = subparsers.add_parser("threads", help="MKL thread scaling of one solve")
    threadsParser.add_argument("dumpDir")
    threadsParser.add_argument("--threads", default="1,4,8,16,36")
    threadsParser.add_argument("--repeats", type=int, default=3)
    threadsParser.add_argument("--child", type=int, default=None, help=argparse.SUPPRESS)
    threadsParser.set_defaults(function=commandThreads)

    laggedParser = subparsers.add_parser("lagged", help="GMRES iterations with a lagged LU preconditioner")
    laggedParser.add_argument("dumpDir")
    laggedParser.add_argument("--maxiter", type=int, default=200)
    laggedParser.add_argument("--restart", type=int, default=50)
    laggedParser.add_argument("--rtol", type=float, default=1.0e-8)
    laggedParser.set_defaults(function=commandLagged)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
