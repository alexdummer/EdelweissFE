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

"""Registry-facing factory for the Panua PARDISO solver.

The implementation lives in the Cython extension
:mod:`edelweissfe.linsolve.panuapardiso.panuapardiso`, which is optional -- it needs the Panua
PARDISO library at build time. The factory below therefore imports it lazily; see
:func:`createSolver`.
"""

from collections.abc import Callable, Mapping

from edelweissfe.linsolve.base import LinearSolver


class PanuaPardisoLinearSolver(LinearSolver):
    """Thin wrapper giving the Cython
    :class:`~edelweissfe.linsolve.panuapardiso.panuapardiso.PanuaPardisoSolver` extension type the
    common :class:`~edelweissfe.linsolve.base.LinearSolver` interface
    (``setJournal``/``setFieldStructure``) without touching the ``.pyx`` itself -- a plain Python
    subclass keeps the Cython extension's build untouched and this wrapper trivially testable without
    a rebuild.

    Also forwards ``factorize``/``solveFactorized`` (the two-method contract
    :class:`~edelweissfe.linsolve.inexactnewton.inexactnewton.InexactNewtonSolver` uses instead of the
    plain ``(A, b) -> x`` call), exactly as
    :class:`~edelweissfe.linsolve.pardiso.PardisoLinearSolver` does: ``inexactnewton`` resolves its
    factorizing delegate through this same registry seam, so a backend that can serve as that
    delegate has to satisfy both contracts, not just the ``LinearSolver`` one.
    """

    def __init__(self, delegate):
        self._delegate = delegate

    def __call__(self, A, b):
        return self._delegate(A, b)

    def factorize(self, A):
        return self._delegate.factorize(A)

    def solveFactorized(self, b):
        return self._delegate.solveFactorized(b)


def createSolver(opts) -> Callable:
    """Create a Panua-PARDISO-backed linear solver.

    The factory the ``linsolver`` registry category resolves for the name ``panuapardiso`` (see
    ``PLAN_INPUT_SYSTEM.md`` §9): every ``linsolve`` subpackage exposes this one signature, so that
    a third party can contribute a linear solver through an entry point.

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``. Only
        ``opts["reuseSymbolicFactorization"]`` is consulted, defaulting to ``False``; ``opts`` is
        tolerated as any non-mapping (the implicit-static solver passes ``""`` when no configuration
        file is given), in which case that default applies.

    Returns
    -------
    Callable
        A :class:`PanuaPardisoLinearSolver` wrapping a
        :class:`~edelweissfe.linsolve.panuapardiso.panuapardiso.PanuaPardisoSolver` instance, callable
        as ``(A, b) -> x``.

    Raises
    ------
    ImportError
        If the optional ``panuapardiso`` extension was not built -- raised here, at construction
        time (matching the pre-refactor behaviour), not deferred to the first solve.
    """

    # Imported inside the function body, not at module scope: this extension is optional and
    # genuinely absent in most installs, so at module scope its absence would break anyone who
    # merely resolves a `linsolver` registry name rather than this one.
    from edelweissfe.linsolve.panuapardiso.panuapardiso import PanuaPardisoSolver

    # Symbolic-factorization reuse across solves is only correct if the caller
    # can guarantee the sparsity pattern stays genuinely stable for the solver
    # instance's entire lifetime; with the MKL backend it has been observed to silently
    # produce wrong (but not NaN, so undetected by the usual failure check) results for
    # some coupled-DOF problems, and nothing about that is MKL-specific. Off by
    # default; opt in explicitly via opts["reuseSymbolicFactorization"] = True once
    # that has been verified safe for the problem at hand.
    reuseSymbolicFactorization = (
        bool(opts.get("reuseSymbolicFactorization", False)) if isinstance(opts, Mapping) else False
    )

    return PanuaPardisoLinearSolver(
        PanuaPardisoSolver(reuseSymbolicFactorization=reuseSymbolicFactorization)
    )
