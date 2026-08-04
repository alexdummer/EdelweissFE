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

"""Registry-facing factory for the inexact-Newton (lagged-LU-preconditioned GMRES) linear solver.

The implementation lives in :mod:`edelweissfe.linsolve.inexactnewton.inexactnewton`; see there for
what this is for and how the refactorization policy works.
"""

from collections.abc import Callable, Mapping


def createSolver(opts) -> Callable:
    """Create an inexact-Newton linear solver.

    The factory the ``linsolver`` registry category resolves for the name ``inexactnewton``: every
    ``linsolve`` subpackage exposes this one signature.

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``. Recognized keys,
        all optional (see :class:`~edelweissfe.linsolve.inexactnewton.inexactnewton.InexactNewtonSolver`
        for the meaning and defaults):

        ``delegate``
            The name of the factorizing backend that supplies the lagged LU: ``"pardiso"`` (default,
            the intended production backend) or ``"superlu"`` (SciPy, dependency-free, for testing or
            installs without the PARDISO extension).
        ``maxReuse``, ``residualGrowthFactor``, ``etaMin``, ``etaMax``, ``ewGamma``, ``ewAlpha``,
        ``gmresRestart``, ``gmresMaxOuter``, ``staleIterationThreshold``, ``verbose``
            The policy and GMRES knobs, forwarded to the solver's constructor.

        As with the other factories, a non-mapping ``opts`` is tolerated (the implicit-static solver
        passes ``""`` when no configuration file is given), in which case every default applies -- the
        default PARDISO delegate with the measured-sweet-spot policy, which is a sensible turnkey
        configuration.

    Returns
    -------
    Callable
        An :class:`~edelweissfe.linsolve.inexactnewton.inexactnewton.InexactNewtonSolver` instance,
        callable as ``(A, b) -> x``.

    Raises
    ------
    ImportError
        If the PARDISO delegate is requested but its optional extension was not built.
    """

    # Imported inside the function body to match every other factory in this package: `config.linsolve`
    # imports the registry, and the registry is what resolves *this* module -- a module-scope import
    # would close that loop. The delegate imports are likewise deferred so that merely resolving this
    # name never forces the optional PARDISO extension to load.
    from edelweissfe.linsolve.inexactnewton.inexactnewton import (
        InexactNewtonSolver,
        _SuperLUFactorization,
    )

    optionMap = opts if isinstance(opts, Mapping) else {}

    delegateName = optionMap.get("delegate", "pardiso")
    if delegateName == "pardiso":
        # Routed through the registry seam rather than importing PardisoSolver directly, so there is
        # one place that knows how a pardiso solver is built. Reuse of the symbolic factorization is
        # switched on: it is safe here because a lagged factorization only ever refactorizes a genuinely
        # new matrix, and it lets PARDISO skip the reordering (phase 11) whenever a refactorization
        # happens to land on an unchanged pattern.
        from edelweissfe.config.linsolve import getLinSolverByName

        delegate = getLinSolverByName("pardiso", {"reuseSymbolicFactorization": True})
    elif delegateName == "superlu":
        delegate = _SuperLUFactorization()
    else:
        raise ValueError("inexactnewton: unknown delegate {!r}; expected 'pardiso' or 'superlu'".format(delegateName))

    kwargs = {}
    for key, cast in (
        ("maxReuse", int),
        ("residualGrowthFactor", float),
        ("etaMin", float),
        ("etaMax", float),
        ("ewGamma", float),
        ("ewAlpha", float),
        ("gmresRestart", int),
        ("gmresMaxOuter", int),
        ("staleIterationThreshold", int),
        ("verbose", bool),
    ):
        if key in optionMap:
            kwargs[key] = cast(optionMap[key])

    return InexactNewtonSolver(delegate, **kwargs)
