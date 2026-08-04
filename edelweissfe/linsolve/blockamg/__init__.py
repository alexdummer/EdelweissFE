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

"""Registry-facing factory for the field-split block-AMG linear solver.

The implementation lives in :mod:`edelweissfe.linsolve.blockamg.blockamg`; see there for the method.
"""

from collections.abc import Callable, Mapping


def createSolver(opts) -> Callable:
    """Create a field-split block-AMG linear solver.

    The factory the ``linsolver`` registry category resolves for the name ``blockamg``. No field
    structure is configured here: it is discovered from the model and pushed in by the nonlinear
    solver (see :class:`~edelweissfe.linsolve.base.FieldStructureAwareLinearSolver`).

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``. All optional (see
        :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`):

        ``outerTol``, ``outerRestart``, ``outerMaxiter``, ``sweeps``, ``symmetric``, ``verbose``
            The outer GMRES and block Gauss-Seidel knobs. ``outerTol``, if given, fixes the outer
            tolerance and disables the Eisenstat--Walker forcing described next.
        ``etaMin``, ``etaMax``, ``ewGamma``, ``ewAlpha``, ``residualGrowthFactor``,
        ``hierarchyStalenessFactor``
            Knobs for the adaptive outer tolerance and the per-field AMG hierarchy reuse across Newton
            iterations -- see :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`.
        ``fieldPreconds``
            Optional mapping of field name (e.g. ``"displacement"``) to an AMGCL preconditioner
            parameter tree, overriding the dimension-based default for that field.

        As with the other factories, a non-mapping ``opts`` is tolerated (the implicit-static solver
        passes ``""`` when no configuration file is given), in which case every default applies.

    Returns
    -------
    Callable
        A :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`, callable as ``(A, b) -> x``.
    """

    from edelweissfe.linsolve.blockamg.blockamg import BlockAMGSolver

    optionMap = opts if isinstance(opts, Mapping) else {}

    kwargs = {}
    for key, cast in (
        ("outerTol", float),
        ("outerRestart", int),
        ("outerMaxiter", int),
        ("sweeps", int),
        ("symmetric", bool),
        ("verbose", bool),
        ("etaMin", float),
        ("etaMax", float),
        ("ewGamma", float),
        ("ewAlpha", float),
        ("residualGrowthFactor", float),
        ("hierarchyStalenessFactor", float),
    ):
        if key in optionMap:
            kwargs[key] = cast(optionMap[key])
    if "fieldPreconds" in optionMap:
        kwargs["fieldPreconds"] = dict(optionMap["fieldPreconds"])

    return BlockAMGSolver(**kwargs)
