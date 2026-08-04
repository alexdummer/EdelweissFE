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

    The factory the ``linsolver`` registry category resolves for the name ``blockamg``.

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``. Because the block
        structure cannot be inferred from ``(A, b)`` alone, a configuration file is **required** and
        must contain ``fields``:

        ``fields``
            The ordered list of field blocks, contiguous and field-major in the DOF vector. Each entry
            is a mapping with ``size`` (int, required), and optionally ``elasticity`` (bool),
            ``components`` (int, for an elasticity block), and ``precond`` (an AMGCL parameter tree
            overriding the default for that block). The field extents are logged by the nonlinear
            solver at equation-system build ("field '...': N dofs, [a, b)").
        ``outerTol``, ``outerRestart``, ``outerMaxiter``, ``sweeps``, ``symmetric``, ``verbose``
            Optional; forwarded to
            :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`.

    Returns
    -------
    Callable
        A :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`, callable as ``(A, b) -> x``.

    Raises
    ------
    ValueError
        If no ``fields`` list is supplied -- the block structure has no safe default.
    """

    from edelweissfe.linsolve.blockamg.blockamg import BlockAMGSolver

    if not isinstance(opts, Mapping) or "fields" not in opts:
        raise ValueError(
            "blockamg requires a linsolverConfigFile with a 'fields' list describing the block "
            "structure (e.g. the displacement and damage field sizes); none was given"
        )

    kwargs = {}
    for key, cast in (
        ("outerTol", float),
        ("outerRestart", int),
        ("outerMaxiter", int),
        ("sweeps", int),
        ("symmetric", bool),
        ("verbose", bool),
    ):
        if key in opts:
            kwargs[key] = cast(opts[key])

    return BlockAMGSolver(list(opts["fields"]), **kwargs)
