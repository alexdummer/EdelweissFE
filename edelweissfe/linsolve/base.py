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

"""Base classes / mixins for linear solvers that need more context than the ``(A, b) -> x`` call
carries.

Most linear solvers need only the matrix and the right hand side. A *field-split* solver additionally
needs to know which DOFs belong to which physical field -- information the matrix does not carry, but
the :class:`~edelweissfe.numerics.dofmanager.DofManager` does. Rather than hand-specify it in a
configuration file (brittle, and it changes with the mesh / adaptivity), the nonlinear solver pushes
it in through the mixin below once the equation system is (re)built.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldBlock:
    """One physical field's contiguous block in the DOF vector.

    Attributes
    ----------
    name
        The field name, e.g. ``"displacement"`` or ``"nonlocal damage"``.
    start, stop
        The half-open DOF range ``[start, stop)`` of the field (fields are contiguous and field-major).
    dimension
        The nodal dimension of the field: the number of components per node (e.g. ``3`` for a 3D
        displacement, ``1`` for a scalar damage field). Determines the field's near null-space -- a
        vector field's rigid-body translations, a scalar field's constant.
    """

    name: str
    start: int
    stop: int
    dimension: int


class FieldStructureAwareLinearSolver:
    """Mixin for linear solvers that require the field-block structure of the DOF vector.

    The nonlinear solver calls :meth:`setFieldStructure` once the ``DofManager`` is built, and again
    after any rebuild (e.g. adaptive refinement changing the DOF count). Solvers that do not need this
    simply do not inherit the mixin and are never asked; the nonlinear solver dispatches on
    ``isinstance``. Subclasses read :attr:`_fieldStructure` (``None`` until set).
    """

    _fieldStructure: "list[FieldBlock] | None" = None

    def setFieldStructure(self, fields: "list[FieldBlock]") -> None:
        """Receive the ordered field blocks of the DOF vector (in DOF order)."""
        self._fieldStructure = list(fields)
