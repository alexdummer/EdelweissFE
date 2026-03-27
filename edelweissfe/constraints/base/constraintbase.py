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

from abc import ABC, abstractmethod

import numpy as np

from edelweissfe.models.femodel import FEModel
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable


class ConstraintBase(ABC):
    @abstractmethod
    def __init__(self, name: str, options: dict, model: FEModel):
        """The constraint base class.

        Constraints can act on nodal variables, and scalar variables.
        If scalar variables are required, the can be created on demand by
        defining
        :func:`~ConstraintBase.getNumberOfAdditionalNeededScalarVariables` and
        :func:`~ConstraintBase.assignAdditionalScalarVariables`, which are called at the beginning of an analysis.

        If scalar variables are used, EdelweissFE expects the layout of the external load vector PExt
        (and the stiffness) to be of the form

        .. code-block:: console

            [ node 1 - dofs field 1,
              node 1 - dofs field 2,
              node 1 - ... ,
              node 1 - dofs field n,
              node 2 - dofs field 1,
              ... ,
              node N - dofs field n,
              scalar variable 1,
              scalar variable 2,
              ... ,
              scalar variable J ].

        Parameters
        ----------
        name
            The name of the constraint.
        options
            A dictionary containing the options for the constraint.
        model
            A dictionary containing the model tree.
        """

        self.scalarVariables = []

    @property
    @abstractmethod
    def nodes(self) -> list:
        """The nodes this constraint is acting on."""

    @property
    @abstractmethod
    def fieldsOnNodes(self) -> list:
        """The fields on the nodes this constraint is acting on."""

    @property
    @abstractmethod
    def nDof(self) -> int:
        """The total number of degrees of freedom this constraint is associated with."""

    def getVIJContributionSize(self) -> int:
        """Return the number of entries this constraint contributes to the VIJ (COO) system matrix.

        By default this is ``nDof**2``, which corresponds to a full dense block.
        Constraints whose nonzero pattern is much sparser than a full block should override
        this method (together with :meth:`initializeVIJContribution`) to avoid allocating
        O(N²) entries when only O(N) entries are actually nonzero.

        Returns
        -------
        int
            Number of VIJ entries for this constraint.
        """

        return self.nDof**2

    def initializeVIJContribution(self, idcs: np.ndarray, I: np.ndarray, J: np.ndarray, offset: int) -> None:
        """Fill the global ``I`` and ``J`` index arrays for this constraint's VIJ contribution.

        The default implementation writes a full dense ``nDof × nDof`` block, identical to
        the behaviour used for finite elements.  Constraints that override
        :meth:`getVIJContributionSize` to return a smaller value **must** also override
        this method so that exactly ``getVIJContributionSize()`` ``(I, J)`` pairs are
        written starting at ``offset``.

        Parameters
        ----------
        idcs
            Global DOF indices for this constraint (length = ``nDof``).
        I
            The global row-index array of the VIJ triple (written in-place).
        J
            The global column-index array of the VIJ triple (written in-place).
        offset
            First position in ``I`` / ``J`` that belongs to this constraint.
        """

        n = len(idcs)
        VIJLocations = np.tile(idcs, (n, 1))
        I[offset : offset + n**2] = VIJLocations.flatten()
        J[offset : offset + n**2] = VIJLocations.flatten("F")

    def getNumberOfAdditionalNeededScalarVariables(
        self,
    ) -> int:
        """This method is called to determine the additional number of scalar variables
        this Constraint needs.

        Returns
        -------
        int
            Number of ScalarVariables required.

        """

        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        """Assign a list of scalar variables associated with this constraint.

        Parameters
        ----------
        list
            The list of ScalarVariables associated with this constraint.

        """

        self.scalarVariables = scalarVariables

    @abstractmethod
    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        V: np.ndarray,
        timeStep: TimeStep,
    ):
        """Apply the constraint.  Add the contributions to the external load vector and the system matrix.

        Parameters
        ----------
        U_np
            The current total solution vector.
        dU
            The current increment since the last time the constraint was applied.
        PExt
            The external load vector.
        K
            The system (stiffness) matrix.
        dT
            The time increment.
        time
            The current step and total time.
        """
