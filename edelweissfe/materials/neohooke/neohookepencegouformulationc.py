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
#  Daniel Reitmair daniel.reitmair@uibk.ac.at
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

import numpy as np
import numpy.linalg as lin

from edelweissfe.materials.neohooke.pencegoubase import NeoHookeanPenceGouBaseMaterial
from edelweissfe.utils.voigtnotation import doVoigtStress


class NeoHookeanWcMaterial(NeoHookeanPenceGouBaseMaterial):
    """Neo-Hookean Wb material according to [1] with the following energy density function.

    W_c (I1, J) = mu/2 * (I1 - 3) + 3mu²/(3K - 2mu) * (J^(2/3 - K/mu) - 1).

    [1] Pence, T. J., & Gou, K. (2015). On compressible versions of the incompressible neo-Hookean material.
        Mathematics and Mechanics of Solids, 20(2), 157–182. https://doi.org/10.1177/1081286514544258

    Parameters
    ----------
    materialProperties
        The numpy array containing the material properties for the requested material."""

    def computeKirchhoff(
        self,
        stress: np.ndarray,
        dStress_dDeformationGradient: np.ndarray,
        deformationGradient: np.ndarray,
        time: float,
        dTime: float,
    ):
        """Computes the stresses for a 3D material/2D material with plane strain.

        Parameters
        ----------
        stress
            Vector containing the stresses.
        dStress_dDeformationGradient
            Matrix containing dStress/dStrain.
        deformationGradient
            The deformation gradient at time step t.
        time
            Array of step time and total time.
        dTime
            Current time step size."""

        F = deformationGradient
        B = F @ F.T  # left Cauchy-Green tensor
        invF = lin.inv(F)  # inverse
        J = lin.det(F)
        # NOTE: kept as trace(F) to preserve existing behavior; formulations Wa/Wb use trace(B) for I1 instead.
        I1 = np.trace(F)
        muBar = self._mu * J ** (2 / 3 - self._K / self._mu)
        lambdaBar = (self._K / self._mu - 2 / 3) * muBar
        stress[:] = doVoigtStress(3, self._mu * B - muBar * np.eye(3))
        dStress_dDeformationGradient[:] = self._mu * (
            np.einsum("ik,jl->ijkl", np.eye(3), F) + np.einsum("il,jk->ijkl", F, np.eye(3))
        ) + lambdaBar * np.einsum("ij,lk->ijkl", np.eye(3), invF)
        self._energy[0] = self._mu / 2 * (I1 - 3) + 3 * self._mu**2 / (3 * self._K - 2 * self._mu) * (
            J ** (2 / 3 - self._K / self._mu) - 1
        )
