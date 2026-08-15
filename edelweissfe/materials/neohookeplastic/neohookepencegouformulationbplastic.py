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

from edelweissfe.materials.neohookeplastic.pencegouplasticbase import (
    Ieye,
    Ieye4D,
    NeoHookeanPenceGouPlasticBaseMaterial,
)
from edelweissfe.utils.plasticityutils import dTensorExp_dA, tensorExp


class NeoHookeanWbPlasticMaterial(NeoHookeanPenceGouPlasticBaseMaterial):
    """Neo-Hookean Wb material according to [1] with the following energy density function.

    W_b (I1, J) = mu/2 * (I1/J^(2/3) - 3) + K/8 * (J² + 1/J² - 2).

    [1] Pence, T. J., & Gou, K. (2015). On compressible versions of the incompressible neo-Hookean material.
        Mathematics and Mechanics of Solids, 20(2), 157–182. https://doi.org/10.1177/1081286514544258

    Parameters
    ----------
    materialProperties
        The numpy array containing the material properties for the requested material."""

    def _energyAndStressAndElasticTangent(self, Btr, dF, invFold, Bold):
        """Compute the energy, the Kirchhoff stress and the elastic tangent.

        Parameters
        ----------
        Btr
            The trial left Cauchy-Green tensor.
        dF
            The change of the deformation gradient since the last step.
        invFold
            The inverse of the deformation gradient from the last step.
        Bold
            The left Cauchy-Green tensor from the last step.

        Returns
        -------
        double
            The energy density.
        np.ndarray
            The Kirchhoff stress.
        np.ndarray
            The elastic tangent dKirchhoff/dDeformationGradient."""

        K, mu = self._K, self._mu
        J = np.sqrt(lin.det(Btr))
        I1 = np.trace(Btr)
        invB = lin.inv(Btr)
        lambdaHat = K / 4 * (J**2 + 1 / J**2) + mu * I1 / (9 * J ** (2 / 3))
        muBar = mu / (3 * J ** (2 / 3))
        lambdaBar = K / 4 * (J**2 - 1 / J**2) - muBar * I1
        T = 3 * muBar * Btr + lambdaBar * np.eye(3)
        dB_dF = np.einsum("ik,np,jp,ln->ijkl", Ieye, Bold, dF, invFold) + np.einsum(
            "io,on,jk,ln->ijkl", dF, Bold, Ieye, invFold
        )
        CTauF = muBar * (
            -np.einsum("ij,mn,mnkl->ijkl", Btr, invB.T, dB_dF)
            + 3 * dB_dF
            - np.einsum("ij,mn,mnkl->ijkl", Ieye, Ieye, dB_dF)
        ) + lambdaHat * np.einsum("ij,mn,mnkl->ijkl", np.eye(3), invB.T, dB_dF)
        energy = mu / 2 * (I1 / J ** (2 / 3) - 3) + K / 8 * (J**2 + 1 / J**2 - 2)
        return (energy, T, CTauF)

    def _kirchhoffStress(self, e):
        """Compute the Kirchhoff stress.

        Parameters
        ----------
        e
            The spatial logarithmic Hencky strain.

        Returns
        -------
        np.ndarray
            The Kirchhoff stress."""

        K, mu = self._K, self._mu
        B, _ = tensorExp(2 * e)
        I1 = B.trace()
        J = np.sqrt(lin.det(B))
        muBar = K / 4 * (J**2 - 1 / J**2) - mu * I1 / (3 * J ** (2 / 3))
        T = mu / J ** (2 / 3) * B + muBar * Ieye
        return T

    def _dKirchhoff_dE(self, e):
        """Compute the derivative of the Kirchhoff stress w.r.t. the spatial logarithmic Hencky strain.

        Parameters
        ----------
        e
            The spatial logarithmic Hencky strain.

        Returns
        -------
        np.ndarray
            The derivative of the Kirchhoff stress w.r.t. the spatial Hencky strain."""

        K, mu = self._K, self._mu
        B, n = tensorExp(2 * e)
        J = np.sqrt(lin.det(B))
        dExpE = np.einsum("ijmn,mnkl->ijkl", dTensorExp_dA(2 * e, n), Ieye4D)
        I1 = B.trace()
        p = 2 * mu / J ** (2 / 3)
        lambdaBar = K / 2 * (J**2 + 1 / J**2) + p * I1 / 9
        dT_de = (
            p * dExpE
            - p / 3 * np.einsum("ij,kl->ijkl", B, Ieye)
            + lambdaBar * np.einsum("ij,kl->ijkl", Ieye, Ieye)
            - p / 3 * np.einsum("ij,mn,mnkl->ijkl", Ieye, Ieye, dExpE)
        )
        return dT_de
