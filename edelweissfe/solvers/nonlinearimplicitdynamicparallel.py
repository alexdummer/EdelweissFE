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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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

import os
from multiprocessing import cpu_count

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.numerics.dofmanager import DofVector, VIJSystemMatrix
from edelweissfe.solvers.base.dirichlet import applyDirichletK as applyDirichletsFast
from edelweissfe.solvers.base.parallelelementcomputation import (
    computeElementsForImplicitDynamicsInParallel,
)
from edelweissfe.solvers.nonlinearimplicitdynamic import NID
from edelweissfe.timesteppers.timestep import TimeStep


class NIDParallel(NID):
    """This is the parallel Nonlinear Implicit Dynamic -- solver.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NIDPSolver"

    def solveStep(self, step, model, fieldOutputController, outputmanagers):
        # determine number of threads
        self.numThreads = cpu_count()

        if "OMP_NUM_THREADS" in os.environ:
            self.numThreads = int(os.environ["OMP_NUM_THREADS"])  # higher priority than .inp settings
        # else:
        #     if "NISTSolver" in step.actions["options"]:
        #         self.numThreads = int(step.actions["options"][self.identification].get('numThreads', self.numThreads))

        self.journal.message("Using {:} threads".format(self.numThreads), self.identification)
        return super().solveStep(step, model, fieldOutputController, outputmanagers)

    @performancetiming.timeit("dirichlet")
    def applyDirichletK(self, K, dirichlets):
        return applyDirichletsFast(self, K, dirichlets)

    @performancetiming.timeit("elements")
    def computeElements(
        self,
        elements: list,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        K: VIJSystemMatrix,
        M: DofVector,
        F: DofVector,
        timeStep: TimeStep,
    ) -> tuple[DofVector, VIJSystemMatrix, DofVector]:
        return computeElementsForImplicitDynamicsInParallel(self, elements, U_np, dU, P, K, M, F, timeStep)
