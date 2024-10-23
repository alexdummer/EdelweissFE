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
"""
Created on Sun Jan  8 20:37:35 2017

@author: matthias

Fri Oct 6 2018:

    This solver is now deprecated since it is replaced by its sucessor mk2.
    This version (mk1) directly accesses the underlying MarmotElement, and thus
    it is able to completely release the GIL throughout the complete prange loop.
    Naturally, it is compatible only with cython elements based on MarmotElements.
    However, it seems that there is no measurable performance advantage over mk2, which
    is not dependent on a  underlying MarmotElement (and is thus also compatible with python or
    cython elements).
    This solver remains for teaching purposes to demonstrate how to interact with cpp objects.
"""

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.solvers.base.parallelelementcomputation import (
    computeElementsInParallelForMarmotElements,
)
from edelweissfe.solvers.nonlinearimplicitstaticparallelmk2 import NISTParallel


class NISTParallelForMarmotElements(NISTParallel):

    identification = "NISTPSolver"

    @performancetiming.timeit("elements")
    def computeElements(self, elements, Un1, dU, P, K, F, timeStep):
        return computeElementsInParallelForMarmotElements(self, elements, Un1, dU, P, K, F, timeStep)
