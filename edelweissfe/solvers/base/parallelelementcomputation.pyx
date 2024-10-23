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

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.utils.exceptions import CutbackRequest

from cython.parallel cimport parallel, prange, threadid
from libc.stdlib cimport free, malloc
from libcpp.string cimport string

from edelweissfe.elements.marmotelement.element cimport (
    MarmotElement,
    MarmotElementWrapper,
)

import os
from multiprocessing import cpu_count

from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase


def computeElementsInParallel(nls: NonlinearSolverBase, elements, Un1, dU, P, K, F, timeStep):

    cdef:
        double[::1] time = np.array([timeStep.stepTime, timeStep.totalTime])
        double dT = timeStep.timeIncrement

    cdef:
        int elNDof, elNumber, elIdxInVIJ, elIdxInPe, threadID, currentIdxInU
        int desiredThreads = nls.numThreads
        int nElements = len(elements.values())
        list elList = list(elements.values())

        long[::1] I             = K.I
        double[::1] K_mView     = K
        double[::1] UN1_mView   = Un1
        double[::1] dU_mView    = dU
        double[::1] P_mView     = P
        double[::1] F_mView     = F

        # oversized Buffers for parallel computing:
        # tables [nThreads x max(elements.ndof) ] for U & dU (can be overwritten during parallel computing)
        maxNDofOfAnyEl      = nls.theDofManager.largestNumberOfElNDof
        double[:, ::1] UN1e = np.empty((desiredThreads, maxNDofOfAnyEl), )
        double[:, ::1] dUe  = np.empty((desiredThreads, maxNDofOfAnyEl), )
        # oversized buffer for Pe ( size = sum(elements.ndof) )
        double[::1] Pe = np.zeros(nls.theDofManager.accumulatedElementNDof)

        # lists (indices and nDofs), which can be accessed parallely
        int[::1] elIndicesInVIJ         = np.empty( (nElements,), dtype=np.intc )
        int[::1] elIndexInPe            = np.empty( (nElements,), dtype=np.intc )
        int[::1] elNDofs                = np.empty( (nElements,), dtype=np.intc )

        int i, j = 0

    #TODO: this could be done once (__init__) and stored permanently in a cdef class
    for i in range(nElements):
        # prepare all lists for upcoming parallel element computing
        el                      = elList[i]
        elIndicesInVIJ[i]       = nls.theDofManager.idcsOfHigherOrderEntitiesInVIJ[el]
        elNDofs[i]              = el.nDof
        # each element gets its place in the Pe buffer
        elIndexInPe[i] = j
        j += elNDofs[i]

    for i in prange(nElements,
                schedule='dynamic',
                num_threads=desiredThreads,
                nogil=True):

        threadID    = threadid()
        elIdxInVIJ  = elIndicesInVIJ[i]
        elIdxInPe   = elIndexInPe[i]
        elNDof      = elNDofs[i]

        for j in range (elNDof):
            # copy global U & dU to buffer memories for element eval.
            currentIdxInU =     I [ elIndicesInVIJ[i] +  j ]
            UN1e[threadID, j] = UN1_mView[ currentIdxInU ]
            dUe[threadID, j] =  dU_mView[ currentIdxInU ]

        # for accessing the element in the list, and for passing the parameters
        # we have to enable the gil.
        # This prevents a parallel execution in the meanwhile,
        # so we hope the method computeYourself AGAIN releases the gil INSIDE.
        # Otherwise, a truly parallel execution won't happen at all!
        with gil:
            elList[i].computeYourself(K_mView[elIdxInVIJ : elIdxInVIJ + elNDof ** 2],
                                      Pe[elIdxInPe : elIdxInPe + elNDof],
                                      UN1e[threadID, :],
                                      dUe[threadID, :],
                                      time,
                                      dT)

    #successful elements evaluation: condense oversize Pe buffer -> P
    for i in range(nElements):
        elIdxInVIJ =    elIndicesInVIJ[i]
        elIdxInPe =     elIndexInPe[i]
        elNDof =   elNDofs[i]
        for j in range (elNDof):
            P_mView[ I[elIdxInVIJ + j] ] +=      Pe[ elIdxInPe + j ]
            F_mView[ I[elIdxInVIJ + j] ] += abs( Pe[ elIdxInPe + j ] )


    return P, K, F

def computeElementsInParallelForMarmotElements(nls: NonlinearSolverBase, elements, Un1, dU, P, K, F, timeStep):

    cdef:
        double[::1] time = np.array([timeStep.stepTime, timeStep.totalTime])
        double dT = timeStep.timeIncrement

    cdef:
        int elNDofPerEl, elNumber, elIdxInVIJ, elIdxInPe, threadID, currentIdxInU
        int desiredThreads = nls.numThreads
        int nElements = len(elements.values())
        list elList = list(elements.values())

        long[::1] I             = K.I
        double[::1] K_mView     = K
        double[::1] UN1_mView   = Un1
        double[::1] dU_mView    = dU
        double[::1] P_mView     = P
        double[::1] F_mView     = F

        double[:, ::1] pNewDTVector = np.ones( (desiredThreads, 1), order='C' )  * 1e36 # as many pNewDTs as threads

        # oversized Buffers for parallel computing:
        # tables [nThreads x max(elements.ndof) ] for U & dU (can be overwritten during parallel computing)
        maxNDofOfAnyEl      = nls.theDofManager.largestNumberOfElNDof
        double[:, ::1] UN1e = np.empty((desiredThreads, maxNDofOfAnyEl), )
        double[:, ::1] dUe  = np.empty((desiredThreads, maxNDofOfAnyEl), )
        # oversized buffer for Pe ( size = sum(elements.ndof) )
        double[::1] Pe = np.zeros(nls.theDofManager.accumulatedElementNDof)


        MarmotElementWrapper backendBasedCythonElement
        # lists (cpp elements + indices and nDofs), which can be accessed parallely
        MarmotElement** cppElements =      <MarmotElement**> malloc ( nElements * sizeof(MarmotElement*) )
        int[::1] elIndicesInVIJ         = np.empty( (nElements,), dtype=np.intc )
        int[::1] elIndexInPe            = np.empty( (nElements,), dtype=np.intc )
        int[::1] elNDofs                = np.empty( (nElements,), dtype=np.intc )

        int i,j=0

    for i in range(nElements):
        # prepare all lists for upcoming parallel element computing
        backendBasedCythonElement   = elList[i]
        backendBasedCythonElement._initializeStateVarsTemp()
        cppElements[i]              = backendBasedCythonElement.marmotElement
        elIndicesInVIJ[i]           = nls.theDofManager.idcsOfHigherOrderEntitiesInVIJ[backendBasedCythonElement]
        elNDofs[i]                  = backendBasedCythonElement.nDof
        # each element gets its place in the Pe buffer
        elIndexInPe[i] = j
        j += elNDofs[i]

    try:
        for i in prange(nElements,
                    schedule='dynamic',
                    num_threads=desiredThreads,
                    nogil=True):

            threadID =      threadid()
            elIdxInVIJ =    elIndicesInVIJ[i]
            elIdxInPe =     elIndexInPe[i]
            elNDofPerEl =   elNDofs[i]

            for j in range (elNDofPerEl):
                # copy global U & dU to buffer memories for element eval.
                currentIdxInU =     I [ elIndicesInVIJ[i] +  j ]
                UN1e[threadID, j] = UN1_mView[ currentIdxInU ]
                dUe[threadID, j] =  dU_mView[ currentIdxInU ]

            (<MarmotElement*>
                 cppElements[i] )[0].computeYourself(&UN1e[threadID, 0],
                                                    &dUe[threadID, 0],
                                                    &Pe[elIdxInPe],
                                                    &K_mView[elIdxInVIJ],
                                                    &time[0],
                                                    dT,
                                                    pNewDTVector[threadID, 0])

            if pNewDTVector[threadID, 0] <= 1.0:
                break

        minPNewDT = np.min(pNewDTVector)
        if minPNewDT < 1.0:
            raise CutbackRequest("An element requests for a cutback", minPNewDT)

        #successful elements evaluation: condense oversize Pe buffer -> P
        P_mView[:] = 0.0
        F_mView[:] = 0.0
        for i in range(nElements):
            elIdxInVIJ =    elIndicesInVIJ[i]
            elIdxInPe =     elIndexInPe[i]
            elNDofPerEl =   elNDofs[i]
            for j in range (elNDofPerEl):
                P_mView[ I[elIdxInVIJ + j] ] +=      Pe[ elIdxInPe + j ]
                F_mView[ I[elIdxInVIJ + j] ] += abs( Pe[ elIdxInPe + j ] )
    finally:
        free( cppElements )

    return P, K, F


def computeElementsForExplicitDynamicsInParallel(nls, elements, Un1, dU, P, M, timeStep):

    # workaround for the fact that no stiffness matrix is present in the NED solver
    # but we want to use the indices
    K = nls.theDofManager.constructVIJSystemMatrix()
    cdef:
        double[::1] time = np.array([timeStep.stepTime, timeStep.totalTime])
        double dT = timeStep.timeIncrement

    cdef:
        int elNDof, elNumber, elIdxInVIJ, elIdxInPe, threadID, currentIdxInU
        int desiredThreads = nls.numThreads
        int nElements = len(elements.values())
        list elList = list(elements.values())

        long[::1] I             = K.I
        double[::1] UN1_mView   = Un1
        double[::1] dU_mView    = dU
        double[::1] P_mView     = P
        double[::1] M_mView     = M

        # oversized Buffers for parallel computing:
        # tables [nThreads x max(elements.ndof) ] for U & dU (can be overwritten during parallel computing)
        maxNDofOfAnyEl      = nls.theDofManager.largestNumberOfElNDof
        double[:, ::1] UN1e = np.empty((desiredThreads, maxNDofOfAnyEl), )
        double[:, ::1] dUe  = np.empty((desiredThreads, maxNDofOfAnyEl), )
        # oversized buffer for Pe ( size = sum(elements.ndof) )
        double[::1] Pe = np.zeros(nls.theDofManager.accumulatedElementNDof)
        double[::1] Me = np.zeros(nls.theDofManager.accumulatedElementNDof)
        double[::1] Ke = np.zeros(K.size)
        # lists (indices and nDofs), which can be accessed parallely
        int[::1] elIndicesInVIJ         = np.empty( (nElements,), dtype=np.intc )
        int[::1] elIndexInPe            = np.empty( (nElements,), dtype=np.intc )
        int[::1] elNDofs                = np.empty( (nElements,), dtype=np.intc )

        int i, j = 0

    # delete K again
    del K

    #TODO: this could be done once (__init__) and stored permanently in a cdef class
    for i in range(nElements):
        # prepare all lists for upcoming parallel element computing
        el                      = elList[i]
        elIndicesInVIJ[i]       = nls.theDofManager.idcsOfHigherOrderEntitiesInVIJ[el]
        elNDofs[i]              = el.nDof
        # each element gets its place in the Pe buffer
        elIndexInPe[i] = j
        j += elNDofs[i]

    for i in prange(nElements,
                schedule='dynamic',
                num_threads=desiredThreads,
                nogil=True):

        threadID    = threadid()
        elIdxInVIJ  = elIndicesInVIJ[i]
        elIdxInPe   = elIndexInPe[i]
        elNDof      = elNDofs[i]

        for j in range (elNDof):
            # copy global U & dU to buffer memories for element eval.
            currentIdxInU =     I [ elIndicesInVIJ[i] +  j ]
            UN1e[threadID, j] = UN1_mView[ currentIdxInU ]
            dUe[threadID, j] =  dU_mView[ currentIdxInU ]

        # for accessing the element in the list, and for passing the parameters
        # we have to enable the gil.
        # This prevents a parallel execution in the meanwhile,
        # so we hope the method computeYourself AGAIN releases the gil INSIDE.
        # Otherwise, a truly parallel execution won't happen at all!
        with gil:
            elList[i].computeYourself(Ke[elIdxInVIJ : elIdxInVIJ + elNDof**2],
                                      Pe[elIdxInPe : elIdxInPe + elNDof],
                                      UN1e[threadID, :],
                                      dUe[threadID, :],
                                      time,
                                      dT)
            elList[i].computeLumpedInertia(Me[elIdxInPe : elIdxInPe + elNDof])

    #successful elements evaluation: condense oversize Pe buffer -> P
    for i in range(nElements):
        elIdxInVIJ =    elIndicesInVIJ[i]
        elIdxInPe =     elIndexInPe[i]
        elNDof =   elNDofs[i]
        for j in range (elNDof):
            P_mView[ I[elIdxInVIJ + j] ] +=      Pe[ elIdxInPe + j ]
            M_mView[ I[elIdxInVIJ + j] ] +=      Me[ elIdxInPe + j ]


    return P, M
