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

def applyDirichletK(nls, K, dirichlets):
    """Apply the dirichlet bcs on the global stiffnes matrix
    Is called by solveStep() before solving the global sys.
    http://stackoverflux.com/questions/12129948/scipy-sparse-set-row-to-zeros

    Cythonized version for speed!

    Parameters
    ----------
    K
        The system matrix.
    dirichlets
        The list of dirichlet boundary conditions.

    Returns
    -------
    VIJSystemMatrix
        The modified system matrix.
    """

    cdef int  i, j
    cdef int [::1] indices_, indptr_,
    cdef long[::1] dirichletIndices
    cdef double[::1] data_

    indices_ = K.indices
    indptr_ = K.indptr
    data_ = K.data

    for dirichlet in dirichlets: # for each bc
        dirichletIndices = nls.findDirichletIndices(dirichlet)
        for i in dirichletIndices: # for each node dof in the BC
            for j in range ( indptr_[i] , indptr_ [i+1] ): # iterate along row
                if i == indices_ [j]:
                    data_[ j ] = 1.0 # diagonal entry
                else:
                    data_[ j ] = 0.0 # off diagonal entry

    K.eliminate_zeros()

    return K
