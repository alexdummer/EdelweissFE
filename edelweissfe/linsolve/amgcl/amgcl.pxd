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

from libcpp.string cimport string


cdef extern from "amgcl-wrapper.hpp":
    cdef cppclass LinearSolver:
        LinearSolver(const char* json_params) except +
        void solve(int n,
                   const int* ptr,
                   const int* col,
                   const double* val,
                   const double* rhs,
                   double* x,
                   int& iters,
                   double& error) except +
        void set_nullspace(const double* B, int rows, int cols) except +
        void build(int n, const int* ptr, const int* col, const double* val) except +
        void applyPreconditioner(int n, const double* rhs, double* x) except +
        string report() except +

    # Same interface, backed by amgcl::backend::builtin<float> -- half the memory traffic in the
    # smoother apply (§19.3), at the cost of matrix/hierarchy precision. rhs/x stay double at this
    # boundary too; the narrowing/widening happens inside the C++ wrapper.
    cdef cppclass LinearSolverFloat:
        LinearSolverFloat(const char* json_params) except +
        void solve(int n,
                   const int* ptr,
                   const int* col,
                   const float* val,
                   const double* rhs,
                   double* x,
                   int& iters,
                   double& error) except +
        void set_nullspace(const double* B, int rows, int cols) except +
        void build(int n, const int* ptr, const int* col, const float* val) except +
        void applyPreconditioner(int n, const double* rhs, double* x) except +
        string report() except +
