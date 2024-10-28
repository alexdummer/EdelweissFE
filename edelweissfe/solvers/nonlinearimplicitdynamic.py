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

import numpy as np
import scipy

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.config.linsolve import getDefaultLinSolver, getLinSolverByName
from edelweissfe.config.timing import createTimingDict
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.csrgenerator import CSRGenerator
from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import (
    ConditionalStop,
    CutbackRequest,
    DivergingSolution,
    ReachedMaxIncrements,
    ReachedMaxIterations,
    ReachedMinIncrementSize,
    StepFailed,
)
from edelweissfe.utils.fieldoutput import FieldOutputController


def computeParamentersForGeneralizedAlpha(rho, alphaM, alphaF, beta, gamma):
    """Compute the parameters for the generalized alpha method.

    Parameters
    ----------
    rho
        The rho parameter.
    alphaM
        The alphaM parameter.
    alphaF
        The alphaF parameter.
    beta
        The beta parameter.
    gamma
        The gamma parameter.

    Returns
    -------
    tuple[float,float,float,float]
        The computed parameters.
    """

    if alphaM != alphaM:
        alphaM = (2.0 * rho - 1.0) / (rho + 1.0)
    if alphaF != alphaF:
        alphaF = rho / (rho + 1.0)
    if beta != beta:
        beta = (1 - alphaM + alphaF) ** 2 / 4
    if gamma != gamma:
        gamma = 0.5 - alphaM + alphaF

    return alphaM, alphaF, beta, gamma


class NID(NonlinearSolverBase):
    """This is the Nonlinear Implicit Dynamic -- solver.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NIDSolver"

    SolverSpecificOptions = {
        "rho": 0.9,
        "alphaM": np.nan,
        "alphaF": np.nan,
        "beta": np.nan,
        "gamma": np.nan,
        "alphaR": 0.0,
        "betaR": 0.0,
        "linsolver": "pardiso",
        "extrapolation": "off",
        "defaultMaxIter": 10,
        "defaultCriticalIter": 5,
        "defaultMaxGrowingIter": 10,
    }

    def __init__(self, jobInfo, journal, **kwargs):
        self.journal = journal

        self.fieldCorrectionTolerances = jobInfo["fieldCorrectionTolerance"]
        self.fluxResidualTolerances = jobInfo["fluxResidualTolerance"]
        self.fluxResidualTolerancesAlt = jobInfo["fluxResidualToleranceAlternative"]

        self.options = self.SolverSpecificOptions.copy()
        self._updateOptions(kwargs, journal)

    def solveStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: dict[str, OutputManagerBase],
    ) -> tuple[bool, FEModel]:
        """Public interface to solve for a step.

        Parameters
        ----------
        stepNumber
            The step number.
        step
            The dictionary containing the step definition.
        stepActions
            The dictionary containing all step actions.
        model
            The  model tree.
        fieldOutputController
            The field output controller.
        """

        self.journal.message("Creating monolithic equation system", self.identification, 0)
        self.theDofManager = DofManager(
            model.nodeFields.values(),
            model.scalarVariables.values(),
            model.elements.values(),
            model.constraints.values(),
            model.nodeSets.values(),
        )
        self.journal.message(
            "total size of eq. system: {:}".format(self.theDofManager.nDof),
            self.identification,
            0,
        )

        self.journal.printSeperationLine()

        presentVariableNames = list(self.theDofManager.idcsOfFieldsInDofVector.keys())

        if self.theDofManager.idcsOfScalarVariablesInDofVector:
            presentVariableNames += [
                "scalar variables",
            ]

        nVariables = len(presentVariableNames)
        self.iterationHeader = ("{:^25}" * nVariables).format(*presentVariableNames)
        self.iterationHeader2 = (" {:<10}  {:<10}  ").format("||R||∞", "||ddU||∞") * nVariables
        self.iterationMessageTemplate = "{:11.2e}{:1}{:11.2e}{:1} "

        U = self.theDofManager.constructDofVector()
        V = self.theDofManager.constructDofVector()
        A = self.theDofManager.constructDofVector()
        K = self.theDofManager.constructVIJSystemMatrix()

        self.csrGenerator = CSRGenerator(K)

        self.computationTimes = createTimingDict()

        try:
            self._updateOptions(step.actions["options"]["NISTSolver"].options, self.journal)
        except KeyError:
            pass

        extrapolation = self.options["extrapolation"]
        self.linSolver = (
            getLinSolverByName(self.options["linsolver"]) if "linsolver" in self.options else getDefaultLinSolver()
        )

        maxIter = step.maxIter
        criticalIter = step.criticalIter
        maxGrowingIter = step.maxGrowIter

        # nodes = model.nodes
        # elements = model.elements
        # constraints = model.constraints

        U = self.theDofManager.constructDofVector()
        V = self.theDofManager.constructDofVector()
        A = self.theDofManager.constructDofVector()
        P = self.theDofManager.constructDofVector()
        dU = self.theDofManager.constructDofVector()

        for fieldName, field in model.nodeFields.items():
            U = self.theDofManager.writeNodeFieldToDofVector(U, field, "U")

        for variable in model.scalarVariables.values():
            U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]] = variable.value

        prevTimeStep = None

        self.applyStepActionsAtStepStart(model, step.actions)

        try:
            for timeStep in step.getTimeStep():
                statusInfoDict = {
                    "step": step.number,
                    "inc": timeStep.number,
                    "iters": None,
                    "converged": False,
                    "time inc": timeStep.timeIncrement,
                    "time end": timeStep.totalTime,
                    "notes": "",
                }

                self.journal.printSeperationLine()
                self.journal.message(
                    "increment {:}: {:8f}, {:8f}; time {:10f} to {:10f}".format(
                        timeStep.number,
                        timeStep.stepProgressIncrement,
                        timeStep.stepProgress,
                        timeStep.totalTime - timeStep.timeIncrement,
                        timeStep.totalTime,
                    ),
                    self.identification,
                    level=1,
                )
                self.journal.message(self.iterationHeader, self.identification, level=2)
                self.journal.message(self.iterationHeader2, self.identification, level=2)

                try:
                    U, V, A, dU, P, iterationCounter, incrementResidualHistory = self.solveIncrement(
                        U,
                        V,
                        A,
                        dU,
                        P,
                        K,
                        step.actions,
                        model,
                        timeStep,
                        prevTimeStep,
                        extrapolation,
                        maxIter,
                        maxGrowingIter,
                    )

                except CutbackRequest as e:
                    self.journal.message(str(e), self.identification, 1)
                    step.discardAndChangeIncrement(max(e.cutbackSize, 0.25))
                    prevTimeStep = None

                    statusInfoDict["iters"] = np.inf
                    statusInfoDict["notes"] = str(e)

                    for man in outputmanagers:
                        man.finalizeFailedIncrement(
                            statusInfoDict=statusInfoDict,
                            currentComputingTimes=self.computationTimes,
                        )

                except (ReachedMaxIterations, DivergingSolution) as e:
                    self.journal.message(str(e), self.identification, 1)
                    step.discardAndChangeIncrement(0.25)
                    prevTimeStep = None

                    statusInfoDict["iters"] = np.inf
                    statusInfoDict["notes"] = str(e)

                    for man in outputmanagers:
                        man.finalizeFailedIncrement(
                            statusInfoDict=statusInfoDict,
                            currentComputingTimes=self.computationTimes,
                        )
                else:
                    prevTimeStep = timeStep
                    if iterationCounter >= criticalIter:
                        step.preventIncrementIncrease()

                    # write results to nodes:
                    for fieldName, field in model.nodeFields.items():
                        self.theDofManager.writeDofVectorToNodeField(U, field, "U")
                        self.theDofManager.writeDofVectorToNodeField(P, field, "P")
                        self.theDofManager.writeDofVectorToNodeField(dU, field, "dU")

                    for variable in model.scalarVariables.values():
                        variable.value = U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]]

                    model.advanceToTime(timeStep.totalTime)

                    self.journal.message(
                        "Converged in {:} iteration(s)".format(iterationCounter),
                        self.identification,
                        1,
                    )

                    statusInfoDict["iters"] = iterationCounter
                    statusInfoDict["converged"] = True

                    fieldOutputController.finalizeIncrement()
                    for man in outputmanagers:
                        man.finalizeIncrement(
                            currentComputingTimes=self.computationTimes,
                            statusInfoDict=statusInfoDict,
                        )

        except (ReachedMaxIncrements, ReachedMinIncrementSize):
            self.journal.errorMessage("Incrementation failed", self.identification)
            raise StepFailed()

        except ConditionalStop:
            self.journal.message("Conditional Stop", self.identification)
            self.applyStepActionsAtStepEnd(model, step.actions)

        else:
            self.applyStepActionsAtStepEnd(model, step.actions)

        finally:
            prettyTable = performancetiming.makePrettyTable()
            self.journal.printPrettyTable(prettyTable, self.identification)
            performancetiming.times.clear()

    def solveIncrement(
        self,
        U_n: DofVector,
        V_n: DofVector,
        A_n: DofVector,
        dU: DofVector,
        P: DofVector,
        K: VIJSystemMatrix,
        stepActions: list,
        model: FEModel,
        timeStep: TimeStep,
        prevTimeStep: TimeStep,
        extrapolation: str,
        maxIter: int,
        maxGrowingIter: int,
    ) -> tuple[DofVector, DofVector, DofVector, int, dict]:
        """Implicit generalized alpha time-integration to solve for an increment.

        Parameters
        ----------
        Un
            The old solution vector.
        Vn
            The old velocity vector.
        An
            The old acceleration vector.
        dU
            The old solution increment.
        P
            The old reaction vector.
        K
            The system matrix to be used.
        elements
            The dictionary containing all elements.
        stepActions
            The list of active step actions.
        model
            The model tree.
        increment
            The increment.
        lastIncrementSize
            The size of the previous increment.
        extrapolation
            The type of extrapolation to be used.
        maxIter
            The maximum number of iterations to be used.
        maxGrowingIter
            The maximum number of growing residuals until the Newton-Raphson is terminated.

        Returns
        -------
        tuple[DofVector,DofVector,DofVector,int,dict]
            A tuple containing
                - the new solution vector
                - the solution increment
                - the new reaction vector
                - the number of required iterations
                - the history of residuals per field
        """

        # incNumber, incrementSize, stepProgress, dT, stepTime, totalTime = timeStep
        iterationCounter = 0
        incrementResidualHistory = dict.fromkeys(self.theDofManager.idcsOfFieldsInDofVector, (0.0, 0))

        elements = model.elements
        constraints = model.constraints

        R = self.theDofManager.constructDofVector()
        F = self.theDofManager.constructDofVector()
        M = self.theDofManager.constructDofVector()
        PExt = self.theDofManager.constructDofVector()
        U_np = self.theDofManager.constructDofVector()
        A_np = self.theDofManager.constructDofVector()
        V_np = self.theDofManager.constructDofVector()
        U_int = self.theDofManager.constructDofVector()
        dU_int = self.theDofManager.constructDofVector()
        A_int = self.theDofManager.constructDofVector()
        V_int = self.theDofManager.constructDofVector()
        ddU = None

        dirichlets = stepActions["dirichlet"].values()
        nodeforces = stepActions["nodeforces"].values()
        distributedLoads = stepActions["distributedload"].values()
        bodyForces = stepActions["bodyforce"].values()

        self.applyStepActionsAtIncrementStart(model, timeStep, stepActions)

        dU, isExtrapolatedIncrement = self.extrapolateLastIncrement(
            extrapolation, timeStep, dU, dirichlets, prevTimeStep, model
        )

        # time integration parameters
        alphaM, alphaF, beta, gamma = computeParamentersForGeneralizedAlpha(
            self.options.get("rho"),
            self.options.get("alphaM"),
            self.options.get("alphaF"),
            self.options.get("beta"),
            self.options.get("gamma"),
        )

        # Rayleigh damping parameters
        alphaR = self.options.get("alphaR")
        betaR = self.options.get("betaR")

        timeStep_int = TimeStep(
            timeStep.number,
            timeStep.stepProgressIncrement,
            timeStep.stepProgress,
            timeStep.timeIncrement * (1 - alphaF),
            timeStep.stepTime,
            timeStep.totalTime - (1 - alphaF) * timeStep.timeIncrement,
        )
        dT = timeStep.timeIncrement
        while True:
            for geostatic in stepActions["geostatic"].values():
                geostatic.applyAtIterationStart()

            U_np[:] = U_n
            U_np += dU

            # update acceleration
            A_np[:] = A_n
            if dT != 0:
                A_np[:] *= -(0.5 - beta) / beta
                A_np[:] += 1 / beta / dT / dT * dU
                A_np[:] -= 1 / beta / dT * V_n
            A_int[:] = (1 - alphaM) * A_np + alphaM * A_n

            # update velocity
            V_np[:] = V_n
            V_np[:] += (1 - gamma) * dT * A_n
            V_np[:] += gamma * dT * A_np
            V_int[:] = (1 - alphaF) * V_np + alphaF * V_n

            # inetrmediate displacement
            U_int[:] = (1 - alphaF) * U_np + alphaF * U_n
            dU_int[:] = U_int - U_n

            P[:] = K[:] = M[:] = F[:] = PExt[:] = 0.0

            # intermediate time

            P, K, M, F = self.computeElements(elements, U_int, dU_int, P, K, M, F, timeStep_int)
            PExt, K = self.assembleLoads(nodeforces, distributedLoads, bodyForces, U_int, PExt, K, timeStep_int)
            PExt, K = self.assembleConstraints(constraints, U_int, dU_int, PExt, K, timeStep_int)

            R[:] = P
            R += PExt

            K_ = self.assembleStiffnessCSR(K)
            R[:] = R - M.T * (A_int + alphaR * V_int) - K_ * V_int * betaR

            # check for zero increment
            if dT != 0:
                K_ *= 1 - alphaF
                K_ += (1 - alphaF) * (1 + gamma) / beta / dT * K_ * betaR
                K_ = self.addVectorToCSRDiagonal(
                    K_,
                    (1 - alphaM) * (1.0 / beta / dT / dT + gamma / beta / dT * alphaR) * M,
                )

            K_ = self.applyDirichletK(K_, dirichlets)

            if iterationCounter == 0 and not isExtrapolatedIncrement and dirichlets:
                # first iteration? apply dirichlet bcs and unconditionally solve
                R = self.applyDirichlet(timeStep, R, dirichlets)
            else:
                # iteration cycle 1 or higher, time to check the convergence
                for dirichlet in dirichlets:
                    R[self.findDirichletIndices(dirichlet)] = 0.0

                converged, nodesWithLargestResidual = self.checkConvergence(
                    R, ddU, F, iterationCounter, incrementResidualHistory
                )

                if converged:
                    # compute elements once more with correct increment
                    P[:] = K[:] = M[:] = F[:] = 0.0
                    P, K, M, F = self.computeElements(elements, U_np, dU, P, K, M, F, timeStep)
                    break

                if self.checkDivergingSolution(incrementResidualHistory, maxGrowingIter):
                    self.printResidualOutlierNodes(nodesWithLargestResidual)
                    raise DivergingSolution("Residual grew {:} times, cutting back".format(maxGrowingIter))

                if iterationCounter == maxIter:
                    self.printResidualOutlierNodes(nodesWithLargestResidual)
                    raise ReachedMaxIterations("Reached max. iterations in current increment, cutting back")

            ddU = self.linearSolve(K_, R)
            dU += ddU
            iterationCounter += 1

        return U_np, V_np, A_np, dU, P, iterationCounter, incrementResidualHistory

    def addVectorToCSRDiagonal(self, csr: scipy.sparse.csr_matrix, vec: DofVector):

        indices_ = csr.indices
        indptr_ = csr.indptr
        data_ = csr.data

        for i in range(vec.size):  # for each node dof in the BC
            for j in range(indptr_[i], indptr_[i + 1]):  # iterate along row
                if i == indices_[j]:
                    data_[j] += vec[i]  # diagonal entry
        return csr

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
        """Loop over all elements, and evalute them.
        Is is called by solveStep() in each iteration.

        Parameters
        ----------
        elements
            The list of finite elements.
        U_np
            The current solution vector.
        dU
            The current solution increment vector.
        P
            The reaction vector.
        K
            The system matrix.
        M
            The lumped mass matrix.
        F
            The vector of accumulated fluxes for convergence checks.
        timeStep
            The time step.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix,DofVector]
            - The modified reaction vector.
            - The modified system matrix.
            - The modified accumulated flux vector.
        """

        time = np.array([timeStep.stepTime, timeStep.totalTime])
        dT = timeStep.timeIncrement

        for el in elements.values():
            Ke = K[el]
            Pe = np.zeros(el.nDof)
            Me = np.zeros(el.nDof)

            el.computeYourself(Ke, Pe, U_np[el], dU[el], time, dT)
            el.computeLumpedInertia(Me)

            P[el] += Pe
            M[el] += Me
            F[el] += abs(Pe)

        return P, K, M, F
