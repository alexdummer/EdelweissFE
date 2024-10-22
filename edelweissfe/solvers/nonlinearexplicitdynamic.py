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

from time import time as getCurrentTime

import numpy as np
import scipy

from edelweissfe.config.timing import createTimingDict
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
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


class NED:
    """This is the Nonlinear Explicit Dynamic -- solver.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NEDSolver"

    NEDOptions = {
        "runge-kutta-2": False,
        # "defaultCriticalIter": 5,
        # "defaultMaxGrowingIter": 10,
        # "extrapolation": "linear",
        # "linsolver": "pardiso",
    }

    def __init__(self, jobInfo, journal, **kwargs):
        self.journal = journal

        self.options = self.NEDOptions.copy()
        self._updateOptions(kwargs, journal)

    def _updateOptions(self, updatedOptions: dict, journal):
        """Update options of the solver using a string dict

        Parameters
        ----------
        updatedOptions
            The options dictionary.
        journal
            The journal module.
        """

        for k, v in updatedOptions.items():
            if k in self.NEDOptions:
                journal.message("Updating option {:}={:}".format(k, v), self.identification)
                self.options[k] = type(self.NEDOptions[k])(updatedOptions[k])
            else:
                raise AttributeError("Invalid option {:} for {:}".format(k, self.identification))

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

        # nVariables = len(presentVariableNames)

        self.computationTimes = createTimingDict()

        try:
            self._updateOptions(step.actions["options"]["NEDSolver"].options, self.journal)
        except KeyError:
            pass

        M = self.theDofManager.constructDofVector()  # initialize lumped mass matrix
        U = self.theDofManager.constructDofVector()  # initialize displacement vector
        V = self.theDofManager.constructDofVector()  # initilize velocity vector
        P = self.theDofManager.constructDofVector()  # initialize reaction vector
        dU = self.theDofManager.constructDofVector()  # initialize displacement increment vector

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
                # self.journal.message(self.iterationHeader, self.identification, level=2)
                # self.journal.message(self.iterationHeader2, self.identification, level=2)

                try:
                    U, dU, V, P = self.solveIncrement(
                        U,
                        dU,
                        V,
                        P,
                        M,
                        step.actions,
                        model,
                        timeStep,
                        prevTimeStep,
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

                    # write results to nodes:
                    for fieldName, field in model.nodeFields.items():
                        self.theDofManager.writeDofVectorToNodeField(U, field, "U")
                        self.theDofManager.writeDofVectorToNodeField(P, field, "P")
                        self.theDofManager.writeDofVectorToNodeField(dU, field, "dU")

                    for variable in model.scalarVariables.values():
                        variable.value = U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]]

                    model.advanceToTime(timeStep.totalTime)

                    # self.journal.message(
                    #     "Converged in {:} iteration(s)".format(iterationCounter),
                    #     self.identification,
                    #     1,
                    # )

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
            self.journal.printTable(
                [("Time in {:}".format(k), " {:10.4f}s".format(v)) for k, v in self.computationTimes.items()],
                self.identification,
            )

    def solveIncrement(
        self,
        U_n: DofVector,
        dU_: DofVector,
        V: DofVector,
        P: DofVector,
        M: scipy.sparse.diags,
        stepActions: list,
        model: FEModel,
        timeStep: TimeStep,
        prevTimeStep: TimeStep,
    ) -> tuple[DofVector, DofVector, DofVector, DofVector]:
        """Standard explicit update scheme to solve for an increment.

        Parameters
        ----------
        Un
            The old solution vector.
        dU
            The old solution increment.
        V
            The old velocity vector.
        P
            The old reaction vector.
        M
            The lumped mass matrix to be used.
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

        elements = model.elements
        # constraints = model.constraints
        dU = self.theDofManager.constructDofVector()
        R = self.theDofManager.constructDofVector()
        PExt = self.theDofManager.constructDofVector()
        U_np = self.theDofManager.constructDofVector()

        dirichlets = stepActions["dirichlet"].values()
        # nodeforces = stepActions["nodeforces"].values()
        distributedLoads = stepActions["distributedload"].values()
        bodyForces = stepActions["bodyforce"].values()

        self.applyStepActionsAtIncrementStart(model, timeStep, stepActions)

        for geostatic in stepActions["geostatic"].values():
            geostatic.applyAtIterationStart()

        if timeStep.timeIncrement == 0.0:
            return U_n, dU_, V, P

        dU[:] = dU_
        U_np[:] = U_n

        if prevTimeStep is None:

            prevTimeStep = TimeStep(
                timeStep.number,
                timeStep.stepProgressIncrement,
                timeStep.stepProgress,
                timeStep.timeIncrement * 0,
                timeStep.stepTime,
                timeStep.totalTime - timeStep.timeIncrement,
            )
        # if self.options["runge-kutta-2"]:
        #
        #     P[:] = M[:] = PExt[:] = 0.0
        #     P, M = self.computeElements(elements, U_np, dU, P, M, prevTimeStep)
        #     PExt = self.computeDistributedLoads(distributedLoads, U_np, PExt, prevTimeStep)
        #     PExt = self.computeBodyForces(bodyForces, U_np, PExt, prevTimeStep)
        #
        #     # first stage
        #     MInverse = 1.0 / M.T
        #
        #     R[:] = P + PExt
        #
        #     # enforce dirichlet boundary conditions
        #     for dirichlet in dirichlets:
        #         R[self.findDirichletIndices(dirichlet)] = 0.0
        #
        #     # update velocity vector with lumped mass matrix
        #     v1 = MInverse * R * timeStep.timeIncrement
        #
        #     # for dirichlet in dirichlets:
        #     #     v1[self.findDirichletIndices(dirichlet)] = (
        #     #     dirichlet.getDelta(timeStep).flatten() / timeStep.timeIncrement
        #     # )
        #     dU1 = v1 * timeStep.timeIncrement
        #     # stage 2
        #     U_np[:] = U_n + 0.5 * dU1
        #     dU[:] = 0.5 * dU1
        #
        #     ts = TimeStep(
        #         timeStep.number,
        #         timeStep.stepProgressIncrement,
        #         timeStep.stepProgress,
        #         timeStep.timeIncrement / 2,
        #         timeStep.stepTime,
        #         timeStep.totalTime - timeStep.timeIncrement / 2,
        #     )
        #     P[:] = M[:] = PExt[:] = 0.0
        #     P, M = self.computeElements(elements, U_np, dU, P, M, ts)
        #     PExt = self.computeDistributedLoads(distributedLoads, U_np, PExt, ts)
        #     PExt = self.computeBodyForces(bodyForces, U_np, PExt, ts)
        #
        #     R[:] = P + PExt
        #
        #     # enforce dirichlet boundary conditions
        #     for dirichlet in dirichlets:
        #         R[self.findDirichletIndices(dirichlet)] = 0.0
        #
        #     # update velocity vector with lumped mass matrix
        #     v2 = MInverse * R * timeStep.timeIncrement
        #     #
        #     # for dirichlet in dirichlets:
        #     #     v2[self.findDirichletIndices(dirichlet)] = (
        #     #     dirichlet.getDelta(timeStep).flatten() / timeStep.timeIncrement
        #     # )
        #     dU2 = v2 * timeStep.timeIncrement
        #
        #     V += 0.5 * (v1 + v2)
        #
        #     # # stage 3
        #     # U_np[:] = U_n
        #     # U_np += 1 / 2 * dU2
        #     # dU[:] = 1 / 2 * dU2
        #     # ts = TimeStep(
        #     #     timeStep.number,
        #     #     timeStep.stepProgressIncrement,
        #     #     timeStep.stepProgress,
        #     #     timeStep.timeIncrement / 2,
        #     #     timeStep.stepTime,
        #     #     timeStep.totalTime - timeStep.timeIncrement / 2,
        #     # )
        #     #
        #     # P[:] = M[:] = PExt[:] = 0.0
        #     # P, M = self.computeElements(elements, U_np, dU, P, M, ts)
        #     # PExt = self.computeDistributedLoads(distributedLoads, U_np, PExt, ts)
        #     # PExt = self.computeBodyForces(bodyForces, U_np, PExt, ts)
        #     #
        #     # # first stage
        #     # MInverse = 1.0 / M.T
        #     #
        #     # R[:] = P + PExt
        #     #
        #     # # enforce dirichlet boundary conditions
        #     # for dirichlet in dirichlets:
        #     #     R[self.findDirichletIndices(dirichlet)] = 0.0
        #     #
        #     # # update velocity vector with lumped mass matrix
        #     # v3 = MInverse * R * timeStep.timeIncrement
        #     # dU3 = v3 * timeStep.timeIncrement
        #     #
        #     #
        #     # # stage 3
        #     # U_np[:] = U_n
        #     # U_np += dU3
        #     # dU[:] = dU3
        #     # ts =  timeStep
        #     #
        #     # P[:] = M[:] = PExt[:] = 0.0
        #     # P, M = self.computeElements(elements, U_np, dU, P, M, ts)
        #     # PExt = self.computeDistributedLoads(distributedLoads, U_np, PExt, ts)
        #     # PExt = self.computeBodyForces(bodyForces, U_np, PExt, ts)
        #     #
        #     # # first stage
        #     # MInverse = 1.0 / M.T
        #     #
        #     # R[:] = P + PExt
        #     #
        #     # # enforce dirichlet boundary conditions
        #     # for dirichlet in dirichlets:
        #     #     R[self.findDirichletIndices(dirichlet)] = 0.0
        #     #
        #     # # update velocity vector with lumped mass matrix
        #     # v4 = MInverse * R * timeStep.timeIncrement
        #     # dU4 = v4 * timeStep.timeIncrement
        #     # V += 1 / 6 * (v1 + 2*v2 + 2*v3 + v4)
        # else:

        # ts
        P[:] = M[:] = PExt[:] = 0.0
        P, M = self.computeElements(elements, U_np, dU, P, M, prevTimeStep)
        PExt = self.computeDistributedLoads(distributedLoads, U_np, PExt, prevTimeStep)
        PExt = self.computeBodyForces(bodyForces, U_np, PExt, prevTimeStep)

        R[:] = P + PExt

        # enforce dirichlet boundary conditions
        for dirichlet in dirichlets:
            R[self.findDirichletIndices(dirichlet)] = 0.0

        # update velocity vector with lumped mass matrix
        V += 1.0 / M.T * R * 0.5 * (timeStep.timeIncrement + prevTimeStep.timeIncrement)

        for dirichlet in dirichlets:
            V[self.findDirichletIndices(dirichlet)] = dirichlet.getDelta(timeStep).flatten() / timeStep.timeIncrement

        # update displacement increment vector
        inc = V * timeStep.timeIncrement
        dU[:] = inc
        # enforce dirichlet boundary conditions
        for dirichlet in dirichlets:
            dU[self.findDirichletIndices(dirichlet)] = dirichlet.getDelta(timeStep).flatten()

        # update displacement vector
        U_np[:] = U_n + dU

        P[:] = M[:] = PExt[:] = 0.0
        P, M = self.computeElements(elements, U_np, dU, P, M, timeStep)
        PExt = self.computeDistributedLoads(distributedLoads, U_np, PExt, timeStep)
        PExt = self.computeBodyForces(bodyForces, U_np, PExt, timeStep)

        P += PExt

        return U_np, dU, V, P

    def computeDistributedLoads(
        self,
        distributedLoads: list[StepActionBase],
        U_np: DofVector,
        PExt: DofVector,
        timeStep: TimeStep,
    ) -> DofVector:
        """Loop over all distributed loads acting on elements, and evaluate them.
        Assembles into the global external load vector.

        Parameters
        ----------
        distributedLoads
            The list of distributed loads.
        U_np
            The current solution vector.
        PExt
            The external load vector to be augmented.
        timeStep
            The current time step.

        Returns
        -------
        DofVector
            The augmented load vector.
        """

        tic = getCurrentTime()

        time = np.array([timeStep.stepTime, timeStep.totalTime])
        dT = timeStep.timeIncrement

        for dLoad in distributedLoads:
            load = dLoad.getCurrentLoad(timeStep)
            for faceID, elementSet in dLoad.surface.items():
                for el in elementSet:
                    Pe = np.zeros(el.nDof)
                    Ke = np.zeros((el.nDof, el.nDof)).ravel()
                    el.computeDistributedLoad(dLoad.loadType, Pe, Ke, faceID, load, U_np[el], time, dT)

                    PExt[el] += Pe

        toc = getCurrentTime()
        self.computationTimes["distributed loads"] += toc - tic
        return PExt

    def computeBodyForces(
        self,
        bodyForces: list[StepActionBase],
        U_np: DofVector,
        PExt: DofVector,
        timeStep: TimeStep,
    ) -> DofVector:
        """Loop over all body forces loads acting on elements, and evaluate them.
        Assembles into the global external load vector and the system matrix.

        Parameters
        ----------
        distributedLoads
            The list of distributed loads.
        U_np
            The current solution vector.
        PExt
            The external load vector to be augmented.
        increment
            The increment.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            The augmented load vector and system matrix.
        """

        tic = getCurrentTime()

        time = np.array([timeStep.stepTime, timeStep.totalTime])
        dT = timeStep.timeIncrement

        for bForce in bodyForces:
            force = bForce.getCurrentLoad(timeStep)
            for el in bForce.elementSet:
                Pe = np.zeros(el.nDof)
                Ke = np.zeros((el.nDof, el.nDof)).ravel()

                el.computeBodyForce(Pe, Ke, force, U_np[el], time, dT)

                PExt[el] += Pe

        toc = getCurrentTime()
        self.computationTimes["body forces"] += toc - tic
        return PExt

    def computeElements(
        self,
        elements: list,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        M: DofVector,
        timeStep: TimeStep,
    ) -> tuple[DofVector, DofVector]:
        """Loop over all elements, and evalute them.
        Is is called by solveStep() in each iteration.

        Parameters
        ----------
        elements
            The list of finite elements.
        U_n
            The current solution vector.
        dU
            The  solution increment vector.
        P
            The reaction vector.
        timeStep
            The time step.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix,DofVector]
            - The modified reaction vector.
            - The modified system matrix.
            - The modified accumulated flux vector.
        """

        tic = getCurrentTime()

        time = np.array([timeStep.stepTime, timeStep.totalTime])
        dT = timeStep.timeIncrement
        P[:] = M[:] = 0.0
        for el in elements.values():
            Pe = np.zeros(el.nDof)
            Me = np.zeros(el.nDof)
            Ke = np.zeros((el.nDof, el.nDof)).ravel()
            el.computeYourself(Ke, Pe, U_np[el], dU[el], time, dT)
            el.computeLumpedInertia(Me)

            P[el] += Pe
            M[el] += Me
        toc = getCurrentTime()
        self.computationTimes["elements"] += toc - tic

        return P, M

    def assembleConstraints(
        self,
        constraints: list[ConstraintBase],
        U_np: DofVector,
        dU: DofVector,
        PExt: DofVector,
        timeStep: TimeStep,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Loop over all elements, and evaluate them.
        Is is called by solveStep() in each iteration.

        Parameters
        ----------
        constraints
            The list of constraints.
        U_np
            The current solution vector.
        dU
            The current solution increment vector.
        PExt
            The external load vector.
        K
            The system matrix.
        dT
            The time increment.
        time
            The step and total time.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix,DofVector]
            - The modified external load vector.
            - The modified system matrix.
        """

        tic = getCurrentTime()

        for constraint in constraints.values():
            # Kc = K[constraint].reshape(constraint.nDof, constraint.nDof, order="F")
            Pc = np.zeros(constraint.nDof)

            constraint.applyConstraint(U_np[constraint], dU[constraint], Pc, timeStep)

            # instead of PExt[constraint] += Pe, np.add.at allows for repeated indices
            np.add.at(PExt, PExt.entitiesInDofVector[constraint], Pc)

        toc = getCurrentTime()
        self.computationTimes["constraints"] += toc - tic

        return PExt

    def assembleLoads(
        self,
        nodeForces: list[StepActionBase],
        distributedLoads: list[StepActionBase],
        bodyForces: list[StepActionBase],
        U_np: DofVector,
        PExt: DofVector,
        K: VIJSystemMatrix,
        timeStep: TimeStep,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Assemble all loads into a right hand side vector.

        Parameters
        ----------
        nodeForces
            The list of concentrated (nodal) loads.
        distributedLoads
            The list of distributed (surface) loads.
        bodyForces
            The list of body (volumetric) loads.
        U_np
            The current solution vector.
        PExt
            The external load vector.
        K
            The system matrix.
        timeStep
            The current time step.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            - The augmented external load vector.
            - The augmented system matrix.
        """
        for cLoad in nodeForces:
            PExt[
                self.theDofManager.idcsOfFieldsOnNodeSetsInDofVector[cLoad.field][cLoad.nodeSet]
            ] += cLoad.getCurrentLoad(timeStep).flatten()
        PExt = self.computeDistributedLoads(distributedLoads, U_np, PExt, timeStep)
        PExt = self.computeBodyForces(bodyForces, U_np, PExt, timeStep)

        return PExt

    def applyStepActionsAtStepStart(self, model: FEModel, stepActions: dict[str, StepActionBase]):
        """Called when all step actions should be appliet at the start a step.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The dictionary of active step actions.
        """

        for stepActionType in stepActions.values():
            for action in stepActionType.values():
                action.applyAtStepStart(model)

    def applyStepActionsAtStepEnd(self, model: FEModel, stepActions: dict[str, StepActionBase]):
        """Called when all step actions should finish a step.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The dictionary of active step actions.
        """

        for stepActionType in stepActions.values():
            for action in stepActionType.values():
                action.applyAtStepEnd(model)

    def applyStepActionsAtIncrementStart(
        self, model: FEModel, timeStep: TimeStep, stepActions: dict[str, StepActionBase]
    ):
        """Called when all step actions should be applied at the start of a step.

        Parameters
        ----------
        model
            The model tree.
        increment
            The time increment.
        stepActions
            The dictionary of active step actions.
        """

        for stepActionType in stepActions.values():
            for action in stepActionType.values():
                action.applyAtIncrementStart(model, timeStep)

    def findDirichletIndices(self, dirichlet):
        nSet = dirichlet.nSet
        field = dirichlet.field
        components = dirichlet.components

        fieldIndices = self.theDofManager.idcsOfFieldsOnNodeSetsInDofVector[field][nSet]

        return fieldIndices.reshape((len(nSet), -1))[:, components].flatten()
