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
        "defaultMaxIter": 10,
        "defaultCriticalIter": 5,
        "defaultMaxGrowingIter": 10,
        "extrapolation": "linear",
        "linsolver": "pardiso",
    }

    def __init__(self, jobInfo, journal, **kwargs):
        self.journal = journal

        # self.fieldCorrectionTolerances = jobInfo["fieldCorrectionTolerance"]
        # self.fluxResidualTolerances = jobInfo["fluxResidualTolerance"]
        # self.fluxResidualTolerancesAlt = jobInfo["fluxResidualToleranceAlternative"]

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
                self.options[k] = type(self.NISTOptions[k])(updatedOptions[k])
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

        nVariables = len(presentVariableNames)
        self.iterationHeader = ("{:^25}" * nVariables).format(*presentVariableNames)
        self.iterationHeader2 = (" {:<10}  {:<10}  ").format("||R||∞", "||ddU||∞") * nVariables
        self.iterationMessageTemplate = "{:11.2e}{:1}{:11.2e}{:1} "

        # U = self.theDofManager.constructDofVector()
        # V = self.theDofManager.constructDofVector()
        # K = self.theDofManager.constructVIJSystemMatrix()

        # self.csrGenerator = CSRGenerator(K)

        self.computationTimes = createTimingDict()

        try:
            self._updateOptions(step.actions["options"]["NISTSolver"].options, self.journal)
        except KeyError:
            pass

        # extrapolation = self.options["extrapolation"]
        # self.linSolver = (
        #     getLinSolverByName(self.options["linsolver"]) if "linsolver" in self.options else getDefaultLinSolver()
        # )

        # maxIter = step.maxIter
        # criticalIter = step.criticalIter
        # maxGrowingIter = step.maxGrowIter

        # nodes = model.nodes
        # elements = model.elements
        # constraints = model.constraints
        M = scipy.sparse.diags(np.zeros(self.theDofManager.nDof), format="diag")  # initialize lumped mass matrix
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
                self.journal.message(self.iterationHeader, self.identification, level=2)
                self.journal.message(self.iterationHeader2, self.identification, level=2)

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

                    # statusInfoDict["iters"] = iterationCounter
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
        dU: DofVector,
        V: DofVector,
        P: DofVector,
        M: scipy.sparse.diags,
        stepActions: list,
        model: FEModel,
        timeStep: TimeStep,
        prevTimeStep: TimeStep,
        extrapolation: str,
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
        constraints = model.constraints

        R = self.theDofManager.constructDofVector()
        F = self.theDofManager.constructDofVector()
        PExt = self.theDofManager.constructDofVector()
        U_np = self.theDofManager.constructDofVector()

        dirichlets = stepActions["dirichlet"].values()
        nodeforces = stepActions["nodeforces"].values()
        distributedLoads = stepActions["distributedload"].values()
        bodyForces = stepActions["bodyforce"].values()

        self.applyStepActionsAtIncrementStart(model, timeStep, stepActions)

        # dU, isExtrapolatedIncrement = self.extrapolateLastIncrement(
        #     extrapolation, timeStep, dU, dirichlets, prevTimeStep, model
        # )

        for geostatic in stepActions["geostatic"].values():
            geostatic.applyAtIterationStart()

        U_np[:] = U_n

        P[:] = M[:] = F[:] = PExt[:] = 0.0

        P, M, F = self.computeElements(elements, U_np, dU, P, F, timeStep)
        PExt = self.assembleLoads(nodeforces, distributedLoads, bodyForces, U_np, PExt, timeStep)
        PExt = self.assembleConstraints(constraints, U_np, dU, PExt, timeStep)

        R[:] = P
        R += PExt
        U_np += dU

        for dirichlet in dirichlets:
            R[self.findDirichletIndices(dirichlet)] = 0.0

        V += M.inv() * R * timeStep.timeIncrement
        for dirichlet in dirichlets:
            V[self.findDirichletIndices(dirichlet)] = dirichlet.getDelta(timeStep).flatten() / timeStep.timeIncrement

        dU = V * timeStep.timeIncrement

        U_np += dU

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

                    el.computeDistributedLoadWithoutStiffness(dLoad.loadType, Pe, faceID, load, U_np[el], time, dT)

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

                el.computeBodyForceWithoutStiffness(Pe, force, U_np[el], time, dT)

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
        M: scipy.sparse.diags,
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

        for el in elements.values():
            Pe = np.zeros(el.nDof)

            el.computeYourselfWithoutStiffness(Pe, U_np[el], dU[el], time, dT)

            P[el] += Pe

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
        PExt = self.computeDistributedLoadsWithoutStiffness(distributedLoads, U_np, PExt, timeStep)
        PExt = self.computeBodyForcesWithoutStiffness(bodyForces, U_np, PExt, timeStep)

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
