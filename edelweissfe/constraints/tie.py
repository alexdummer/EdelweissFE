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

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from edelweissfe.constraints.base.multipointconstraintbase import (
    MultiPointConstraintBase,
)
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.facetcontactgeometry import line2ClosestPoint, tria3ClosestPoint
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
An Abaqus-style surface-to-surface tie constraint, bonding the nodes of a slave surface rigidly to
a deformable master surface via master-slave DOF elimination (multi-point constraint
condensation). Both surfaces are represented by flat contact facet elements
(:mod:`~edelweissfe.elements.contactsurfaceelement`, typically created via
:mod:`~edelweissfe.generators.surfaceelementgenerator`). Each slave node is projected onto its
closest master facet in the reference configuration; the clamped closest-point weights are frozen
and every displacement component of the slave node is constrained to the identically-weighted
master interpolation. The constraint is enforced exactly -- no penalty parameter, no Lagrange
multiplier DOFs -- and adds zero stiffness, which in particular leaves the critical time step of
explicit dynamics untouched.

This constraint is a :class:`~edelweissfe.models.meshdependent.MeshDependent`: if either surface's
source solid elements are refined mid-run (e.g. by :mod:`~edelweissfe.modelmodifiers.adaptivity.
hadaptivity`), it regenerates that side's facets and re-projects the tied records -- no separate
wiring needed. Unlike the penalty contact constraint, a tie has no per-increment tick of its own
that runs *before* the DofManager/system matrix is rebuilt (its only hook,
:meth:`getMultiPointConstraints`, is called from inside that rebuild -- too late to safely swap in
freshly regenerated facet elements), so it reconciles via the model's push notification instead,
synchronously as part of the mesh-mutating modifier's own call. The re-projection never re-adjusts
slave node coordinates regardless of the ``adjust`` setting: that snap is a setup-time convenience
for removing an initial geometric gap, not something to repeat on an already-loaded, already-tied
node.
"""


def _facetCharacteristicSize(facetCoords: np.ndarray) -> float:
    """Mean edge length of a facet (Line2: its one edge; Tria3: its three edges) -- the local
    length scale used for the default position tolerance when none is given explicitly.

    Parameters
    ----------
    facetCoords
        The facet's node coordinates, shape (2, nDim) for a Line2 or (3, nDim) for a Tria3.

    Returns
    -------
    float
        The facet's mean edge length.
    """

    nNodes = facetCoords.shape[0]
    edgeLengths = [np.linalg.norm(facetCoords[i] - facetCoords[(i + 1) % nNodes]) for i in range(nNodes)]
    return np.mean(edgeLengths)


@dataclass(frozen=True)
class TieSchema:
    """L2: the options this constraint accepts, owned by this module and never mutated from
    outside it.
    """

    slaveSurface: str | None = schemaField(
        description="The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet) "
        "forming the slave surface; its nodes are tied. For quadratic (hexa20/quad8) faces, generate "
        "the facets with triangulation=midside on BOTH surfaces -- the corner triangulation excludes "
        "the midside nodes from the facet node list entirely, leaving them untied.",
        dtype=str,
        default=None,
        required=True,
    )
    masterSurface: str | None = schemaField(
        description="The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet) "
        "forming the master surface.",
        dtype=str,
        default=None,
        required=True,
    )
    positionTolerance: float | None = schemaField(
        description="Slave nodes whose reference-configuration closest-point distance to the "
        "master surface exceeds this tolerance are left untied (recorded in the constraint's "
        "untiedSlaveNodes), matching Abaqus' *TIE default behavior of silently dropping "
        "out-of-range slave nodes. If not given and adjust=True (the default combination), a "
        "tolerance is computed per slave node as positionToleranceFactor times the characteristic "
        "edge length of that node's closest master facet -- see positionToleranceFactor. If not "
        "given and adjust=False, every slave node ties unconditionally, however far the closest "
        "master point is (see adjust). Set this explicitly to an absolute distance to override "
        "either default.",
        dtype=float,
        default=None,
    )
    positionToleranceFactor: float = schemaField(
        description="Used only when positionTolerance is not given and adjust=True: the default "
        "tolerance for a slave node is this fraction of its closest master facet's own "
        "characteristic (mean) edge length, so it scales with local mesh density instead of being "
        "one fixed number for the whole surface. 0.25 comfortably exceeds the sub-percent gaps "
        "expected between two compatible discretizations of the same surface (mismatched density, "
        "curvature/interpolation error), while remaining well below the facet-size-or-larger gaps "
        "that indicate the surfaces don't actually correspond (e.g. a slave surface extending "
        "beyond the master surface's actual extent -- a partial-bond-length or otherwise "
        "partially-overlapping pair of surfaces). Each node's computed default is frozen the first "
        "time that node is evaluated (construction, or first appearance on a later reconcile) and "
        "reused from then on, so a subsequent AMR refinement of the master surface elsewhere cannot "
        "retroactively shrink an already-tied node's tolerance and untie it. Not used at all with "
        "adjust=False -- see positionTolerance and adjust.",
        dtype=float,
        default=0.25,
    )
    adjust: bool = schemaField(
        description="Move each tied slave node onto its closest master point at construction "
        "(Abaqus-like default). If False, any initial geometric gap between the surfaces is "
        "preserved rigidly (the displacements are tied, not the positions), AND the computed "
        "default position tolerance (see positionTolerance/positionToleranceFactor) does not apply: "
        "every slave node ties unconditionally regardless of distance. adjust=False is itself the "
        "signal that a real, deliberate gap exists and should be tied across regardless of size -- "
        "second-guessing that with a mesh-size-derived cutoff would contradict the setting just "
        "chosen; give an explicit positionTolerance instead if you want a cutoff together with "
        "adjust=False. Note that adjusting modifies the nodal coordinates before the element "
        "geometry is initialized; avoid adjusting nodes that also belong to an already-generated "
        "contact surface of another constraint.",
        dtype=bool,
        default=True,
    )


class Constraint(MultiPointConstraintBase, MeshDependent):
    """
    An Abaqus-style surface-to-surface tie constraint via master-slave DOF elimination.

    Theoretical background
    -----------------------
    Each slave node is projected onto its closest master facet once, in the reference
    configuration, using the clamped closest-point search shared with the small-sliding contact
    formulation. The resulting non-negative facet weights :math:`N_a` are frozen, and every
    displacement component of the slave node is constrained linearly to the master facet nodes:

    .. math::
        u_s = \\sum_a N_a \\, u_{m_a}.

    The constraint records are collected by the solver and enforced by condensing the slave DOFs
    out of the equation system (see
    :class:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation`) -- the
    identical mechanism serves the implicit solvers (system matrix transformation) and explicit
    dynamics (mass/force folding onto the masters, direct kinematic slaving). The enforcement is
    exact for arbitrary meshes; the interpolation across a master facet is the facet's own linear
    one, so a linear displacement field is transferred exactly (patch test) for matching and
    non-matching meshes alike, on straight-edged hexa20 faces included.

    Currently only the 'displacement' field is tied. Available for spatialdomain = 3D (Tria3
    facets) and 2D (Line2 facets).
    """

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = TieSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        slaveSurface: ElementSet,
        masterSurface: ElementSet,
        *,
        configuration: TieSchema = TieSchema(),
    ):
        self.name = name
        self.nDim = model.domainSize

        self._journal = Journal()
        self._slaveSurfaceSetName = slaveSurface.name
        self._masterSurfaceSetName = masterSurface.name
        self._positionTolerance = configuration.positionTolerance
        self._positionToleranceFactor = configuration.positionToleranceFactor
        #: The constraint's originally configured adjust setting -- distinct from the local
        #: ``adjust`` parameter _buildTiedRecords receives on each call, which reconcile() always
        #: pins to False (never re-snap an already-loaded node) regardless of this. Used to decide
        #: whether the computed default tolerance applies at all: see _buildTiedRecords.
        self._adjustConfigured = configuration.adjust
        #: Per-node computed default tolerance, frozen the first time a node is evaluated (see
        #: _buildTiedRecords). Only touched when positionTolerance is None and adjust=True.
        self._defaultToleranceCache = {}

        slaveFacetElements = list(slaveSurface)
        masterFacetElements = list(masterSurface)

        masterNodes = {node for el in masterFacetElements for node in el.nodes}
        slaveNodesForCheck = {node for el in slaveFacetElements for node in el.nodes}
        if not masterNodes.isdisjoint(slaveNodesForCheck):
            raise ValueError(
                f"Constraint '{name}': slave surface '{self._slaveSurfaceSetName}' and master "
                f"surface '{self._masterSurfaceSetName}' share nodes -- a node cannot be tied to "
                "itself."
            )

        self.tiedRecords, self.untiedSlaveNodes = self._buildTiedRecords(
            model, slaveFacetElements, masterFacetElements, adjust=configuration.adjust
        )

        # A tie has no per-increment tick of its own that runs before the DofManager/VIJSystemMatrix
        # rebuild -- getMultiPointConstraints() is only called from inside that rebuild, too late to
        # safely swap in newly regenerated facet elements (the just-built system wouldn't know about
        # them). So, unlike nodeToDeformableSurfacePenalty, a tie reconciles via the push escape
        # hatch: model.notifyModelChanged() calls onModelChanged() synchronously, from inside the
        # model modifier's own updateModel(), strictly before the rebuild decision is even made.
        model.registerObserver(self)

    @classmethod
    def fromConstraintDefinition(cls, name: str, definition: dict, model: FEModel) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition. See
        :class:`~edelweissfe.constraints.base.multipointconstraintbase.MultiPointConstraintBase`
        for why this is separate from ``__init__``."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(
            name,
            model,
            model.elementSets[configuration.slaveSurface],
            model.elementSets[configuration.masterSurface],
            configuration=configuration,
        )

    def onModelChanged(self, model: FEModel, changeType, change) -> None:
        if change is not None:
            self.reconcile(model, change)

    def _buildTiedRecords(self, model: FEModel, slaveFacetElements, masterFacetElements, adjust: bool):
        """Project every unique slave-surface node onto its closest master facet (reference
        configuration) and freeze the resulting clamped weights. With ``adjust``, additionally snap
        each tied node onto its projected point, removing any initial geometric gap -- a setup-time
        convenience, never applied on a reconcile-triggered re-projection.

        Slave nodes already claimed as slaves by another multi-point constraint of the model are
        skipped: a DOF may be condensed out only once, and a second record for it would be rejected
        by the condensation operator. Dropping the tie record is exact, not a silencer -- the typical
        case is a hanging node created by adaptive refinement of the slave surface, whose masters are
        the coarse-trace nodes of that very surface and are themselves tied, so its interpolated
        motion already is the tied motion and the tie equation is redundant.

        When positionTolerance is not given AND the constraint was configured with adjust=True, each
        node's default tolerance (positionToleranceFactor times its closest master facet's
        characteristic size) is computed once, the first time that node is evaluated, and cached in
        self._defaultToleranceCache from then on -- NOT recomputed from the facet size current at
        reconcile time (refining the master surface (AMR) shrinks nearby facets and thus the
        tolerance, which would otherwise silently untie an already-tied node purely because unrelated
        mesh refinement happened nearby). With adjust=False, the default tolerance does not apply at
        all (every node ties unconditionally, as before this feature was added): adjust=False is
        itself the user's explicit signal that a real, deliberate gap exists and should be preserved
        and tied across regardless of size (e.g. AMR_TieRefineShear's persistent 0.05 gap) -- second-
        guessing that with a mesh-size-derived cutoff would contradict the very setting the user
        chose. positionTolerance remains available to add an explicit, absolute cutoff on top of
        adjust=False if that combination is ever wanted."""

        slaveNodes = list(dict.fromkeys(node for el in slaveFacetElements for node in el.nodes))

        alreadyClaimedNodes = set()
        for constraint in model.multiPointConstraints.values():
            if constraint is not self:
                alreadyClaimedNodes |= constraint.claimedSlaveNodes()

        closestPointFunction = tria3ClosestPoint if self.nDim == 3 else line2ClosestPoint
        masterFacetCoords = [np.array([n.coordinates for n in el.nodes]) for el in masterFacetElements]
        masterFacetCharacteristicSizes = [_facetCharacteristicSize(coords) for coords in masterFacetCoords]

        # Projecting every slave node onto its closest master facet by brute force is O(nSlave *
        # nMaster) and, on a large tied surface re-projected after every AMR refinement, dominates
        # the whole refinement step. Index the master facets by centroid in a k-d tree instead and,
        # per slave node, run the exact closest-point test only on the facets that could possibly win.
        # The candidate set is a *superset* of the brute-force winner (see maxFacetRadius below), and
        # iterating it in ascending facet order with a strict "<" reproduces the brute-force choice
        # exactly (same first-of-equals tie-break) -- so this is an acceleration, not an approximation.
        if masterFacetCoords:
            centroids = np.array([coords.mean(axis=0) for coords in masterFacetCoords])
            facetTree = cKDTree(centroids)
            # The farthest any facet vertex sits from that facet's own centroid. If a facet's true
            # closest point beats a known distance d, its centroid lies within d + maxFacetRadius of
            # the slave node (triangle inequality), so a ball query of that radius, seeded with the
            # exact distance to the nearest-centroid facet (an upper bound on the global minimum),
            # is guaranteed to include the brute-force winner.
            maxFacetRadius = max(
                float(np.linalg.norm(coords - coords.mean(axis=0), axis=1).max()) for coords in masterFacetCoords
            )

        tiedRecords = []
        untiedSlaveNodes = []
        nSkippedClaimedNodes = 0

        for slaveNode in slaveNodes:
            if slaveNode in alreadyClaimedNodes:
                nSkippedClaimedNodes += 1
                continue

            bestWeights = None
            bestFacetIdx = None
            bestDistance = np.inf

            if masterFacetCoords:
                xs = slaveNode.coordinates
                _, seedIdx = facetTree.query(xs, k=1)
                _, seedDistance = closestPointFunction(xs, *masterFacetCoords[int(seedIdx)])
                candidates = set(facetTree.query_ball_point(xs, seedDistance + maxFacetRadius))
                candidates.add(int(seedIdx))
                for facetIdx in sorted(candidates):
                    weights, distance = closestPointFunction(xs, *masterFacetCoords[facetIdx])
                    if distance < bestDistance:
                        bestDistance = distance
                        bestWeights = weights
                        bestFacetIdx = facetIdx

            # Abaqus' *TIE always enforces some tolerance (an explicit POSITION TOLERANCE, or an
            # internally computed default) and silently drops slave nodes outside it -- it never
            # ties unconditionally regardless of distance. Mirror that when adjust=True (the case
            # this is actually meant for: closing small gaps between surfaces that are meant to
            # coincide). Freeze that computed default the first time this node is evaluated (see the
            # cache note in this method's docstring) rather than recomputing it from whatever facet
            # size a later reconcile happens to see. With adjust=False, skip this entirely -- see the
            # docstring for why.
            if self._positionTolerance is not None:
                effectiveTolerance = self._positionTolerance
            elif self._adjustConfigured:
                if slaveNode not in self._defaultToleranceCache:
                    self._defaultToleranceCache[slaveNode] = (
                        self._positionToleranceFactor * masterFacetCharacteristicSizes[bestFacetIdx]
                    )
                effectiveTolerance = self._defaultToleranceCache[slaveNode]
            else:
                effectiveTolerance = np.inf

            if bestDistance > effectiveTolerance:
                untiedSlaveNodes.append(slaveNode)
                continue

            if adjust and bestDistance > 0.0:
                slaveNode.coordinates[:] = bestWeights @ masterFacetCoords[bestFacetIdx]

            tiedRecords.append((slaveNode, masterFacetElements[bestFacetIdx].nodes, bestWeights))

        if nSkippedClaimedNodes:
            self._journal.message(
                "{:} slave node(s) are already slaves of another multi-point constraint "
                "(e.g. hanging nodes); their redundant tie records were dropped".format(nSkippedClaimedNodes),
                self.name,
            )

        return tiedRecords, untiedSlaveNodes

    def reconcile(self, model: FEModel, change) -> bool:
        """Regenerate whichever side's facets were affected by ``change`` (via its recorded
        :attr:`~edelweissfe.models.femodel.FEModel.contactFacetRecipes`) and re-project the tied
        records from scratch against the rebuilt surfaces -- never adjusting coordinates, since the
        tied nodes are already loaded mid-run."""

        slaveRecipe = model.contactFacetRecipes.get(self._slaveSurfaceSetName)
        masterRecipe = model.contactFacetRecipes.get(self._masterSurfaceSetName)
        touchedSlave = slaveRecipe is not None and change.touchesSurface(slaveRecipe[0])
        touchedMaster = masterRecipe is not None and change.touchesSurface(masterRecipe[0])
        if not (touchedSlave or touchedMaster):
            return False

        if touchedSlave:
            buildContactFacets(model, *slaveRecipe, self._journal)
        if touchedMaster:
            buildContactFacets(model, *masterRecipe, self._journal)

        slaveFacetElements = list(model.elementSets[self._slaveSurfaceSetName])
        masterFacetElements = list(model.elementSets[self._masterSurfaceSetName])
        self.tiedRecords, self.untiedSlaveNodes = self._buildTiedRecords(
            model, slaveFacetElements, masterFacetElements, adjust=False
        )
        return True

    def claimedSlaveNodes(self) -> set:
        """The tied slave nodes of this constraint. Overrides the base implementation, since a tie
        keeps its records in :attr:`tiedRecords` rather than in the base class' ``_records``."""

        return {slaveNode for slaveNode, _, _ in self.tiedRecords}

    def getMultiPointConstraints(self, dofManager) -> list[tuple[int, list[tuple[int, float]]]]:
        fieldVariableIndices = dofManager.idcsOfFieldVariablesInDofVector

        records = []
        for slaveNode, masterNodes, weights in self.tiedRecords:
            slaveDofIndices = fieldVariableIndices[slaveNode.fields["displacement"]]
            masterDofIndices = [fieldVariableIndices[node.fields["displacement"]] for node in masterNodes]

            for component in range(self.nDim):
                records.append(
                    (
                        slaveDofIndices[component],
                        [
                            (masterDofIdcs[component], weight)
                            for masterDofIdcs, weight in zip(masterDofIndices, weights)
                            if weight != 0.0
                        ],
                    )
                )

        return records
