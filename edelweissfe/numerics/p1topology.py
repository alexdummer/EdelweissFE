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
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""Corner/midside node topology for building a P1 (linear) restriction operator over a quadratic
serendipity displacement mesh (§22, the p-multigrid enabler).

The projection :math:`P` is purely topological: identity on corner nodes, ½/½ on each exclusive
midside node from its two edge-endpoint corners (the P1 function expressed in the serendipity
basis) -- so building it only requires classifying every node of a vector field as a corner or an
exclusive midside, with its two edge-endpoint corners if the latter.
"""

import numpy as np

#: ``(nSpatialDimensions, nNodes) -> (cornerLocalIndices, [(midsideLocal, cornerALocal, cornerBLocal), ...])``
#: the only two quadratic-serendipity element families in this codebase that carry exclusive
#: midside nodes (verified against the actual shape functions, both the pure-Python
#: ``edelweissfe.elements.displacementelement`` and Marmot's ``DisplacementFiniteElement``, and
#: cross-checked against ``edelweissfe.adaptivity.hex20shapefunctions.EDGES`` for Hexa20).
_QUAD8_TOPOLOGY = ([0, 1, 2, 3], [(4, 0, 1), (5, 1, 2), (6, 2, 3), (7, 3, 0)])
_HEXA20_TOPOLOGY = (
    list(range(8)),
    [
        (8, 0, 1),
        (9, 1, 2),
        (10, 2, 3),
        (11, 3, 0),
        (12, 4, 5),
        (13, 5, 6),
        (14, 6, 7),
        (15, 7, 4),
        (16, 0, 4),
        (17, 1, 5),
        (18, 2, 6),
        (19, 3, 7),
    ],
)
_QUADRATIC_TOPOLOGY = {(2, 8): _QUAD8_TOPOLOGY, (3, 20): _HEXA20_TOPOLOGY}

#: node counts verified, against this repo's full element-type inventory (every ``*element,
#: type=...`` across ``testfiles/``), to carry no midside nodes at all whenever the
#: ``(nSpatialDimensions, nNodes)`` pair is not one of the two quadratic families above -- e.g.
#: ``C3D8``/``GC3D8*`` (3, 8), ``GCPS4`` (2, 4), ``T2D2`` (2, 2). A contact facet has no
#: ``nSpatialDimensions`` of its own at all and is always pure-corner regardless of node count,
#: since a facet element cannot itself carry a midside concept.
_KNOWN_LINEAR_NODE_COUNTS = {1, 2, 3, 4, 6, 8}


def classifyElementTopology(nSpatialDimensions, nNodes: int):
    """Return ``(cornerLocalIndices, edgeEndpointsLocal)`` for one element's local node numbering.

    Parameters
    ----------
    nSpatialDimensions
        The element's spatial dimension, or ``None`` if unavailable (e.g. a contact facet, always
        treated as pure-corner regardless of node count).
    nNodes
        The element's node count.

    Returns
    -------
    tuple
        ``cornerLocalIndices`` (list of int) and ``edgeEndpointsLocal`` (list of
        ``(midsideLocalIndex, cornerALocalIndex, cornerBLocalIndex)``, empty for a pure-corner
        element).

    Raises
    ------
    ValueError
        If ``nNodes`` is neither a recognized quadratic-serendipity count for
        ``nSpatialDimensions`` nor among the verified pure-corner node counts -- silently
        misclassifying an unrecognized element's midside nodes as corners would quietly degrade
        the resulting P1 operator, so this fails loudly instead.
    """
    key = (nSpatialDimensions, nNodes)
    if key in _QUADRATIC_TOPOLOGY:
        return _QUADRATIC_TOPOLOGY[key]
    if nSpatialDimensions is None or nNodes in _KNOWN_LINEAR_NODE_COUNTS:
        return list(range(nNodes)), []
    raise ValueError(
        "p1topology: unrecognized element topology (nSpatialDimensions={:}, nNodes={:}) -- neither "
        "a known quadratic-serendipity family ((2, 8) Quad8, (3, 20) Hexa20) nor a verified linear "
        "node count. Add its corner/midside topology before including it in a P1 map.".format(
            nSpatialDimensions, nNodes
        )
    )


def buildP1Map(model, fieldName: str):
    """Classify every node of ``fieldName`` (a vector field, e.g. ``"displacement"``) as a corner
    or an exclusive midside, in the same order as ``model.nodeFields[fieldName].nodes`` (the
    field's own DOF-vector row order, see ``DofManager._reserveSpaceForNodeFields``).

    A node is a corner iff it is a corner node of *at least one* element it belongs to -- load-
    bearing under AMR, where a hanging node can be a midside of a coarse element and a corner of
    the fine elements replacing it; corner status wins, and every element's own contribution only
    ever sets ``isCorner`` to ``True``, never clears it.

    Parameters
    ----------
    model
        The model tree.
    fieldName
        The vector field to classify (e.g. ``"displacement"``).

    Returns
    -------
    isCorner : np.ndarray
        Boolean, shape ``(nNodes,)``.
    edgeEndpoints : np.ndarray
        Int, shape ``(nNodes, 2)``, ``-1`` for corner rows; for an exclusive midside row, the two
        edge-endpoint corner rows (both guaranteed corners themselves, by construction).
    """
    field = model.nodeFields[fieldName]
    nodeRows = {node: i for i, node in enumerate(field.nodes)}
    nNodes = len(field.nodes)
    isCorner = np.zeros(nNodes, dtype=bool)
    # provisional -- a node classified as a midside by the element seen first may turn out to be a
    # corner of a *different* element processed later (the AMR corner-wins case), so isCorner and
    # the edge-endpoint candidates are collected in one pass over every element first, and only
    # resolved into the final per-row result afterward -- writing straight into a shared
    # `edgeEndpoints` array during this loop would leave a stale entry for exactly that case.
    provisionalEdges: dict[int, frozenset] = {}

    for element in model.elements.values():
        if not any(fieldName in nodeFields for nodeFields in element.fields):
            continue

        try:
            dim = element.nSpatialDimensions
        except AttributeError:
            dim = None
        elNodes = element.nodes
        cornersLocal, edgesLocal = classifyElementTopology(dim, len(elNodes))
        elRows = [nodeRows.get(node) for node in elNodes]

        for c in cornersLocal:
            row = elRows[c]
            if row is not None:
                isCorner[row] = True

        for midLocal, cALocal, cBLocal in edgesLocal:
            row = elRows[midLocal]
            if row is None:
                continue
            candidate = frozenset((elRows[cALocal], elRows[cBLocal]))
            existing = provisionalEdges.get(row)
            if existing is not None and existing != candidate:
                raise AssertionError(
                    "p1topology: inconsistent edge endpoints recorded for the same midside node -- "
                    "a mesh-topology bug, not a numerical tolerance issue."
                )
            provisionalEdges[row] = candidate

    edgeEndpoints = -np.ones((nNodes, 2), dtype=int)
    for row in range(nNodes):
        if isCorner[row]:
            continue
        candidate = provisionalEdges.get(row)
        if candidate is None:
            raise AssertionError(
                "p1topology: node {:} of field '{:}' is neither a corner of any element nor has "
                "recorded edge endpoints -- every non-corner node must be an exclusive midside of "
                "at least one element.".format(row, fieldName)
            )
        edgeEndpoints[row] = tuple(candidate)

    return isCorner, edgeEndpoints
