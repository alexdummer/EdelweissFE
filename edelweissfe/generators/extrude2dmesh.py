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
A mesh generator that extrudes a 2D quad mesh to 3D.

The source mesh is read with meshio and may contain linear quads (quad)
or quadratic serendipity quads (quad8).

Arguments:
- meshFile: 2D mesh file path, readable by meshio.
- thickness: total extrusion thickness in z-direction.
- nElThickness: number of elements through thickness.
- elType: target 3D element type (for example C3D8 or C3D20).
"""

import meshio
import numpy as np

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

module = Module("extrude2dmesh", "Extrude a 2D quad mesh to a 3D mesh.")

inputLanguage = InputLanguage()

keyword = "modelGenerator"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredArg("meshFile", "Path to the 2D mesh file.", str)
module.addRequiredArg("thickness", "Total thickness in z direction.", float)
module.addRequiredArg("nElThickness", "Number of elements through thickness.", int)
module.addRequiredArg("elType", "Target 3D element type.", str)
module.addOptionalArg("elProvider", "Element provider.", str, None)

documentation = [module]

identification = "extrude2dmesh"


def _set_entry_to_indices(set_entry) -> np.ndarray:
    if set_entry is None:
        return np.array([], dtype=int)

    if isinstance(set_entry, (list, tuple)):
        indices = [_set_entry_to_indices(entry) for entry in set_entry]
        if not indices:
            return np.array([], dtype=int)
        return np.unique(np.concatenate(indices)).astype(int)

    indices = np.asarray(set_entry)
    if indices.dtype == bool:
        return np.where(indices)[0].astype(int)
    return indices.astype(int).ravel()


@caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
@castKwargsValuesAndAddDefaults(module)
def generateModelData(generatorDefinition: dict, model: FEModel, journal: Journal, *args, **kwargs) -> dict:
    kwargs = CaseInsensitiveDict(kwargs)

    name = generatorDefinition.get("name", "extrude2dmesh")

    mesh_file = kwargs["meshFile"]
    thickness = kwargs["thickness"]
    n_el_thickness = kwargs["nElThickness"]
    element_class = getElementClass(kwargs["elType"], kwargs["elProvider"])

    if n_el_thickness < 1:
        raise ValueError("nElThickness must be >= 1")

    mesh_2d = meshio.read(mesh_file)

    supported_cell_types = {"quad", "quad8"}
    selected_blocks = [
        (block_idx, block.type, np.asarray(block.data, dtype=int))
        for block_idx, block in enumerate(mesh_2d.cells)
        if block.type in supported_cell_types
    ]

    if not selected_blocks:
        raise ValueError("No supported 2D cells found in mesh. Supported cell types: quad, quad8")

    base_cell_type = selected_blocks[0][1]
    if any(cell_type != base_cell_type for _, cell_type, _ in selected_blocks):
        raise ValueError("Mixed 2D cell types are not supported. Use either quad or quad8 blocks only.")

    if base_cell_type == "quad":
        expected_n_nodes_3d = 8
        n_layers = n_el_thickness + 1
    else:
        expected_n_nodes_3d = 20
        n_layers = 2 * n_el_thickness + 1

    test_element = element_class(kwargs["elType"], 0)
    if test_element.nNodes != expected_n_nodes_3d:
        raise ValueError(
            "3D element type has {} nodes, but extrusion of {} requires {} nodes".format(
                test_element.nNodes, base_cell_type, expected_n_nodes_3d
            )
        )

    points = np.asarray(mesh_2d.points, dtype=float)
    if points.shape[1] == 2:
        points_3d = np.column_stack((points[:, 0], points[:, 1], np.zeros(points.shape[0], dtype=float)))
    elif points.shape[1] == 3:
        points_3d = points.copy()
    else:
        raise ValueError("Mesh points must have 2 or 3 coordinates")

    corner_node_ids = set()
    if base_cell_type == "quad8":
        for _, _, block_data in selected_blocks:
            corner_node_ids.update(np.ravel(block_data[:, :4]).tolist())

    node_labels_by_point_and_layer = {}
    nodes_created = []

    current_node_label = 1
    if model.nodes:
        current_node_label += max(model.nodes.keys())

    for layer in range(n_layers):
        if n_layers == 1:
            z_shift = 0.0
        else:
            z_shift = thickness * float(layer) / float(n_layers - 1)

        for point_id, point in enumerate(points_3d):
            # For quadratic serendipity extrusion, odd layers only host corner nodes.
            if base_cell_type == "quad8" and layer % 2 == 1 and point_id not in corner_node_ids:
                continue

            coordinates = np.array([point[0], point[1], point[2] + z_shift], dtype=float)
            node = Node(current_node_label, coordinates)
            model.nodes[current_node_label] = node
            node_labels_by_point_and_layer[(point_id, layer)] = current_node_label
            nodes_created.append(node)
            current_node_label += 1

    current_element_label = 1
    if model.elements:
        current_element_label += max(model.elements.keys())

    all_new_elements = []
    extruded_elements_by_block = []
    for block_idx, (_, _, block_data) in enumerate(selected_blocks):
        block_elements = []
        block_mappings = []

        for conn_2d in block_data:
            extruded_elements_for_source_element = []
            for k in range(n_el_thickness):
                if base_cell_type == "quad":
                    l_bottom = k
                    l_top = k + 1

                    node_labels = [
                        node_labels_by_point_and_layer[(conn_2d[0], l_bottom)],
                        node_labels_by_point_and_layer[(conn_2d[1], l_bottom)],
                        node_labels_by_point_and_layer[(conn_2d[2], l_bottom)],
                        node_labels_by_point_and_layer[(conn_2d[3], l_bottom)],
                        node_labels_by_point_and_layer[(conn_2d[0], l_top)],
                        node_labels_by_point_and_layer[(conn_2d[1], l_top)],
                        node_labels_by_point_and_layer[(conn_2d[2], l_top)],
                        node_labels_by_point_and_layer[(conn_2d[3], l_top)],
                    ]
                else:
                    l_bottom = 2 * k
                    l_mid = 2 * k + 1
                    l_top = 2 * k + 2

                    node_labels = [
                        node_labels_by_point_and_layer[(conn_2d[0], l_bottom)],
                        node_labels_by_point_and_layer[(conn_2d[1], l_bottom)],
                        node_labels_by_point_and_layer[(conn_2d[2], l_bottom)],
                        node_labels_by_point_and_layer[(conn_2d[3], l_bottom)],
                        node_labels_by_point_and_layer[(conn_2d[0], l_top)],
                        node_labels_by_point_and_layer[(conn_2d[1], l_top)],
                        node_labels_by_point_and_layer[(conn_2d[2], l_top)],
                        node_labels_by_point_and_layer[(conn_2d[3], l_top)],
                        node_labels_by_point_and_layer[(conn_2d[4], l_bottom)],
                        node_labels_by_point_and_layer[(conn_2d[5], l_bottom)],
                        node_labels_by_point_and_layer[(conn_2d[6], l_bottom)],
                        node_labels_by_point_and_layer[(conn_2d[7], l_bottom)],
                        node_labels_by_point_and_layer[(conn_2d[0], l_mid)],
                        node_labels_by_point_and_layer[(conn_2d[1], l_mid)],
                        node_labels_by_point_and_layer[(conn_2d[2], l_mid)],
                        node_labels_by_point_and_layer[(conn_2d[3], l_mid)],
                        node_labels_by_point_and_layer[(conn_2d[4], l_top)],
                        node_labels_by_point_and_layer[(conn_2d[5], l_top)],
                        node_labels_by_point_and_layer[(conn_2d[6], l_top)],
                        node_labels_by_point_and_layer[(conn_2d[7], l_top)],
                    ]

                new_element = element_class(kwargs["elType"], current_element_label)
                new_element.setNodes([model.nodes[label] for label in node_labels])

                model.elements[current_element_label] = new_element
                block_elements.append(new_element)
                all_new_elements.append(new_element)
                extruded_elements_for_source_element.append(new_element)
                current_element_label += 1

            block_mappings.append(extruded_elements_for_source_element)

        extruded_elements_by_block.append(block_mappings)

        model.elementSets["{}_block{}".format(name, block_idx + 1)] = ElementSet(
            "{}_block{}".format(name, block_idx + 1), block_elements
        )

    model.elementSets["{}_all".format(name)] = ElementSet("{}_all".format(name), all_new_elements)

    model._populateNodeFieldVariablesFromElements()

    model.nodeSets["{}_all".format(name)] = NodeSet(
        "{}_all".format(name),
        [node for node in nodes_created if len(node.fields)],
    )

    back_surface_nodes = []
    front_surface_nodes = []
    for point_id in range(points_3d.shape[0]):
        back_label = node_labels_by_point_and_layer.get((point_id, 0), None)
        if back_label is not None:
            back_surface_nodes.append(model.nodes[back_label])

        front_label = node_labels_by_point_and_layer.get((point_id, n_layers - 1), None)
        if front_label is not None:
            front_surface_nodes.append(model.nodes[front_label])

    model.nodeSets["{}_back".format(name)] = NodeSet("{}_back".format(name), back_surface_nodes)
    model.nodeSets["{}_front".format(name)] = NodeSet("{}_front".format(name), front_surface_nodes)

    for set_name, point_indices in (mesh_2d.point_sets or {}).items():
        base_point_indices = _set_entry_to_indices(point_indices)

        extruded_nodes = []
        for layer in range(n_layers):
            for point_id in base_point_indices:
                node_label = node_labels_by_point_and_layer.get((int(point_id), layer), None)
                if node_label is not None:
                    extruded_nodes.append(model.nodes[node_label])

        model.nodeSets[set_name] = NodeSet(set_name, extruded_nodes)

    for set_name, set_entries in (mesh_2d.cell_sets or {}).items():
        extruded_elements = []
        for selected_block_idx, (original_block_idx, _, _) in enumerate(selected_blocks):
            if original_block_idx >= len(set_entries):
                continue

            base_element_indices = _set_entry_to_indices(set_entries[original_block_idx])
            for element_idx in base_element_indices:
                if element_idx < 0 or element_idx >= len(extruded_elements_by_block[selected_block_idx]):
                    continue
                extruded_elements.extend(extruded_elements_by_block[selected_block_idx][element_idx])

        model.elementSets[set_name] = ElementSet(set_name, extruded_elements)

    journal.message(
        "Extruded 2D mesh '{}' to 3D with {} nodes and {} elements.".format(
            mesh_file,
            len(nodes_created),
            len(all_new_elements),
        ),
        identification,
    )

    return model
