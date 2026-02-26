#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
Utilities to pre-process and edit Sionna scenes.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import mitsuba as mi

import sionna
from .constants import DEFAULT_THICKNESS
from .radio_materials.itu import ITU_MATERIALS_PROPERTIES
from .scene_object import SceneObject
from .utils.meshes import remove_mesh_duplicate_vertices


def process_xml(xml_string: str,
                merge_shapes: bool = True,
                merge_shapes_exclude_regex: str | None = None,
                default_thickness: float = DEFAULT_THICKNESS) -> str:
    """
    Preprocess the XML string describing the scene

    This function adds an instruction to merge shapes that share the same radio
    material to speed-up ray tracing.

    :param xml_string: XML string containing the scene.

    :param merge_shapes: If set to `True`, shapes that share
        the same radio material are merged.

    :param merge_shapes_exclude_regex: Optional regex to exclude shapes from
        merging. Only used if ``merge_shapes`` is set to `True`.

    :param default_thickness: Default thickness [m] of radio materials
    """

    # Compile the regex if not 'None'
    if merge_shapes_exclude_regex is not None:
        regex = re.compile(merge_shapes_exclude_regex)
    else:
        regex = None

    root = ET.fromstring(xml_string)

    # 1. Replace BSDFs with radio BSDFs
    # If a BSDF node in the XML scene has a special name starting with
    # `mat-itu_` or `itu_`, we automatically convert that BSDF to an
    # `itu-radio-material` plugin.
    # All other nodes are left untouched, assuming that they already are
    # radio materials.
    #
    # We don't need to process BSDFs nested in other BSDFs, e.g. a `diffuse`
    # inside of a `twosided`. We just process the outermost BSDF element.
    for bsdf in root.findall("./bsdf") + root.findall(".//shape/bsdf"):
        bsdf_type = bsdf.attrib.get("type")
        mat_id = bsdf.attrib.get("id")

        # Ensure compatibility with Sionna v0.x ITU radio materials
        name = mat_id
		if name is None:
            bsdf_string = ET.tostring(bsdf, encoding='unicode')
            raise ValueError(f"Found bsdf without name: {bsdf_string[:100]}")
        if name.startswith("mat-"):
            name = name[4:]
        if name.startswith("itu_") or name.startswith("itu-"):
            bsdf_type = "itu-radio-material"
            props = {}

            # Preserve user-defined thickness, if any
            thickness = default_thickness
            # Returns first match
            thickness_prop = bsdf.find("float[@name='thickness']")
            if thickness_prop is not None:
                thickness = float(thickness_prop.get("value", thickness))

            # Read user-defined material type, if any
            itu_type = name[4:]
            type_prop = bsdf.find("string[@name='type']")
            if type_prop is not None:
                itu_type = type_prop.get("value")

            props["type"] = ("string", itu_type)
            props["thickness"] = ("float", thickness)

            # TODO: we could consider saving some information about the original
            # "visual" BSDFs if that allows users to customize the look of their
            # scenes easily from Blender.
            bsdf.clear()
            bsdf.attrib["type"] = bsdf_type
            if mat_id is not None:
                bsdf.attrib["id"] = mat_id
            for k, (t, v) in props.items():
                bsdf.append(ET.Element(t, {"name": k, "value": str(v)}))
        elif (bsdf_type != "itu-radio-material") \
             and (name in ITU_MATERIALS_PROPERTIES):
            raise ValueError(
                f"Found material with name \"{mat_id}\"."
                " ITU material names must start with \"itu_\","
                " e.g. \"mat-itu_concrete\" or \"itu_wet_ground\"."
            )

    # 2. Wrap shapes into a `merge` shape if requested
    merge_node = ET.Element("shape", {"type": "merge", "id": "merged-shapes"})
    merge_node_empty = True
    for shape in root.findall("shape"):
        # Add the shape to the merge node only if requested
        if (merge_shapes and (
                (merge_shapes_exclude_regex is None) or
                (not regex.search(shape.attrib.get("id", "")))
            )):
            root.remove(shape)
            merge_node.append(shape)
            merge_node_empty = False

    if not merge_node_empty:
        root.append(merge_node)

    ET.indent(root, space="    ")
    return ET.tostring(root).decode("utf-8")

def edit_scene_shapes(
    scene: sionna.rt.Scene,
    add: (SceneObject |
                   list[SceneObject] |
                   dict |
                   None)=None,
    remove: (str |
             SceneObject |
             list[SceneObject | str] |
             None)=None,
    return_dict: bool = False
) -> dict | mi.Scene:
    """
    Builds a *new* Mitsuba Scene object identicaly to `scene`, but which
    includes the shapes listed in `add` but not the shapes listed in `remove`.

    The shapes and other plugins that are left untouched carry over to the new
    scene (same objects).

    :param scene: Scene to edit

    :param add: Object, or list /dictionary of objects to be added

    :param remove: Name or object, or list/dictionary of objects or names
            to be added

    :param return_dict: If `True`, then the new scene is returned as a
        dictionnary. Otherwise, it is returned as a Mitsuba scene.
    """

    mi_scene = scene.mi_scene

    # Result scene as a dict
    result = {
        "type": "scene",
    }

    # Local utility to add an object to `result`
    def add_with_id(obj, fallback_id):
        if obj is None:
            return
        key = obj.id() or fallback_id
        assert key not in result
        result[key] = obj


    # Add the visual components of the scene (sensors, integrator, environment
    # map) to `result`.
    for i, sensor in enumerate(mi_scene.sensors()):
        add_with_id(sensor, f"sensor-{i}")
    add_with_id(mi_scene.environment(), "envmap")
    add_with_id(mi_scene.integrator(), "integrator")

    # Build the sets of object ids to remove

    # Set of object ids to remove
    ids_to_remove = set()
    # In case some given `Shape` objects don't have IDs, we keep a separate set
    other_to_remove = set()
    if remove is not None:
        if isinstance(remove, (mi.Shape, str, SceneObject)):
            remove = [remove]

        for v in remove:
            if isinstance(v, SceneObject):
                v = v.mi_mesh

            if isinstance(v, str):
                o = scene.objects.get(v)
                if o:
                    mi_id = o.mi_mesh.id()
                    ids_to_remove.add(mi_id)
            elif isinstance(v, mi.Shape):
                v_id = v.id()
                if v_id:
                    ids_to_remove.add(v_id)
                else:
                    # Shape doesn't have an ID, we still want to remove it
                    other_to_remove.add(v)
            else:
                raise ValueError(f"Cannot remove object of type ({type(v)})."
                                  " The `remove` argument should be a list"
                                  " containing either shape instances or shape"
                                  " IDs.")

    # Add all shapes of the current scene to `result`, except the ones we want
    # to exclude.
    n_shapes = 0
    for shape in mi_scene.shapes():
        shape_id = shape.id()
        if (shape_id in ids_to_remove) or (shape in other_to_remove):
            continue

        if not shape_id:
            shape_id = f"shape-{n_shapes}"
        assert shape_id not in result
        result[shape_id] = shape
        n_shapes += 1

    # Add the objects provided by the user though `add` to `result`
    if add is not None:
        if isinstance(add, (mi.Object, dict, SceneObject)):
            add = [add]

        for a in add:
            if isinstance(a, SceneObject):
                a = a.mi_mesh

            if isinstance(a, dict):
                new_id = a.get("id")
            elif isinstance(a, mi.Object):
                new_id = a.id()
            else:
                raise ValueError(f"Cannot add object of type ({type(a)})."
                                  " The `add` argument should be a list"
                                  " containing either a dict to be loaded by"
                                  " `mi.load_dict()` or an existing Mitsuba"
                                  " object instance.")

            if not new_id:
                if isinstance(a, mi.Shape):
                    new_id = f"shape-{n_shapes}"
                    n_shapes += 1
                else:
                    new_id = f"object-{len(result)}"

            if new_id in result:
                raise ValueError(f"Cannot add object of type ({type(a)}) with"
                                 f" ID \"{new_id}\" because this ID is already"
                                 " used in the scene.")

            result[new_id] = a

    if return_dict:
        return result
    else:
        return mi.load_dict(result)

def extend_scene_with_mesh(scene: mi.Scene, mesh: mi.Mesh):
    """
    Add a mesh to a Mitsuba scene

    This function takes a Mitsuba scene and a mesh, and adds the mesh
    to the scene.

    :param scene: The Mitsuba scene to which the mesh will be added.
    :param mesh: The Mitsuba mesh to be added to the scene.

    :return: New Mitsuba scene with the mesh added
    """

    # Result scene as a dict
    result = {
        "type": "scene",
    }

    # Add to `result` all shapes of the current shapes
    for s in scene.shapes():
        shape_id = s.id()
        result[shape_id] = s

    # Add the shape
    result[mesh.id()] = mesh

    return mi.load_dict(result)

def remove_objects_duplicate_vertices(scene: mi.Scene):
    """
    Remove duplicate vertices from all objects in a Mitsuba scene

    The scene is updated in place.

    :param scene: The Mitsuba scene from which to remove duplicate vertices
    """

    for s in scene.shapes():
        remove_mesh_duplicate_vertices(s)
