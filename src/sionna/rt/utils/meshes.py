#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Meshes-related utilities.
"""

import os
import mitsuba as mi
import drjit as dr
import numpy as np

import sionna
import sionna.rt
from .geometry import rotation_matrix


def clone_mesh(mesh: mi.Mesh, name: str | None = None,
               props: mi.Properties | None = None) -> mi.Mesh:
    """
    Clone a Mitsuba mesh object, preserving some of its important properties.

    :param name: Name (id) of the cloned object.
        If :py:class:`None`, the clone will be named as
        ``<original_name>-clone``.

    :param props: Pre-populated properties to use in the new Mitsuba mesh.
        Allows overriding the BSDF, emitter, etc.
    """
    # Clone name
    clone_name = name if (name is not None) else f"{mesh.id()}-clone"

    # Set ID and radio material for the clone
    if props is None:
        props = mi.Properties()
    props.set_id(clone_name)
    if "flip_normals" not in props:
        props['flip_normals'] = mesh.has_flipped_normals()
    props["face_normals"] = mesh.has_face_normals()

    # Instantiate clone mesh
    result = mi.Mesh(name=clone_name,
                     vertex_count=mesh.vertex_count(),
                     face_count=mesh.face_count(),
                     props=props,
                     has_vertex_normals=mesh.has_vertex_normals(),
                     has_vertex_texcoords=mesh.has_vertex_texcoords())

    # Copy mesh parameters
    params = mi.traverse(result)
    params["vertex_positions"] = mesh.vertex_positions_buffer()
    params["vertex_normals"] = mesh.vertex_normals_buffer()
    params["vertex_texcoords"] = mesh.vertex_texcoords_buffer()
    params["faces"] = mesh.faces_buffer()
    params.update()

    return result

def load_mesh(fname: str, flip_normals: bool = True) -> mi.Mesh:
    # pylint: disable=line-too-long
    """
    Load a mesh from a file

    This function loads a mesh from a given file and returns it as a Mitsuba mesh.
    The file must be in either PLY or OBJ format.

    :param fname: Filename of the mesh to be loaded
    :param flip_normals: Whether to invert the normals of the mesh.

    :return: Mitsuba mesh object representing the loaded mesh
    """

    mesh_type = os.path.splitext(fname)[1][1:]
    if mesh_type not in ('ply', 'obj'):
        raise ValueError("Invalid mesh type. Supported types: `ply` and `obj`")

    mi_mesh = mi.load_dict({
        'type': mesh_type,
        'filename': fname,
        'flip_normals': flip_normals,
        'face_normals': True,
        # We need to add a radio material to the object to enable shooting
        # and bouncing of rays.
        'bsdf': sionna.rt.RadioMaterial("dummy", 1.0)
    })

    return mi_mesh

def transform_mesh(mesh: mi.Mesh,
                   translation: mi.Point3f | None = None,
                   rotation: mi.Point3f | None = None,
                   scale: mi.Point3f | None = None):
    # pylint: disable=line-too-long
    r"""
    In-place transformation of a mesh by applying translation, rotation, and scaling

    The order of the transformations is as follows:

    1. Scaling
    2. Rotation
    3. Translation

    Before applying the transformations, the mesh is centered.

    If the mesh stores vertex normals, Mitsuba recomputes them from the
    updated geometry when parameters are committed.

    :param mesh: Mesh to be edited. The mesh is modified in-place.
    :param translation: Translation vector to apply
    :param rotation: Rotation angles [rad] specified through three angles
                     :math:`(\alpha, \beta, \gamma)` corresponding to a 3D rotation as defined in :eq:`rotation`
    :param scale: Scaling vector for scaling along the x, y, and z axes
    """

    params = mi.traverse(mesh)

    # Read vertex positions
    v = params["vertex_positions"]
    v = dr.unravel(mi.Point3f, v)

    # Center the mesh
    c = 0.5 * (mesh.bbox().min + mesh.bbox().max)
    v -= c

    # Scale the mesh
    if scale is not None:
        scale = mi.Point3f(scale)
        v *= scale

    # Rotate the mesh
    if rotation is not None:
        rotation = mi.Point3f(rotation)
        rot_matrix = rotation_matrix(rotation)
        v = rot_matrix @ v

    # Translate
    if translation is None:
        translation = c
    else:
        translation = mi.Point3f(translation)
        translation += c
    v += translation

    params["vertex_positions"] = dr.ravel(v)
    params.update()

def remove_mesh_duplicate_vertices(mesh: mi.Mesh):
    """
    Remove duplicate vertices from a mesh

    Vertices are considered duplicates when they share the same position.
    Face indices are updated accordingly. Texture coordinates of the retained
    representatives are preserved when present, and vertex normals are
    recomputed if the mesh stores them. The mesh is updated in place.

    :param mesh: Mitsuba mesh from which to remove duplicate vertices
    """

    vertices = dr.unravel(mi.Point3f, mesh.vertex_positions_buffer()).numpy()
    faces = dr.unravel(mi.Point3u, mesh.faces_buffer()).numpy()

    # Find unique vertices by position and their indices
    unique_vertices, unique_indices, inverse_indices\
        = np.unique(vertices, axis=1, return_inverse=True, return_index=True)

    # Update faces to use new vertex indices
    new_faces = inverse_indices[faces]

    # Update the mesh
    params = mi.traverse(mesh)
    params["vertex_positions"] = dr.ravel(mi.Point3f(unique_vertices))
    params["faces"] = dr.ravel(mi.Point3u(new_faces))
    if mesh.has_vertex_texcoords():
        texcoords = dr.unravel(mi.Vector2f,
                               mesh.vertex_texcoords_buffer()).numpy()
        new_texcoords = texcoords[:, unique_indices]
        params["vertex_texcoords"] = dr.ravel(mi.Vector2f(new_texcoords))
    params.update()
    if mesh.has_vertex_normals():
        mesh.recompute_vertex_normals()

    # Removing duplicate vertices invalidates Mitsuba's directed edge table,
    # so we need to rebuild it.
    build_mesh_directed_edges(mesh)

def build_mesh_directed_edges(mesh: mi.Mesh):
    """
    Build Mitsuba's directed edge table, used for wedge and silhouette queries

    Must be called after any change to the mesh topology: Mitsuba only rebuilds
    the table automatically for meshes with gradient tracking enabled, and
    ``opposite_dedge()`` returns stale adjacency rather than reporting that the
    table is outdated.

    :param mesh: Mitsuba mesh for which to build the directed edge table
    """
    if mesh.face_count() == 0:
        return

    # Building directed edges can throw ignorable warnings, e.g. for
    # non-manifold vertices where more than two faces meet an edge.
    mi.set_log_level(mi.LogLevel.Error)
    try:
        mesh.build_directed_edges()
    finally:
        mi.set_log_level(mi.LogLevel.Warn)
