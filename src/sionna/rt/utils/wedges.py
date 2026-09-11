#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Wedge utilities of Sionna RT"""

import drjit as dr
import mitsuba as mi
from dataclasses import dataclass
from typing import Tuple

from sionna.rt.constants import EPSILON_FLOAT


@dataclass
class WedgeGeometry:
    """
    Dataclass storing wedge geometry information

    :param shape: Mesh that owns the wedge faces
    :param prim0: Primitive index of the first face (0-face)
    :param primn: Primitive index of the second face (n-face)
    :param local_edge: Local edge index (0, 1, or 2) within the primitive that
        was passed to :py:func:`wedge_geometry` at construction time. This index
        is **not** updated by :py:meth:`swap_faces`, so after a face swap it may
        no longer index an edge of the current ``prim0``. Use it together with
        the construction-time primitive, or treat ``(o, e_hat, length)`` as the
        geometric edge identity after orientation changes.
    :param o: Origin of the edge
    :param e_hat: Normalized edge direction vector
    :param length: Edge length
    :param n0: Normal of the first face
    :param nn: Normal of the second face
    """
    shape : mi.MeshPtr
    prim0: mi.UInt
    primn: mi.UInt
    local_edge: mi.UInt
    o: mi.Point3f
    e_hat: mi.Vector3f
    length: mi.Float
    n0: mi.Normal3f
    nn: mi.Normal3f

    @classmethod
    def build_with_size(cls, size: int) -> "WedgeGeometry":
        return dr.zeros(cls, size)

    def swap_faces(self, flip: mi.Bool):
        """
        Flips the 0 and n faces

        When ``flip`` is true, swaps ``prim0``/``primn`` and ``n0``/``nn``, and
        reverses the edge parameterization (``o``, ``e_hat``). ``local_edge`` is
        left unchanged: it remains the local edge index of the primitive used
        when this record was built, not of the post-swap ``prim0``.

        :param flip: Mask indicating which wedges to flip
        """
        n0_ = dr.select(flip, self.nn, self.n0)
        nn_ = dr.select(flip, self.n0, self.nn)
        self.n0 = n0_
        self.nn = nn_

        prim0_ = dr.select(flip, self.primn, self.prim0)
        primn_ = dr.select(flip, self.prim0, self.primn)
        self.prim0 = prim0_
        self.primn = primn_

        self.o = dr.select(flip, self.o + self.length * self.e_hat,
                           self.o)
        self.e_hat = dr.select(flip, -self.e_hat, self.e_hat)


def wedge_other_face(mesh: mi.MeshPtr | mi.Mesh,
                     prim: mi.UInt,
                     local_edge: mi.UInt,
                     active: bool | mi.Bool = True
                     ) -> mi.UInt:
    # pylint: disable=line-too-long
    r"""
    Returns the primitive index of the face adjacent to a given face along a specified edge

    For a given face (primitive) and one of its edges, this function returns the
    primitive index of the adjacent face that shares the same edge. If no adjacent
    face exists (i.e., the edge is on the boundary), the original primitive index
    is returned.

    :param mesh: The mesh containing the faces and edges
    :param prim: Primitive (face) index
    :param local_edge: Local edge index within the face (0, 1, or 2)
    :param active: (Optional) Mask indicating which computations are active

    :return: Primitive index of the adjacent face, or original primitive if no adjacent face
    """

    # Global edge index within the mesh
    edge0 = 3 * prim + local_edge
    # Other face
    edge1 = mesh.opposite_dedge(edge0, active=active)
    primn = dr.select(edge1 != 0xFFFFFFFF, edge1 // 3, prim)

    return primn

def wedge_geometry(mesh: mi.MeshPtr | mi.Mesh,
                   prim0: mi.UInt,
                   local_edge0: mi.UInt,
                   active: bool | mi.Bool =True) -> WedgeGeometry:
    # pylint: disable=line-too-long
    r"""
    Returns the wedge geometry

    This function computes the geometric properties of a wedge (edge) formed by two
    adjacent faces in a mesh.

    The normals are oriented such that the interior angle of the wedge is <= π.
    The edge vector is oriented such that n0 × e = t0, where t0 is oriented
    toward the interior of the wedge.

    :param mesh: The mesh containing the faces and edges
    :param prim0: Primitive index of the first face
    :param local_edge0: Local edge index within ``prim0`` (0, 1, or 2). Stored
        as :py:attr:`WedgeGeometry.local_edge` and kept even if
        :py:meth:`WedgeGeometry.swap_faces` later exchanges the face indices.
    :param active: (Optional) Mask indicating which computations are active

    :return: WedgeGeometry object. For zero-length edges, ``length`` is ``0``
        and ``e_hat`` falls back to :math:`(0, 0, 1)` so inactive or degenerate
        lanes stay finite under vectorized evaluation.
    """

    # Index of the other face of the wedge
    primn = wedge_other_face(mesh, prim0, local_edge0, active)

    # Indices of the edge endpoints
    v_ind = mesh.edge_indices(prim0, local_edge0, active)

    # Coordinates of the edge endpoints
    e0 = mesh.vertex_position(v_ind.x, active)
    e1 = mesh.vertex_position(v_ind.y, active)

    # Non-normalized edge vector
    d = e1 - e0

    # Edge length
    length = dr.norm(d)

    # Edge vector (placeholder direction for collapsed edges)
    e_hat = dr.select(length > 0, d * dr.rcp(length), mi.Vector3f(0, 0, 1))

    # Normal to the faces
    n0 = mesh.face_normal(prim0, active)
    nn = mesh.face_normal(primn, active)

    # If the wedge is a screen, then the normals are opposite
    nn[prim0 == primn] *= -1

    # The normals are oriented such that the interior angle of the
    # wedge is <= pi.
    # To ensure this, we need a point on each face.

    # Read the vertices of the 0-face primitive
    prim0_indices = mesh.face_indices(prim0, active)
    v00 = mesh.vertex_position(prim0_indices.x, active)
    v01 = mesh.vertex_position(prim0_indices.y, active)
    v02 = mesh.vertex_position(prim0_indices.z, active)
    f0 = (v00 + v01 + v02)/3.

    # Read the vertices of the 1-face primitive
    primn_indices = mesh.face_indices(primn, active)
    vn0 = mesh.vertex_position(primn_indices.x, active)
    vn1 = mesh.vertex_position(primn_indices.y, active)
    vn2 = mesh.vertex_position(primn_indices.z, active)
    fn = (vn0 + vn1 + vn2)/3.

    flip_n0 = dr.dot(n0, fn - e0) > EPSILON_FLOAT
    n0 = dr.select(flip_n0, -n0, n0)
    flip_nn = dr.dot(nn, f0 - e0) > EPSILON_FLOAT
    nn = dr.select(flip_nn, -nn, nn)

    # The edge vector is oriented such that n0 x e = t0, where t0
    # is oriented toward the interior of the wedge.
    t0 = dr.cross(n0, e_hat)
    flip_e = dr.dot(t0, f0 - e0) < -EPSILON_FLOAT
    e_hat = dr.select(flip_e, -e_hat, e_hat)
    o = dr.select(flip_e, e1, e0)

    o = mi.Point3f(o)
    e_hat = mi.Vector3f(e_hat)
    n0 = mi.Normal3f(n0)
    nn = mi.Normal3f(nn)
    wedge = WedgeGeometry(mesh, prim0, primn, local_edge0, o, e_hat, length, n0, nn)
    return wedge

def wedge_interior_angle(n0: mi.Normal3f, nn: mi.Normal3f) -> mi.Float:
    r"""
    Computes the interior angle [rad] of a wedge with face normals n0 and nn

    :param n0: Normal of the first face
    :param nn: Normal of the second face
    """

    interior_angle = dr.safe_acos(-dr.dot(n0, nn))
    return interior_angle

def sample_wedge_diffraction_point(
        si: mi.SurfaceInteraction3f,
        ray_origin: mi.Point3f,
        ki_world: mi.Vector3f,
        sample1: mi.Float,
        edge_diffraction: bool,
        active: bool | mi.Bool =True
    ) -> Tuple[mi.Bool, WedgeGeometry, mi.Point3f]:
    # pylint: disable=line-too-long
    r"""
    Samples a diffraction point on the silhouette edge of a mesh from a surface interaction

    This function identifies potential diffraction points by sampling along the silhouette
    of the mesh.

    :param si: Surface interaction containing information about the intersected surface
    :param ray_origin: Origin of the incident ray
    :param ki_world: Direction of propagation of the incident wave in the world frame
    :param sample1: Random number in :math:`[0,1)` used to sample a point on the
        silhouette edge
    :param edge_diffraction: If set to `True`, then diffraction on free floating edges
        is computed. If `False`, only diffraction on edges between different primitives
        is considered.
    :param active: Mask indicating which rays are active. Defaults to `True`.

    :return: A tuple ``(valid, wedges, diff_point)`` where:
        - ``valid``: Boolean mask indicating if a valid diffraction point was found
        - ``wedges``: Wedge geometry for the diffraction point. Faces may be
          swapped so the incident direction sees the 0-face; ``local_edge``
          still indexes the silhouette-sampled primitive (see
          :py:class:`WedgeGeometry`).
        - ``diff_point``: 3D position of the diffraction point on the silhouette
          edge
    """

    # Angle [rad] between the normal and the vector along which the point
    # on the surface is offset to define the silhouette
    theta = 0.5*dr.pi - 0.05

    # Offset [scene units] by which the point on the surface is offset
    # to define the silhouette
    delta = 0.1

    mesh = mi.MeshPtr(si.shape)

    # Normal of the surface
    n_world = si.n

    # Tangent vector perpendicular to the surface normal
    t = dr.normalize(ki_world - dr.dot(ki_world, n_world) * n_world)
    # Vector along which the point on the surface is offset to define
    # the silhouette
    d = dr.cos(theta) * n_world + dr.sin(theta) * t
    # Viewpoint
    viewpoint = si.p + delta * d

    # Sample the silhouette
    ss = si.shape.primitive_silhouette_projection(viewpoint=viewpoint, si=si,
            flags=mi.DiscontinuityFlags.PerimeterType, sample=sample1,
            active=active)
    valid = active & ss.is_valid()

    # Diffracting primitive
    prim0 = ss.prim_index
    # Diffracting edge
    local_edge0 = ss.projection_index
    # Diffracting point
    diff_point = ss.p

    wedges = wedge_geometry(mesh, prim0, local_edge0, valid)
    wedges.swap_faces(dr.dot(ki_world, wedges.n0) > 0)

    # Reject collapsed edges (zero length yields a placeholder e_hat)
    valid &= wedges.length > 0

    # The ray origin must be on the exterior of the wedge
    k = ray_origin - diff_point
    valid &= (dr.dot(k, wedges.n0) > EPSILON_FLOAT)\
        | (dr.dot(k, wedges.nn) > EPSILON_FLOAT)

    if not edge_diffraction:
        valid &= wedges.primn != wedges.prim0

    return valid, wedges, diff_point
