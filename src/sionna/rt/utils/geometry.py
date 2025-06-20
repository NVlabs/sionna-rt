#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Geometry utilities of Sionna RT"""

import drjit as dr
import mitsuba as mi
from typing import Tuple
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def phi_hat(phi : mi.Float) -> mi.Vector3f:
    # pylint: disable=line-too-long
    r"""
    Computes the spherical unit vector :math:`\hat{\boldsymbol{\varphi}}(\theta, \varphi)`
    as defined in :eq:`spherical_vecs`

    :param phi: Azimuth angle :math:`\varphi` [rad]
    """
    width = dr.width(phi)
    sin_phi, cos_phi = dr.sincos(phi)
    v = mi.Vector3f(-sin_phi,
                    cos_phi,
                    dr.zeros(mi.Float, width))
    return v

def theta_hat(theta : mi.Float, phi : mi.Float) -> mi.Vector3f:
    # pylint: disable=line-too-long
    r"""
    Computes the spherical unit vector :math:`\hat{\boldsymbol{\theta}}(\theta, \varphi)`
    as defined in :eq:`spherical_vecs`

    :param theta: Zenith angle :math:`\theta` [rad]
    :param phi: Azimuth angle :math:`\varphi` [rad]
    """
    sin_theta, cos_theta = dr.sincos(theta)
    sin_phi, cos_phi = dr.sincos(phi)
    v = mi.Vector3f(cos_theta*cos_phi,
                    cos_theta*sin_phi,
                     -sin_theta)
    return v

def theta_phi_from_unit_vec(v : mi.Vector3f) -> Tuple[mi.Float, mi.Float]:
    # pylint: disable=line-too-long
    r"""
    Computes zenith and azimuth angles (:math:`\theta,\varphi`)
    from unit-norm vectors as described in :eq:`theta_phi`

    :param v: Unit vector

    :return: Zenith angle :math:`\theta` [rad] and azimuth angle :math:`\varphi` [rad]
    """

    # Clip z for numerical stability
    z = dr.clip(v.z, -1, 1)
    theta = dr.safe_acos(z)
    phi = dr.atan2(v.y, v.x)
    return theta, phi

def r_hat(theta : mi.Float, phi : mi.Float) -> mi.Vector3f:
    r"""
    Computes the spherical unit vetor :math:`\hat{\mathbf{r}}(\theta, \phi)`
    as defined in :eq:`spherical_vecs`

    :param theta: Zenith angle :math:`\theta` [rad]
    :param phi: Azimuth angle :math:`\varphi` [rad]
    """
    sin_phi, cos_phi = dr.sincos(phi)
    sin_theta, cos_theta = dr.sincos(theta)
    v = mi.Vector3f(sin_theta*cos_phi,
                    sin_theta*sin_phi,
                    cos_theta)
    return v

def rotation_matrix(angles : mi.Point3f) -> mi.Matrix3f:
    # pylint: disable=line-too-long
    r"""
    Computes the rotation matrix as defined in :eq:`rotation`

    The closed-form expression in (7.1-4) [TR38901]_ is used.

    :param angles: Angles for the rotations :math:`(\alpha,\beta,\gamma)`
        [rad] that define rotations about the axes :math:`(z, y, x)`,
        respectively
    """

    a = angles.x
    b = angles.y
    c = angles.z
    sin_a, cos_a = dr.sincos(a)
    sin_b, cos_b = dr.sincos(b)
    sin_c, cos_c = dr.sincos(c)

    r_11 = cos_a*cos_b
    r_12 = cos_a*sin_b*sin_c - sin_a*cos_c
    r_13 = cos_a*sin_b*cos_c + sin_a*sin_c

    r_21 = sin_a*cos_b
    r_22 = sin_a*sin_b*sin_c + cos_a*cos_c
    r_23 = sin_a*sin_b*cos_c - cos_a*sin_c

    r_31 = -sin_b
    r_32 = cos_b*sin_c
    r_33 = cos_b*cos_c

    rot_mat = mi.Matrix3f([[r_11, r_12, r_13],
                           [r_21, r_22, r_23],
                           [r_31, r_32, r_33]])

    return rot_mat

def triangulate_elevation(elevation: mi.TensorXf) -> mi.Mesh:
    # Treat elevation data as values at the centers of cells. We want the
    # corners. The corners are the average of the surrounding centers
    padded = np.pad(elevation.numpy(), 1, constant_values=float("nan"))
    windows = sliding_window_view(padded, window_shape=(2, 2), axis=(0, 1))
    elevation = np.nanmean(windows, axis=(-1, -2))
    num_rows, num_cols = elevation.shape
    vertices_x, vertices_y = dr.meshgrid(
        dr.linspace(mi.Float, 0, 1, num_cols),
        dr.linspace(mi.Float, 0, 1, num_rows)
    )
    vertices = mi.Point3f(vertices_x, vertices_y, elevation.ravel())
    texcoords = vertices.xy

    vertex_indices = dr.arange(mi.UInt, num_rows * num_cols)
    faces_count = 2 * (num_rows - 1) * (num_cols - 1)
    faces = dr.empty(mi.Vector3u, faces_count)
    is_last_column = (vertex_indices + 1) % num_cols == 0
    is_last_row = vertex_indices // num_cols == num_rows - 1
    ii = dr.compress(~is_last_column & ~is_last_row)
    first_half_mask = dr.arange(mi.UInt, faces_count) < faces_count / 2
    # TODO verify that this isn't backwards?
    dr.scatter(target=faces,
            #    value=mi.Vector3u(ii, ii + 1, ii + num_cols + 1),
               value=mi.Vector3u(ii, ii + num_cols + 1, ii + 1),
               index=dr.compress(first_half_mask))
    dr.scatter(target=faces,
            #    value=mi.Vector3u(ii, ii + num_cols + 1, ii + num_cols),
               value=mi.Vector3u(ii, ii + num_cols, ii + num_cols + 1),
               index=dr.compress(~first_half_mask))

    mesh = mi.Mesh(name="triangulated_elevation",
                   face_count=dr.width(faces),
                   vertex_count=dr.width(vertices),
                   has_vertex_texcoords=True)
    params = mi.traverse(mesh)
    params["vertex_positions"] = dr.ravel(vertices)
    params["vertex_texcoords"] = dr.ravel(texcoords)
    params["faces"] = dr.ravel(faces)
    params.update()

    return mesh
