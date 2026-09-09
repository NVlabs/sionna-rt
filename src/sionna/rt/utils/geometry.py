#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Geometry utilities of Sionna RT"""

import drjit as dr
import mitsuba as mi
from typing import Tuple
from sionna.rt.constants import EPSILON_FLOAT
from sionna.rt.utils.misc import safe_atan2, isclose


def phi_hat(phi: mi.Float) -> mi.Vector3f:
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

def theta_hat(theta: mi.Float, phi: mi.Float) -> mi.Vector3f:
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

def theta_phi_from_unit_vec(v: mi.Vector3f) -> Tuple[mi.Float, mi.Float]:
    # pylint: disable=line-too-long
    r"""
    Computes zenith and azimuth angles (:math:`\theta,\varphi`)
    from unit-norm vectors as described in :eq:`theta_phi`

    At the poles, i.e., for :math:`v = (0, 0, \pm 1)`, the azimuth angle is not
    defined and is set to :math:`\varphi = 0`.

    :param v: Unit vector

    :return: Zenith angle :math:`\theta` [rad] and azimuth angle :math:`\varphi` [rad]
    """

    # Norm of the horizontal component, equal to sin(theta) for a unit vector.
    # The zero case is handled explicitly to avoid an infinite gradient.
    rho_sqr = dr.square(v.x) + dr.square(v.y)
    rho = dr.select(rho_sqr > 0., dr.sqrt(rho_sqr), 0.)

    # atan2(rho, z) is accurate for all theta, unlike acos(z) which loses all
    # precision at the poles where 1-|z| underflows
    theta = safe_atan2(rho, v.z)

    # Near the poles, rho is dominated by round-off and the azimuth carries no
    # information. Pinning it to zero keeps the result deterministic.
    phi = dr.select(rho < EPSILON_FLOAT, mi.Float(0.), safe_atan2(v.y, v.x))

    return theta, phi

def r_hat(theta: mi.Float, phi: mi.Float) -> mi.Vector3f:
    r"""
    Computes the spherical unit vector :math:`\hat{\mathbf{r}}(\theta, \phi)`
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

def rotation_matrix(angles: mi.Point3f) -> mi.Matrix3f:
    # pylint: disable=line-too-long
    r"""
    Computes the rotation matrix as defined in :eq:`rotation`

    The closed-form expression in (7.1-4) :cite:p:`TR38901_RT` is used.

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

def look_at_orientation(position: mi.Point3f,
                        target: mi.Point3f,
                        target_desc: str = "object") -> mi.Point3f:
    # pylint: disable=line-too-long
    r"""
    Computes the orientation angles :math:`(\alpha, \beta, \gamma)` that make
    the local x-axis point from ``position`` toward ``target``

    Given the direction :math:`\mathbf{x} = \texttt{target} - \texttt{position}`
    with spherical angles :math:`\theta` and :math:`\varphi` :eq:`theta_phi`,
    the returned angles are
    :math:`\left(\varphi, \theta-\frac{\pi}{2}, 0\right)`, which define a
    rotation as in :eq:`rotation` with the local z-axis kept in the vertical
    plane containing :math:`\mathbf{x}`, i.e., the world z-axis is used as
    up direction.

    For a vertical :math:`\mathbf{x}`, the rotation about the local x-axis is
    not determined by the look-at direction alone, and :math:`\varphi` is
    ill-conditioned. In that case, the azimuth is set to :math:`\varphi=0`,
    which amounts to using :math:`-\hat{\mathbf{x}}` as up direction when
    looking upward, and :math:`\hat{\mathbf{x}}` when looking downward.

    :param position: Position from which to look
    :param target: Position to look at
    :param target_desc: Description of the entity being oriented, used to build
        the error message raised when ``target`` coincides with ``position``

    :return: Orientation angles :math:`(\alpha, \beta, \gamma)` [rad]

    :raises ValueError: If ``target`` coincides with ``position``, for which
        the look-at direction is undefined
    """

    x = mi.Point3f(target) - mi.Point3f(position)
    if dr.any(isclose(dr.norm(x), mi.Float(0.))):
        raise ValueError("The look-at target coincides with the "
                         f"{target_desc} position")
    x = dr.normalize(x)

    theta, phi = theta_phi_from_unit_vec(x)

    beta = theta - dr.pi/2. # Rotation around y-axis
    gamma = dr.zeros(mi.Float, dr.width(phi)) # Rotation around x-axis

    return mi.Point3f(phi, beta, gamma)

def rotate_vector_around_axis(x: mi.Vector3f,
                              u: mi.Vector3f,
                              theta: mi.Float) -> mi.Vector3f:
    # pylint: disable=line-too-long
    r"""
    Rotates vector :math:`\mathbf{x}` around axis :math:`\mathbf{u}` by angle
    :math:`\theta` using Rodrigues' rotation formula

    The rotation is computed using:

    .. math::
        \mathbf{R}_{\mathbf{u}}(\theta)\mathbf{x} = \mathbf{u}(\mathbf{u} \cdot \mathbf{x}) + \cos(\theta)(\mathbf{u} \times \mathbf{x}) \times \mathbf{u} + \sin(\theta)(\mathbf{u} \times \mathbf{x})

    :param x: Vector to rotate :math:`\mathbf{x}`
    :param u: Unit axis vector around which to rotate :math:`\mathbf{u}`
    :param theta: Rotation angle [rad]

    :return: Rotated vector
    """

    sin_theta, cos_theta = dr.sincos(theta)

    # u · x (scalar projection of x onto u)
    u_dot_x = dr.dot(u, x)

    # u × x (cross product)
    u_cross_x = dr.cross(u, x)

    # (u × x) × u (double cross product)
    u_cross_x_cross_u = dr.cross(u_cross_x, u)

    # Apply Rodrigues' formula: R_u(θ)x = u(u·x) + cos(θ)(u×x)×u + sin(θ)(u×x)
    rotated = u * u_dot_x + cos_theta * u_cross_x_cross_u\
                + sin_theta * u_cross_x

    return rotated

def vector_plane_reflection(u: mi.Vector3f,
                            n: mi.Vector3f | mi.Normal3f,
                            active: mi.Bool) -> mi.Vector3f:
    # pylint: disable=line-too-long
    r"""
    Computes the mirror image of a vector with respect to a plane

    Given a vector :math:`\mathbf{u}` and a plane with normal :math:`\mathbf{n}`,
    this function computes the mirror image of the vector using:

    .. math::
        \mathbf{u}' = \mathbf{u} - 2(\mathbf{u} \cdot \mathbf{n})\mathbf{n}

    :param u: Vector to mirror :math:`\mathbf{u}`
    :param n: Unit normal vector of the plane :math:`\mathbf{n}`
    :param active: Boolean mask indicating which elements to process

    :return: Mirrored vector :math:`\mathbf{u}'`
    """

    u_prime = dr.select(active,
                        u - 2 * dr.dot(u, n) * n,
                        u)
    return u_prime

def point_plane_reflection(p: mi.Point3f,
                           n: mi.Vector3f | mi.Normal3f,
                           v: mi.Point3f,
                           active: mi.Bool) -> mi.Point3f:
    # pylint: disable=line-too-long
    r"""
    Computes the mirror image of a point with respect to a plane

    Given a point :math:`\mathbf{p}`, a plane with normal :math:`\mathbf{n}`,
    and a point :math:`\mathbf{v}` on the plane, this function computes the
    mirror image of the point using:

    .. math::
        \mathbf{p}' = \mathbf{p} - 2((\mathbf{p} - \mathbf{v}) \cdot \mathbf{n})\mathbf{n}

    :param p: Point to mirror :math:`\mathbf{p}`
    :param n: Unit normal vector of the plane :math:`\mathbf{n}`
    :param v: Point on the plane :math:`\mathbf{v}`
    :param active: Boolean mask indicating which elements to process

    :return: Mirrored point :math:`\mathbf{p}'`
    """

    p_prime = dr.select(active,
                        p - 2 * dr.dot(p - v, n) * n,
                        p)
    return p_prime
