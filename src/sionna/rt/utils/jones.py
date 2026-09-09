#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utilities for Jones calculus"""

import drjit as dr
import mitsuba as mi

from .geometry import theta_phi_from_unit_vec, theta_hat


def implicit_basis_vector(k: mi.Vector3f) -> mi.Vector3f:
    # pylint: disable=line-too-long
    r"""
    Returns a reference frame basis vector for a Jones vector, representing a
    transverse wave propagating in direction ``k``

    The spherical basis vector :math:`\hat{\boldsymbol{\theta}}(\theta, \varphi)`
    :eq:`spherical_vecs` is used as basis vector, where the zenith and azimuth angles
    are obtained from the unit vector ``k``.

    :param k: A unit vector corresponding to the direction of propagation of a transverse wave
    :returns: A basis vector orthogonal to ``k``
    """

    theta, phi = theta_phi_from_unit_vec(k)
    v = theta_hat(theta, phi)

    # Near poles, using `v` as implicit basis vector may be unstable
    near_pole = dr.abs(k.z) > 1 - 5e-7 # \approx sqrt(1 - 1e-6)
    fallback_reference = mi.Vector3f(-1., 0., 0.)
    # Ensure orthogonality to k
    fallback = fallback_reference - dr.dot(fallback_reference, k)*k
    fallback = dr.normalize(fallback)

    return dr.select(near_pole, fallback, v)

def transverse_basis_from_normal(
    k: mi.Vector3f,
    normal: mi.Vector3f
) -> mi.Vector3f:
    r"""
    Returns a deterministic transverse basis vector for a wave interacting
    with a surface.

    Away from normal incidence, the transverse electric direction is
    :math:`\hat{\mathbf{k}}\times\hat{\mathbf{n}}`. At normal or near-normal
    incidence, this cross product is degenerate, and an arbitrary deterministic
    vector orthogonal to ``k`` is used instead.

    :param k: Direction of propagation as a unit vector
    :param normal: Surface normal as a unit vector
    :returns: A unit vector orthogonal to ``k``
    """

    basis = dr.cross(k, normal)
    degenerate = dr.squared_norm(basis) < 1e-12

    # Project a deterministic Cartesian axis onto the plane transverse to k.
    # Selecting y when k is close to x keeps the projection well-conditioned.
    reference = dr.select(dr.abs(k.x) < 0.9,
                          mi.Vector3f(1., 0., 0.),
                          mi.Vector3f(0., 1., 0.))
    fallback = reference - dr.dot(reference, k)*k
    basis = dr.select(degenerate, fallback, basis)
    return dr.normalize(basis)


def jones_matrix_rotator(
    k: mi.Vector3f,
    s_current: mi.Vector3f,
    s_target: mi.Vector3f
) -> mi.Matrix2f:
    r"""
    Constructs the 2D change-of-basis matrix to rotate the reference frame of
    a Jones vector representing a transverse wave propagating in direction
    ``k`` from basis vector ``s_current`` to basis
    vector ``s_target``

    :param k: Direction of propagation as a unit vector
    :param s_current: Current basis vector as a unit vector
    :param s_target: Target basis vector as a unit vector
    """

    c = dr.dot(s_current, s_target)
    s = dr.dot(k, dr.cross(s_current, s_target))

    rotator = mi.Matrix2f([[c,  s],
                           [-s, c]])
    return rotator

def jones_matrix_rotator_flip_forward(k: mi.Vector3f) -> mi.Matrix2f:
    # pylint: disable=line-too-long
    r"""
    Constructs the 2D change-of-basis matrix that flips the direction of
    propagation of the reference frame of a Jones vector representing a
    transverse wave from the basis vector corresponding to ``k`` to the
    one corresponding to ``-k``

    This is useful to evaluate the antenna pattern of a receiver, as the pattern
    needs to be rotated to match the frame in which the incident wave is
    represented.

    Note that the rotation matrix returned by this function is a diagonal
    matrix:

    .. math::

        \mathbf{R} =
            \begin{bmatrix}
                \begin{array}{c c}
                    c & 0 \\
                    0   & -c
                \end{array}
            \end{bmatrix}

    where:

    .. math::

        c = \mathbf{s_c}^\textsf{T} \mathbf{s_t}

    and :math:`\mathbf{s_c}` and :math:`\mathbf{s_t}` are the basis
    vectors corresponding to ``k`` and ``-k``,
    respectively, and computed using :func:`~sionna.rt.utils.implicit_basis_vector`.

    :param k: Current direction of propagation as a unit vector
    """

    s_current = implicit_basis_vector(k)
    s_target = implicit_basis_vector(-k)
    a = dr.dot(s_current, s_target)
    rotator = mi.Matrix2f(a,  0.0,
                          0.0, -a)
    return rotator

def to_world_jones_rotator(
    to_world: mi.Matrix3f,
    k_local: mi.Vector3f
) -> mi.Matrix2f:
    # pylint: disable=line-too-long
    r"""
    Constructs the 2D change-of-basis matrix to rotate the reference frame of
    a Jones vector representing a transverse wave with ``k_local`` as
    direction of propagation from the local implicit frame to the world implicit
    frame

    :param to_world: Change-of-basis matrix from the local to the world frame
    :param k_local: Direction of propagation in the local frame as a unit vector
    """

    # To-world transform for the incident field
    k_world = to_world@k_local
    # Current S basis vector
    s_current = implicit_basis_vector(k_local)
    s_current = to_world@s_current
    # Target S basis vector
    s_target = implicit_basis_vector(k_world)
    # Rotation matrix
    rotator = jones_matrix_rotator(k_world, s_current, s_target)

    return rotator

def jones_matrix_to_world_implicit(
    c1: mi.Complex2f,
    c2: mi.Complex2f,
    to_world: mi.Matrix3f,
    k_in_local: mi.Vector3f,
    k_out_local: mi.Vector3f,
) -> mi.Matrix4f:
    # pylint: disable=line-too-long
    r"""
    Builds the Jones matrix that models a specular reflection or a refraction

    ``c1`` and ``c2`` are Fresnel coefficients that depend on the
    composition of the scatterer.
    ``k_in_local`` and ``k_out_local`` are the direction of propagation of the
    incident and scattered wave, respectively, represented in the local frame of
    the interaction.
    Note that in the local frame of the interaction, the z-axis vector
    corresponds to the normal to the scatterer surface at the interaction point.

    The returned matrix operates on the incident wave represented in the
    implicit world frame. The resulting scattered wave is also represented in
    the implicit world frame.
    This is ensured by applying a left and right rotation matrix to the 2x2
    diagonal matrix containing the Fresnel coefficients, which operates on the
    local frame having the transverse electric component as basis vector:

    .. math::

        \mathbf{J} = \mathbf{R_O} \mathbf{D} \mathbf{R_I}^\textsf{T}

    where:

    .. math::

        \mathbf{D} =
            \begin{bmatrix}
                \begin{array}{c c}
                    \texttt{c1} & 0 \\
                    0   & \texttt{c2}
                \end{array}
            \end{bmatrix}

    and :math:`\mathbf{R_I}` (:math:`\mathbf{R_O}`) is the change-of-basis
    matrix from the local frame using the transverse electric direction as basis
    vector to the world implicit frame for the incident (scattered) wave.

    This function returns the :math:`4 \times 4` real-valued matrix equivalent
    to :math:`\mathbf{J}`:

    .. math::

        \mathbf{M} =
            \begin{bmatrix}
                \begin{array}{c c}
                    \Re\{\mathbf{J}\} & -\Im\{\mathbf{J}\} \\
                    \Im\{\mathbf{J}\} &  \Re\{\mathbf{J}\}
                \end{array}
            \end{bmatrix}

    where :math:`\mathbf{M}` is the returned matrix and :math:`\Re\{\mathbf{J}\}`
    and :math:`\Im\{\mathbf{J}\}` the real and imaginary components of
    :math:`\mathbf{J}`, respectively.

    :param c1: First complex-valued Fresnel coefficient
    :param c2: Second complex-valued Fresnel coefficient
    :param to_world: Change-of-basis matrix from the local to the world frame
    :param k_in_local: Direction of propagation of the incident wave in the local frame as a unit vector
    :param k_out_local: Direction of propagation of the scattered wave in the local frame as a unit vector
    """

    # TE direction. The normal is always z+ in the local frame.
    si_target_local = transverse_basis_from_normal(
        k_in_local, mi.Vector3f(0., 0., 1.))
    so_current_local = si_target_local

    k_in_world = to_world @ k_in_local
    k_out_world = to_world @ k_out_local

    # Rotator for the incident field
    to_local = to_world.T
    si_current_world = implicit_basis_vector(k_in_world)
    si_current_local = to_local @ si_current_world
    in_rotator = jones_matrix_rotator(k_in_local, si_current_local,
                                      si_target_local)

    # Rotator for the scattered field
    so_current_world = to_world @ so_current_local
    so_target_world = implicit_basis_vector(k_out_world)
    out_rotator = jones_matrix_rotator(k_out_world, so_current_world,
                                       so_target_world)

    # Compute the Jones matrix

    c1_real = c1.real
    c1_imag = c1.imag

    c2_real = c2.real
    c2_imag = c2.imag

    # Real component
    real = mi.Matrix2f(c1_real * out_rotator[0,0], c2_real * out_rotator[0,1],
                       c1_real * out_rotator[1,0], c2_real * out_rotator[1,1])
    real @= in_rotator

    # Imaginary component
    imag = mi.Matrix2f(c1_imag * out_rotator[0,0], c2_imag * out_rotator[0,1],
                       c1_imag * out_rotator[1,0], c2_imag * out_rotator[1,1])
    imag @= in_rotator

    # The Jones matrix is returned as a 4x4 real-valued matrix
    return mi.Matrix4f(
        real[0,0], real[0,1], -imag[0,0], -imag[0,1],
        real[1,0], real[1,1], -imag[1,0], -imag[1,1],
        imag[0,0], imag[0,1],  real[0,0],  real[0,1],
        imag[1,0], imag[1,1],  real[1,0],  real[1,1]
    )

def jones_vec_dot(u: mi.Vector4f, v: mi.Vector4f) -> mi.Complex2f:
    # pylint: disable=line-too-long
    r"""
    Computes the dot product of two Jones vectors :math:`\mathbf{u}` and
    :math:`\mathbf{v}`

    A Jones vector is assumed to be represented by a real-valued vector of
    four dimensions, obtained by concatenating its real and imaginary components.
    The returned array is complex-valued.

    More precisely, the following formula is implemented:

    .. math::

        \begin{multline}
        a = \mathbf{u}^\textsf{H} \mathbf{v}\\
          = \left( \Re\{\mathbf{u}\}^\textsf{T} \Re\{\mathbf{v}\}\\
          + \Im\{\mathbf{u}\}^\textsf{T} \Im\{\mathbf{v}\} \right)\\
          + j\left( \Re\{\mathbf{u}\}^\textsf{T} \Im\{\mathbf{v}\}\\
          - \Im\{\mathbf{u}\}^\textsf{T} \Re\{\mathbf{v}\} \right)
        \end{multline}

    :param u: First input vector
    :param v: Second input vector
    """

    u_real = mi.Vector2f(u.x, u.y)
    u_imag = mi.Vector2f(u.z, u.w)

    v_real = mi.Vector2f(v.x, v.y)
    v_imag = mi.Vector2f(v.z, v.w)

    a_real = dr.dot(u_real, v_real) + dr.dot(u_imag, v_imag)
    a_imag = dr.dot(u_real, v_imag) - dr.dot(u_imag, v_real)
    a = mi.Complex2f(a_real, a_imag)
    return a
