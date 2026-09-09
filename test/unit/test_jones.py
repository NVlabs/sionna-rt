#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import drjit as dr
import mitsuba as mi
import numpy as np
from scipy.constants import speed_of_light
from scipy.spatial.transform import Rotation as scipy_rotation

from sionna.rt import RadioMaterial
from sionna.rt.utils import jones_matrix_rotator, to_world_jones_rotator,\
    jones_vec_dot, implicit_basis_vector, jones_matrix_rotator_flip_forward,\
    transverse_basis_from_normal, jones_matrix_to_world_implicit,\
    complex_relative_permittivity, itu_coefficients_single_layer_slab

#############################################################
# Constants
#############################################################

# Threshold for the relative squared error above which a test fails
MAX_RSE = 1e-8

#############################################################
# Utilities
#############################################################

def batch_matvec(m, v):
    """
    Batchified matvec op
    """

    return (m@np.expand_dims(v, axis=-1))[:,:,0]

def vec_mi_to_np(v):
    """
    Cast a mi.Vector3f to an equivalent numpy array
    """
    return np.transpose(v.numpy(), [1,0])

def vec_np_to_mi(v):
    """
    Cast a numpy array with shape [batch_size, 3] to a mi.Vector3f
    """
    return mi.Vector3f(v[:,0], v[:,1], v[:,2])

def mat_mi_to_np(m):
    """
    Casts a JonesMatrix object to an equivalent complex-valued numpy array
    """
    a = np.transpose(m.numpy(), [2, 0, 1])
    return a

def mat2_mi_to_np(m):
    """
    Cast a :class:`mi.Matrix2f` to an array of shape ``[batch, 2, 2]``.
    """
    a = np.array(m.numpy())
    if a.ndim == 2:
        return a[None, ...]
    return np.transpose(a, [2, 0, 1])

def mat4_mi_to_np(m):
    """
    Cast a :class:`mi.Matrix4f` to an array of shape ``[batch, 4, 4]``.
    """
    a = np.array(m.numpy())
    if a.ndim == 2:
        return a[None, ...]
    return np.transpose(a, [2, 0, 1])

def complex2f_to_np(c):
    """Cast a :class:`mi.Complex2f` to a complex NumPy array."""
    return np.asarray(c.real.numpy()) + 1j * np.asarray(c.imag.numpy())

def embed_complex_jones(J):
    r"""
    Embed a complex :math:`2 \times 2` Jones matrix into the real
    :math:`4 \times 4` layout used by Sionna RT:

    .. math::

        M = \begin{bmatrix} \Re J & -\Im J \\ \Im J & \Re J \end{bmatrix}.
    """
    return np.block([[J.real, -J.imag],
                     [J.imag,  J.real]])

def ref_complex_jones_world_implicit(c1, c2, to_world, k_in_local, k_out_local):
    r"""
    Independent complex :math:`2 \times 2` reference for
    :func:`~sionna.rt.utils.jones_matrix_to_world_implicit`.

    Rebuilds :math:`J = R_O D R_I` from the public rotator helpers without
    calling the embedding under test.
    """
    te_local = transverse_basis_from_normal(k_in_local, mi.Vector3f(0., 0., 1.))

    k_in_world = to_world @ k_in_local
    k_out_world = to_world @ k_out_local

    si_current_local = to_world.T @ implicit_basis_vector(k_in_world)
    in_rotator = jones_matrix_rotator(k_in_local, si_current_local, te_local)

    so_current_world = to_world @ te_local
    out_rotator = jones_matrix_rotator(k_out_world, so_current_world,
                                       implicit_basis_vector(k_out_world))

    R_I = mat2_mi_to_np(in_rotator)
    R_O = mat2_mi_to_np(out_rotator)
    c1_np = np.atleast_1d(complex2f_to_np(c1))
    c2_np = np.atleast_1d(complex2f_to_np(c2))

    J = np.zeros((R_I.shape[0], 2, 2), dtype=np.complex128)
    for i in range(R_I.shape[0]):
        D = np.diag([c1_np[min(i, c1_np.size - 1)],
                     c2_np[min(i, c2_np.size - 1)]])
        J[i] = R_O[i] @ D @ R_I[i]
    return J

def max_rel_se(u, v, axes=-1):
    """
    Computes the relative max squared error (SE) between `u` and `v` by reducing along `axes`.
    `u` is the reference value.
    """

    if axes is None:
        rse = np.square(np.abs(u-v)) / np.square(np.abs(u))
    else:
        rse = np.sum(np.square(np.abs(u-v)), axis=axes) / np.sum(np.square(np.abs(u)), axis=axes)
    return np.max(rse)

def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v)
    return [float(x) for x in v]

def _vec3(v):
    x, y, z = _unit(v) if not isinstance(v, mi.Vector3f) else (v.x, v.y, v.z)
    if isinstance(v, mi.Vector3f):
        return dr.normalize(v)
    return mi.Vector3f(x, y, z)

def _to_world_mi(matrix3=None):
    if matrix3 is None:
        return mi.Matrix3f(1.0)
    m = np.asarray(matrix3, dtype=np.float64)
    # Mitsuba Matrix3f expects columns or a flat layout compatible with float.
    return mi.Matrix3f(m)

#############################################################
# Tests
#############################################################

def test_jones_matrix_rotator():
    r"""
    Tests the `jones_matrix_rotator()` utility
    """

    batch_size = 100

    # Random vector
    u = np.random.normal(size=[batch_size,3])

    # Current basis
    s1 = np.random.normal(size=[batch_size,3])
    s1 = s1/np.linalg.norm(s1, axis=1, keepdims=True)

    # Forward direction
    fwd = np.cross(u,s1)
    fwd = fwd/np.linalg.norm(fwd, axis=1, keepdims=True)

    # Target basis (must be orthogonal to forward)
    s2 = np.random.normal(size=[batch_size,3])
    s2 -= np.sum(s2*fwd, axis=1, keepdims=True)*fwd
    s2 = s2/np.linalg.norm(s2, axis=1, keepdims=True)

    # p1 and p2 components
    p1 = np.cross(fwd,s1)
    p2 = np.cross(fwd,s2)

    # u in the (s1, p1, k)
    # Drop the k-dim as it is 0
    u_in = np.stack([np.sum(u*s1, axis=1),
                    np.sum(u*p1, axis=1)],
                    axis=-1)
    # u in the (s2, p2, k) basis
    # Drop the k-dim as it is 0
    u_out_ref = np.stack([np.sum(u*s2, axis=1),
                        np.sum(u*p2, axis=1)],
                        axis=-1)

    # Compute change-of-basis matrix
    fwd_mi = vec_np_to_mi(fwd)
    s1_mi = vec_np_to_mi(s1)
    s2_mi = vec_np_to_mi(s2)
    P = jones_matrix_rotator(fwd_mi, s1_mi, s2_mi)
    P = mat_mi_to_np(P)

    # Apply the change of basis
    u_out = batch_matvec(P, u_in)

    assert max_rel_se(u_out_ref, u_out, axes=(-1,)) < MAX_RSE


def test_transverse_basis_from_normal():
    """Test the transverse basis at generic and face-normal incidence."""

    k = np.array([[0.0, 0.0, -1.0],
                  [1.0, 0.0, 0.0],
                  [1.0, 1e-8, 0.0],
                  [0.0, 1.0, 0.0]])
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    normal = np.array([[0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0]])

    basis = transverse_basis_from_normal(vec_np_to_mi(k),
                                         vec_np_to_mi(normal))
    basis = vec_mi_to_np(basis)

    assert np.all(np.isfinite(basis))
    assert np.allclose(np.linalg.norm(basis, axis=-1), 1.0)
    assert np.allclose(np.sum(k*basis, axis=-1), 0.0, atol=1e-7)

    # Preserve the +x convention used for normal incidence in the local
    # interaction frame, where k = -z and the surface normal is +z.
    assert np.allclose(basis[0], [1.0, 0.0, 0.0])

    # Away from normal incidence, the conventional k x n basis is preserved.
    expected = np.cross(k[-1], normal[-1])
    expected /= np.linalg.norm(expected)
    assert np.allclose(basis[-1], expected)


def test_to_world_jones_rotator():
    r"""
    Tests `to_world_jones_rotator()`
    """

    batch_size = 100

    np.random.seed(42)

    # Generate random to-world transform
    to_world_angles = np.random.uniform(low=0., high=0.5*np.pi, size=[batch_size,3])
    to_world = scipy_rotation.from_euler('xyz', to_world_angles).as_matrix()
    to_world_mi = mi.Matrix3f(np.transpose(to_world, [1, 2, 0]))

    # Generate random local forward direction
    k_local = np.random.normal(size=[batch_size, 3])
    k_local = k_local / np.linalg.norm(k_local, axis=-1, keepdims=True)
    k_local_mi = vec_np_to_mi(k_local)
    #
    k_world = batch_matvec(to_world, k_local)
    k_world_mi = vec_np_to_mi(k_world)

    # Compute implicit S and P basis vector for the local frame
    s_local_mi = implicit_basis_vector(k_local_mi)
    s_local = vec_mi_to_np(s_local_mi)
    p_local = np.cross(k_local, s_local)

    # Compute implicit S and P basis vector for the world frame
    s_world_mi = implicit_basis_vector(k_world_mi)
    s_world = vec_mi_to_np(s_world_mi)
    p_world = np.cross(k_world, s_world)

    # Generate random local Jones vectors
    jones_local = np.random.random(size=[batch_size, 2])
    # 3D field vector in local frame
    field_vec_local = np.concatenate([jones_local,
                                    np.zeros([batch_size, 1])], axis=1)
    field_vec_local = field_vec_local[:,:1]*s_local + field_vec_local[:,1:2]*p_local
    # Field vector in world frame
    field_vec_world = batch_matvec(to_world, field_vec_local)

    # Compute the rotator using the function to test, and rotates the Jones vector
    rotator_mi = to_world_jones_rotator(to_world_mi, k_local_mi)
    rotator = mat_mi_to_np(rotator_mi)
    jones_world = batch_matvec(rotator, jones_local)

    # Test that `jones_world` is accurate
    ref_jones_world_x = np.sum(field_vec_world*s_world, axis=-1)
    assert max_rel_se(ref_jones_world_x, jones_world[:,0]) < MAX_RSE
    ref_jones_world_y = np.sum(field_vec_world*p_world, axis=-1)
    assert max_rel_se(ref_jones_world_y, jones_world[:,1]) < MAX_RSE

def test_jones_vec_dot():
    """Test `jones_vec_dot()` against numpy"""

    def batch_dot_product(vectors_a, vectors_b):
        # Batch dot product using np.einsum
        return np.einsum('ij,ij->i', np.conj(vectors_a), vectors_b)

    np.random.seed(42)
    batch_size = 100

    u = np.random.normal(size=[batch_size, 4])
    v = np.random.normal(size=[batch_size, 4])

    u_np = u[:,:2] + 1j*u[:,2:]
    v_np = v[:,:2] + 1j*v[:,2:]

    u_mi = mi.Vector4f(u[:,0], u[:,1], u[:,2], u[:,3])
    v_mi = mi.Vector4f(v[:,0], v[:,1], v[:,2], v[:,3])

    p_mi = jones_vec_dot(u_mi, v_mi)
    p_np = p_mi.numpy()
    ref_p = batch_dot_product(u_np,v_np)
    assert max_rel_se(ref_p, p_np, axes=None) < MAX_RSE

def test_jones_matrix_rotator_flip_forward():
    r"""Test `jones_matrix_rotator_flip_forward()`"""

    batch_size = 100

    np.random.seed(42)

    # Forward direction
    k = np.random.normal(size=[batch_size, 3])
    k = k / np.linalg.norm(k, axis=-1, keepdims=True)
    k_mi = vec_np_to_mi(k)

    # Basis for the current forward direction
    s = implicit_basis_vector(k_mi).numpy().T
    p = np.cross(k, s)

    # Basis for the flipped forward direction
    sf = implicit_basis_vector(-k_mi).numpy().T
    pf = np.cross(-k, sf)

    # Rotator that flip the forward direction
    flip_rotator = np.transpose(jones_matrix_rotator_flip_forward(k_mi).numpy(),
                                [2,0,1])

    # Generate random Jones vector, assumed to be represented
    # in the non-flipped basis
    j_real = np.random.normal(size=[batch_size, 2])
    j_imag = np.random.normal(size=[batch_size, 2])
    j = j_real + 1j*j_imag

    # Rotate to the basis with flipped fwd
    jf = batch_matvec(flip_rotator, j)

    # `j` and `jf` should correspond to the same vector
    v1 = j[:,:1]*s + j[:,1:]*p
    v2 = jf[:,:1]*sf + jf[:,1:]*pf
    assert max_rel_se(v1, v2) < MAX_RSE


@pytest.mark.parametrize("ki_local, reflection, rotated", [
    # Oblique specular reflection, identity surface frame
    ([0.3, 0.4, -0.8], True, False),
    # Exact normal-incidence reflection (TE basis is degenerate)
    ([0.0, 0.0, -1.0], True, False),
    # Near-normal reflection
    ([1e-8, 0.0, -1.0], True, False),
    # Transmission (same local direction of propagation)
    ([0.3, 0.4, -0.8], False, False),
    # Exact normal-incidence transmission
    ([0.0, 0.0, -1.0], False, False),
    # Oblique reflection with a rotated surface frame
    ([0.3, 0.4, -0.8], True, True),
    # Normal reflection with a rotated surface frame
    ([0.0, 0.0, -1.0], True, True),
])
def test_jones_matrix_to_world_implicit(ki_local, reflection, rotated):
    r"""
    Independent golden for :func:`~sionna.rt.utils.jones_matrix_to_world_implicit`.

    Rebuilds the complex :math:`2 \times 2` Jones matrix from the public rotator
    helpers, embeds it into the real :math:`4 \times 4` layout, and compares
    against the production embedding — including exact and near-normal
    incidence, where the TE basis falls back to a deterministic transverse
    vector.
    """
    np.random.seed(42)

    ki_mi = _vec3(ki_local)
    if reflection:
        ko_mi = dr.normalize(mi.reflect(-ki_mi))
    else:
        ko_mi = ki_mi

    if rotated:
        to_world_np = scipy_rotation.from_euler(
            'xyz', [0.4, -0.7, 1.1]).as_matrix()
    else:
        to_world_np = np.eye(3)

    to_world = _to_world_mi(to_world_np)

    # Lossy, non-trivial Fresnel-like coefficients (distinct TE/TM).
    c1 = mi.Complex2f(-0.5767383, 0.03351247)
    c2 = mi.Complex2f(0.09469027, -0.02514344)

    M = jones_matrix_to_world_implicit(c1, c2, to_world, ki_mi, ko_mi)
    M_np = mat4_mi_to_np(M)
    assert np.all(np.isfinite(M_np))

    J_ref = ref_complex_jones_world_implicit(c1, c2, to_world, ki_mi, ko_mi)
    M_ref = np.stack([embed_complex_jones(J) for J in J_ref], axis=0)
    assert max_rel_se(M_ref, M_np) < MAX_RSE

    # Applying M to a random implicit-world Jones vector must match J @ e.
    e_c = (np.random.normal(size=(J_ref.shape[0], 2))
           + 1j * np.random.normal(size=(J_ref.shape[0], 2)))
    e_real = np.concatenate([e_c.real, e_c.imag], axis=-1)
    out_real = batch_matvec(M_np, e_real)
    out_c = out_real[:, :2] + 1j * out_real[:, 2:]
    out_c_ref = batch_matvec(J_ref, e_c)
    assert max_rel_se(out_c_ref, out_c) < MAX_RSE


def test_jones_matrix_to_world_implicit_identity_and_zero_coeffs():
    r"""
    Special-coefficient checks for the world-implicit embedding.

    With :math:`c_1 = c_2 = 1` the map is a pure change of basis and must
    preserve the field amplitude. With :math:`c_1 = c_2 = 0` the output must
    vanish.
    """
    np.random.seed(7)

    ki_mi = _vec3([0.2, -0.5, -0.8])
    ko_mi = dr.normalize(mi.reflect(-ki_mi))
    to_world = _to_world_mi(
        scipy_rotation.from_euler('xyz', [0.2, 0.5, -0.3]).as_matrix())

    e_c = np.array([[0.7 - 0.2j, -0.3 + 0.5j]])
    e_real = np.concatenate([e_c.real, e_c.imag], axis=-1)

    M_one = mat4_mi_to_np(
        jones_matrix_to_world_implicit(mi.Complex2f(1.0, 0.0),
                                      mi.Complex2f(1.0, 0.0),
                                      to_world, ki_mi, ko_mi))
    out_one = batch_matvec(M_one, e_real)
    out_one_c = out_one[:, :2] + 1j * out_one[:, 2:]
    assert np.allclose(np.linalg.norm(out_one_c, axis=-1),
                       np.linalg.norm(e_c, axis=-1), atol=1e-6)

    M_zero = mat4_mi_to_np(
        jones_matrix_to_world_implicit(mi.Complex2f(0.0, 0.0),
                                      mi.Complex2f(0.0, 0.0),
                                      to_world, ki_mi, ko_mi))
    out_zero = batch_matvec(M_zero, e_real)
    assert np.allclose(out_zero, 0.0, atol=1e-12)


@pytest.mark.parametrize("ki_local, reflection, rotated", [
    ([0.3, 0.4, -0.8], True, False),
    ([0.0, 0.0, -1.0], True, False),
    ([0.3, 0.4, -0.8], False, True),
    ([0.0, 0.0, -1.0], False, False),
])
def test_specular_reflection_transmission_matrix_matches_jones_embedding(
        ki_local, reflection, rotated):
    r"""
    Material-level Fresnel/Jones golden.

    :meth:`~sionna.rt.RadioMaterial._specular_reflection_transmission_matrix`
    must select the ITU slab reflection or transmission coefficients and embed
    them through :func:`~sionna.rt.utils.jones_matrix_to_world_implicit`. This
    pins the TE/TM-to-world-implicit chain used by specular paths without going
    through the path or radio-map solvers.
    """
    material = RadioMaterial(name="jones-fresnel-golden",
                             relative_permittivity=5.24,
                             conductivity=0.01,
                             thickness=0.1)

    fc = 3.5e9
    wavelength = speed_of_light / fc
    omega = 2.0 * np.pi * fc
    eta = complex_relative_permittivity(mi.Float(5.24), mi.Float(0.01),
                                        mi.Float(omega))

    ki_mi = _vec3(ki_local)
    cos_theta_i = float(np.asarray((-ki_mi.z).numpy()).reshape(-1)[0])
    if reflection:
        ko_mi = dr.normalize(mi.reflect(-ki_mi))
    else:
        ko_mi = ki_mi

    if rotated:
        to_world_np = scipy_rotation.from_euler(
            'xyz', [-0.3, 0.8, 0.2]).as_matrix()
    else:
        to_world_np = np.eye(3)

    to_world = _to_world_mi(to_world_np)

    r_te, r_tm, t_te, t_tm = itu_coefficients_single_layer_slab(
        mi.Float(cos_theta_i), eta, mi.Float(0.1), mi.Float(wavelength))

    mat = material._specular_reflection_transmission_matrix(
        to_world, ki_mi, ko_mi, mi.Bool(reflection),
        r_te, r_tm, t_te, t_tm)

    c1 = r_te if reflection else t_te
    c2 = r_tm if reflection else t_tm
    J_ref = ref_complex_jones_world_implicit(c1, c2, to_world, ki_mi, ko_mi)
    M_ref = np.stack([embed_complex_jones(J) for J in J_ref], axis=0)
    M_np = mat4_mi_to_np(mat)

    assert np.all(np.isfinite(M_np))
    assert max_rel_se(M_ref, M_np) < MAX_RSE
