#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import numpy as np
import drjit as dr
import mitsuba as mi

import sionna.rt
from sionna.rt import load_scene, Transmitter, Receiver, \
                      PlanarArray, PathSolver, RadioMaterial, r_hat

@pytest.mark.parametrize('synthetic_array', [True, False])
def test_cir_computation(synthetic_array):
    """Verify that CIR is correctly computed
       TX and RX are placed such that some paths are invalid
    """
    scene = load_scene(sionna.rt.scene.simple_reflector)
    scene.add(Transmitter("tx-1", [-10,0,10]))
    scene.add(Receiver("rx-1", [10,0,10]))
    scene.add(Transmitter("tx-2", [-10,1,10]))
    scene.add(Receiver("rx-2", [10,-1,10]))
    scene.tx_array = PlanarArray(num_cols=3, num_rows=3,
                                pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_cols=1, num_rows=2,
                                pattern="iso", polarization="VH")
    scene.get("tx-1").velocity = [-10, 10, 10]
    scene.get("rx-2").velocity = [-5, 8, 6]
    p_solver = PathSolver()
    paths = p_solver(scene, los=True, specular_reflection=True,
                    diffuse_reflection=False, refraction=False,
                    diffraction=True, edge_diffraction=True,
                    synthetic_array=synthetic_array)

    sampling_frequency = 10**6
    num_time_steps = 10
    a_rt, _ = paths.cir(sampling_frequency=sampling_frequency,
                    num_time_steps=num_time_steps,
                    normalize_delays=False, out_type="numpy")

    a = paths.a[0].numpy() + 1j*paths.a[1].numpy()
    tau = paths.tau.numpy()
    doppler = paths.doppler.numpy()
    if synthetic_array:
        # Add dimensions for broadcasting
        num_rx, num_tx, num_paths = tau.shape
        tau = np.reshape(tau, [num_rx, 1, num_tx, 1, num_paths])
        doppler = np.reshape(doppler, [num_rx, 1, num_tx, 1, num_paths])

    # Compute baseband euivalent channel coefficients
    a_b = a*np.exp(-1j*2*np.pi*scene.frequency.numpy()*tau)

    # Apply Doppler
    t = np.arange(0, num_time_steps)/sampling_frequency
    doppler = np.expand_dims(doppler, axis=-1)
    a_ref = np.expand_dims(a_b, -1)*np.exp(1j*2*np.pi*doppler*t)
    err = np.max(np.abs(a_rt-a_ref)/np.where(a_rt==0, 1, np.abs(a_rt)))
    assert err < 1e-3

@pytest.mark.parametrize('synthetic_array', [True, False])
def test_cir_delay_normalization(synthetic_array):
    """Test that normalization of path delays works as expected.
       TX and RX are placed such that some paths are invalid.
    """
    scene = load_scene(sionna.rt.scene.simple_reflector)
    scene.add(Transmitter("tx-1", [-10,0,10]))
    scene.add(Receiver("rx-1", [10,0,10]))
    scene.add(Transmitter("tx-2", [-10,1,10]))
    scene.add(Receiver("rx-2", [10,-1,10]))
    scene.tx_array = PlanarArray(num_cols=3, num_rows=3,
                                pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_cols=1, num_rows=2,
                                pattern="iso", polarization="VH")
    p_solver = PathSolver()

    # Compute paths
    paths = p_solver(scene, los=True, specular_reflection=True,
                    diffuse_reflection=False, refraction=False,
                    diffraction=True, edge_diffraction=True,
                    synthetic_array=synthetic_array)
    _, tau = paths.cir(normalize_delays=False, out_type="numpy")
    _, tau_norm = paths.cir(normalize_delays=True, out_type="numpy")

    if not synthetic_array:
        num_rx, _, num_tx, _, _ = tau.shape
        for i in range(num_rx):
            for j in range(num_tx):
                z = tau[i,:,j,:]
                z = np.where(z<0, np.inf, z)
                t_min = np.min(z)
                z -= t_min
                z = np.where(z==np.inf, -1, z)
                assert np.array_equal(z, tau_norm[i,:,j,:])
    else:
        num_rx, num_tx, _ = tau.shape
        for i in range(num_rx):
            for j in range(num_tx):
                z = tau[i,j]
                z = np.where(z<0, np.inf, z)
                t_min = np.min(z)
                z -= t_min
                z = np.where(z==np.inf, -1, z)
                assert np.array_equal(z, tau_norm[i,j])

def test_cir_doppler_vs_geometry_updates():
    scene = load_scene(sionna.rt.scene.simple_reflector, merge_shapes=False)
    scene.add(Transmitter("tx", [-10,0,10]))
    scene.add(Receiver("rx", [10,0,10]))
    scene.tx_array = PlanarArray(num_cols=1, num_rows=1,
                                 pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array
    v_tx = np.array([3., 0., -3.])
    v_rx = np.array([-3., 0., -3.])
    v_ref =  np.array([0., 0., 3.])
    scene.get("tx").velocity = v_tx
    scene.get("rx").velocity = v_rx
    scene.get("reflector").velocity = v_ref

    p_solver = PathSolver()

    paths = p_solver(scene, los=True, specular_reflection=True,
                    diffuse_reflection=False, refraction=False,
                    diffraction=False, edge_diffraction=False,
                    synthetic_array=False)

    # Doppler-based time evolution
    sampling_frequency = 10**4
    num_time_steps = 10
    a, _ = paths.cir(sampling_frequency=sampling_frequency,
                    num_time_steps=num_time_steps,
                    normalize_delays=False, out_type="numpy")
    a = np.squeeze(a)

    # Geometrical time evolution
    d_tx = v_tx/sampling_frequency
    d_rx = v_rx/sampling_frequency
    d_ref = v_ref/sampling_frequency
    a_geo = np.zeros([2,0])
    for _ in range(num_time_steps):
        paths = p_solver(scene, los=True, specular_reflection=True,
                    diffuse_reflection=False, refraction=False,
                    diffraction=False, edge_diffraction=False,
                    synthetic_array=False)
        a_geo_, _ = paths.cir(normalize_delays=False, out_type="numpy")
        a_geo_ = np.squeeze(a_geo_, axis=(0,1,2,3))
        a_geo = np.concatenate([a_geo, a_geo_], axis=1)
        scene.get("tx").position  += d_tx
        scene.get("rx").position  += d_rx
        scene.get("reflector").position  += d_ref

    assert np.max(np.abs(a - a_geo)/ np.abs(a)) < 0.005

@pytest.mark.parametrize('synthetic_array', [True, False])
def test_cir_reverse_direction(synthetic_array):

    scene = load_scene(sionna.rt.scene.box)

    scene.tx_array = PlanarArray(num_rows=4, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=8, num_cols=1, pattern="iso", polarization="V")

    scene.add(Transmitter("tx-1",
                        position=(4, 3, 1.5)))
    scene.add(Transmitter("tx-2",
                        position=(4, 3, 2.5)))
    scene.add(Transmitter("tx-3",
                        position=(4, 3, 3.5)))

    scene.add(Receiver("rx-1",
                    position=(-4, 0, 2.0)))
    scene.add(Receiver("rx-2",
                    position=(-4, 0, 3.0)))

    scene.get("tx-1").velocity = [1.0, 2.0, 3.0]
    scene.get("tx-2").velocity = [-4.0, 5.0, 6.0]
    scene.get("tx-3").velocity = [7.0, -8.0, 9.0]
    scene.get("rx-1").velocity = [-2.0, 3.0, -4.0]
    scene.get("rx-2").velocity = [5.0, 6.0, -7.0]

    p_solver = PathSolver()
    paths = p_solver(scene, synthetic_array=synthetic_array)

    sampling_frequency = 1e3
    num_time_steps = 3
    a, tau = paths.cir(sampling_frequency=sampling_frequency,
                    num_time_steps=num_time_steps, out_type="numpy",
                    reverse_direction=False)

    a_r, tau_r = paths.cir(sampling_frequency=sampling_frequency,
                    num_time_steps=num_time_steps, out_type="numpy",
                    reverse_direction=True)

    # Reverse direction using numpy
    a_ref = np.transpose(a, (2, 3, 0, 1, 4, 5))
    if synthetic_array:
        tau_ref = np.transpose(tau, (1, 0, 2))
    else:
        tau_ref = np.transpose(tau, (2, 3, 0, 1, 4))

    assert a_ref.shape == a_r.shape
    assert tau_ref.shape == tau_r.shape

    assert np.allclose(a_ref, a_r)
    assert np.allclose(tau_ref, tau_r)

def test_aoa_aod():
    scene = load_scene(sionna.rt.scene.box)

    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    tx_position = [0, 0, 1.0]
    rx_position = [0, 0, 1.0]


    tx = Transmitter(name="tx", position=tx_position)
    rx = Receiver(name="rx", position=rx_position)

    scene.add(tx)
    scene.add(rx)

    p_solver = PathSolver()

    MAX_DEPTH = [1, 2]

    for max_depth in MAX_DEPTH:
        paths = p_solver(
            scene=scene,
            max_depth=max_depth,
            los=False,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=False,
            diffraction=False,
            synthetic_array=True,
            seed=1,
        )

        # If paths have an even number of bounces, then the direction of arrival
        # should be opposite to the direction of departure
        int_types = paths.interactions.numpy()[:,0,0,:]
        int_types = int_types > 0
        depth = np.sum(int_types, axis=0)
        flip_direction = (-1)**(depth+1)

        theta_r = paths.theta_r.array
        phi_r = paths.phi_r.array
        theta_t = paths.theta_t.array
        phi_t = paths.phi_t.array

        k_tx = r_hat(theta_t, phi_t)
        k_rx = r_hat(theta_r, phi_r)

        ktx_dot_krx = dr.dot(k_tx, k_rx).numpy()
        max_d = np.max(np.abs(ktx_dot_krx - flip_direction))
        assert max_d < 1e-5

@pytest.mark.parametrize("phi",  np.linspace(-np.pi*0.5, np.pi*0.5, 10))
def test_synthetic_array(phi):
    """Test that synthetic and real arrays elad to similar channel impulse
    in the far field responses"""

    # Load empty scene
    scene = load_scene()
    wavelength = scene.frequency.numpy()[0]

    # Solver
    solver = PathSolver()

    # Set arrays
    scene.tx_array = PlanarArray(num_rows=3,
                                 num_cols=3,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="tr38901",
                                 polarization="VH")
    scene.rx_array = PlanarArray(num_rows=3,
                                 num_cols=3,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="dipole",
                                 polarization="cross")

    d = 100.0
    # Transmitter and receiver
    x = d*dr.cos(phi)
    y = d*dr.sin(phi)
    rx = Receiver("rx", position=mi.Point3f(0., 0., 0.))
    tx = Transmitter("tx", position=mi.Point3f(x, y, 0))
    tx.look_at(rx)
    scene.add(tx)
    scene.add(rx)

    # Synthetic array
    paths = solver(scene, synthetic_array = True)
    a_real, a_imag = paths.a
    a_real = a_real.numpy()
    a_imag = a_imag.numpy()
    a = a_real + 1j*a_imag
    a_synthetic = a[0,:,0,:,0]
    tau_synthetic = paths.tau.numpy()[0,0,0]
    del paths, a_real, a_imag, a

    # Non-synthetic array
    paths = solver(scene, synthetic_array = False)
    a_real, a_imag = paths.a
    a_real = a_real.numpy()
    a_imag = a_imag.numpy()
    a = a_real + 1j*a_imag
    a_non_synthetic = a[0,:,0,:,0]
    tau_non_synthetic = paths.tau.numpy()[0,:,0,:,0]
    del paths, a_real, a_imag, a

    # Apply phase shift due to different propagation delays between the antenna for
    # the simulation using non-synthetic arrays
    a_non_synthetic = a_non_synthetic*np.exp(-1j*2.*np.pi*(tau_non_synthetic-tau_synthetic)*wavelength)

    max_err = np.max(np.abs(a_synthetic - a_non_synthetic)/np.abs(a_synthetic))
    assert max_err < 1e-2


def _diffracted_gains(scattering_coefficient):
    r"""
    Computes the complex diffracted path gains on the ``simple_wedge`` scene for
    a given scattering coefficient.

    Only diffraction is enabled (no LoS, no specular/diffuse reflection, no
    refraction), so all returned paths are diffracted paths. The gains are
    returned sorted by propagation delay so that they can be matched across
    solver runs that use different scattering coefficients (path index ordering
    is not stable across runs, but the geometry -- and hence the delays -- is).

    :param scattering_coefficient: Scattering coefficient :math:`S \in [0, 1]`
    :return: Complex-valued diffracted path gains, sorted by delay
    """
    scene = load_scene(sionna.rt.scene.simple_wedge, merge_shapes=False)

    # Only the scattering coefficient changes across runs; the geometry and the
    # (complex) permittivity stay identical.
    scene.get("wedge").radio_material.scattering_coefficient = \
        scattering_coefficient

    scene.tx_array = PlanarArray(num_cols=1, num_rows=1,
                                 pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_cols=1, num_rows=1,
                                 pattern="iso", polarization="V")

    # Transmitter and receiver placed on opposite sides of the wedge so that a
    # diffracted path exists.
    scene.add(Transmitter("tx", [-10.0, -10.0, 0.0]))
    scene.add(Receiver("rx", [10.0, -15.0, 9.0]))

    # These runs are compared path-by-path, so use deterministic candidate
    # generation in addition to an explicit common random seed.
    p_solver = PathSolver(deterministic=True)
    paths = p_solver(scene,
                     los=False,
                     specular_reflection=False,
                     diffuse_reflection=False,
                     refraction=False,
                     diffraction=True,
                     edge_diffraction=True,
                     synthetic_array=True,
                     seed=42)

    a = paths.a[0].numpy() + 1j * paths.a[1].numpy()
    tau = paths.tau.numpy()

    # Single tx/rx with a single antenna each: drop the leading dimensions.
    a = a.reshape(-1)
    tau = tau.reshape(-1)

    # Keep only valid paths (non-negative delay) and sort by delay so that the
    # same physical path can be matched across runs.
    valid = tau >= 0.0
    a = a[valid]
    tau = tau[valid]
    order = np.argsort(tau)
    return a[order], tau[order]


def test_diffraction_depends_on_scattering_coefficient():
    r"""
    The diffracted field must depend on the scattering coefficient ``S``.

    When ``S > 0`` the Fresnel reflection coefficients are reduced by
    :math:`R = \sqrt{1 - S^2}`, and these reduced coefficients are used in the
    reflection terms of the UTD diffraction coefficients. With ``S = 0`` the
    reduction factor is ``R = 1`` and with ``S = 1`` it is ``R = 0``, which
    removes the reflection terms of the diffraction coefficient. The resulting
    fields must therefore differ.
    """
    a0, tau0 = _diffracted_gains(0.0)
    a1, tau1 = _diffracted_gains(1.0)

    # The geometry is unchanged, so the same paths (and delays) must be found.
    assert a0.size > 0, "Expected at least one diffracted path"
    assert a0.size == a1.size
    assert np.allclose(tau0, tau1, atol=1e-9)

    # The reflection terms must contribute, so removing them (S = 1) must change
    # the field.
    rel_change = np.linalg.norm(a1 - a0) / np.linalg.norm(a0)
    assert rel_change > 1e-3, \
        "Diffracted field does not change with the scattering coefficient"


def test_diffraction_incident_term_survives_at_s1():
    r"""
    At ``S = 1`` the reflection terms of the diffraction coefficient vanish but
    the incident (shadow-filling) term ``-(d1 + d2)`` must remain: a
    fully-diffuse wedge still casts a shadow. Only the reflection terms are
    reduced by :math:`R = \sqrt{1 - S^2}`, not the entire diffracted field.
    """
    a1, _ = _diffracted_gains(1.0)
    assert a1.size > 0
    assert np.all(np.abs(a1) > 0.0), \
        "Incident diffraction term must not vanish at S = 1"


def test_diffraction_reflection_term_scales_as_sqrt_1_minus_s2():
    r"""
    The diffracted field is affine in ``r = sqrt(1 - S^2)``:

        E(S) = A + r * B

    where ``A`` is the incident term and ``B`` the reflection term. Sampling at
    ``r = 1`` (S = 0), ``r = 0`` (S = 1) and ``r = 0.5`` (S = sqrt(0.75)) must
    therefore satisfy

        E(r=0.5) == 0.5 * E(r=1) + 0.5 * E(r=0)

    which verifies that *only* the reflection terms are scaled, and that they
    are scaled by exactly ``sqrt(1 - S^2)``.
    """
    r_half = 0.5
    s_half = np.sqrt(1.0 - r_half ** 2)  # S such that sqrt(1 - S^2) = 0.5

    a_r1, tau_r1 = _diffracted_gains(0.0)        # r = 1
    a_r0, tau_r0 = _diffracted_gains(1.0)        # r = 0
    a_rh, tau_rh = _diffracted_gains(s_half)     # r = 0.5

    # Same geometry across runs.
    assert a_r1.size > 0
    assert a_r1.size == a_r0.size == a_rh.size
    assert np.allclose(tau_r1, tau_r0, atol=1e-9)
    assert np.allclose(tau_r1, tau_rh, atol=1e-9)

    expected = r_half * a_r1 + (1.0 - r_half) * a_r0
    assert np.allclose(a_rh, expected, rtol=1e-4, atol=1e-12), \
        "Reflection term is not scaled by sqrt(1 - S^2)"


@pytest.mark.parametrize("ki_local", [
    [0.0, 0.0, -1.0],    # Normal incidence on the 0-face
    [0.0, -1.0, 0.0],    # Normal incidence on the n-face
    [0.0, 1e-8, -1.0],   # Near-normal incidence on the 0-face
    [0.0, -1.0, 1e-8],   # Near-normal incidence on the n-face
])
def test_diffraction_matrix_face_normal_incidence_is_finite(ki_local):
    """The diffraction Jones matrix must be finite at face-normal incidence."""

    material = RadioMaterial(name="test-material",
                             relative_permittivity=5.0,
                             conductivity=0.01)

    # Right-angle wedge with its edge along x. The 0-face normal is z+ and the
    # n-face normal is y+.
    si = mi.SurfaceInteraction3f()
    si.dn_du = mi.Vector3f(1.0, 0.0, 0.0)
    si.dn_dv = mi.Vector3f(0.0, 1.0, 0.0)
    si.dp_du = mi.Vector3f(1.0, 1.0, 0.0)

    ki_local = dr.normalize(mi.Vector3f(ki_local))
    ko_local = dr.normalize(mi.Vector3f(0.0, 0.6, 0.8))
    matrix = material._diffraction_matrix(
        mi.Matrix3f(1.0),
        ki_local,
        ko_local,
        si,
        mi.Complex2f(5.0, -0.1),
        mi.Float(100.0),
    )

    assert np.all(np.isfinite(matrix.numpy()))


def test_diffraction_matrix_shadow_boundary_is_finite_and_nonzero():
    """UTD coefficient stays finite and non-zero on a reflection shadow boundary.

    The old ``cot()`` helper forced poles to zero, which zeroed the singular
    UTD term exactly on the boundary. The regularized ``cot_times_f_utd``
    product must leave a finite, non-vanishing diffraction matrix there, with
    nearby off-boundary samples of comparable magnitude.
    """

    material = RadioMaterial(name="test-material-isb",
                             relative_permittivity=5.0,
                             conductivity=0.01)

    # Right-angle wedge, edge along x; 0-face normal z+, n-face normal y+.
    si = mi.SurfaceInteraction3f()
    si.dn_du = mi.Vector3f(1.0, 0.0, 0.0)
    si.dn_dv = mi.Vector3f(0.0, 1.0, 0.0)
    si.dp_du = mi.Vector3f(2.0, 2.0, 0.0)

    # Incident from the 0-face side along -z. With phi = phi' = π/2 the
    # reflection term d4 hits a cotangent pole (sum_phi = π).
    ki_local = dr.normalize(mi.Vector3f(0.0, 0.0, -1.0))
    ko_on = dr.normalize(mi.Vector3f(0.0, 0.0, 1.0))

    def eval_matrix(ko):
        return material._diffraction_matrix(
            mi.Matrix3f(1.0),
            ki_local,
            ko,
            si,
            mi.Complex2f(5.0, -0.1),
            mi.Float(100.0),
        ).numpy()

    m_on = eval_matrix(ko_on)
    assert np.all(np.isfinite(m_on))
    norm_on = np.linalg.norm(m_on)
    assert norm_on > 1e-3

    for eps in (1e-3, 1e-4):
        for sign in (+1.0, -1.0):
            m = eval_matrix(dr.normalize(mi.Vector3f(0.0, sign * eps, 1.0)))
            assert np.all(np.isfinite(m))
            assert np.isclose(np.linalg.norm(m), norm_on, rtol=0.5)


@pytest.mark.parametrize('tx_polarization', ['V', 'VH'])
@pytest.mark.parametrize('rx_polarization', ['V', 'VH'])
@pytest.mark.parametrize('synthetic_array', [True, False])
def test_empty_paths_shapes(tx_polarization, rx_polarization, synthetic_array):
    """Empty path tensors must match the public fused antenna shape contract.

    Covers all combinations of single/dual polarization on TX and RX with
    synthetic and non-synthetic arrays. For non-synthetic arrays the contract
    is ``[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]``, where
    ``num_*_ant`` includes polarization pattern ports. Synthetic arrays keep
    ``[num_rx, num_tx, num_paths]`` for non-coefficient tensors.
    """
    scene = load_scene()
    # Unequal physical array sizes so fused vs unfused shapes are distinct
    # whenever either side is dual-polarized.
    scene.tx_array = PlanarArray(num_rows=1, num_cols=2,
                                 pattern="iso",
                                 polarization=tx_polarization)
    scene.rx_array = PlanarArray(num_rows=1, num_cols=3,
                                 pattern="iso",
                                 polarization=rx_polarization)
    scene.add(Transmitter("tx", position=[0.0, 0.0, 1.0]))
    scene.add(Receiver("rx", position=[10.0, 0.0, 1.0]))

    num_tx_ant = scene.tx_array.num_ant
    num_rx_ant = scene.rx_array.num_ant
    assert num_tx_ant == scene.tx_array.array_size * (
        2 if tx_polarization == 'VH' else 1)
    assert num_rx_ant == scene.rx_array.array_size * (
        2 if rx_polarization == 'VH' else 1)

    # Disable every interaction so the solver returns an empty path set.
    paths = PathSolver()(
        scene,
        los=False,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=synthetic_array,
    )

    a_shape = (1, num_rx_ant, 1, num_tx_ant, 0)
    if synthetic_array:
        link_shape = (1, 1, 0)
    else:
        link_shape = a_shape

    a_real, a_imag = paths.a
    assert tuple(a_real.shape) == a_shape
    assert tuple(a_imag.shape) == a_shape
    assert tuple(paths.tau.shape) == link_shape
    assert tuple(paths.valid.shape) == link_shape
    assert tuple(paths.doppler.shape) == link_shape
    assert tuple(paths.theta_t.shape) == link_shape
    assert tuple(paths.phi_t.shape) == link_shape
    assert tuple(paths.theta_r.shape) == link_shape
    assert tuple(paths.phi_r.shape) == link_shape
    # The direction vectors are internal, but they follow the same shape
    # contract with an extra dimension for the (x,y,z) components
    assert tuple(paths._k_tx.shape) == link_shape + (3,)
    assert tuple(paths._k_rx.shape) == link_shape + (3,)

    max_depth = paths.interactions.shape[0]
    components_shape = (max_depth,) + link_shape
    assert tuple(paths.interactions.shape) == components_shape
    assert tuple(paths.objects.shape) == components_shape
    assert tuple(paths.primitives.shape) == components_shape
    assert tuple(paths.vertices.shape) == components_shape + (3,)

    a_cir, tau_cir = paths.cir(num_time_steps=2, out_type="numpy")
    assert a_cir.shape == a_shape + (2,)
    assert tau_cir.shape == link_shape


@pytest.mark.parametrize('synthetic_array', [True, False])
def test_angles_match_internal_directions(synthetic_array):
    """The reported angles must agree with the internal direction vectors.

    The direction vectors are the primitive quantities and the angles exposed
    to the user are derived from them, so ``r_hat(theta_t, phi_t)`` must give
    back the stored direction of departure, and likewise for the direction of
    arrival.
    """
    scene = load_scene(sionna.rt.scene.simple_reflector)
    scene.tx_array = PlanarArray(num_rows=1, num_cols=2,
                                 pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=3,
                                 pattern="iso", polarization="VH")
    scene.add(Transmitter("tx", [-25, 0.1, 50]))
    scene.add(Receiver("rx", [25, -0.3, 50]))

    paths = PathSolver()(scene, max_depth=1,
                         synthetic_array=synthetic_array)

    k_tx = paths._k_tx.numpy()
    k_rx = paths._k_rx.numpy()
    assert k_tx.shape == tuple(paths.theta_t.shape) + (3,)
    assert k_rx.shape == tuple(paths.theta_r.shape) + (3,)

    ref_tx = r_hat(paths.theta_t.array,
                   paths.phi_t.array).numpy().T.reshape(k_tx.shape)
    ref_rx = r_hat(paths.theta_r.array,
                   paths.phi_r.array).numpy().T.reshape(k_rx.shape)
    assert np.allclose(k_tx, ref_tx, atol=1e-5)
    assert np.allclose(k_rx, ref_rx, atol=1e-5)

    # The directions must be unit vectors
    assert np.allclose(np.linalg.norm(k_tx, axis=-1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(k_rx, axis=-1), 1.0, atol=1e-5)


def test_xpd_coefficient_traversal_updates_matrix():
    r"""
    A Mitsuba traversal write of ``xpd_coefficient`` must update the XPD Jones
    matrix used for diffuse reflection.

    The coefficient is a differentiable scene parameter exposed via
    ``RadioMaterial.traverse()``. Updates go through ``mi.traverse`` /
    ``SceneParameters.update()`` and bypass the property setter, so the matrix
    must be derived from the live coefficient rather than a stale cache.
    """
    kx = 0.25
    expected_a = np.sqrt(1.0 - kx)
    expected_b = np.sqrt(kx)

    material = RadioMaterial(name="xpd-traverse", relative_permittivity=5.0)
    # Default K_x = 0 yields the identity-like XPD block.
    m0 = material._xpd_matrix().numpy()
    assert np.isclose(m0[0, 0, 0], 1.0)
    assert np.isclose(m0[1, 0, 0], 0.0)

    params = mi.traverse(material)
    params['xpd_coefficient'] = mi.Float(kx)
    params.update()

    m_traverse = material._xpd_matrix().numpy()
    assert np.isclose(m_traverse[0, 0, 0], expected_a)
    assert np.isclose(m_traverse[0, 1, 0], -expected_b)
    assert np.isclose(m_traverse[1, 0, 0], expected_b)
    assert np.isclose(m_traverse[1, 1, 0], expected_a)

    # Traversal and the public setter must agree.
    material_setter = RadioMaterial(name="xpd-setter",
                                    relative_permittivity=5.0)
    material_setter.xpd_coefficient = kx
    m_setter = material_setter._xpd_matrix().numpy()
    assert np.allclose(m_traverse, m_setter)
