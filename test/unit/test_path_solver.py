#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import numpy as np
import mitsuba as mi
from scipy.constants import speed_of_light

import sionna.rt
from sionna.rt import (load_scene, PathSolver, Transmitter, Receiver,
                       PlanarArray, ITURadioMaterial, InteractionType,
                       DirectivePattern)
from sionna.rt.utils import (itu_coefficients_single_layer_slab,
                             complex_relative_permittivity, rotation_matrix,
                             jones_vec_dot)
from sionna.rt.antenna_pattern import antenna_pattern_to_world_implicit


@pytest.fixture
def solver_and_scene():
    return PathSolver(), load_scene()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_depth": -1},
        {"max_num_paths_per_src": 0},
        {"max_num_paths_per_src": -5},
        {"samples_per_src": 0},
        {"samples_per_src": -1},
        {"seed": -1},
    ],
)
def test_path_solver_rejects_out_of_range_args(solver_and_scene, kwargs):
    solver, scene = solver_and_scene
    with pytest.raises(ValueError):
        solver(scene, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_depth": 1.5},
        {"max_depth": True},
        {"max_num_paths_per_src": 1.0},
        {"max_num_paths_per_src": True},
        {"samples_per_src": "1000"},
        {"samples_per_src": False},
        {"seed": 1.2},
        {"seed": True},
        {"synthetic_array": 1},
        {"los": 0},
        {"specular_reflection": "True"},
        {"diffuse_reflection": 1},
        {"refraction": None},
        {"diffraction": 0},
        {"edge_diffraction": 1},
        {"diffraction_lit_region": "yes"},
    ],
)
def test_path_solver_rejects_wrong_type_args(solver_and_scene, kwargs):
    solver, scene = solver_and_scene
    with pytest.raises(TypeError):
        solver(scene, **kwargs)


def test_path_solver_rejects_non_scene():
    solver = PathSolver()
    with pytest.raises(TypeError, match="`scene` must be an instance of Scene"):
        solver(scene="not-a-scene")


def test_path_solver_accepts_zero_max_depth(solver_and_scene):
    """max_depth=0 is valid and means LoS-only candidate generation."""
    solver, scene = solver_and_scene
    # Argument validation must accept max_depth=0. The incomplete scene then
    # fails in all_set(), which runs only after the argument checks.
    with pytest.raises(ValueError, match="Transmitter array not set"):
        solver(scene, max_depth=0, samples_per_src=1, max_num_paths_per_src=1)


###########################################
# Sanity checks
###########################################

#############################
# Utilities
#############################

def _iso_v_arrays(scene):
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                 pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                 pattern="iso", polarization="V")


def _complex_a(paths):
    return paths.a[0].numpy() + 1j * paths.a[1].numpy()


def _cval(c):
    """Convert a Mitsuba / Dr.Jit complex scalar to a Python complex."""
    return complex(c.real.numpy()[0], c.imag.numpy()[0])


def _friis(wavelength, distance):
    return wavelength / (4.0 * np.pi * distance)


def _assign_itu_brick(obj, thickness, scattering_coefficient=0.0,
                      scattering_pattern="lambertian", **pattern_kwargs):
    mat = ITURadioMaterial(
        name=f"brick-{obj.name}",
        itu_type="brick",
        thickness=thickness,
        scattering_coefficient=scattering_coefficient,
        scattering_pattern=scattering_pattern,
        **pattern_kwargs,
    )
    obj.radio_material = mat
    return mat


def _valid_paths(paths):
    """Return flattened complex gains, delays and a validity mask."""
    a = _complex_a(paths).reshape(-1)
    tau = paths.tau.numpy().reshape(-1)
    valid = tau >= 0.0
    return a, tau, valid


#############################
# Tests
#############################


def test_los_friis_sanity():
    r"""
    Free-space LoS amplitude and delay match Friis / geometric delay.

    With matched isotropic vertically polarized antennas,
    :math:`|a| = \lambda/(4\pi d)` and :math:`\tau = d/c`.
    """
    distance = 10.0
    scene = load_scene()
    _iso_v_arrays(scene)
    scene.add(Transmitter("tx", [0.0, 0.0, 0.0]))
    scene.add(Receiver("rx", [distance, 0.0, 0.0]))
    scene.get("tx").look_at(scene.get("rx"))
    scene.get("rx").look_at(scene.get("tx"))

    paths = PathSolver()(
        scene,
        max_depth=0,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
    )

    a, tau, valid = _valid_paths(paths)
    assert np.count_nonzero(valid) == 1
    assert np.all(paths.interactions.numpy() == InteractionType.NONE)

    wavelength = float(scene.wavelength.numpy()[0])
    a0 = a[valid][0]
    tau0 = tau[valid][0]

    assert np.isclose(tau0, distance / speed_of_light, atol=1e-12)
    assert np.isclose(np.abs(a0), _friis(wavelength, distance), rtol=1e-5)
    assert np.abs(np.angle(a0)) < 1e-5


def test_specular_reflection_sanity():
    r"""
    Single specular bounce on ``simple_reflector`` with ITU brick (5 cm).

    Path length equals the image distance; the complex gain matches the TM
    ITU-R P.2040 slab reflection coefficient times the Friis factor
    :math:`\lambda/(4\pi r)`.
    """
    thickness = 0.05
    tx_pos = np.array([-10.0, 0.0, 10.0])
    rx_pos = np.array([10.0, 0.0, 10.0])
    hit = np.array([0.0, 0.0, 0.0])

    scene = load_scene(sionna.rt.scene.simple_reflector, merge_shapes=False)
    mat = _assign_itu_brick(scene.get("reflector"), thickness=thickness, scattering_coefficient=0.0)
    _iso_v_arrays(scene)
    scene.add(Transmitter("tx", tx_pos.tolist()))
    scene.add(Receiver("rx", rx_pos.tolist()))
    scene.get("tx").look_at(hit.tolist())
    scene.get("rx").look_at(hit.tolist())

    paths = PathSolver(deterministic=True)(
        scene,
        max_depth=1,
        los=False,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
    )

    a, tau, valid = _valid_paths(paths)
    assert np.count_nonzero(valid) == 1
    assert np.all(paths.interactions.numpy() == InteractionType.SPECULAR)

    # Image of the TX through the z=0 plane.
    r_img = float(np.linalg.norm(rx_pos - np.array([-10.0, 0.0, -10.0])))
    cos_theta = abs((hit - tx_pos)[2] / np.linalg.norm(hit - tx_pos))

    wavelength = float(scene.wavelength.numpy()[0])
    frequency = float(scene.frequency.numpy()[0])
    eta = complex_relative_permittivity(
        mi.Float(float(mat.relative_permittivity.numpy()[0])),
        mi.Float(float(mat.conductivity.numpy()[0])),
        mi.Float(2.0 * np.pi * frequency),
    )
    _, r_tm, _, _ = itu_coefficients_single_layer_slab(
        mi.Float(cos_theta), eta, mi.Float(thickness), mi.Float(wavelength))
    a_ref = _cval(r_tm) * _friis(wavelength, r_img)

    a0 = a[valid][0]
    assert np.isclose(tau[valid][0], r_img / speed_of_light, atol=1e-12)
    assert np.isclose(a0, a_ref, rtol=1e-4, atol=1e-12)


def test_transmission_sanity():
    r"""
    Normal-incidence transmission through ``simple_reflector`` with ITU brick
    (3 cm). TX and RX sit on opposite sides of the slab.

    The geometric delay is the Euclidean distance; the complex gain matches the
    ITU-R P.2040 slab transmission coefficient times
    :math:`-\lambda/(4\pi d)` (Jones / antenna-frame sign convention).
    """
    thickness = 0.03
    tx_pos = np.array([0.0, 0.0, 5.0])
    rx_pos = np.array([0.0, 0.0, -5.0])
    distance = float(np.linalg.norm(rx_pos - tx_pos))

    scene = load_scene(sionna.rt.scene.simple_reflector, merge_shapes=False)
    mat = _assign_itu_brick(scene.get("reflector"), thickness=thickness, scattering_coefficient=0.0)
    _iso_v_arrays(scene)
    scene.add(Transmitter("tx", tx_pos.tolist()))
    scene.add(Receiver("rx", rx_pos.tolist()))
    scene.get("tx").look_at(scene.get("rx"))
    scene.get("rx").look_at(scene.get("tx"))

    paths = PathSolver(deterministic=True)(
        scene,
        max_depth=1,
        los=False,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=True,
        diffraction=False,
    )

    a, tau, valid = _valid_paths(paths)
    assert np.count_nonzero(valid) == 1
    assert np.all(paths.interactions.numpy() == InteractionType.REFRACTION)

    wavelength = float(scene.wavelength.numpy()[0])
    frequency = float(scene.frequency.numpy()[0])
    eta = complex_relative_permittivity(
        mi.Float(float(mat.relative_permittivity.numpy()[0])),
        mi.Float(float(mat.conductivity.numpy()[0])),
        mi.Float(2.0 * np.pi * frequency),
    )
    # Normal incidence: TE and TM transmission coefficients coincide.
    _, _, t_te, _ = itu_coefficients_single_layer_slab(
        mi.Float(1.0), eta, mi.Float(thickness), mi.Float(wavelength))
    a_ref = -_cval(t_te) * _friis(wavelength, distance)

    a0 = a[valid][0]
    assert np.isclose(tau[valid][0], distance / speed_of_light, atol=1e-12)
    assert np.isclose(a0, a_ref, rtol=1e-4, atol=1e-12)


def _diffuse_power_riemann_ref(
        tx_pos,
        rx_pos,
        wavelength,
        eta,
        thickness,
        scattering_coefficient,
        alpha_r,
        tx_to_world,
        rx_to_world,
        tx_pattern,
        rx_pattern,
        num_cells=50,
):
    r"""
    Incoherent diffuse power at the RX by a Riemann sum over the unit square
    reflector on :math:`z=0` (:math:`[-0.5,0.5]^2`).

    Each patch of area :math:`\Delta A` contributes with the shoot-and-bounce
    solid angle :math:`\Omega = \cos\theta_i\,\Delta A / r_\text{tx}^2` and the
    same field scaling as the path solver
    (:math:`S\,\Gamma\sqrt{f_s\Omega}\,\lambda/(4\pi r_\text{rx})`), using the
    directive pattern and the TM ITU slab reflection coefficient for
    :math:`\Gamma` (matched to vertically polarized iso antennas on this
    geometry). Powers are summed non-coherently,
    :math:`P=\sum_p|\Delta a_p|^2`.
    """
    xs = np.linspace(-0.5, 0.5, num_cells, endpoint=False) + 0.5 / num_cells
    ys = np.linspace(-0.5, 0.5, num_cells, endpoint=False) + 0.5 / num_cells
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    qx = xx.reshape(-1).astype(np.float64)
    qy = yy.reshape(-1).astype(np.float64)
    qz = np.zeros_like(qx)
    dA = (1.0 / num_cells) ** 2

    v_i = np.stack([qx - tx_pos[0], qy - tx_pos[1], qz - tx_pos[2]], axis=0)
    r_tx = np.linalg.norm(v_i, axis=0)
    ki = v_i / r_tx
    cos_ti = -ki[2]  # n = z-hat

    v_s = np.stack([rx_pos[0] - qx, rx_pos[1] - qy, rx_pos[2] - qz], axis=0)
    r_rx = np.linalg.norm(v_s, axis=0)
    ks = v_s / r_rx
    cos_ts = ks[2]

    visible = (cos_ti > 0.0) & (cos_ts > 0.0)
    ki = ki[:, visible]
    ks = ks[:, visible]
    r_tx = r_tx[visible]
    r_rx = r_rx[visible]
    cos_ti = cos_ti[visible]

    omega = cos_ti * dA / np.square(r_tx)

    # Directive scattering pattern in the reflector-local frame (z = n).
    sp = DirectivePattern(alpha_r=alpha_r)
    ki_mi = mi.Vector3f(ki[0].astype(np.float32),
                        ki[1].astype(np.float32),
                        ki[2].astype(np.float32))
    ks_mi = mi.Vector3f(ks[0].astype(np.float32),
                        ks[1].astype(np.float32),
                        ks[2].astype(np.float32))
    fs = np.asarray(sp(ki_mi, ks_mi).numpy(), dtype=np.float64)

    # TM slab reflection coefficient -> gamma = |R_tm|
    _, r_tm, _, _ = itu_coefficients_single_layer_slab(
        mi.Float(cos_ti.astype(np.float32)),
        eta,
        mi.Float(thickness),
        mi.Float(wavelength),
    )
    gamma = np.abs(
        np.asarray(r_tm.real.numpy(), dtype=np.float64)
        + 1j * np.asarray(r_tm.imag.numpy(), dtype=np.float64)
    )

    e_i = antenna_pattern_to_world_implicit(
        tx_pattern, tx_to_world, ki_mi, direction="out")
    scale = (scattering_coefficient * gamma
             * np.sqrt(np.maximum(fs * omega, 0.0))).astype(np.float32)
    e_scaled = e_i * mi.Float(scale)

    e_r = antenna_pattern_to_world_implicit(
        rx_pattern, rx_to_world, -ks_mi, direction="in")
    a_c = jones_vec_dot(e_r, e_scaled)
    a_np = (np.asarray(a_c.real.numpy(), dtype=np.float64)
            + 1j * np.asarray(a_c.imag.numpy(), dtype=np.float64))
    a_np *= wavelength / (4.0 * np.pi * r_rx)
    return float(np.sum(np.abs(a_np) ** 2))


def test_diffuse_reflection_sanity():
    r"""
    Diffuse-only bounce on ``simple_reflector`` with ITU brick (5 cm),
    :math:`S = 0.7` and a directive scattering pattern.

    The non-coherent path-solver power :math:`\sum_i |a_i|^2` must match a
    Riemann sum of the diffusely scattered field over the reflector (EM primer
    / Degli-Esposti model), which validates the surface integral realized by
    shoot-and-bounce sampling.
    """
    thickness = 0.05
    scattering_coefficient = 0.7
    alpha_r = 10
    samples_per_src = 1_000_000
    tx_pos = np.array([-10.0, 0.0, 10.0])
    rx_pos = np.array([10.0, 0.0, 10.0])

    scene = load_scene(sionna.rt.scene.simple_reflector, merge_shapes=False)
    mat = _assign_itu_brick(
        scene.get("reflector"),
        thickness=thickness,
        scattering_coefficient=scattering_coefficient,
        scattering_pattern="directive",
        alpha_r=alpha_r,
    )
    _iso_v_arrays(scene)
    scene.add(Transmitter("tx", tx_pos.tolist()))
    scene.add(Receiver("rx", rx_pos.tolist()))
    scene.get("tx").look_at([0.0, 0.0, 0.0])
    scene.get("rx").look_at([0.0, 0.0, 0.0])

    paths = PathSolver(deterministic=True)(
        scene,
        max_depth=1,
        samples_per_src=samples_per_src,
        max_num_paths_per_src=200_000,
        los=False,
        specular_reflection=False,
        diffuse_reflection=True,
        refraction=False,
        diffraction=False,
        seed=1,
    )

    a, tau, valid = _valid_paths(paths)
    assert np.count_nonzero(valid) > 0
    assert np.all(paths.interactions.numpy() == InteractionType.DIFFUSE)

    vertices = paths.vertices.numpy()[0, 0, 0]
    d_geo = (np.linalg.norm(vertices - tx_pos, axis=-1)
             + np.linalg.norm(rx_pos - vertices, axis=-1))
    assert np.allclose(tau[valid], d_geo[valid] / speed_of_light, atol=1e-12)

    power_rt = float(np.sum(np.abs(a[valid]) ** 2))

    wavelength = float(scene.wavelength.numpy()[0])
    frequency = float(scene.frequency.numpy()[0])
    eta = complex_relative_permittivity(
        mi.Float(float(mat.relative_permittivity.numpy()[0])),
        mi.Float(float(mat.conductivity.numpy()[0])),
        mi.Float(2.0 * np.pi * frequency),
    )
    power_ref = _diffuse_power_riemann_ref(
        tx_pos=tx_pos,
        rx_pos=rx_pos,
        wavelength=wavelength,
        eta=eta,
        thickness=thickness,
        scattering_coefficient=scattering_coefficient,
        alpha_r=alpha_r,
        tx_to_world=rotation_matrix(scene.get("tx").orientation),
        rx_to_world=rotation_matrix(scene.get("rx").orientation),
        tx_pattern=scene.tx_array.antenna_pattern.patterns[0],
        rx_pattern=scene.rx_array.antenna_pattern.patterns[0],
        num_cells=50,
    )

    assert np.isclose(power_rt, power_ref, rtol=5e-2)


def test_diffuse_probability_compensation_unbiased():
    r"""
    Mixed specular/diffuse sampling must match diffuse-only power after
    :math:`1/\sqrt{p}` compensation.

    With only diffuse reflection enabled the event probability is normalized to
    one. With both specular and diffuse enabled the diffuse probability is
    :math:`S^2` (metal, all energy reflected). The field calculator scales the
    ray-tube solid angle by :math:`1/p` so the diffuse amplitude carries
    :math:`1/\sqrt{p}`; the non-coherent powers must therefore agree.
    """
    thickness = 0.05
    scattering_coefficient = 0.7
    samples_per_src = 1_000_000
    tx_pos = [-10.0, 0.0, 10.0]
    rx_pos = [10.0, 0.0, 10.0]

    def diffuse_power(specular_reflection):
        scene = load_scene(sionna.rt.scene.simple_reflector, merge_shapes=False)
        _assign_itu_brick(
            scene.get("reflector"),
            thickness=thickness,
            scattering_coefficient=scattering_coefficient,
            scattering_pattern="lambertian",
        )
        _iso_v_arrays(scene)
        scene.add(Transmitter("tx", tx_pos))
        scene.add(Receiver("rx", rx_pos))
        scene.get("tx").look_at([0.0, 0.0, 0.0])
        scene.get("rx").look_at([0.0, 0.0, 0.0])

        paths = PathSolver(deterministic=True)(
            scene,
            max_depth=1,
            samples_per_src=samples_per_src,
            max_num_paths_per_src=500_000,
            los=False,
            specular_reflection=specular_reflection,
            diffuse_reflection=True,
            refraction=False,
            diffraction=False,
            seed=1,
        )
        a, _, valid = _valid_paths(paths)
        diffuse = paths.interactions.numpy().reshape(-1) == InteractionType.DIFFUSE
        mask = valid & diffuse
        assert np.count_nonzero(mask) > 0
        return float(np.sum(np.abs(a[mask]) ** 2))

    power_diffuse_only = diffuse_power(specular_reflection=False)
    power_mixed = diffuse_power(specular_reflection=True)

    print(power_mixed, power_diffuse_only)
    assert np.isclose(power_mixed, power_diffuse_only, rtol=0.05)