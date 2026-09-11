#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile
import time

import mitsuba as mi
import numpy as np
import drjit as dr
import pytest
import sionna.rt as rt
from sionna.rt import load_scene, Transmitter, PlanarArray, ITURadioMaterial,\
    Receiver, PathSolver, RadioMapSolver, BackscatteringPattern, PlanarRadioMap,\
    MeshRadioMap
from sionna.rt.utils import dbm_to_watt, load_mesh, transform_mesh, WedgeGeometry


####################################################
# Utilities
####################################################

def paths_to_coverage_map(paths, is_mesh=False):
    """
    Converts paths into the equivalent coverage map values.
    The coverage map is assumed to be a ssquare.
    """
    # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    a_real, a_imag = paths.a
    a = a_real.numpy() + 1j*a_imag.numpy()

    # Transmit precoding
    # Assume default precoding
    # [num_rx, num_rx_ant, num_tx, num_paths]
    a /= np.sqrt(a.shape[3])
    a = np.sum(a, axis=3)

    # Sum energy of paths
    a = np.square(np.abs(a))
    # [num_rx, num_tx]
    a = np.sum(a, axis=(1, 3))

    # Swap dims
    # [num_tx, num_rx]
    a = a.T

    if not is_mesh:
        # Reshape to coverage map
        n = int(np.sqrt(a.shape[1]))
        shape = [a.shape[0], n, n]
        a = np.reshape(a, shape)

    return a

def default_array(num_rows: int = 1, num_cols: int = 1,
                  vertical_spacing: float = 0.5, horizontal_spacing: float = 0.5,
                  pattern: str = "iso", polarization: str = "V"):
    return PlanarArray(num_rows=num_rows,
                       num_cols=num_cols,
                       vertical_spacing=vertical_spacing,
                       horizontal_spacing=horizontal_spacing,
                       pattern=pattern,
                       polarization=polarization)


####################################################
# Public argument validation
####################################################

@pytest.fixture
def radio_map_solver_and_scene():
    scene = load_scene()
    scene.tx_array = default_array()
    scene.add(Transmitter(name="tx", position=mi.Point3f(0.0, 0.0, 10.0)))
    return RadioMapSolver(), scene

def test_radio_map_solver_rejects_malformed_precoding_vec(
        radio_map_solver_and_scene):
    solver, scene = radio_map_solver_and_scene
    common = dict(samples_per_tx=10, max_depth=0, size=mi.Point2f(1, 1),
                  center=mi.Point3f(0, 0, 0), orientation=mi.Point3f(0),
                  cell_size=mi.Point2f(1, 1))

    with pytest.raises(TypeError, match="tuple or list"):
        solver(scene, precoding_vec=mi.TensorXf([1.0]), **common)

    with pytest.raises(ValueError, match="shape"):
        solver(scene,
               precoding_vec=(mi.TensorXf([[1.0, 2.0]]),
                              mi.TensorXf([[0.0, 0.0]])),
               **common)

    with pytest.raises(ValueError, match="length"):
        solver(scene,
               precoding_vec=(mi.Float([1.0, 0.0]), mi.Float([0.0, 0.0])),
               **common)


def test_radio_map_solver_rejects_stale_modified_scene(
        radio_map_solver_and_scene):
    solver, scene = radio_map_solver_and_scene
    measurement_surface = mi.Mesh("measurement-surface", 3, 1)
    params = mi.traverse(measurement_surface)
    params["vertex_positions"] = [0.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0]
    params["faces"] = [0, 1, 2]
    params.update()

    with pytest.raises(ValueError, match="does not contain"):
        solver(scene,
               measurement_surface=measurement_surface,
               modified_scene=scene.mi_scene,
               samples_per_tx=10,
               max_depth=0)


def test_radio_map_solver_rejects_non_mesh_measurement_surface(
        radio_map_solver_and_scene):
    solver, scene = radio_map_solver_and_scene
    with pytest.raises(TypeError, match="Mitsuba Mesh"):
        solver(scene,
               measurement_surface=mi.load_dict({"type": "sphere"}),
               samples_per_tx=10,
               max_depth=0)


def test_mesh_radio_map_rejects_degenerate_triangles(
        radio_map_solver_and_scene):
    _, scene = radio_map_solver_and_scene
    # Collinear vertices -> zero-area triangle
    measurement_surface = mi.Mesh("degenerate-measurement-surface", 3, 1)
    params = mi.traverse(measurement_surface)
    params["vertex_positions"] = [0.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  2.0, 0.0, 0.0]
    params["faces"] = [0, 1, 2]
    params.update()

    # Default: skip the expensive area check
    MeshRadioMap(scene, measurement_surface)

    with pytest.raises(ValueError, match="degenerate"):
        MeshRadioMap(scene, measurement_surface, check_cells_area=True)

    solver = RadioMapSolver()
    with pytest.raises(ValueError, match="degenerate"):
        solver(scene,
               measurement_surface=measurement_surface,
               check_cells_area=True,
               samples_per_tx=10,
               max_depth=0)


def test_mesh_radio_map_rejects_empty_mesh(radio_map_solver_and_scene):
    _, scene = radio_map_solver_and_scene
    measurement_surface = mi.Mesh("empty-measurement-surface", 0, 0)
    with pytest.raises(ValueError, match="at least one triangle"):
        MeshRadioMap(scene, measurement_surface)


def test_samples_per_tx_wedge():
    """Samples are counted independently for each transmitter and wedge."""

    tx_indices = mi.UInt([0, 0, 0, 1, 1, 1])
    wedge_indices = mi.UInt([0, 0, 1, 0, 1, 1])

    counts = RadioMapSolver._samples_per_tx_wedge(
        tx_indices, wedge_indices, num_txs=2, num_wedges=2)

    assert dr.all(counts == mi.UInt([2, 2, 1, 1, 2, 2]))


def test_multitransmitter_diffraction_map_normalization():
    """Adding colocated transmitters does not reduce each diffraction map."""

    def compute_map(num_tx):
        scene = load_scene(rt.scene.simple_wedge)
        scene.tx_array = default_array()
        for tx_ind in range(num_tx):
            scene.add(Transmitter(name=f"tx-{tx_ind}",
                                  position=[1.0, 1.0, 0.0]))

        radio_map = RadioMapSolver()(
            scene=scene,
            center=mi.Point3f(-1.0, -2.5, -10.0),
            orientation=mi.Point3f(0.0),
            size=mi.Point2f(1.0, 1.0),
            cell_size=mi.Point2f(0.1, 0.1),
            samples_per_tx=200_000,
            max_depth=1,
            los=False,
            specular_reflection=False,
            diffuse_reflection=False,
            refraction=False,
            diffraction=True,
            seed=42,
        )
        return radio_map.path_gain.numpy()

    single_tx_map = compute_map(1)
    two_tx_map = compute_map(2)

    single_tx_power = np.sum(single_tx_map[0])
    two_tx_power = np.mean(np.sum(two_tx_map, axis=(1, 2)))
    relative_power = two_tx_power / single_tx_power

    # The maps use independent Monte Carlo samples, hence the loose tolerance.
    # Global per-wedge counting gives approximately 0.5 and fails this check.
    assert np.isclose(relative_power, 1.0, rtol=0.3)


def test_diffraction_shadow_region_angular_weight():
    """Shadow-only and full-cone sampling estimate the same shadow map."""

    def compute_map(lit_region):
        scene = load_scene(rt.scene.simple_wedge)
        scene.tx_array = default_array()
        scene.add(Transmitter(name="tx", position=[1.0, 1.0, 0.0]))

        radio_map = RadioMapSolver()(
            scene=scene,
            center=mi.Point3f(-1.0, -2.5, -10.0),
            orientation=mi.Point3f(0.0),
            size=mi.Point2f(1.0, 1.0),
            cell_size=mi.Point2f(0.1, 0.1),
            samples_per_tx=200_000,
            max_depth=1,
            los=False,
            specular_reflection=False,
            diffuse_reflection=False,
            refraction=False,
            diffraction=True,
            diffraction_lit_region=lit_region,
            seed=42,
        )
        return np.sum(radio_map.path_gain.numpy())

    full_cone_power = compute_map(True)
    shadow_power = compute_map(False)

    assert np.isclose(shadow_power, full_cone_power, rtol=0.3)


def _numpy_diffraction_integration_weight(
        n0, e_hat, wedge_o, source, q, k_world, si_p, si_n, eps=1e-5):
    r"""Finite-difference geometric Jacobian
    :math:`\|\partial_{\phi} s \times \partial_{\ell} s\|` for the diffraction
    reparameterization used by
    :meth:`RadioMap._diffraction_integration_weight`.
    """
    n0 = np.asarray(n0, dtype=np.float64)
    e_hat = np.asarray(e_hat, dtype=np.float64)
    e_hat = e_hat / np.linalg.norm(e_hat)
    t0 = np.cross(n0, e_hat)
    t0 = t0 / np.linalg.norm(t0)
    # Columns: (t0, n0, e_hat) — edge-local to world.
    rot = np.column_stack([t0, n0, e_hat])

    wedge_o = np.asarray(wedge_o, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    k_world = np.asarray(k_world, dtype=np.float64)
    si_p = np.asarray(si_p, dtype=np.float64)
    si_n = np.asarray(si_n, dtype=np.float64)

    k_local = rot.T @ k_world
    k_local = k_local / np.linalg.norm(k_local)
    phi0 = np.arctan2(k_local[1], k_local[0])
    ell0 = np.linalg.norm(q - wedge_o)

    def compute_s(ell, phi):
        w = np.dot(source - wedge_o, e_hat)
        source_proj = wedge_o + w * e_hat
        v = source - wedge_o
        nrm = np.linalg.norm(ell * e_hat - v)
        sin_theta = np.linalg.norm(source - source_proj) / nrm
        cos_theta = (ell - w) / nrm
        d_local = np.array([sin_theta * np.cos(phi),
                            sin_theta * np.sin(phi),
                            cos_theta], dtype=np.float64)
        d_world = rot @ d_local
        u = si_p - wedge_o
        scale = np.dot(si_n, u - ell * e_hat) / np.dot(si_n, d_world)
        return wedge_o + ell * e_hat + scale * d_world

    j_ell = (compute_s(ell0 + eps, phi0) - compute_s(ell0 - eps, phi0)) / (2 * eps)
    j_phi = (compute_s(ell0, phi0 + eps) - compute_s(ell0, phi0 - eps)) / (2 * eps)
    return float(np.linalg.norm(np.cross(j_phi, j_ell)))


def test_diffraction_integration_weight_matches_finite_difference():
    """AD Jacobian of the diffraction surface map matches a finite-difference
    geometric reference.
    """
    scene = load_scene()
    scene.tx_array = default_array()
    scene.add(Transmitter(name="tx", position=mi.Point3f(2.0, 0.5, 0.0)))

    rm = PlanarRadioMap(
        scene,
        cell_size=mi.Point2f(1.0, 1.0),
        center=mi.Point3f(-1.0, -2.5, -10.0),
        orientation=mi.Point3f(0.0),
        size=mi.Point2f(2.0, 2.0),
    )

    # Right-angle edge along +z; 0-face normal +y.
    n0 = np.array([0.0, 1.0, 0.0])
    e_hat = np.array([0.0, 0.0, 1.0])
    wedge_o = np.array([0.0, 0.0, 0.0])
    source = np.array([2.0, 0.5, 0.0])
    q = np.array([0.0, 0.0, 1.0])  # diffraction point on the edge
    # Observation on the measurement plane below the edge.
    si_p = np.array([-1.0, -2.5, -10.0])
    si_n = np.array([0.0, 0.0, 1.0])
    k_world = si_p - q
    k_world = k_world / np.linalg.norm(k_world)

    wedges = WedgeGeometry.build_with_size(1)
    wedges.n0 = mi.Normal3f(n0)
    wedges.nn = mi.Normal3f(1.0, 0.0, 0.0)
    wedges.e_hat = mi.Vector3f(e_hat)
    wedges.o = mi.Point3f(wedge_o)
    wedges.length = mi.Float(2.0)

    si = mi.SurfaceInteraction3f()
    si.p = mi.Point3f(si_p)
    si.n = mi.Normal3f(si_n)

    w_ad = float(rm._diffraction_integration_weight(
        wedges,
        mi.Point3f(source),
        mi.Point3f(q),
        mi.Vector3f(k_world),
        si,
    ).numpy()[0])
    w_fd = _numpy_diffraction_integration_weight(
        n0, e_hat, wedge_o, source, q, k_world, si_p, si_n)

    assert np.isfinite(w_ad) and w_ad > 0.0
    assert np.isclose(w_ad, w_fd, rtol=1e-3)

    # Flipping the measurement normal cancels in the plane-intersection ratio.
    si.n = mi.Normal3f(-si_n)
    w_flipped = float(rm._diffraction_integration_weight(
        wedges,
        mi.Point3f(source),
        mi.Point3f(q),
        mi.Vector3f(k_world),
        si,
    ).numpy()[0])
    assert np.isclose(w_ad, w_flipped, rtol=1e-6)


def test_planar_radio_map_path_gain_invariant_to_normal_flip():
    """Planar non-diffracted weights use |cos θ|, so flipping the surface
    normal leaves the accumulated path gain unchanged.
    """
    scene = load_scene()
    scene.tx_array = default_array()
    scene.add(Transmitter(name="tx", position=mi.Point3f(0.0, 0.0, 10.0)))

    def make_rm():
        return PlanarRadioMap(
            scene,
            cell_size=mi.Point2f(1.0, 1.0),
            center=mi.Point3f(0.0, 0.0, 0.0),
            orientation=mi.Point3f(0.0),
            size=mi.Point2f(2.0, 2.0),
        )

    e_fields = [mi.Vector4f(1.0, 0.0, 0.0, 0.0)]
    array_w = [mi.Float(1.0)]
    k_world = mi.Vector3f(0.0, 0.0, -1.0)
    solid_angle = mi.Float(0.25)
    tx_indices = mi.UInt(0)
    active = mi.Bool(True)

    def accumulate(rm, normal_z):
        si = mi.SurfaceInteraction3f()
        si.uv = mi.Point2f(0.5, 0.5)
        si.p = mi.Point3f(0.0, 0.0, 0.0)
        si.n = mi.Normal3f(0.0, 0.0, normal_z)
        si.sh_frame = mi.Frame3f(si.n)
        rm.add_paths(
            e_fields, array_w, si, k_world, tx_indices, active,
            diffracted_paths=False, solid_angle=solid_angle)

    rm_pos = make_rm()
    rm_neg = make_rm()
    accumulate(rm_pos, +1.0)
    accumulate(rm_neg, -1.0)

    g_pos = rm_pos.path_gain.numpy()
    g_neg = rm_neg.path_gain.numpy()
    assert np.sum(g_pos) > 0.0
    assert np.allclose(g_pos, g_neg, rtol=1e-6, atol=0.0)


def test_mesh_radio_map_path_gain_invariant_to_normal_flip():
    """Mesh non-diffracted weights use |n·k|, so flipping the surface normal
    leaves the accumulated path gain unchanged.
    """
    scene = load_scene()
    scene.tx_array = default_array()
    scene.add(Transmitter(name="tx", position=mi.Point3f(0.0, 0.0, 10.0)))

    measurement_surface = mi.Mesh("normal-flip-ms", 3, 1)
    params = mi.traverse(measurement_surface)
    params["vertex_positions"] = [0.0, 0.0, 0.0,
                                  2.0, 0.0, 0.0,
                                  0.0, 2.0, 0.0]
    params["faces"] = [0, 1, 2]
    params.update()

    e_fields = [mi.Vector4f(1.0, 0.0, 0.0, 0.0)]
    array_w = [mi.Float(1.0)]
    k_world = mi.Vector3f(0.0, 0.0, -1.0)
    solid_angle = mi.Float(0.25)
    tx_indices = mi.UInt(0)
    active = mi.Bool(True)

    def accumulate(normal_z):
        rm = MeshRadioMap(scene, measurement_surface)
        si = mi.SurfaceInteraction3f()
        si.prim_index = mi.UInt(0)
        si.p = mi.Point3f(0.5, 0.5, 0.0)
        si.n = mi.Normal3f(0.0, 0.0, normal_z)
        rm.add_paths(
            e_fields, array_w, si, k_world, tx_indices, active,
            diffracted_paths=False, solid_angle=solid_angle)
        return rm.path_gain.numpy()

    g_pos = accumulate(+1.0)
    g_neg = accumulate(-1.0)
    assert np.sum(g_pos) > 0.0
    assert np.allclose(g_pos, g_neg, rtol=1e-6, atol=0.0)


def test_stop_threshold_passes_through_measurement_surface():
    """A negligible stop_threshold must not kill rays at measurement crossings.

    Russian roulette shares the same continue mask; it is not compared here
    because enabling it consumes extra sampler state and changes path samples.
    """

    scene = load_scene(rt.scene.box, merge_shapes=False)
    scene.objects["box"].radio_material = ITURadioMaterial(
        "concrete", "concrete", 0.1)
    scene.tx_array = default_array()
    scene.add(Transmitter(name="tx", position=mi.Point3f(0.0, 0.0, 2.5)))

    common = dict(
        max_depth=3,
        cell_size=mi.Point2f(0.5, 0.5),
        center=mi.Point3f(0.0, 0.0, 1.5),
        orientation=mi.Point3f(0.0),
        size=mi.Point2f(8.0, 8.0),
        samples_per_tx=100_000,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        seed=42,
    )
    solver = RadioMapSolver()
    g_ref = solver(scene, stop_threshold=None, **common).path_gain.numpy()
    g_th = solver(scene, stop_threshold=-1000.0, **common).path_gain.numpy()

    # Same seed and a threshold that never triggers on gain: maps must match.
    # Before the fix, any stop_threshold killed rays at every measurement hit.
    assert np.allclose(g_ref, g_th, rtol=1e-5, atol=0.0)
    assert np.sum(g_ref) > 0.0


def validate_cm(los=False,
                specular_reflection=False,
                diffuse_reflection=False,
                refraction=False,
                diffraction=False,
                rm_center_z=1,
                tx_pattern="iso",
                rx_pattern="iso",
                tx_pol="V"):
    """Compares coverage map against exact path calculation for
    different propagation phenomena.
    """

    scene = load_scene(rt.scene.simple_reflector, merge_shapes=False)

    scene.get("reflector").radio_material = ITURadioMaterial("mat-concrete", "concrete", thickness=0.1)
    rm = scene.get("reflector").radio_material
    scene.get("reflector").radio_material.scattering_coefficient = mi.Float(np.sqrt(0.5))
    scene.get("reflector").radio_material.xpd_coefficient = mi.Float(0.3)

    scene.tx_array = default_array(pattern=tx_pattern, polarization=tx_pol)
    scene.rx_array = default_array(pattern=rx_pattern, polarization="VH")

    delta = 0.1
    width = 3
    tx = Transmitter(name="tx",
                     position=mi.Point3f(0, 0, .1),
                     orientation=mi.Point3f(0, 0, 0))
    scene.add(tx)

    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=1,
                   cell_size=mi.Point2f(delta, delta),
                   center=mi.Point3f(0, 0, rm_center_z),
                   orientation=mi.Point3f(0., 0., 0.),
                   size=mi.Point2f(width, width),
                   samples_per_tx=int(1e7),
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction,
                   diffraction=diffraction,
                   edge_diffraction=diffraction,
                   diffraction_lit_region=diffraction)

    cell_centers = rm.cell_centers.numpy()
    cell_centers = np.reshape(cell_centers, [-1,3])
    for i, pos in enumerate(cell_centers):
        scene.add(Receiver(name=f"rx-{i}",
                           position=mi.Point3f(pos),
                           orientation=mi.Point3f(0., 0., 0.)))

    solver = PathSolver()
    paths = solver(scene,
                   samples_per_src=int(1e4),
                   max_num_paths_per_src=int(1e7),
                   max_depth=1,
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction,
                   diffraction=diffraction,
                   edge_diffraction=diffraction,
                   diffraction_lit_region=diffraction)

    rm_theo = paths_to_coverage_map(paths)[0]
    rm_rt = rm.path_gain.numpy()[0]

    err = np.where(rm_theo == 0.0,
                   0.0,
                   np.abs(rm_rt - rm_theo) / rm_theo)

    nmse_db = 10*np.log10(np.mean(np.abs(err)**2))
    return rm, rm_theo, nmse_db


#############################################################
# Tests
#############################################################

def test_planar_radio_map_non_divisible_grid():
    """Cell geometry stays consistent when size is not a multiple of cell_size."""

    scene = load_scene()
    scene.tx_array = default_array()
    scene.add(Transmitter(name="tx", position=mi.Point3f(0.0, 0.0, 10.0)))

    size = mi.Point2f(10.0, 10.0)
    cell_size = mi.Point2f(3.0, 3.0)
    center = mi.Point3f(0.0, 0.0, 1.5)
    orientation = mi.Point3f(0.0)

    rm = PlanarRadioMap(scene, cell_size, center, orientation, size)

    # Requested 10 / 3 cells expands to 4 cells spanning 12 m.
    assert rm.cells_per_dim.x[0] == 4
    assert rm.cells_per_dim.y[0] == 4
    assert dr.allclose(rm.size, mi.Point2f(12.0, 12.0))
    assert dr.allclose(rm.cell_size, cell_size)

    half_extent = 0.5 * np.array([rm.size.x[0], rm.size.y[0]])
    centers = rm.cell_centers.numpy()
    assert np.all(centers[..., 0] >= -half_extent[0])
    assert np.all(centers[..., 0] <= half_extent[0])
    assert np.all(centers[..., 1] >= -half_extent[1])
    assert np.all(centers[..., 1] <= half_extent[1])

    # Centers use the requested cell size on the snapped rectangle.
    expected_x = (np.arange(4) + 0.5) * 3.0 - 6.0
    expected_y = (np.arange(4) + 0.5) * 3.0 - 6.0
    assert np.allclose(centers[0, :, 0], expected_x)
    assert np.allclose(centers[:, 0, 1], expected_y)

    # Global indexing agrees with local UV binning for every cell center.
    flat_centers = centers.reshape(-1, 3)
    global_ind = rm._global_to_cell_ind(mi.Point3f(flat_centers.T))
    local_uv = mi.Point2f(
        (flat_centers[:, 0] + half_extent[0]) / rm.size.x[0],
        (flat_centers[:, 1] + half_extent[1]) / rm.size.y[0],
    )
    local_flat = rm._local_to_cell_ind(local_uv)
    expected_flat = global_ind.y * rm.cells_per_dim.x[0] + global_ind.x
    assert dr.all(local_flat == mi.Int(expected_flat))
    assert dr.all(global_ind.x == mi.UInt(np.tile(np.arange(4), 4)))
    assert dr.all(global_ind.y == mi.UInt(np.repeat(np.arange(4), 4)))

    # Area normalization uses the requested cell area (9), not size/cells (6.25).
    cell_area = rm.cell_size.x[0] * rm.cell_size.y[0]
    assert np.isclose(cell_area, 9.0)
    expected_norm = (scene.wavelength / (4.0 * np.pi))**2 / cell_area
    assert dr.allclose(rm._normalization_factor, expected_norm)

    # Covered measurement area matches the snapped rectangle.
    assert np.isclose(
        rm.size.x[0] * rm.size.y[0],
        rm.cells_per_dim.x[0] * rm.cells_per_dim.y[0] * cell_area,
    )


def test_sample_cells_num_valid_when_no_cell_matches():
    """Over-constrained sampling reports zero valid cells, not cell 0."""

    scene = load_scene()
    scene.tx_array = default_array()
    scene.add(Transmitter(name="tx", position=mi.Point3f(0.0, 0.0, 10.0)))

    rm = PlanarRadioMap(
        scene,
        cell_size=mi.Point2f(1.0, 1.0),
        center=mi.Point3f(0.0, 0.0, 1.5),
        orientation=mi.Point3f(0.0),
        size=mi.Point2f(10.0, 10.0),
    )

    # Untraced map has zero path gain in every cell, so no cell meets this floor.
    cells, num_valid = rm.sample_cells(num_cells=4, min_val_db=-50.0)
    assert tuple(cells.shape) == (1, 4)
    assert tuple(num_valid.shape) == (1,)
    assert int(num_valid.numpy()[0]) == 0

    pos, cell_ids, num_valid = rm.sample_positions(num_pos=4, min_val_db=-50.0)
    assert tuple(pos.shape) == (1, 4, 3)
    assert tuple(cell_ids.shape) == (1, 4, 2)
    assert int(num_valid.numpy()[0]) == 0
    assert np.all(np.isnan(pos.numpy()))


def test_random_positions():
    """test that random positions have a valid path loss and min/max
    distance is correctly set."""

    cell_size = mi.Point2f([4., 5.])
    batch_size = 100
    tx_pos = mi.Point3f(-210,73,105) # Top of Frauenkirche

    scene = load_scene(rt.scene.munich)

    tx = Transmitter(name="tx", position=tx_pos)
    scene.add(tx)

    scene.tx_array = default_array(num_rows=4, num_cols=4)
    scene.rx_array = default_array(num_rows=4, num_cols=4)

    # Position of the measurement plane
    radio_map_pos = dr.copy(scene.transmitters["tx"].position)
    radio_map_pos.z = 1.5

    ### Check with centering set to True

    # Generate the radio map
    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=5,
                   center=radio_map_pos,
                   orientation=(0., 0., 0.),
                   size=(500., 500.),
                   cell_size=cell_size,
                   los=True,
                   specular_reflection=True,
                   diffuse_reflection=True,
                   refraction=True,
                   diffraction=True,
                   samples_per_tx=int(1e7))

    samples_pos, samples_cell_ind, num_valid = rm.sample_positions(
        batch_size, min_val_db=-110, center_pos=True)
    assert int(num_valid.numpy()[0]) == batch_size
    samples_pos = samples_pos.numpy()
    cell_centers = rm.cell_centers.numpy()
    samples_pos = np.squeeze(samples_pos, axis=0) # Only a single transmitter
    samples_cell_ind = samples_cell_ind.numpy()
    samples_cell_ind = np.squeeze(samples_cell_ind, axis=0)
    # Check that the transmitter is always at the center of the cell
    for p, cell_ind in zip(samples_pos, samples_cell_ind):
        cell_center = cell_centers[cell_ind[0], cell_ind[1]]
        d = np.linalg.norm(p - cell_center)
        assert d == 0.

    ### Check with centering set to False

    # Generate the radio map
    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=5,
                   center=radio_map_pos,
                   orientation=(0., 0., 0.),
                   size=(500., 500.),
                   cell_size=cell_size,
                   los=True,
                   specular_reflection=True,
                   diffuse_reflection=True,
                   refraction=True,
                   diffraction=True,
                   samples_per_tx=int(1e7))
    samples_pos, samples_cell_ind, num_valid = rm.sample_positions(
        batch_size, min_val_db=-110, center_pos=False)
    assert int(num_valid.numpy()[0]) == batch_size
    samples_pos = samples_pos.numpy()
    cell_centers = rm.cell_centers.numpy()
    samples_pos = np.squeeze(samples_pos, axis=0) # Only a single transmitter
    samples_cell_ind = samples_cell_ind.numpy()
    samples_cell_ind = np.squeeze(samples_cell_ind, axis=0)
    # Check that the transmitter is always in the cell
    for p, cell_ind in zip(samples_pos, samples_cell_ind):
        cell_center = cell_centers[cell_ind[0], cell_ind[1]]
        d = np.abs(p - cell_center)
        assert (d[0] <= cell_size.x*0.5) and (d[1] <= cell_size.y*0.5)\
                and (d[2] == 0.0)

    ### Test min and max distance

    batch_size = 1000
    d_min = 150
    d_max = 300

    # max distance offset due to cell size quantization
    # dist can be off at most by factor 0.5 of diagonal
    d_cell = 0.5*dr.norm(cell_size).numpy()[0]
    low = d_min - d_cell
    high = d_max + d_cell

    samples_pos, _, num_valid = rm.sample_positions(batch_size,
                                        min_dist=d_min,
                                        max_dist=d_max,
                                        center_pos=False)
    assert int(num_valid.numpy()[0]) == batch_size
    samples_pos = samples_pos.numpy()
    samples_pos = np.squeeze(samples_pos, axis=0) # Only a single transmitter
    tx_pos = tx.position.numpy().T[0]
    for p in samples_pos:
        d = np.linalg.norm(p - tx_pos)
        assert (d > low) and (d < high)

    ### Test TX associations

    # Remove transmitter
    scene.remove("tx")

    # Add the first transmitter
    tx0 = Transmitter(name='tx0',
                      position=mi.Point3f(150, -100, 20),
                      orientation=mi.Point3f(0., 0., dr.pi*5/6),
                      power_dbm=44)
    scene.add(tx0)

    # Add the second transmitter
    tx1 = Transmitter(name='tx1',
                      position=mi.Point3f(-150, -100, 20),
                      orientation=mi.Point3f(0., 0., dr.pi/60),
                      power_dbm=44)
    scene.add(tx1)

    # Add the third transmitter
    tx2 = Transmitter(name='tx2',
                      position=mi.Point3f(0, 150 * dr.tan(dr.pi/3) - 100, 20),
                      orientation=mi.Point3f(0., 0., -dr.pi/2),
                      power_dbm=44)
    scene.add(tx2)

    # Compute radio map
    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=5,
                   cell_size=cell_size,
                   los=True,
                   specular_reflection=True,
                   diffuse_reflection=True,
                   refraction=True,
                   diffraction=True,
                   samples_per_tx=int(1e7))

    metric = 'sinr'
    sm = getattr(rm, 'sinr').numpy()
    cell_to_tx_ideal = np.argmax(sm, axis=0)
    cell_to_tx_ideal[np.max(sm, axis=0)==0] = -1

    _, samples_cell_ind, num_valid = rm.sample_positions(batch_size,
                                              center_pos=False,
                                              tx_association=True,
                                              metric=metric)
    samples_cell_ind = samples_cell_ind.numpy()
    num_valid = num_valid.numpy()

    for tx in range(sm.shape[0]):
        assert int(num_valid[tx]) == batch_size
        for cell_ind in samples_cell_ind[tx]:
            assert cell_to_tx_ideal[cell_ind[0], cell_ind[1]] == tx

def test_sinr_map():
    """Test SINR map"""

    cm_cell_size = np.array([4., 5.])

    scene = load_scene(rt.scene.munich)

    # Add the first transmitter
    tx0 = Transmitter(name='tx0',
                        position=mi.Point3f(150, -100, 20),
                        orientation=mi.Point3f(0., 0., dr.pi*5/6),
                        power_dbm=44)
    scene.add(tx0)

    # Add the second transmitter
    tx1 = Transmitter(name='tx1',
                    position=mi.Point3f(-150, -100, 20),
                    orientation=mi.Point3f(0., 0., dr.pi/60),
                    power_dbm=44)
    scene.add(tx1)

    # Add the third transmitter
    tx2 = Transmitter(name='tx2',
                    position=mi.Point3f(0, 150 * dr.tan(dr.pi/3) - 100, 20),
                    orientation=mi.Point3f(0., 0., -dr.pi/2),
                    power_dbm=44)
    scene.add(tx2)

    scene.tx_array = default_array(num_rows=4, num_cols=4)
    scene.rx_array = default_array(num_rows=4, num_cols=4)

    rx_pos = dr.copy(scene.transmitters["tx0"].position)
    rx_pos.z = 1.5

    # generate coverage map
    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=5,
                   center=rx_pos,
                   orientation=mi.Point3f(0., 0., 0.),
                   size=mi.Point2f(500., 500.),
                   cell_size=cm_cell_size,
                   los=True,
                   specular_reflection=True,
                   diffuse_reflection=True,
                   refraction=True,
                   samples_per_tx=int(1e7))

    # retrieve path gain map
    path_gain = rm.path_gain.numpy()

    # retrieve transmit power
    tx_power = [dbm_to_watt(tx.power_dbm).numpy()[0]
                for tx in scene.transmitters.values()]

    # retrieve noise power
    n0 = scene.thermal_noise_power.numpy()[0]

    num_tx = rm.num_tx
    sinr_map_per_tx_ideal = np.zeros(path_gain.shape)

    # compute the SINR via Numpy operations
    for tx in range(num_tx):
        interf_mat = np.zeros(path_gain.shape[1:])
        for tx_interf in range(num_tx):
            # compute interference
            if tx_interf != tx:
                interf_mat += tx_power[tx_interf] * \
                    path_gain[tx_interf, ::]
        # SINR per tx,  assuming that all tiles connect to transmitter tx
        sinr_map_per_tx_ideal[tx, ::] = tx_power[tx] * \
            path_gain[tx, ::] / (interf_mat + n0)

    # Tile to tx association
    tile_to_tx_ideal = np.argmax(sinr_map_per_tx_ideal, axis=0)
    tile_to_tx_ideal[np.max(sinr_map_per_tx_ideal, axis=0) == 0] = -1

    # Get SINR map and RT association from RT
    sinr_map_per_tx = rm.sinr.numpy()
    tile_to_tx = rm.tx_association('sinr').numpy()

    err_sinr_map_per_tx = np.sort(np.abs((sinr_map_per_tx - sinr_map_per_tx_ideal)\
                                        /sinr_map_per_tx_ideal).flatten())
    err_sinr_map_per_tx = err_sinr_map_per_tx[np.isfinite(err_sinr_map_per_tx)]

    err_tile_to_tx = np.max(abs(tile_to_tx_ideal - tile_to_tx))

    # check that 90-th percentile SINR has small error
    assert err_sinr_map_per_tx[int(.99*len(err_sinr_map_per_tx))] < .01
    assert err_tile_to_tx == 0

def test_los():
    """Test that LoS pathgain map is close to exact path calculation"""
    _, _, nmse_db = validate_cm(los=True, tx_pol="V")
    assert nmse_db <  -20
    _, _, nmse_db = validate_cm(los=True, tx_pol="cross")
    assert nmse_db < -20

def test_specular_reflection():
    """Test that reflection pathgain map is close to exact path calculation"""
    _, _, nmse_db = validate_cm(specular_reflection=True, tx_pol="V")
    assert nmse_db <  -20
    _, _, nmse_db = validate_cm(specular_reflection=True, tx_pol="cross")
    assert nmse_db <  -20

def test_scattering():
    """Test that scattering pathgain map is close to exact path calculation"""
    _, _, nmse_db = validate_cm(diffuse_reflection=True, tx_pol="V")
    assert nmse_db <  -20
    _, _, nmse_db = validate_cm(diffuse_reflection=True, tx_pol="cross")
    assert nmse_db <  -20

def test_refraction():
    """Test that refraction pathgain map is close to exact path calculation"""
    _, _, nmse_db = validate_cm(refraction=True, tx_pol="V", rm_center_z=-1)
    assert nmse_db <  -20
    _, _, nmse_db = validate_cm(refraction=True, tx_pol="cross", rm_center_z=-1)
    assert nmse_db <  -20

def test_edge_diffraction():
    """Test that diffraction pathgain map is close to exact path calculation"""
    _, _, nmse_db = validate_cm(diffraction=True, tx_pol="V", rm_center_z=-1)
    assert nmse_db < -20
    _, _, nmse_db = validate_cm(diffraction=True, tx_pol="cross", rm_center_z=-1)
    assert nmse_db < -20

def test_wedge_diffraction():
    """Test that diffraction pathgain map is close to exact path calculation
    for a wedge"""

    scene = load_scene(rt.scene.simple_wedge)
    scene.tx_array = default_array()
    scene.rx_array = default_array(polarization="VH")

    # Unique transmitter
    tx = Transmitter(name="tx",
                    position=[1.0, 1.0, 0.0],
                    orientation=[0,0,0])
    scene.add(tx)

    # Compute the diffracted field energy using the radio map
    rm = RadioMapSolver()(scene=scene,
                        center=mi.Point3f(-1., -2.5, -10.),
                        orientation=mi.Point3f(0., 0., 0.),
                        size=mi.Point2f(1., 1.),
                        cell_size=mi.Point2f(0.1, 0.1),
                        samples_per_tx=10**8,
                        los=False,
                        specular_reflection=False,
                        diffuse_reflection=False,
                        refraction=False,
                        diffraction=True,
                        diffraction_lit_region=True)
    rm_np = rm.path_gain.numpy()[0]

    # Compute the diffracted field using the paths solver
    cell_centers = np.reshape(rm.cell_centers.numpy(), [-1,3])
    for i, rx_pos in enumerate(cell_centers):
        scene.add(Receiver(name=f"rx-{i}",
                        position=rx_pos,
                        orientation=mi.Point3f(0., 0., 0.)))
    paths = PathSolver()(scene,
                        samples_per_src=10**5,
                        max_num_paths_per_src=1000,
                        max_depth=1,
                        los=False,
                        specular_reflection=False,
                        diffuse_reflection=False,
                        refraction=False,
                        diffraction=True)
    a, _ = paths.cir(out_type="numpy")
    paths_rm = np.sum(np.square(np.abs(a)), axis=1)
    paths_rm = np.squeeze(paths_rm)
    paths_rm = np.reshape(paths_rm, [10, 10])

    err = np.abs(rm_np - paths_rm) / paths_rm
    err_db = 10.*np.log10(err)
    assert np.max(err_db) < -10


def test_wedge_diffraction_deterministic():
    """Test that diffraction pathgain map is consistent accross runs for
    a given seed."""

    scene = load_scene(rt.scene.simple_street_canyon)
    scene.tx_array = default_array()
    scene.rx_array = default_array(polarization="VH")

    # Unique transmitter
    tx = Transmitter(name="tx",
                     position=[-45, 11.8, 35],
                     orientation=[0, 0, 0])
    scene.add(tx)

    solver = RadioMapSolver()
    radio_maps = []

    # Compute the diffracted field energy using the radio map
    t0 = time.time()
    for i in range(10):
        rm = solver(scene=scene,
                    samples_per_tx=int(1e7),
                    cell_size=(0.5, 0.5),
                    los=True,
                    specular_reflection=False,
                    diffuse_reflection=False,
                    refraction=False,
                    diffraction=True,
                    diffraction_lit_region=True)
        radio_maps.append(rm.path_gain)

    dr.eval(radio_maps)
    dr.sync_thread()
    t1 = time.time()
    print(f"Average runtime: {1000 * (t1 - t0) / len(radio_maps):.2f} ms / map")

    # Check that the radio maps are consistent
    for i in range(1, len(radio_maps)):
        assert np.allclose(radio_maps[i], radio_maps[0])



def test_diffraction_street_canyon():
    """Test that diffraction pathgain map is close to exact path calculation
    for the simple_street_canyon scene"""

    scene = load_scene(rt.scene.simple_street_canyon)

    tx = Transmitter(name="tx",
                     position=[-45, 11.8, 35],
                     orientation=[0, 0, 0])
    scene.add(tx)
    scene.tx_array = default_array()

    rmap = RadioMapSolver()(scene=scene,
                            samples_per_tx=10**9,
                            cell_size=(0.5, 0.5),
                            los=False,
                            specular_reflection=False,
                            refraction=False,
                            diffraction=True,
                            diffraction_lit_region=True)

    rx_indices = [(120, 120), (170, 50), (160, 60), (120, 200), (220, 100),
                  (175, 350)]
    for i, p in enumerate(rx_indices):
        rx_pos = rmap.cell_centers[p[0], p[1]]
        scene.add(
            Receiver(name=f"rx-{i}", position=rx_pos.numpy(),
                     display_radius=3.0)
        )

    scene.rx_array = default_array(polarization="VH")

    paths = PathSolver()(scene=scene,
                         los=False,
                         samples_per_src=10**6,
                         specular_reflection=False,
                         refraction=False,
                         diffraction=True,
                         diffraction_lit_region=True)

    a, _ = paths.cir(out_type="numpy")
    a = np.squeeze(a)
    paths_en = np.sum(np.square(np.abs(a)), axis=(1,2))

    for i, ind in enumerate(rx_indices):
        g_rm = rmap.path_gain.numpy()[0, ind[0], ind[1]]
        g_paths = paths_en[i]
        err = np.abs(g_rm - g_paths) / g_paths
        err_db = 10. * np.log10(err)
        assert np.max(err_db) < -5

def test_box_01():
    """Test that field scattered and reflected fields jointly match for
    max_depth=1 in the box scene.
    This test also applies orientation, directive antenna patterns, as well as
    depolariation during scattering.
    """
    los = False
    specular_reflection = True
    diffuse_reflection = True
    refraction = False
    max_depth = 1 # Test only works for max_depth=1
    delta = 1

    scene = load_scene(rt.scene.box, merge_shapes=False)
    scene.objects["box"].radio_material = ITURadioMaterial("concrete", "concrete", 0.1)
    scene.objects["box"].radio_material.scattering_coefficient = dr.sqrt(0.5)
    scene.objects["box"].radio_material.scattering_pattern = \
        BackscatteringPattern(alpha_r=30, alpha_i=10, lambda_=0.5)

    scene.tx_array = default_array(pattern="tr38901")
    scene.rx_array = default_array(polarization="VH")

    scene.add(Transmitter(name="tx",
                          position=mi.Point3f(1.1, 0.8, 2),
                          orientation=mi.Point3f(0,0,0)))
    scene.get("tx").look_at(mi.Point3f(5,5,5))

    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=max_depth,
                   cell_size=mi.Point2f(delta, delta),
                   samples_per_tx=int(1e8),
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction)

    for i, pos in enumerate(np.reshape(rm.cell_centers, [-1,3])):
        scene.add(Receiver(name=f"rx-{i}",
                           position=pos,
                           orientation=mi.Point3f(0, 0, 0)))

    solver = PathSolver()
    paths = solver(scene,
                   samples_per_src = 10000,
                   max_num_paths_per_src=int(1e7),
                   max_depth=max_depth,
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction,
                   diffraction=False)

    a = paths_to_coverage_map(paths)[0]
    nmse_db = 10 * np.log10(np.mean( ((rm.path_gain[0] - a) / a) ** 2 ))
    assert nmse_db < -20

def test_box_02():
    """Test a multiple reflections and LoS in the box scene.
    It includes directive antenna pattern and a complex material"""
    scene = load_scene(rt.scene.box, merge_shapes=False)
    scene.objects["box"].radio_material = ITURadioMaterial("concrete", "concrete", 0.1)
    scene.objects["box"].radio_material.scattering_coefficient = 0.2


    scene.tx_array = default_array(pattern="tr38901")
    scene.rx_array = default_array(polarization="VH")

    los = True
    specular_reflection = True
    diffuse_reflection = True
    refraction = False
    width = 9
    num_cells_x = 20
    delta = width / num_cells_x
    max_depth = 5

    scene.add(Transmitter(name="tx",
                          position=mi.Point3f(-3, -0.3, 4.),
                          orientation=mi.Point3f(0,0,0)))

    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=max_depth,
                   cell_size=mi.Point2f(delta, delta),
                   center=mi.Point3f(0.0,0.0,0.5),
                   orientation=mi.Point3f(0.,0.,0),
                   size=mi.Point2f(width, width),
                   samples_per_tx=int(1e7),
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction)


    for i, pos in enumerate(np.reshape(rm.cell_centers, [-1,3])):
        scene.add(Receiver(name=f"rx-{i}",
                           position=pos,
                           orientation=mi.Point3f(0, 0, 0)))

    solver = PathSolver()
    paths = solver(scene,
                   samples_per_src = int(1e4),
                   max_num_paths_per_src=int(2e7),
                   max_depth=max_depth,
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction,
                   diffraction=False)

    a = paths_to_coverage_map(paths)[0]
    nmse_db = 10*np.log10(np.mean( ((rm.path_gain[0]-a)/a)**2 ))
    assert nmse_db < -20.

@pytest.mark.parametrize("los", [True, False])
def test_mesh_radio_map(los):

    specular_reflection = True
    diffuse_reflection = True
    refraction=False
    max_depth = 2

    # Setup the scene
    scene = load_scene(rt.scene.box, merge_shapes=False)
    scene.objects["box"].radio_material = ITURadioMaterial("concrete", "concrete", 0.1)
    scene.objects["box"].radio_material.scattering_coefficient = dr.sqrt(0.5)
    scene.objects["box"].radio_material.scattering_pattern =\
        BackscatteringPattern(alpha_r=30, alpha_i=10, lambda_=0.5)

    scene.tx_array = default_array()
    scene.rx_array = default_array(polarization="VH")

    scene.add(Transmitter(name="tx",
                position=mi.Point3f(4, 3, 4.0),
                orientation=mi.Point3f(0,0,0),
                display_radius=0.1))
    scene.get("tx").look_at(mi.Point3f(0,0,0))


    # Load the measurement surface
    fname = os.path.join(os.path.dirname(__file__),
                         "../data/subdivided_cube.ply")
    ms = load_mesh(fname)
    transform_mesh(ms,
                   translation=mi.Point3f(0, 0, 2.5),
                   scale=mi.Point3f(1.5, 1.5, 1))

    # Compute the radio map
    rm_solver = RadioMapSolver()
    rm_solver.loop_mode = "evaluated"
    rm = rm_solver(scene,
                measurement_surface=ms,
                samples_per_tx=int(1e7),
                max_depth=max_depth,
                los=los,
                specular_reflection=specular_reflection,
                diffuse_reflection=diffuse_reflection,
                refraction=refraction)

    # Check for NaN or Inf
    assert not (dr.any(dr.isinf(rm.path_gain)) or dr.any(dr.isnan(rm.path_gain)))

    # Add receivers at the cell centers
    for i, pos in enumerate(rm.cell_centers.numpy().T):
        scene.remove(f"rx-{i}")
        scene.add(Receiver(name=f"rx-{i}",
                            position=pos,
                            display_radius=0.1,
                            orientation=mi.Point3f(0, 0, 0)))

    # Compute radio map using the path solver
    solver = PathSolver()
    paths = solver(scene,
                    samples_per_src = 10000,
                    max_num_paths_per_src=int(1e7),
                    max_depth=max_depth,
                    los=los,
                    specular_reflection=specular_reflection,
                    diffuse_reflection=diffuse_reflection,
                    refraction=refraction,
                    diffraction=False)
    a = paths_to_coverage_map(paths, is_mesh=True)[0]

    nmse_db = 10*np.log10(np.mean( ((rm.path_gain[0]-a)/a)**2 ))
    assert nmse_db < -20


def test_mesh_radio_map_diffraction_multiple_intersections():
    """Measurement-surface hits do not re-apply diffraction."""

    # Two large parallel measurement planes separated by only 2 cm. A
    # diffracted ray can hit both planes, whose gains should be nearly equal.
    extent = 20.0
    xy = [(-extent, -extent),
          ( extent, -extent),
          (-extent,  extent),
          ( extent,  extent)]
    vertices = []
    for z in (-9.99, -10.01):
        vertices.extend((x, y, z) for x, y in xy)
    vertices = np.asarray(vertices, dtype=np.float32)

    measurement_surface = mi.Mesh("two-plane-measurement-surface", 8, 4)
    params = mi.traverse(measurement_surface)
    params["vertex_positions"] = vertices.ravel()
    params["faces"] = [0, 1, 2, 1, 3, 2,
                       4, 5, 6, 5, 7, 6]
    params.update()

    scene = load_scene(rt.scene.simple_wedge)
    scene.tx_array = default_array()
    scene.add(Transmitter(name="tx", position=[1.0, 1.0, 0.0]))

    solver = RadioMapSolver()
    solver.loop_mode = "evaluated"
    radio_map = solver(
        scene=scene,
        measurement_surface=measurement_surface,
        samples_per_tx=500_000,
        max_depth=1,
        los=False,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=True,
        seed=42,
    )

    gains = radio_map.path_gain.numpy()[0]
    first_plane_power = np.mean(gains[:2])
    second_plane_power = np.mean(gains[2:])

    assert first_plane_power > 0.0
    assert np.isclose(second_plane_power, first_plane_power, rtol=0.5)


@pytest.mark.parametrize("color_map", [None, "rainbow", np.random.rand(30, 3)])
def test_show_association(color_map):
    scene = load_scene(rt.scene.box_two_screens)
    scene.tx_array = default_array()
    scene.rx_array = default_array(polarization="VH")
    bbox = scene.mi_scene.bbox()
    scene_size = bbox.extents()

    # Many transmitters to make sure that we are not limited by the number of
    # colors in the colormap.
    safe_range = 0.8
    rng = np.random.default_rng(seed=3445)
    for tx_i in range(20):
        p = mi.Point3f(rng.uniform(safe_range * bbox.min.x, safe_range * bbox.max.x),
                       rng.uniform(safe_range * bbox.min.y, safe_range * bbox.max.y),
                       safe_range * bbox.max.z)
        tx = Transmitter(name=f"tx-{tx_i}", position=p)
        scene.add(tx)

    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   cell_size=mi.Point2f(0.1, 0.1),
                   center=mi.Point3f(0, 0, 0.2 * bbox.center().z),
                   orientation=mi.Point3f(0., 0., 0.),
                   size=mi.Point2f(scene_size.x, scene_size.y),
                   samples_per_tx=int(1e4),
                   max_depth=5)

    if color_map is None:
        suffix = "default"
        # Check error handling if given an insufficiently large listed color map
        with pytest.raises(ValueError, match=r"The color map has 8 entries.*"):
            rm.show_association(color_map="Dark2")
    elif isinstance(color_map, str):
        suffix = "named"
    else:
        suffix = "custom"

    fig = rm.show_association(color_map=color_map)
    fig.tight_layout()

    # For visual inspection
    if False:
        fname = os.path.join(tempfile.gettempdir(),
                             f"test_show_association_{suffix}.png")
        fig.savefig(fname)
        print(f"Saved figure to: {fname}")

    # For visual inspection
    if False:
        fname = os.path.join(tempfile.gettempdir(),
                             f"test_show_association_preview_{suffix}.png")
        bbox = scene.mi_scene.bbox()
        to_world = mi.ScalarTransform4f().look_at(
            origin=mi.ScalarVector3f(2, 2, 3) * bbox.max,
            target=mi.ScalarVector3f(1, 1, 0) * bbox.center(),
            up=[0, 0, 1],
        )
        scene.render_to_file(camera=to_world, filename=fname, radio_map=rm,
                             clip_at=0.5 * bbox.max.z)
        print(f"Saved rendering to: {fname}")
