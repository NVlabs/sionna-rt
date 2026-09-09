#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Direct unit tests for antenna arrays, antenna patterns and radio devices."""

import pytest
import numpy as np
import mitsuba as mi
import drjit as dr
from scipy.constants import speed_of_light

from sionna.rt import (PlanarArray, AntennaArray, Transmitter, Receiver,
                       PathSolver, load_scene)
from sionna.rt.antenna_pattern import (AntennaPattern, PolarizedAntennaPattern,
                                       antenna_pattern_to_world_implicit,
                                       v_iso_pattern)
from sionna.rt.utils import rotation_matrix, jones_vec_dot



###########################################
# AntennaPattern.patterns validation
###########################################

class _PatternHolder(AntennaPattern):
    """Minimal concrete AntennaPattern for setter tests."""

    def __init__(self):
        super().__init__()
        self.patterns = [lambda theta, phi: (mi.Complex2f(1, 0),
                                             mi.Complex2f(0, 0))]


def test_antenna_pattern_rejects_invalid_patterns_lists():
    pattern = _PatternHolder()

    with pytest.raises(ValueError, match="must be a list"):
        pattern.patterns = lambda theta, phi: (mi.Complex2f(1, 0),
                                               mi.Complex2f(0, 0))

    with pytest.raises(ValueError, match="length 1 or 2"):
        pattern.patterns = []

    with pytest.raises(ValueError, match="length 1 or 2"):
        pattern.patterns = [lambda t, p: None,
                            lambda t, p: None,
                            lambda t, p: None]

    with pytest.raises(ValueError, match="must be a callable"):
        pattern.patterns = ["not-callable"]


###########################################
# RadioDevice validation
###########################################

def test_radio_device_rejects_orientation_and_look_at_together():
    with pytest.raises(ValueError, match="Only one of `orientation` or "
                                         "`look_at`"):
        Transmitter("tx", position=[0, 0, 0],
                    orientation=[0, 0, 0], look_at=[1, 0, 0])


def test_radio_device_rejects_coincident_look_at():
    tx = Transmitter("tx", position=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="coincides with the radio device"):
        tx.look_at([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="coincides with the radio device"):
        Receiver("rx", position=[0.0, 0.0, 0.0], look_at=[0.0, 0.0, 0.0])


@pytest.mark.parametrize("color", [
    (1.0, 0.0),           # wrong length
    (0.0, 0.0, 0.0, 1.0),
    (-0.1, 0.0, 0.0),     # out of range
    (0.0, 1.1, 0.0),
])
def test_radio_device_rejects_invalid_color(color):
    with pytest.raises(ValueError, match="[Cc]olor"):
        Transmitter("tx", position=[0, 0, 0], color=color)


@pytest.mark.parametrize("position, target, expected", [
    # +x is the default device heading: theta=π/2, phi=0 → (0, 0, 0)
    ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    # +y: theta=π/2, phi=π/2 → (π/2, 0, 0)
    ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [np.pi / 2, 0.0, 0.0]),
    # -x: theta=π/2, phi=π → (π, 0, 0)
    ([0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [np.pi, 0.0, 0.0]),
    # +z: theta=0 → (0, -π/2, 0)
    ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -np.pi / 2, 0.0]),
    # -z: theta=π → (0, π/2, 0)
    ([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, np.pi / 2, 0.0]),
    # Oblique direction from a non-origin position
    ([1.0, 2.0, 3.0], [2.0, 3.0, 4.0],
     [np.pi / 4, np.arccos(1.0 / np.sqrt(3.0)) - np.pi / 2, 0.0]),
])
def test_radio_device_look_at_sets_orientation(position, target, expected):
    """look_at sets (α, β, γ) = (φ, θ−π/2, 0) for the direction to target."""
    tx = Transmitter("tx", position=position)
    tx.look_at(target)
    assert np.allclose(tx.orientation.numpy()[:, 0], expected, atol=1e-5)

    # Same orientation when looking at another radio device at `target`.
    rx = Receiver("rx", position=target)
    tx.look_at(rx)
    assert np.allclose(tx.orientation.numpy()[:, 0], expected, atol=1e-5)


###########################################
# Antenna-pattern normalization golden
###########################################

@pytest.mark.parametrize("pattern, expected_directivity", [
    ("iso", 1.0),
    ("dipole", 1.5),
    ("hw_dipole", 1.643),
])
def test_antenna_pattern_normalization_golden(pattern, expected_directivity):
    r"""
    Integrated directional gain equals :math:`4\pi\eta_\text{rad}`, and
    registered patterns that are normalized for unit radiation efficiency
    satisfy :math:`\eta_\text{rad}\approx 1` with the analytic directivity.
    """
    array = PlanarArray(num_rows=1, num_cols=1, pattern=pattern,
                        polarization="V")
    d, g_max, eta_rad = array.antenna_pattern.compute_gain(
        verbose=False, num_samples=2000)

    d = float(d.numpy()[0])
    g_max = float(g_max.numpy()[0])
    eta_rad = float(eta_rad.numpy()[0])

    # By definition η_rad = ∫ G(θ,φ) sinθ dθ dφ / (4π), so the sphere
    # integral of directional gain must equal 4π η_rad.
    assert np.isclose(g_max / d, eta_rad, rtol=1e-5)

    # Analytic radiation efficiency ≈ 1 for these registered patterns.
    assert np.isclose(eta_rad, 1.0, rtol=2e-2)
    assert np.isclose(d, expected_directivity, rtol=2e-2)


###########################################
# Friis link-budget golden (antenna path)
###########################################

def test_friis_link_budget_iso_antennas():
    r"""
    Matched isotropic V-pol antennas on a free-space LoS link recover the
    Friis amplitude :math:`|a|=\lambda/(4\pi d)`.

    Complements the path-solver Friis sanity check by also asserting that the
    world-implicit antenna patterns alone contribute unit power on boresight.
    """
    distance = 10.0
    scene = load_scene()
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                 pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                 pattern="iso", polarization="V")
    scene.add(Transmitter("tx", [0.0, 0.0, 0.0]))
    scene.add(Receiver("rx", [distance, 0.0, 0.0]))
    scene.get("tx").look_at(scene.get("rx"))
    scene.get("rx").look_at(scene.get("tx"))

    # Antenna patterns alone: |f_tx · f_rx| = 1 on this boresight link.
    k = dr.normalize(mi.Vector3f(1.0, 0.0, 0.0))
    tx_to_world = rotation_matrix(scene.get("tx").orientation)
    rx_to_world = rotation_matrix(scene.get("rx").orientation)
    f_tx = antenna_pattern_to_world_implicit(
        scene.tx_array.antenna_pattern.patterns[0],
        tx_to_world, k, direction="out")
    f_rx = antenna_pattern_to_world_implicit(
        scene.rx_array.antenna_pattern.patterns[0],
        rx_to_world, -k, direction="in")
    coupling = jones_vec_dot(f_rx, f_tx)
    coupling_np = complex(coupling.real.numpy()[0], coupling.imag.numpy()[0])
    assert np.isclose(np.abs(coupling_np), 1.0, rtol=1e-5)

    paths = PathSolver()(
        scene,
        max_depth=0,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
    )
    a = paths.a[0].numpy() + 1j * paths.a[1].numpy()
    a0 = a.reshape(-1)[paths.tau.numpy().reshape(-1) >= 0][0]
    wavelength = float(scene.wavelength.numpy()[0])
    friis = wavelength / (4.0 * np.pi * distance)
    assert np.isclose(np.abs(a0), friis, rtol=1e-5)
    assert np.abs(np.angle(a0)) < 1e-5
    # Sanity: carrier frequency used by the empty scene is physical.
    assert wavelength == pytest.approx(speed_of_light / float(scene.frequency.numpy()[0]))
