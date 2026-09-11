#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import numpy as np
import mitsuba as mi

import sionna.rt
from sionna.rt import load_scene, Transmitter, Receiver, \
                      PlanarArray, PathSolver, subcarrier_frequencies


@pytest.mark.parametrize('synthetic_array', [True, False])
def test_cfr_computation(synthetic_array):
    """Verify that CFR is correctly computed
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
    num_subcarriers = 128
    subcarrier_spacing = 30e3
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
    h_rt = paths.cfr(
                    frequencies=frequencies,
                    sampling_frequency=sampling_frequency,
                    num_time_steps=num_time_steps,
                    normalize=False,
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

    # Compute channel frequency response
    e = np.exp(-1j*2*np.pi*np.expand_dims(tau, -1)*frequencies.numpy())
    e = np.expand_dims(e, -2)
    a_ref = np.expand_dims(a_ref, -1)
    h_ref = np.sum(a_ref*e, axis=-3)

    # Compute maximum relative error
    err = np.max(np.abs(h_rt-h_ref))

    assert err < 1e-6

@pytest.mark.parametrize('synthetic_array', [True, False])
def test_cfr_normalization(synthetic_array):
    """
    Check that normalization of the CFR across a slot is correctly applied
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
    num_subcarriers = 128
    subcarrier_spacing = 30e3
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
    h_rt = paths.cfr(
                    frequencies=frequencies,
                    sampling_frequency=sampling_frequency,
                    num_time_steps=num_time_steps,
                    normalize=True,
                    normalize_delays=False, out_type="numpy")
    norm = np.mean(np.abs(h_rt)**2, axis=(1,3,4,5))
    err = np.max(np.abs(norm-1))

    assert err < 1e-3


@pytest.mark.parametrize('synthetic_array', [True, False])
def test_cfr_reverse_direction(synthetic_array):
    """Test reverse direction with Doppler time evolution."""

    scene = load_scene()
    scene.tx_array = PlanarArray(num_rows=2, num_cols=1,
                                 pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=3, num_cols=1,
                                 pattern="iso", polarization="V")

    scene.add(Transmitter("tx-1", [0.0, 0.0, 1.0]))
    scene.add(Transmitter("tx-2", [0.0, 1.0, 2.0]))
    scene.add(Transmitter("tx-3", [0.0, -1.0, 3.0]))
    scene.add(Receiver("rx-1", [10.0, 0.0, 1.5]))
    scene.add(Receiver("rx-2", [10.0, 2.0, 2.5]))

    scene.get("tx-1").velocity = [1.0, 2.0, 3.0]
    scene.get("tx-2").velocity = [-4.0, 5.0, 6.0]
    scene.get("tx-3").velocity = [7.0, -8.0, 9.0]
    scene.get("rx-1").velocity = [-2.0, 3.0, -4.0]
    scene.get("rx-2").velocity = [5.0, 6.0, -7.0]

    paths = PathSolver()(scene, los=True, specular_reflection=False,
                         diffuse_reflection=False, refraction=False,
                         diffraction=False, synthetic_array=synthetic_array)

    frequencies = subcarrier_frequencies(4, 30e3)
    kwargs = {
        "frequencies": frequencies,
        "sampling_frequency": 1e3,
        "num_time_steps": 3,
        "normalize": False,
        "normalize_delays": False,
        "out_type": "numpy",
    }
    h = paths.cfr(reverse_direction=False, **kwargs)
    h_r = paths.cfr(reverse_direction=True, **kwargs)

    h_ref = np.transpose(h, (2, 3, 0, 1, 4, 5))
    assert h_ref.shape == h_r.shape
    assert np.allclose(h_ref, h_r)


@pytest.mark.parametrize('synthetic_array', [True, False])
def test_cfr_normalize_empty_paths(synthetic_array):
    """Empty paths with normalize=True must stay all-zero and finite."""
    scene = load_scene()
    scene.tx_array = PlanarArray(num_rows=1, num_cols=2,
                                 pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=3,
                                 pattern="iso", polarization="VH")
    scene.add(Transmitter("tx", position=[0.0, 0.0, 1.0]))
    scene.add(Receiver("rx", position=[10.0, 0.0, 1.0]))

    paths = PathSolver()(
        scene,
        los=False,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=synthetic_array,
    )
    assert paths.tau.shape[-1] == 0

    frequencies = subcarrier_frequencies(8, 30e3)
    h = paths.cfr(
        frequencies=frequencies,
        num_time_steps=2,
        normalize=True,
        normalize_delays=False,
        out_type="numpy",
    )

    assert h.shape == (1, scene.rx_array.num_ant, 1, scene.tx_array.num_ant,
                       2, 8)
    assert np.all(np.isfinite(h))
    assert np.allclose(h, 0.0)


@pytest.mark.parametrize('synthetic_array', [True, False])
def test_cfr_normalize_zero_energy_links(synthetic_array):
    """Zero-energy links stay zero under cfr(normalize=True); others unit-energy.

    Regression for the ``dr.select(c == 0, ...)`` guard: without it,
    ``0 * rsqrt(0)`` yields NaN and poisons the CFR.
    """
    scene = load_scene()
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                 pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                 pattern="iso", polarization="V")
    scene.add(Transmitter("tx", position=[0.0, 0.0, 1.0]))
    scene.add(Receiver("rx-0", position=[10.0, 0.0, 1.0]))
    scene.add(Receiver("rx-1", position=[10.0, 2.0, 1.0]))

    paths = PathSolver()(
        scene,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=synthetic_array,
    )
    assert paths.tau.shape[-1] > 0

    # Keep both links' path layout, but force RX-0 coefficients to zero energy.
    a_real = paths._a_real.numpy().copy()
    a_imag = paths._a_imag.numpy().copy()
    a_real[0, ...] = 0.0
    a_imag[0, ...] = 0.0
    paths._a_real = mi.TensorXf(a_real)
    paths._a_imag = mi.TensorXf(a_imag)

    frequencies = subcarrier_frequencies(16, 30e3)
    h = paths.cfr(
        frequencies=frequencies,
        num_time_steps=3,
        normalize=True,
        normalize_delays=False,
        out_type="numpy",
    )

    assert np.all(np.isfinite(h))
    assert np.allclose(h[0], 0.0)

    # Non-zero link must have unit average energy across antennas/time/freq.
    energy_rx1 = np.mean(np.abs(h[1]) ** 2)
    assert np.isclose(energy_rx1, 1.0, rtol=1e-3)
