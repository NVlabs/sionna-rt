#
# SPDX-FileCopyrightText: Copyright (c) 2026 The Sionna contributors.
# SPDX-License-Identifier: Apache-2.0
#
"""Collision resolution and fail-fast checks for candidate deduplication."""

import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from sionna import rt
from sionna.rt.path_solvers.hash_set import insert_hash
from sionna.rt.path_solvers.sb_candidate_generator import SBCandidateGenerator


def test_full_keys_zero_duplicates_and_mask():
    table = dr.zeros(mi.UInt64, 32)
    zero_seen = dr.zeros(mi.UInt, 1)
    overflow = dr.zeros(mi.UInt, 1)
    # All nonzero keys collide at slot 1, including distinct upper 32 bits.
    keys = mi.UInt64([1, 33, 65, 1 + 2**32, 33, 0, 0, 97])
    mask = mi.Bool([True] * 7 + [False])
    new = insert_hash(table, zero_seen, overflow, keys, mi.UInt(0), 32, mask)
    dr.eval(table, zero_seen, overflow, new)
    selected = keys.numpy()[new.numpy()]
    assert sorted(selected.tolist()) == [0, 1, 33, 65, 1 + 2**32]
    assert int(overflow[0]) == 0

    # Insertion into a previously evaluated table still rejects duplicates.
    new = insert_hash(table, zero_seen, overflow, keys, mi.UInt(0), 32, mask)
    dr.eval(table, zero_seen, overflow, new)
    assert not np.any(new.numpy())


def test_sources_have_independent_tables():
    table = dr.zeros(mi.UInt64, 16)
    zero_seen = dr.zeros(mi.UInt, 2)
    overflow = dr.zeros(mi.UInt, 1)
    keys = mi.UInt64([0, 1, 9, 0, 1, 9])
    sources = mi.UInt([0, 0, 0, 1, 1, 1])
    new = insert_hash(table, zero_seen, overflow, keys, sources, 8, mi.Bool(True))
    dr.eval(table, zero_seen, overflow, new)
    assert np.all(new.numpy())
    assert int(overflow[0]) == 0


def test_collision_chain_with_concurrent_duplicates_and_wraparound():
    table = dr.zeros(mi.UInt64, 128)
    zero_seen = dr.zeros(mi.UInt, 1)
    overflow = dr.zeros(mi.UInt, 1)
    # Starting at the last slot exercises wraparound. Many lanes compete for
    # each key; exactly one representative must survive for every unique key.
    unique = np.array([127 + 128 * i for i in range(48)] + [2**64 - 1],
                      dtype=np.uint64)
    keys_numpy = np.tile(unique, 8)
    np.random.default_rng(7).shuffle(keys_numpy)
    keys = mi.UInt64(keys_numpy)
    new = insert_hash(table, zero_seen, overflow, keys, mi.UInt(0), 128,
                      mi.Bool(True))
    dr.eval(table, zero_seen, overflow, new)
    np.testing.assert_array_equal(np.sort(keys_numpy[new.numpy()]),
                                  np.sort(unique))
    assert int(overflow[0]) == 0


@pytest.mark.parametrize("table_size,max_probes", [(2, 128), (8, 2)])
def test_full_or_clustered_table_reports_exhaustion(table_size, max_probes):
    table = dr.zeros(mi.UInt64, table_size)
    zero_seen = dr.zeros(mi.UInt, 1)
    overflow = dr.zeros(mi.UInt, 1)
    keys = mi.UInt64([1, 1 + table_size, 1 + 2 * table_size])
    new = insert_hash(table, zero_seen, overflow, keys, mi.UInt(0), table_size,
                      mi.Bool(True), max_probes=max_probes)
    dr.eval(table, zero_seen, overflow, new)
    assert int(overflow[0]) == 1
    assert np.count_nonzero(new.numpy()) == 2


def reflector_scene(count=64):
    scene = rt.load_scene(rt.scene.simple_reflector)
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array
    scene.add(rt.Transmitter("tx", [-.1, 0., 1.]))
    for index in range(count):
        scene.add(rt.Receiver(f"rx{index}", [.1, -.1 + .2 * index / count, 1.]))
    return scene


def solve_reflector(scene, **kwargs):
    return rt.PathSolver(**kwargs)(
        scene, max_depth=1, samples_per_src=1000,
        max_num_paths_per_src=100, synthetic_array=False,
        los=False, specular_reflection=True, refraction=False,
        diffuse_reflection=False, diffraction=False, seed=42
    )


def test_reflector_order_independence_under_modulo_collisions(monkeypatch):
    # Force hash pressure without a large dataset. The output cap (100) is
    # above the exact 64-path answer; any loss occurs during deduplication.
    monkeypatch.setattr(SBCandidateGenerator, "MIN_SPEC_COUNT_SIZE", 1)
    scene = reflector_scene()
    baseline = solve_reflector(scene)
    baseline_a, _ = baseline.cir(out_type="numpy")
    assert np.count_nonzero(baseline_a) < 64

    corrected = solve_reflector(scene, resolve_hash_collisions=True,
                                 hash_table_size=100)
    a, _ = corrected.cir(out_type="numpy")
    assert np.count_nonzero(a) == 64
    assert np.all(np.count_nonzero(a, axis=4) == 1)

    receivers = list(scene.receivers.values())
    for receiver in receivers:
        scene.remove(receiver.name)
    for receiver in reversed(receivers):
        scene.add(receiver)
    reversed_paths = solve_reflector(scene, resolve_hash_collisions=True,
                                     hash_table_size=100)
    b, _ = reversed_paths.cir(out_type="numpy")
    np.testing.assert_allclose(a, b[::-1], rtol=1e-5, atol=1e-10)


def test_hash_exhaustion_raises_and_solver_can_be_reused():
    solver = rt.PathSolver(resolve_hash_collisions=True, hash_table_size=1)
    scene = reflector_scene(2)
    kwargs = dict(max_depth=1, samples_per_src=1000, max_num_paths_per_src=100,
                  los=False, refraction=False)
    with pytest.raises(OverflowError, match="hash-table probe limit"):
        solver(scene, **kwargs)
    scene.remove("rx1")
    paths = solver(scene, **kwargs)
    a, _ = paths.cir(out_type="numpy")
    assert np.count_nonzero(a) == 1


def test_output_exhaustion_raises():
    scene = reflector_scene(2)
    with pytest.raises(OverflowError, match="path buffer exhausted"):
        rt.PathSolver(resolve_hash_collisions=True, hash_table_size=100)(
            scene, max_depth=1, samples_per_src=1000,
            max_num_paths_per_src=1, los=False, refraction=False
        )


def test_los_output_exhaustion_raises_before_out_of_bounds_write():
    scene = reflector_scene(2)
    with dr.scoped_set_flag(dr.JitFlag.Debug, True):
        with pytest.raises(OverflowError, match="path buffer exhausted"):
            rt.PathSolver(resolve_hash_collisions=True)(
                scene, max_depth=0, max_num_paths_per_src=1
            )


def test_exact_output_capacity_is_allowed():
    scene = reflector_scene(2)
    paths = rt.PathSolver(resolve_hash_collisions=True)(
        scene, max_depth=0, max_num_paths_per_src=2
    )
    a, _ = paths.cir(out_type="numpy")
    assert np.count_nonzero(a) == 2


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_hash_size(size):
    with pytest.raises(ValueError, match="positive integer"):
        rt.PathSolver(resolve_hash_collisions=True, hash_table_size=size)


def test_incompatible_modes():
    with pytest.raises(ValueError, match="deterministic=True"):
        rt.PathSolver(deterministic=True, resolve_hash_collisions=True)
    with pytest.raises(ValueError, match="requires resolve_hash_collisions"):
        rt.PathSolver(hash_table_size=100)
