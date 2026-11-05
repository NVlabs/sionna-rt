#
# SPDX-FileCopyrightText: Copyright (c) 2026 The Sionna contributors.
# SPDX-License-Identifier: Apache-2.0
#
"""LoS candidates must share the per-source budget with reflected candidates."""

import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from sionna import rt
from sionna.rt.path_solvers.sb_candidate_generator import SBCandidateGenerator


def trace_candidates(scene, sources, targets, capacity, max_depth):
    """Materialize candidate writes while bounds checking is enabled."""
    with dr.scoped_set_flag(dr.JitFlag.Debug, True):
        paths = SBCandidateGenerator()(
            mi_scene=scene.mi_scene, src_positions=sources,
            tgt_positions=targets,
            samples_per_src=1000, max_num_paths_per_src=capacity,
            max_depth=max_depth, los=True, specular_reflection=True,
            diffuse_reflection=False, refraction=False, diffraction=False,
            edge_diffraction=False, seed=42,
        )
        paths.schedule()
        dr.eval()
    return paths


@pytest.mark.parametrize("num_sources", [1, 2])
@pytest.mark.parametrize("capacity", [1, 3])
def test_los_only_respects_capacity(num_sources, capacity, capfd):
    """Exercise both an overflowing buffer and an exactly full buffer."""
    scene = rt.load_scene()
    sources = mi.Point3f(dr.arange(mi.Float, num_sources),
                         dr.zeros(mi.Float, num_sources),
                         dr.full(mi.Float, 1, num_sources))
    targets = mi.Point3f([0, 1, 2], [1, 1, 1], [1, 1, 1])
    paths = trace_candidates(scene, sources, targets, capacity, max_depth=0)
    stored = int(paths.paths_counter[0])
    assert stored == num_sources * capacity
    counts = np.bincount(paths.source_indices.numpy()[:stored],
                         minlength=num_sources)
    np.testing.assert_array_equal(counts, [capacity] * num_sources)
    captured = capfd.readouterr()
    assert "out-of-bounds" not in captured.out + captured.err


def screened_reflector_scene():
    """One source sees both targets; the other reaches them via the ground."""
    return rt.load_scene_from_string('''
        <scene version="3.0.0">
          <bsdf type="itu-radio-material" id="mat-metal">
            <string name="type" value="metal"/>
            <float name="thickness" value="0.01"/>
          </bsdf>
          <shape type="rectangle" id="ground">
            <transform name="to_world"><scale x="10" y="10" z="1"/></transform>
            <ref id="mat-metal" name="bsdf"/>
          </shape>
          <shape type="rectangle" id="screen">
            <transform name="to_world">
              <scale x="0.75" y="10" z="1"/>
              <rotate y="1" angle="90"/>
              <translate z="1.25"/>
            </transform>
            <ref id="mat-metal" name="bsdf"/>
          </shape>
        </scene>
    ''', merge_shapes=False)


@pytest.mark.parametrize("reverse_sources", [False, True])
@pytest.mark.parametrize("capacity", [1, 2])
@pytest.mark.parametrize("max_depth", [0, 1])
def test_los_leaves_capacity_for_other_sources(
    reverse_sources, capacity, max_depth, capfd
):
    """Excess LoS paths cannot consume the other source's reflection slots."""
    scene = screened_reflector_scene()
    positions = np.array([[1, -.2, 1], [-1, 0, 1]], dtype=np.float32)
    expected = [capacity, capacity if max_depth else 0]
    if reverse_sources:
        positions = positions[::-1].copy()
        expected.reverse()
    sources = mi.Point3f(positions.T)
    targets = mi.Point3f([1, 1], [0, .1], [1, 1])
    paths = trace_candidates(scene, sources, targets, capacity, max_depth)
    stored = int(paths.paths_counter[0])
    assert stored == sum(expected)
    counts = np.bincount(paths.source_indices.numpy()[:stored], minlength=2)
    np.testing.assert_array_equal(counts, expected)
    captured = capfd.readouterr()
    assert "out-of-bounds" not in captured.out + captured.err
