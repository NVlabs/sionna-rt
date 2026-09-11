#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the interactive scene preview widget."""

import drjit as dr
import pytest

from sionna import rt
from sionna.rt import PathSolver
from sionna.rt.radio_materials.itu import itu_material
from sionna.rt.scene import Scene, load_scene


def add_example_radio_devices(scene: Scene):
    # Note: hardcoded for `box_two_screens.xml` as an example.
    scene.add(rt.Transmitter("tr-1", position=[-3.0, 0.0, 1.5]))
    scene.add(rt.Receiver("rc-1", position=[3.0, 0.0, 1.5]))
    scene.add(
        rt.Receiver(
            "rc-2", position=[1.0, -2.0, 3.5], color=(0.9, 0.9, 0.2),
            display_radius=0.9
        )
    )

    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="tr38901", polarization="VH"
    )
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="tr38901", polarization="VH"
    )


def get_example_paths(scene: Scene):
    # Ray tracing parameters
    num_samples_per_src = int(1e6)
    max_num_paths = int(1e7)
    max_depth = 3

    solver = PathSolver()
    paths = solver(
        scene,
        max_depth=max_depth,
        max_num_paths_per_src=max_num_paths,
        samples_per_src=num_samples_per_src,
    )

    return paths


@pytest.mark.parametrize("has_paths", (True, False))
def test01_preview_basic(has_paths):
    scene = load_scene(rt.scene.box_two_screens)

    eta_r, sigma = itu_material("metal", 3e9)  # ITU material evaluated at 3GHz
    for sh in scene.mi_scene.shapes():
        material = sh.bsdf()
        material.relative_permittivity = eta_r
        material.conductivity = sigma
        material.scattering_coefficient = 0.01
        material.xpd_coefficient = 0.2

    add_example_radio_devices(scene)
    paths = get_example_paths(scene)

    if not has_paths:
        paths._valid &= False
        assert dr.count(paths.valid) == 0

    # Should work with or without valid paths.
    # Note: we don't verify that the preview widget is actually functional,
    # simply that no exception is thrown.
    scene.preview(paths=paths)


def test_preview_empty_paths_object():
    """Preview accepts a Paths object that contains zero paths."""
    scene = load_scene(rt.scene.box_two_screens)
    add_example_radio_devices(scene)
    paths = PathSolver()(
        scene,
        los=False,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
    )
    assert paths.vertices.shape[-2] == 0
    scene.preview(paths=paths)
