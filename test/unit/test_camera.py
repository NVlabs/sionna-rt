#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import os

import drjit as dr
import mitsuba as mi
import numpy as np

from sionna import rt
from sionna.rt import Camera


def test01_camera_create():
    def check(camera, expected):
        # print('---')
        # print(camera.world_transform.matrix.numpy())
        # print(np.array(expected))
        assert np.allclose(camera.world_transform.matrix.numpy()[:,:,0],
                           expected, atol=1e-5)

    c = Camera(position=[0, 3, 0.5])
    check(c, [
        [0, 0, 1, 0],
        [1, 0, 0, 3],
        [0, 1, 0, 0.5],
        [0, 0, 0, 1],
    ])

    c = Camera(position=[0, 3, 0.5], orientation=(np.pi, 0, 0))
    check(c, [
        [ 0, 0, -1, 0],
        [-1, 0,  0, 3],
        [ 0, 1,  0, 0.5],
        [ 0, 0,  0, 1],
    ])

    # Looking straight up: the roll is not determined by the look-at direction,
    # and the azimuth is set to zero, which makes -x the up direction.
    c = Camera(position=[0, 0, 3], look_at=[0, 0, 10])
    check(c, [
        [0, -1, 0, 0],
        [1,  0, 0, 0],
        [0,  0, 1, 3],
        [0,  0, 0, 1],
    ])

    c.look_at([0, 10, 0])
    expected = mi.ScalarTransform4f().look_at(origin=[0, 0, 3],
                                        target=[0, 10, 0],
                                        up=[0, 0, 1]).matrix.numpy()
    check(c, expected)

    c.position = np.array([1, 2, 3])
    expected[:3, 3] = [1, 2, 3]
    check(c, expected)


def test_camera_look_at_radio_device_and_scene_object():
    scene = rt.load_scene(rt.scene.box, merge_shapes=False)
    tx = rt.Transmitter("tx", position=[4.0, 2.0, 1.0])
    scene.add(tx)
    obj = next(iter(scene.objects.values()))

    c = Camera(position=[0.0, 0.0, 1.0])
    c.look_at(tx)
    expected_tx = mi.ScalarTransform4f().look_at(
        origin=[0.0, 0.0, 1.0],
        target=[4.0, 2.0, 1.0],
        up=[0.0, 0.0, 1.0],
    ).matrix.numpy()
    assert np.allclose(c.world_transform.matrix.numpy()[:, :, 0],
                       expected_tx, atol=1e-5)

    # Looking at a scene object must produce a finite transform, including when
    # the target is vertically aligned with the camera.
    c.look_at(obj)
    mat = c.world_transform.matrix.numpy()[:, :, 0]
    assert np.all(np.isfinite(mat))
    assert np.allclose(mat[:3, 3], [0.0, 0.0, 1.0], atol=1e-5)


def test_camera_rejects_coincident_look_at():
    """A coincident target is rejected, as for radio devices and scene objects."""
    with pytest.raises(ValueError, match="coincides with the camera"):
        Camera(position=[0.0, 0.0, 0.0], look_at=[0.0, 0.0, 0.0])

    c = Camera(position=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="coincides with the camera"):
        c.look_at([1.0, 2.0, 3.0])
