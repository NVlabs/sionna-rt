#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import drjit as dr
import mitsuba as mi
import sionna
from sionna.rt import load_scene, SceneObject, ITURadioMaterial


def test_scene_close_and_reuse_radio_material():
    s1 = sionna.rt.Scene()
    s2 = sionna.rt.Scene()

    # Add m1 to s1
    m1 = ITURadioMaterial("m1", "concrete", 1.0)
    s1.add(m1)

    with pytest.raises(ValueError):
        # Should not be able to add the same material twice to the same scene
        s2.add(m1)

    # Should have the ability to "close" the scene
    s1.close()

    # Should allow to add the radio material to another scene
    s2.add(m1)


def test_scene_remove_radio_material():
    s1 = sionna.rt.Scene()
    s2 = sionna.rt.Scene()

    # Add m1 to s1
    m1 = ITURadioMaterial("m1", "concrete", 1.0)
    s1.add(m1)

    with pytest.raises(ValueError):
        # Should not be able to add the same material twice to the same scene
        s2.add(m1)

    # Remove it from s1
    s1.remove("m1")

    # Should allow to add the radio material to another scene
    s2.add(m1)


def test_scene_edit_should_not_add_scene_object_if_has_scene():
    s1 = load_scene(sionna.rt.scene.box_one_screen, merge_shapes=False)
    s2 = sionna.rt.Scene()

    assert len(s2.mi_scene.shapes()) == 0

    box = s1.objects["box"]
    with pytest.raises(ValueError):
        s2.edit(add=box)

    assert len(s2.mi_scene.shapes()) == 0
