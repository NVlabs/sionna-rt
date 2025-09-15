#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os

import pytest
import drjit as dr
import mitsuba as mi
import sionna
from sionna.rt import load_scene, SceneObject, ITURadioMaterial


TEST_DUPLICATE_MATERIAL = ITURadioMaterial("test-duplicate-material", "concrete", 1.0)


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


def test_scene_close_and_reuse_scene_object():
    s1 = load_scene(sionna.rt.scene.box_one_screen, merge_shapes=False)
    s2 = sionna.rt.Scene()

    box = s1.objects["box"]
    box_mat = box.radio_material

    with pytest.raises(ValueError):
        s2.add(box_mat)

    with pytest.raises(ValueError):
        s2.edit(add=box)

    # Should not add box's mi_mesh into s2 since it belongs to s1
    assert len(s2.mi_scene.shapes()) == 0

    # Close s1
    s1.close()

    # Check radio material is removed from s1
    assert box_mat._count_using_objects == 1
    assert box_mat.scene is None

    # Check box is removed from s1
    assert box.scene is None
    assert box.radio_material == box_mat

    # Should allow to add the radio material to another scene
    s2.edit(add=box)
    assert len(s2.mi_scene.shapes()) == 1
    assert box.scene == s2
    assert box_mat.scene == s2


def test_scene_auto_remove_1():
    s1 = sionna.rt.Scene()

    s1.add(TEST_DUPLICATE_MATERIAL)


def test_scene_auto_remove_2():
    s1 = sionna.rt.Scene()

    # Should failed, we don't have __del__ to finalize the scene
    with pytest.raises(ValueError):
        s1.add(TEST_DUPLICATE_MATERIAL)

    TEST_DUPLICATE_MATERIAL._scene = None
    s1.add(TEST_DUPLICATE_MATERIAL)

    assert TEST_DUPLICATE_MATERIAL.scene == s1
    s1.close()
    assert TEST_DUPLICATE_MATERIAL.scene is None


def test_scene_edit_remove():
    s1 = load_scene(sionna.rt.scene.box_one_screen, merge_shapes=False)
    s2 = sionna.rt.Scene()

    box = s1.objects["box"]
    with pytest.raises(ValueError):
        s2.edit(add=box)

    s1.edit(remove="box")
    assert "box" not in s1.objects
    assert box.scene is None
    assert box.radio_material is not None
    assert box.radio_material._count_using_objects == 1
    assert box.radio_material.scene is None
    assert len(s1.mi_scene.shapes()) == 1  # only screen remains

    s2.edit(add=box)
    assert box.scene == s2
    assert box.radio_material is not None
    assert box.radio_material._count_using_objects == 1
    assert box.radio_material.scene == s2
    assert len(s2.mi_scene.shapes()) == 1


def test_scene_object_and_radio_material():
    fname = os.path.join(os.path.dirname(__file__), "../data/subdivided_cube.ply")

    s1 = sionna.rt.Scene()
    mat = ITURadioMaterial("test-material", "concrete", 1.0)
    assert mat._count_using_objects == 0

    so = SceneObject(fname=fname, name="test-object", radio_material=mat)
    assert so.radio_material._count_using_objects == 1

    s1.edit(add=so)
    assert so.radio_material._count_using_objects == 1

    s1.edit(remove="test-object")
    assert so.radio_material._count_using_objects == 1
    assert so.scene is None

    s2 = sionna.rt.Scene()
    s2.edit(add=so)
    assert so.radio_material._count_using_objects == 1


def test_scene_object_and_radio_material_after_scene_close():
    fname = os.path.join(os.path.dirname(__file__), "../data/subdivided_cube.ply")

    s1 = sionna.rt.Scene()
    mat = ITURadioMaterial("test-material", "concrete", 1.0)
    assert mat._count_using_objects == 0

    so = SceneObject(fname=fname, name="test-object", radio_material=mat)
    assert so.radio_material._count_using_objects == 1

    s1.edit(add=so)
    assert so.radio_material._count_using_objects == 1

    # Close s1
    s1.close()
    assert so.radio_material._count_using_objects == 1

    s2 = sionna.rt.Scene()
    s2.edit(add=so)
    s2.close()
    assert so.radio_material._count_using_objects == 1
