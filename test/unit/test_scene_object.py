#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import numpy as np
import drjit as dr
import mitsuba as mi
import sionna
from sionna.rt import load_scene, SceneObject, Transmitter


def test_scene_object_clone():
    scene = load_scene(sionna.rt.scene.box_one_screen,merge_shapes=False)

    screen = scene.objects["screen"]
    screen_clone = screen.clone(name="my-screen-clone")

    assert isinstance(screen_clone, SceneObject)

    # Check the name is correctly set
    assert screen_clone.name == "my-screen-clone"

    # The object should not be in the scene
    assert screen_clone.name not in scene.objects

    # Same radio material
    assert screen_clone.radio_material is screen.radio_material

    # Identical but not shared geometry
    assert dr.all(screen_clone.mi_mesh.faces_buffer() == screen.mi_mesh.faces_buffer())
    assert dr.all(screen_clone.mi_mesh.vertex_positions_buffer() == screen.mi_mesh.vertex_positions_buffer())
    assert screen_clone.mi_mesh.faces_buffer() is not screen.mi_mesh.faces_buffer()
    assert screen_clone.mi_mesh.vertex_positions_buffer() is not screen.mi_mesh.vertex_positions_buffer()

    ##################
    # Test is_mesh
    ##################

    screen = scene.objects["screen"]
    screen_clone = screen.clone(name="my-screen-clone", as_mesh=True)

    assert isinstance(screen_clone, mi.Mesh)

    # Check the name is correctly set
    assert screen_clone.id() == "my-screen-clone"

    # Same radio material
    assert screen_clone.bsdf() is screen.radio_material

    # Identical but not shared geometry
    assert dr.all(screen_clone.faces_buffer() == screen.mi_mesh.faces_buffer())
    assert dr.all(screen_clone.vertex_positions_buffer() == screen.mi_mesh.vertex_positions_buffer())
    assert screen_clone.faces_buffer() is not screen.mi_mesh.faces_buffer()
    assert screen_clone.vertex_positions_buffer() is not screen.mi_mesh.vertex_positions_buffer()


def test_scene_object_look_at_string_and_instance_targets():
    scene = load_scene(sionna.rt.scene.box, merge_shapes=False)
    obj = next(iter(scene.objects.values()))
    # Place the TX far along +x from the object center so the look-at angles
    # are unambiguous.
    obj_pos = np.array(obj.position.numpy()[:, 0])
    tx_pos = obj_pos + np.array([10.0, 0.0, 0.0])
    tx = Transmitter("tx-look", position=tx_pos.tolist())
    scene.add(tx)

    obj.look_at("tx-look")
    orient_named = obj.orientation.numpy()[:, 0].copy()
    obj.look_at(tx)
    orient_inst = obj.orientation.numpy()[:, 0]
    assert np.allclose(orient_named, orient_inst, atol=1e-5)
    # Looking along +x → (α, β, γ) = (0, 0, 0)
    assert np.allclose(orient_inst, [0.0, 0.0, 0.0], atol=1e-4)

    other = obj.clone(name="other-box")
    scene.edit(add=other)
    other.position = mi.Point3f(float(obj_pos[0]),
                                float(obj_pos[1] + 5.0),
                                float(obj_pos[2]))
    obj.look_at("other-box")
    # Looking along +y → α = π/2
    assert np.allclose(obj.orientation.numpy()[:, 0],
                       [np.pi / 2, 0.0, 0.0], atol=1e-4)


def test_scene_object_look_at_string_error_paths():
    scene = load_scene(sionna.rt.scene.box, merge_shapes=False)
    obj = next(iter(scene.objects.values()))

    with pytest.raises(ValueError, match="Unknown target"):
        obj.look_at("does-not-exist")

    mat_name = next(iter(scene.radio_materials))
    with pytest.raises(ValueError, match="Cannot look at"):
        obj.look_at(mat_name)

    detached = obj.clone(name="detached")
    with pytest.raises(ValueError, match="Scene is not set"):
        detached.look_at("tx")


def test_scene_object_velocity_width_check():
    """Velocity validation uses raise (not assert), so it survives python -O."""
    scene = load_scene(sionna.rt.scene.box, merge_shapes=False)
    obj = next(iter(scene.objects.values()))

    obj.velocity = mi.Vector3f(1.0, 2.0, 3.0)
    assert np.allclose(obj.velocity.numpy()[:, 0], [1.0, 2.0, 3.0])

    wide = mi.Vector3f([1.0, 4.0], [2.0, 5.0], [3.0, 6.0])
    assert dr.width(wide) == 2
    with pytest.raises(ValueError, match="Only a single velocity vector"):
        obj.velocity = wide
