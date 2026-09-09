#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import gc
import os
from os.path import join
import tempfile

import pytest

import mitsuba as mi
import drjit as dr
from sionna import rt
from sionna.rt import load_scene, load_scene_from_string, SceneObject, \
                      RadioMaterial, RadioMaterialBase, ITURadioMaterial
from sionna.rt.scene_utils import extend_scene_with_mesh


def register_custom_radio_material():
    class MyTestRadioMaterial(RadioMaterial):
        def __init__(self, props: mi.Properties | None = None):
            self.some_param = props.get("some_param", 0.0)
            del props["some_param"]

            super().__init__(props=props)

    plugin_name = "my-test-radio-material"
    mi.register_bsdf(plugin_name, lambda props: MyTestRadioMaterial(props=props))
    return plugin_name, MyTestRadioMaterial


def test01_scene_preprocessing():
    scene_processed = load_scene(rt.scene.box_two_screens)
    scene_processed = scene_processed.mi_scene

    # Check that all BSDFs in the scene were correctly replaced by our custom radio BSDF.
    for sh in scene_processed.shapes():
        assert isinstance(sh.bsdf(), RadioMaterialBase)
        assert isinstance(sh.bsdf(), RadioMaterial)


def test02_merge_exclude_regex():
    tmp_path = join(tempfile.gettempdir(), "test_scene_02.xml")

    with open(tmp_path, "w") as f:
        f.write("""<scene version="2.1.0">
    <bsdf type="diffuse" id="itu_wood"/>

    <shape type="cube" id="floor">
        <ref name="bsdf" id="itu_wood"/>
    </shape>
    <shape type="cube" id="ceiling">
        <ref name="bsdf" id="itu_wood"/>
    </shape>

    <shape type="cube" id="car-1">
        <ref name="bsdf" id="itu_wood"/>
    </shape>
    <shape type="cube" id="car-2">
        <ref name="bsdf" id="itu_wood"/>
    </shape>
    <shape type="cube" id="car-3">
        <ref name="bsdf" id="itu_wood"/>
    </shape>
</scene>""")


    scene_processed = load_scene(tmp_path,
        merge_shapes_exclude_regex=r"^car-.+")
    scene_processed = scene_processed.mi_scene

    # 1 merged shape + 3 car shapes
    assert len(scene_processed.shapes()) == 4
    for shape in scene_processed.shapes():
        this_id = shape.id()
        assert (this_id == "merged-shapes") or this_id.startswith("car-")

    os.remove(tmp_path)


def test03_scene_add_remove():
    tmp_path = join(tempfile.gettempdir(), "test_scene_03.xml")

    with open(tmp_path, "w") as f:
            f.write("""
        <scene version="2.1.0">

            <emitter type="constant"/>

            <integrator type="path"/>

            <bsdf type="itu-radio-material" id="bsdf1">
                <string name="type" value="metal"/>
            </bsdf>

            <shape type="cube" id="shape1">
                <ref name="bsdf" id="bsdf1"/>
            </shape>

            <shape type="cube" id="shape2">
                <ref name="bsdf" id="bsdf1"/>
            </shape>

        </scene>""")

    scene = load_scene(tmp_path, merge_shapes=False)
    shape_rm = scene.objects["shape1"].radio_material
    original_mi_scene = scene.mi_scene
    scene.edit()
    edited1_mi_scene = scene.mi_scene

    # 1. No change: everything should be preserved
    assert not edited1_mi_scene.sensors()
    assert edited1_mi_scene.environment() == original_mi_scene.environment()
    assert edited1_mi_scene.integrator() == original_mi_scene.integrator()
    assert len(edited1_mi_scene.emitters()) == 1  # Just the envmap
    assert len(edited1_mi_scene.shapes()) == 2
    assert set(s.id() for s in edited1_mi_scene.shapes()) == {"shape1", "shape2"}
    assert set(s.bsdf().id() for s in edited1_mi_scene.shapes()) == {"bsdf1"}
    for s1, s2 in zip(original_mi_scene.shapes(), edited1_mi_scene.shapes()):
        assert s1 == s2


    # 2. Add some shapes and remove some other
    car_rm = ITURadioMaterial("car-mat", "metal", 0.01)
    car_mi = mi.load_dict({
        'type': 'ply',
        'filename': rt.scene.low_poly_car,
        'flip_normals': True,
    })
    assert car_mi.id() == ""  # Default ID
    cars = [
        SceneObject(fname=rt.scene.low_poly_car,
                    name="car1",
                    radio_material=car_rm),
        SceneObject(mi_mesh=car_mi,
                    name="car2",
                    radio_material=scene.radio_materials["bsdf1"])
    ]

    # Scene is edited in-place
    scene.edit(add=cars, remove=["shape1"])
    edited2_mi_scene = scene.mi_scene
    assert not edited2_mi_scene.sensors()
    assert edited2_mi_scene.environment() == original_mi_scene.environment()
    assert edited2_mi_scene.integrator() == original_mi_scene.integrator()
    assert len(edited2_mi_scene.emitters()) == 1  # Just the envmap
    assert len(scene.objects) == (2 - 1) + 2
    assert set(o.name for o in scene.objects.values()) == {"shape2", "car1", "car2"}
    for i, car in enumerate(cars):
        assert car.name == f"car{i+1}"
        assert car.mi_mesh.id() == f"car{i+1}"
        assert car.mi_mesh.bsdf().id() == ("car-mat" if i == 0 else "bsdf1")
    assert scene.get("car1") is cars[0]
    assert scene.get("car2") is cars[1]

    # Check that the radio material of the car is the correct one
    for obj in scene.objects.values():
        # All shapes use the main BSDF except "car1"
        if obj.name == "car1":
            assert obj.radio_material is car_rm
        else:
            assert obj.radio_material is shape_rm

    # 3. Remove some shape that we added earlier
    scene.edit(remove=cars[0])
    assert len(scene.objects) == 2
    assert set(o.name for o in scene.objects.values()) == {"shape2", "car2"}

    # 4. Add a shape with an existing ID
    new_car = SceneObject(fname=rt.scene.low_poly_car,
                          name="car2",
                          radio_material=car_rm)
    with pytest.raises(ValueError, match=r"this ID is already used in the scene"):
        scene.edit(add=new_car)

    os.remove(tmp_path)


def test04_scene_radio_materials():
    tmp_path = join(tempfile.gettempdir(), "test_scene_04.xml")

    # We need to support several ways to specify radio materials:
    # - `diffuse` BSDF with a special name (typically from a Blender export)
    # - `itu-radio-material` or other built-in radio material
    # - A user-defined custom radio material registered before loading the scene.
    custom_rm_type, MyCustomRadioMaterial = register_custom_radio_material()

    scene_content = \
    f"""
    <scene version="2.1.0">

        <!-- Materials -->
        <bsdf type="diffuse" id="itu_custom">
            <float name="thickness" value="0.25"/>
            <string name="type" value="metal"/>
            <rgb name="color" value="0.1, 0.2, 0.3"/>
        </bsdf>

        <bsdf type="diffuse" id="itu_metal">
            <rgb name="reflectance" value="0.3, 0.3, 0.4"/>
        </bsdf>

        <bsdf type="itu-radio-material" id="itu-human">
            <float name="thickness" value="5.65"/>
            <string name="type" value="plasterboard"/>
            <rgb name="reflectance" value="0.5, 0.6, 0.7"/>
        </bsdf>

        <bsdf type="{custom_rm_type}" id="a_custom_material">
            <float name="some_param" value="3.14"/>
            <rgb name="color" value="0.6, 0.7, 0.8"/>
        </bsdf>

        <bsdf type="radio-material" id="a_built_in_material">
            <float name="conductivity" value="0.789"/>
            <rgb name="reflectance" value="0.7, 0.8, 0.9"/>
        </bsdf>


        <!-- Shapes -->
        <shape type="cube" id="obj-1">
            <ref name="bsdf" id="itu_custom"/>
        </shape>

        <shape type="cube" id="obj-2">
            <ref name="bsdf" id="itu_metal"/>
        </shape>

        <shape type="cube" id="obj-3">
            <bsdf type="diffuse" id="itu_concrete">
                <float name="thickness" value="0.30"/>
            </bsdf>
        </shape>

        <shape type="cube" id="obj-4">
            <ref name="arbitrary" id="itu-human"/>
        </shape>

        <shape type="cube" id="obj-5">
            <!-- Reference to a material that was nested in a BSDF -->
            <ref name="arbitrary" id="itu_concrete"/>
        </shape>

        <shape type="cube" id="obj-6">
            <!-- Reference a user-defined custom radio material -->
            <ref name="arbitrary" id="a_custom_material"/>
        </shape>

        <shape type="cube" id="obj-7">
            <!-- Directly use a user-defined custom radio material -->
            <bsdf type="{custom_rm_type}" id="nested_custom_material">
                <float name="some_param" value="-1.23"/>
            </bsdf>
        </shape>

        <shape type="cube" id="obj-8">
            <!-- Reference a built-in radio material -->
            <ref name="arbitrary" id="a_built_in_material"/>
        </shape>

        <shape type="cube" id="obj-9">
            <!-- Directly use a built-in radio material -->
            <bsdf type="radio-material" id="nested_built_in_material">
                <float name="conductivity" value="0.567"/>
                <rgb name="base_color" value="0.8, 0.9, 1.0"/>
            </bsdf>
        </shape>

    </scene>
    """
    with open(tmp_path, "w") as f:
        f.write(scene_content)

    # Load the scene
    scene = load_scene(tmp_path)

    mats = scene.radio_materials
    assert len(mats) == 8
    assert mats.keys() == {
        "itu_custom",
        "itu_metal",
        "itu_concrete",
        "itu-human",
        "a_custom_material",
        "nested_custom_material",
        "a_built_in_material",
        "nested_built_in_material",
    }

    assert mats["itu_custom"].thickness == 0.25
    assert mats["itu_custom"].itu_type == "metal"
    assert dr.allclose(mats["itu_custom"].color, (0.1, 0.2, 0.3))
    assert isinstance(mats["itu_custom"], ITURadioMaterial)

    assert mats["itu_metal"].thickness == rt.constants.DEFAULT_THICKNESS
    assert mats["itu_metal"].itu_type == "metal"
    assert dr.allclose(mats["itu_metal"].color, (0.3, 0.3, 0.4))
    assert isinstance(mats["itu_metal"], ITURadioMaterial)

    assert mats["itu-human"].thickness == 5.65
    assert mats["itu-human"].itu_type == "plasterboard"
    assert dr.allclose(mats["itu-human"].color, (0.5, 0.6, 0.7))
    assert isinstance(mats["itu-human"], ITURadioMaterial)

    assert mats["itu_concrete"].thickness == 0.30
    assert mats["itu_concrete"].itu_type == "concrete"
    assert dr.allclose(mats["itu_concrete"].color, ITURadioMaterial.ITU_MATERIAL_COLORS["concrete"])
    assert isinstance(mats["itu_concrete"], ITURadioMaterial)

    assert mats["a_custom_material"].thickness == rt.constants.DEFAULT_THICKNESS
    assert mats["a_custom_material"].some_param == 3.14
    assert dr.allclose(mats["a_custom_material"].color, (0.6, 0.7, 0.8))
    assert isinstance(mats["a_custom_material"], MyCustomRadioMaterial)

    assert mats["nested_custom_material"].thickness == rt.constants.DEFAULT_THICKNESS
    assert mats["nested_custom_material"].some_param == -1.23
    assert isinstance(mats["nested_custom_material"], MyCustomRadioMaterial)

    assert mats["a_built_in_material"].thickness == rt.constants.DEFAULT_THICKNESS
    assert mats["a_built_in_material"].conductivity == 0.789
    assert dr.allclose(mats["a_built_in_material"].color, (0.7, 0.8, 0.9))
    assert isinstance(mats["a_built_in_material"], RadioMaterial)

    assert mats["nested_built_in_material"].thickness == rt.constants.DEFAULT_THICKNESS
    assert mats["nested_built_in_material"].conductivity == 0.567
    assert dr.allclose(mats["nested_built_in_material"].color, (0.8, 0.9, 1.0))
    assert isinstance(mats["nested_built_in_material"], RadioMaterial)

    os.remove(tmp_path)


def test05_scene_object_scaling():
    tmp_path = join(tempfile.gettempdir(), "test_scene_05.xml")

    with open(tmp_path, "w") as f:
        f.write("""
        <scene version="2.1.0">

            <bsdf type="itu-radio-material" id="bsdf1">
                <float name="thickness" value="5.65"/>
                <string name="type" value="plasterboard"/>
            </bsdf>

            <shape type="cube" id="shape1">
                <ref name="bsdf" id="bsdf1"/>
            </shape>

        </scene>""")

    scene = load_scene(tmp_path, merge_shapes=False)
    cube = scene.objects["shape1"]

    # Helper functions
    def assert_bbox_is(min_should_be, max_should_be):
        shape_bb = cube._mi_mesh.bbox()
        assert dr.allclose(shape_bb.min, min_should_be, atol=1e-5)
        assert dr.allclose(shape_bb.max, max_should_be, atol=1e-5)

    def reset_position():
        cube.position = mi.Point3f(0.0, 0.0, 0.0)  # Center the cube
        cube.look_at(mi.Point3f(1.0, 0.0, 0.0))  # Look in the positive x direction

    reset_position()

    # Sanity check the box bounds before scaling
    assert_bbox_is([-1, -1, -1], [1, 1, 1])
    assert dr.all(cube.scaling == mi.Vector3f(1.0))

    # Scale by a scalar value
    scalar = 10
    cube.scaling = scalar
    assert_bbox_is([-scalar] * 3, [scalar] * 3)
    assert dr.all(cube.scaling == mi.Vector3f(scalar))

    # Reassign scalar value
    scalar = 5
    cube.scaling = scalar
    assert_bbox_is([-scalar] * 3, [scalar] * 3)
    assert dr.all(cube.scaling == mi.Vector3f(scalar))

    # Negative scaling fails
    with pytest.raises(ValueError, match=r"Scaling must be positive"):
        cube.scaling = -1

    # Scale by a vector
    new_scale = mi.Vector3f(2.0, 4.0, 6.0)
    cube.scaling = new_scale
    assert_bbox_is([-new_scale.x, -new_scale.y, -new_scale.z], [new_scale.x, new_scale.y, new_scale.z])
    assert dr.all(cube.scaling == new_scale)

    # Reassign vector value
    new_scale = mi.Vector3f(1.2, 2.3, 3.4)
    cube.scaling = new_scale
    assert_bbox_is([-new_scale.x, -new_scale.y, -new_scale.z], [new_scale.x, new_scale.y, new_scale.z])
    assert dr.all(cube.scaling == new_scale)

    # Negative scaling fails
    with pytest.raises(ValueError, match=r"Scaling must be positive"):
        cube.scaling = mi.Vector3f(-1.0, 1.0, 1.0)

    # Translated and rotated cube scales correctly
    reset_position()
    cube.position = mi.Point3f(3.0, 6.0, 9.0) # Translate somewhere
    cube.look_at(mi.Point3f(-1.0, -1.0, -1.0)) # Rotate it

    new_scale = mi.Vector3f(10.0, 5.0, 15.0) # Scale
    cube.scaling = new_scale

    reset_position() # After resetting position the bbox should be as below
    assert_bbox_is([-new_scale.x, -new_scale.y, -new_scale.z], [new_scale.x, new_scale.y, new_scale.z])

    # Translation and rotation is unaffected by scaling
    reset_position()
    cube.scaling = mi.Vector3f(1.0) # Reset scaling
    cube.position = mi.Point3f(2.0, 4.0, 6.0) # Translate somewhere
    cube.look_at(mi.Point3f(1.0, 1.0, 1.0)) # Rotate it

    scene_params = cube.scene.mi_scene_params
    vp_key = cube._mi_mesh.id() + ".vertex_positions"
    vertices_before_scaling = dr.unravel(mi.Point3f, scene_params[vp_key])

    cube.scaling = mi.Vector3f(1.23, 1.5, 7.0) # Scale it by some amount
    cube.scaling = mi.Vector3f(1.0) # Reset the scale

    # If translation and rotation unaffected then all vertices should be the same
    scene_params = cube.scene.mi_scene_params
    vp_key = cube._mi_mesh.id() + ".vertex_positions"
    vertices_after_scaling = dr.unravel(mi.Point3f, scene_params[vp_key])
    assert dr.allclose(vertices_before_scaling, vertices_after_scaling, atol=1e-5)


def test06_scene_deletion():
    # Scene objects should be correctly garbage collected.
    # In particular, there shouldn't be any reference cycles keeping it live.
    gc.collect()

    scene = load_scene(rt.scene.box_two_screens)
    mat = scene.radio_materials["box-mat"]
    assert mat.scene is scene
    gc.collect()
    whos_len_during = len(dr.whos(as_string=True).split('\n'))

    del scene
    gc.collect()

    assert mat.scene is None
    whos_len_after = len(dr.whos(as_string=True).split('\n'))
    assert whos_len_after < whos_len_during, \
           "Expected to find fewer live DrJit arrays after deleting the scene,"\
           f" but found {whos_len_after} > {whos_len_during}"


def test07_scene_loading_error_messages():

    with pytest.raises(ValueError,
                       match="Found material with name \"mat-concrete\"."
                             " ITU material names must start with"):
        load_scene_from_string("""
            <scene version="2.1.0">
                <bsdf type="diffuse" id="mat-concrete"/>

                <shape type="cube" id="shape1">
                    <ref name="bsdf" id="mat-concrete"/>
                </shape>
            </scene>
        """)

    # Note: even though a `ValueError` is raised in the Python plugin, it gets
    # wrapped / replaced by a `RuntimeError` in the C++-based XML loader.
    with pytest.raises(RuntimeError,
                       match="Missing property \"type\""):
        load_scene_from_string("""
            <scene version="2.1.0">
                <bsdf type="itu-radio-material" id="wet_ground"/>
            </scene>
        """)

    # BSDF with no name or ID.
    with pytest.raises(ValueError,
                       match="found BSDF element without 'name'"):
        load_scene_from_string("""
            <scene version="2.1.0">
                <bsdf type="itu-radio-material"/>
            </scene>
        """)

    # Trying to load a scene with plain visual BSDFs should fail.
    with pytest.raises(ValueError,
                       match=r"Found shape \"shape1\" with associated material \"mat-2f4f4f\", which is not a radio material.*"):
        load_scene_from_string("""
            <scene version="2.1.0">
                <bsdf type="diffuse" id="mat-2f4f4f"/>

                <shape type="cube" id="shape1">
                    <ref name="bsdf" id="mat-2f4f4f"/>
                </shape>

                <shape type="cube" id="shape2">
                    <bsdf type="diffuse" id="mat-itu_concrete"/>
                </shape>
            </scene>
        """)

    # The following should be okay.
    load_scene_from_string("""
        <scene version="2.1.0">
            <bsdf type="diffuse" id="mat-itu_concrete"/>
            <bsdf type="itu-radio-material" id="wet_ground">
                <string name="type" value="wet_ground"/>
            </bsdf>
            <bsdf type="radio-material" id="my-concrete"/>

            <shape type="cube" id="shape1">
                <ref name="bsdf" id="mat-itu_concrete"/>
            </shape>
            <shape type="cube" id="shape2">
                <ref name="bsdf" id="wet_ground"/>
            </shape>
            <shape type="cube" id="shape3">
                <ref name="bsdf" id="my-concrete"/>
            </shape>
        </scene>
    """)


def test_extend_scene_with_mesh_rejects_duplicate_id():
    scene = mi.load_dict({
        "type": "scene",
        "shape-a": {"type": "cube", "id": "shape-a"},
    })
    mesh = mi.load_dict({"type": "cube", "id": "shape-a"})
    with pytest.raises(ValueError, match="already used in the scene"):
        extend_scene_with_mesh(scene, mesh)


def test_extend_scene_with_mesh_adds_unique_id():
    scene = mi.load_dict({
        "type": "scene",
        "shape-a": {"type": "cube", "id": "shape-a"},
    })
    mesh = mi.load_dict({"type": "sphere", "id": "shape-b"})
    extended = extend_scene_with_mesh(scene, mesh)
    ids = {s.id() for s in extended.shapes()}
    assert ids == {"shape-a", "shape-b"}


def test_scene_edit_add_raw_mitsuba_dict():
    scene = load_scene(rt.scene.box, merge_shapes=False)
    n_before = len(scene.objects)
    kept = next(iter(scene.objects.values()))

    scene.edit(add={
        "type": "cube",
        "id": "extra-cube",
        "to_world": mi.ScalarTransform4f.translate([5, 0, 2]).scale(0.25),
        "bsdf": {
            "type": "radio-material",
            "id": "extra-mat",
            "relative_permittivity": 3.0,
            "conductivity": 0.01,
            "thickness": 0.1,
        },
    })

    assert "extra-cube" in scene.objects
    assert len(scene.objects) == n_before + 1
    assert isinstance(scene.objects["extra-cube"], SceneObject)
    # Untouched objects keep identity across the edit.
    assert scene.objects[kept.name] is kept

    with pytest.raises(ValueError, match="already used in the scene"):
        scene.edit(add={
            "type": "cube",
            "id": "extra-cube",
            "bsdf": {
                "type": "radio-material",
                "id": "extra-mat-2",
                "relative_permittivity": 2.0,
                "conductivity": 0.0,
                "thickness": 0.1,
            },
        })


def test_use_mi_scene_restores_on_exception():
    scene = load_scene(rt.scene.box, merge_shapes=False)
    original = scene.mi_scene
    other = mi.load_dict({"type": "scene", "sphere": {"type": "sphere"}})

    with pytest.raises(RuntimeError, match="boom"):
        with scene.use_mi_scene(other):
            assert scene.mi_scene is other
            raise RuntimeError("boom")

    assert scene.mi_scene is original

    # Nested contexts restore correctly as well.
    with scene.use_mi_scene(other):
        assert scene.mi_scene is other
        inner = mi.load_dict({"type": "scene", "cube": {"type": "cube"}})
        with scene.use_mi_scene(inner):
            assert scene.mi_scene is inner
        assert scene.mi_scene is other
    assert scene.mi_scene is original
