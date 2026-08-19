#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import gc
import logging
import os
from os.path import join
import tempfile

import pytest

import mitsuba as mi
import drjit as dr
from sionna import rt
from sionna.rt import load_scene, load_scene_from_string, SceneObject, \
                      RadioMaterial, RadioMaterialBase, ITURadioMaterial, \
                      register_itu_radio_material, register_radio_material, \
                      radio_material_registry


def register_custom_radio_material():
    # Register a custom radio material with a custom property `some_param` that is not part of the built-in radio material.
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


def test08_register_itu_radio_material():
    # Register the material
    register_itu_radio_material(
        "custom_unknown_wood",
        {(0.1, 100.0): (2.5, 0.0, 0.01, 1.0)},
        (0.2, 0.4, 0.6)
    )

    # 1. Loading with diffuse BSDF without 'itu_' or 'mat-' prefix must fail
    xml_str_no_prefix = """
    <scene version="2.1.0">
        <bsdf type="diffuse" id="custom_unknown_wood"/>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="custom_unknown_wood"/>
        </shape>
    </scene>
    """
    with pytest.raises(ValueError, match=r".*ITU material names must start with \"itu_\".*"):
        load_scene_from_string(xml_str_no_prefix)

    # 2. Loading with newer explicit syntax (<bsdf type="itu-radio-material" ...>)
    xml_str_new_syntax = """
    <scene version="2.1.0">
        <bsdf type="itu-radio-material" id="itu_custom_unknown_wood">
            <string name="type" value="custom_unknown_wood"/>
        </bsdf>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="itu_custom_unknown_wood"/>
        </shape>
    </scene>
    """
    scene1 = load_scene_from_string(xml_str_new_syntax)
    assert "itu_custom_unknown_wood" in scene1.radio_materials
    assert isinstance(scene1.radio_materials["itu_custom_unknown_wood"], ITURadioMaterial)

    # 3. Loading with newer explicit syntax with id without 'itu_' prefix
    xml_str_new_syntax2 = """
    <scene version="2.1.0">
        <bsdf type="itu-radio-material" id="custom_unknown_wood">
            <string name="type" value="custom_unknown_wood"/>
        </bsdf>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="custom_unknown_wood"/>
        </shape>
    </scene>
    """
    scene2 = load_scene_from_string(xml_str_new_syntax2)
    assert "custom_unknown_wood" in scene2.radio_materials
    assert isinstance(scene2.radio_materials["custom_unknown_wood"], ITURadioMaterial)

    # 4. Loading with legacy/Blender syntax (<bsdf type="diffuse" id="itu_custom_unknown_wood"/>)
    xml_str_legacy = """
    <scene version="2.1.0">
        <bsdf type="diffuse" id="itu_custom_unknown_wood"/>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="itu_custom_unknown_wood"/>
        </shape>
    </scene>
    """
    scene3 = load_scene_from_string(xml_str_legacy)
    assert "itu_custom_unknown_wood" in scene3.radio_materials
    assert isinstance(scene3.radio_materials["itu_custom_unknown_wood"], ITURadioMaterial)

    # 5. Loading with legacy/Blender syntax with 'mat-' prefix (<bsdf type="diffuse" id="mat-itu_custom_unknown_wood"/>)
    xml_str_legacy_mat = """
    <scene version="2.1.0">
        <bsdf type="diffuse" id="mat-itu_custom_unknown_wood"/>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="mat-itu_custom_unknown_wood"/>
        </shape>
    </scene>
    """
    scene4 = load_scene_from_string(xml_str_legacy_mat)
    assert "itu_custom_unknown_wood" in scene4.radio_materials
    # 6. Loading multiple custom material IDs referencing the same ITU material type with different thicknesses
    xml_str_multiple_ids = """
    <scene version="2.1.0">
        <bsdf type="itu-radio-material" id="my_custom_thick_wood">
            <string name="type" value="custom_unknown_wood"/>
            <float name="thickness" value="0.25"/>
        </bsdf>
        <bsdf type="itu-radio-material" id="my_custom_thin_wood">
            <string name="type" value="custom_unknown_wood"/>
            <float name="thickness" value="0.02"/>
        </bsdf>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="my_custom_thick_wood"/>
        </shape>
        <shape type="cube" id="shape2">
            <ref name="bsdf" id="my_custom_thin_wood"/>
        </shape>
    </scene>
    """
    scene_multiple = load_scene_from_string(xml_str_multiple_ids, merge_shapes=False)
    assert "my_custom_thick_wood" in scene_multiple.radio_materials
    assert "my_custom_thin_wood" in scene_multiple.radio_materials
    mat_thick = scene_multiple.radio_materials["my_custom_thick_wood"]
    mat_thin = scene_multiple.radio_materials["my_custom_thin_wood"]
    assert mat_thick.itu_type == "custom_unknown_wood"
    assert mat_thin.itu_type == "custom_unknown_wood"
    assert dr.allclose(mat_thick.thickness, 0.25)
    assert dr.allclose(mat_thin.thickness, 0.02)

    # 7. Loading ITU material with XML color override
    xml_str_color_override = """
    <scene version="2.1.0">
        <bsdf type="itu-radio-material" id="itu_custom_unknown_wood">
            <string name="type" value="custom_unknown_wood"/>
            <rgb name="color" value="0.9, 0.1, 0.1"/>
        </bsdf>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="itu_custom_unknown_wood"/>
        </shape>
    </scene>
    """
    scene_color = load_scene_from_string(xml_str_color_override)
    mat_color = scene_color.radio_materials["itu_custom_unknown_wood"]
    assert dr.allclose(mat_color.color, (0.9, 0.1, 0.1), atol=1e-3)

    # Register material without specifying color (color=None)
    register_itu_radio_material(
        "custom_no_color_wood",
        {(0.1, 100.0): (2.0, 0.0, 0.01, 1.0)}
    )
    xml_str_no_color = """
    <scene version="2.1.0">
        <bsdf type="itu-radio-material" id="itu_custom_no_color_wood">
            <string name="type" value="custom_no_color_wood"/>
        </bsdf>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="itu_custom_no_color_wood"/>
        </shape>
    </scene>
    """
    scene_no_color = load_scene_from_string(xml_str_no_color)
    assert "itu_custom_no_color_wood" in scene_no_color.radio_materials
    mat_no_color = scene_no_color.radio_materials["itu_custom_no_color_wood"]
    assert mat_no_color.color is not None


def test09_register_radio_material():
    xml_str_unregistered = """
    <scene version="2.1.0">
        <bsdf type="diffuse" id="my_custom_rm"/>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="my_custom_rm"/>
        </shape>
    </scene>
    """

    # Loading without registration must fail because "my_custom_rm" is not a radio material
    with pytest.raises(ValueError, match=r".*which is not a radio material.*"):
        load_scene_from_string(xml_str_unregistered)

    # 1. Loading with new explicit syntax (<bsdf type="radio-material" ...>)
    custom_mat1 = RadioMaterial(name="my_custom_rm1", relative_permittivity=4.5, conductivity=0.03)
    register_radio_material(custom_mat1)
    xml_str_new_syntax = """
    <scene version="2.1.0">
        <bsdf type="radio-material" id="my_custom_rm1"/>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="my_custom_rm1"/>
        </shape>
    </scene>
    """
    scene1 = load_scene_from_string(xml_str_new_syntax, merge_shapes=False)
    assert "my_custom_rm1" in scene1.radio_materials
    mat1 = scene1.radio_materials["my_custom_rm1"]
    assert dr.allclose(mat1.relative_permittivity, 4.5)
    assert dr.allclose(mat1.conductivity, 0.03)
    assert mat1 is not custom_mat1
    assert scene1.objects["shape1"].radio_material is mat1
    radio_material_registry.unregister("my_custom_rm1")

    # 2. Loading with legacy/Blender syntax (<bsdf type="diffuse" ...>)
    custom_mat2 = RadioMaterial(name="my_custom_rm2", relative_permittivity=4.5, conductivity=0.03)
    register_radio_material(custom_mat2)
    xml_str_legacy = """
    <scene version="2.1.0">
        <bsdf type="diffuse" id="my_custom_rm2"/>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="my_custom_rm2"/>
        </shape>
    </scene>
    """
    scene2 = load_scene_from_string(xml_str_legacy, merge_shapes=False)
    assert "my_custom_rm2" in scene2.radio_materials
    mat2 = scene2.radio_materials["my_custom_rm2"]
    assert dr.allclose(mat2.relative_permittivity, 4.5)
    assert dr.allclose(mat2.conductivity, 0.03)
    assert mat2 is not custom_mat2
    radio_material_registry.unregister("my_custom_rm2")

    # 3. Loading with 'mat-' prefix syntax (<bsdf type="diffuse" id="mat-my_custom_rm3"/>)
    custom_mat3 = RadioMaterial(name="my_custom_rm3", relative_permittivity=4.5, conductivity=0.03)
    register_radio_material(custom_mat3)
    xml_str_legacy_mat = """
    <scene version="2.1.0">
        <bsdf type="diffuse" id="mat-my_custom_rm3"/>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="mat-my_custom_rm3"/>
        </shape>
    </scene>
    """
    scene3 = load_scene_from_string(xml_str_legacy_mat, merge_shapes=False)
    assert "my_custom_rm3" in scene3.radio_materials
    mat3 = scene3.radio_materials["my_custom_rm3"]
    assert dr.allclose(mat3.relative_permittivity, 4.5)
    assert dr.allclose(mat3.conductivity, 0.03)
    assert mat3 is not custom_mat3
    radio_material_registry.unregister("my_custom_rm3")

    # 4. Loading custom RadioMaterial with XML color override: the override
    # must apply to an independent clone of the material, and must NOT
    # mutate the registered material itself, since it may be shared by
    # other shapes or scenes.
    custom_mat4 = RadioMaterial(name="my_custom_rm4", relative_permittivity=4.5, conductivity=0.03, color=(0.1, 0.1, 0.1))
    register_radio_material(custom_mat4)
    xml_str_color_override = """
    <scene version="2.1.0">
        <bsdf type="diffuse" id="my_custom_rm4">
            <rgb name="reflectance" value="0.8, 0.2, 0.2"/>
        </bsdf>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="my_custom_rm4"/>
        </shape>
    </scene>
    """
    scene4 = load_scene_from_string(xml_str_color_override, merge_shapes=False)
    mat4 = scene4.radio_materials["my_custom_rm4"]
    assert dr.allclose(mat4.color, (0.8, 0.2, 0.2), atol=1e-3)
    assert mat4 is not custom_mat4
    assert dr.allclose(custom_mat4.color, (0.1, 0.1, 0.1), atol=1e-3)
    radio_material_registry.unregister("my_custom_rm4")

    # 5. Overriding `thickness`, `conductivity`, and `color` of a registered
    # material directly in the scene file: overrides must apply to an independent
    # clone of the material without mutating the registered material itself.
    custom_mat5 = RadioMaterial(name="my_custom_rm5", relative_permittivity=4.5, conductivity=0.03, thickness=0.1, color=(0.1, 0.1, 0.1))
    register_radio_material(custom_mat5)
    xml_str_property_overrides = """
    <scene version="2.1.0">
        <bsdf type="radio-material" id="my_custom_rm5">
            <float name="thickness" value="0.35"/>
            <float name="conductivity" value="0.08"/>
            <rgb name="color" value="0.7, 0.3, 0.3"/>
        </bsdf>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="my_custom_rm5"/>
        </shape>
    </scene>
    """
    scene5 = load_scene_from_string(xml_str_property_overrides, merge_shapes=False)
    mat5 = scene5.radio_materials["my_custom_rm5"]
    assert dr.allclose(mat5.thickness, 0.35)
    assert dr.allclose(mat5.conductivity, 0.08)
    assert dr.allclose(mat5.relative_permittivity, 4.5)
    assert dr.allclose(mat5.color, (0.7, 0.3, 0.3), atol=1e-3)
    # The registered material must remain untouched
    assert dr.allclose(custom_mat5.thickness, 0.1)
    assert dr.allclose(custom_mat5.conductivity, 0.03)
    assert dr.allclose(custom_mat5.color, (0.1, 0.1, 0.1), atol=1e-3)
    radio_material_registry.unregister("my_custom_rm5")

    # 6. Loading a registered material with no override at all must not
    # change any of its properties.
    custom_mat6 = RadioMaterial(name="my_custom_rm6", relative_permittivity=4.5, conductivity=0.03, thickness=0.42, color=(0.3, 0.4, 0.5))
    register_radio_material(custom_mat6)
    xml_str_no_override = """
    <scene version="2.1.0">
        <bsdf type="diffuse" id="my_custom_rm6"/>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="my_custom_rm6"/>
        </shape>
    </scene>
    """
    scene6 = load_scene_from_string(xml_str_no_override, merge_shapes=False)
    mat6 = scene6.radio_materials["my_custom_rm6"]
    assert dr.allclose(mat6.thickness, 0.42)
    assert dr.allclose(mat6.relative_permittivity, 4.5)
    assert dr.allclose(mat6.conductivity, 0.03)
    assert dr.allclose(mat6.color, (0.3, 0.4, 0.5), atol=1e-3)
    assert dr.allclose(custom_mat6.thickness, 0.42)
    assert dr.allclose(custom_mat6.color, (0.3, 0.4, 0.5), atol=1e-3)
    radio_material_registry.unregister("my_custom_rm6")

    # 7. Multiple shapes referencing the same BSDF node (through `<ref>`)
    # that requests overrides must share a single cloned material,
    # rather than each getting their own, unrelated, clone.
    custom_mat7 = RadioMaterial(name="my_custom_rm7", relative_permittivity=4.5, conductivity=0.03, color=(0.1, 0.1, 0.1))
    register_radio_material(custom_mat7)
    xml_str_shared_override = """
    <scene version="2.1.0">
        <bsdf type="diffuse" id="my_custom_rm7">
            <rgb name="reflectance" value="0.2, 0.7, 0.2"/>
        </bsdf>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="my_custom_rm7"/>
        </shape>
        <shape type="cube" id="shape2">
            <ref name="bsdf" id="my_custom_rm7"/>
        </shape>
    </scene>
    """
    scene7 = load_scene_from_string(xml_str_shared_override, merge_shapes=False)
    assert scene7.objects["shape1"].radio_material is scene7.objects["shape2"].radio_material
    assert dr.allclose(scene7.objects["shape1"].radio_material.color, (0.2, 0.7, 0.2), atol=1e-3)
    assert dr.allclose(custom_mat7.color, (0.1, 0.1, 0.1), atol=1e-3)
    radio_material_registry.unregister("my_custom_rm7")

    # 8. Registered `ITURadioMaterial`: overrides must also return an
    # independent, working, clone.
    custom_mat8 = ITURadioMaterial(name="my_custom_rm8", itu_type="concrete",
                                   thickness=0.15, color=(0.1, 0.1, 0.1))
    register_radio_material(custom_mat8)
    xml_str_itu_color_override = """
    <scene version="2.1.0">
        <bsdf type="diffuse" id="my_custom_rm8">
            <rgb name="reflectance" value="0.4, 0.6, 0.1"/>
            <float name="thickness" value="0.22"/>
        </bsdf>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="my_custom_rm8"/>
        </shape>
    </scene>
    """
    scene8 = load_scene_from_string(xml_str_itu_color_override, merge_shapes=False)
    mat8 = scene8.radio_materials["my_custom_rm8"]
    assert isinstance(mat8, ITURadioMaterial)
    assert mat8.itu_type == "concrete"
    assert dr.allclose(mat8.thickness, 0.22)
    assert dr.allclose(mat8.color, (0.4, 0.6, 0.1), atol=1e-3)
    assert mat8 is not custom_mat8
    assert dr.allclose(custom_mat8.thickness, 0.15)
    assert dr.allclose(custom_mat8.color, (0.1, 0.1, 0.1), atol=1e-3)
    radio_material_registry.unregister("my_custom_rm8")

    # 9. A custom RadioMaterialBase subclass that does not implement
    # `clone` must raise a clear error when an override is requested.
    class BareRadioMaterial(RadioMaterialBase):
        def __init__(self, name):
            props = mi.Properties("radio-material")
            props.set_id(name)
            super().__init__(props)

    custom_mat9 = BareRadioMaterial("my_custom_rm9")
    register_radio_material(custom_mat9)
    xml_str_unsupported_override = """
    <scene version="2.1.0">
        <bsdf type="diffuse" id="my_custom_rm9">
            <rgb name="reflectance" value="0.4, 0.6, 0.1"/>
        </bsdf>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="my_custom_rm9"/>
        </shape>
    </scene>
    """
    with pytest.raises(NotImplementedError, match=r".*does not support cloning.*"):
        load_scene_from_string(xml_str_unsupported_override, merge_shapes=False)
    radio_material_registry.unregister("my_custom_rm9")

    # 10. Multiple distinct variations of the same registered material in one scene
    # must be assigned unique names and registered independently in scene.radio_materials.
    custom_mat10 = RadioMaterial(name="my_custom_rm10", relative_permittivity=3.0, conductivity=0.01, thickness=0.1)
    register_radio_material(custom_mat10)
    xml_str_multi_variations = """
    <scene version="2.1.0">
        <bsdf type="diffuse" id="my_custom_rm10">
            <float name="thickness" value="0.1"/>
        </bsdf>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="my_custom_rm10"/>
        </shape>
        <shape type="cube" id="shape2">
            <bsdf type="diffuse" id="my_custom_rm10">
                <float name="thickness" value="0.3"/>
            </bsdf>
        </shape>
    </scene>
    """
    scene10 = load_scene_from_string(xml_str_multi_variations, merge_shapes=False)
    assert "my_custom_rm10" in scene10.radio_materials
    assert "my_custom_rm10_1" in scene10.radio_materials
    mat10_a = scene10.radio_materials["my_custom_rm10"]
    mat10_b = scene10.radio_materials["my_custom_rm10_1"]
    assert dr.allclose(mat10_a.thickness, 0.1)
    assert dr.allclose(mat10_b.thickness, 0.3)
    assert scene10.objects["shape1"].radio_material is mat10_a
    assert scene10.objects["shape2"].radio_material is mat10_b
    radio_material_registry.unregister("my_custom_rm10")

    # 11. Shape merging with registered custom radio materials
    custom_mat11 = RadioMaterial(name="my_custom_rm11", relative_permittivity=3.5, conductivity=0.02)
    register_radio_material(custom_mat11)
    xml_str_merge = """
    <scene version="2.1.0">
        <bsdf type="diffuse" id="my_custom_rm11">
            <float name="thickness" value="0.2"/>
        </bsdf>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="my_custom_rm11"/>
        </shape>
        <shape type="cube" id="shape2">
            <ref name="bsdf" id="my_custom_rm11"/>
        </shape>
    </scene>
    """
    scene11 = load_scene_from_string(xml_str_merge, merge_shapes=True)
    # Shapes sharing the same BSDF should be merged into one shape
    assert len(scene11.objects) == 1
    assert "my_custom_rm11" in scene11.radio_materials
    merged_obj = list(scene11.objects.values())[0]
    assert dr.allclose(merged_obj.radio_material.thickness, 0.2)
    radio_material_registry.unregister("my_custom_rm11")


def test10_register_radio_material_logging(caplog):
    custom_mat = RadioMaterial(name="logged_mat", relative_permittivity=3.0, conductivity=0.01)
    register_radio_material(custom_mat)
    xml_str = """
    <scene version="2.1.0">
        <bsdf type="diffuse" id="logged_mat">
            <float name="thickness" value="0.5"/>
        </bsdf>
        <shape type="cube" id="shape1">
            <ref name="bsdf" id="logged_mat"/>
        </shape>
    </scene>
    """
    with caplog.at_level(logging.INFO, logger="sionna.rt"):
        scene = load_scene_from_string(xml_str, merge_shapes=False)
        assert "logged_mat" in scene.radio_materials
        assert any("Registered radio material 'logged_mat'" in record.message for record in caplog.records)
    radio_material_registry.unregister("logged_mat")


def test11_clone_scattering_pattern():
    mat = RadioMaterial(name="test_mat", relative_permittivity=3.0, conductivity=0.01)
    
    # 1. Clone with string factory name
    cloned_str = mat.clone(name="test_mat_str", scattering_pattern="lambertian")
    assert isinstance(cloned_str.scattering_pattern, rt.ScatteringPattern)

    # 2. Clone with ScatteringPattern instance
    sp_instance = mat.scattering_pattern
    cloned_inst = mat.clone(name="test_mat_inst", scattering_pattern=sp_instance)
    assert cloned_inst.scattering_pattern is sp_instance

    # 3. Clone with invalid scattering pattern raises ValueError via setter
    with pytest.raises(ValueError, match="Not an instance of ScatteringPattern"):
        mat.clone(name="test_mat_invalid", scattering_pattern=12345)

    # 4. ITURadioMaterial clone with invalid scattering pattern
    itu_mat = ITURadioMaterial(name="test_itu", itu_type="concrete", thickness=0.1)
    with pytest.raises(ValueError, match="Not an instance of ScatteringPattern"):
        itu_mat.clone(name="test_itu_invalid", scattering_pattern=12345)


def test12_clone_shallow_parameter_sharing():
    # Verify that non-overridden properties share differentiable references
    # in the computation graph, but updating one instance via setter does not
    # implicitly mutate other distinct cloned instances.
    sigma = mi.Float(0.1)
    dr.enable_grad(sigma)
    mat = RadioMaterial(
        name="orig_mat",
        relative_permittivity=3.5,
        conductivity=sigma,
        thickness=0.1
    )
    # Create two clones sharing conductivity but overriding thickness
    cloned1 = mat.clone(name="cloned1", thickness=0.5)
    cloned2 = mat.clone(name="cloned2", thickness=0.8)

    # 1. Verify gradient backpropagation from cloned1 to mat and cloned2
    loss = dr.square(cloned1.conductivity - 1.0)
    dr.backward(loss)
    # Gradient of (x - 1)^2 at x=0.1 is 2*(0.1 - 1.0) = -1.8
    assert dr.allclose(dr.grad(mat.conductivity), -1.8, atol=1e-4)
    assert dr.allclose(dr.grad(cloned1.conductivity), -1.8, atol=1e-4)
    assert dr.allclose(dr.grad(cloned2.conductivity), -1.8, atol=1e-4)

    # 2. Applying a gradient step on mat re-binds mat.conductivity
    grad = dr.grad(mat.conductivity)
    new_val = dr.detach(mat.conductivity - 0.2 * grad)
    mat.conductivity = new_val

    # mat has the new value (~0.46)
    assert dr.allclose(mat.conductivity, 0.46, atol=1e-4)
    # Clones are separate Python objects and retain their original value (0.1)
    assert dr.allclose(cloned1.conductivity, 0.1, atol=1e-4)
    assert dr.allclose(cloned2.conductivity, 0.1, atol=1e-4)

    # To update clones for subsequent forward passes, re-bind explicitly:
    cloned1.conductivity = mat.conductivity
    assert dr.allclose(cloned1.conductivity, 0.46, atol=1e-4)



