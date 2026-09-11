#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import drjit as dr
import mitsuba as mi
import numpy as np

from sionna import rt
from sionna.rt import load_scene, ITURadioMaterial, RadioMaterial,\
    InteractionType, INVALID_SHAPE, INVALID_PRIMITIVE
from sionna.rt.constants import NO_JONES_MATRIX
from sionna.rt.path_solvers.sb_candidate_generator import SBCandidateGenerator
from sionna.rt.path_solvers.sb_deterministic import SBDeterministicCandidateGenerator


GENERATOR_TYPES = [SBCandidateGenerator, SBDeterministicCandidateGenerator]

############################################################
# Utilities
############################################################

def load_box_scene(material_name, thickness, scattering_coefficient):

    scene = load_scene(rt.scene.box, merge_shapes=False)
    box = scene.get("box")

    box.radio_material = ITURadioMaterial(name=f"mat-{material_name}",
                                itu_type=material_name,
                                thickness=thickness,
                                scattering_coefficient=scattering_coefficient)

    return scene

def load_box_one_screen_scene(material_name, thickness, scattering_coefficient):

    scene = load_scene(rt.scene.box_one_screen, merge_shapes=False)
    screen = scene.get("screen")

    screen.radio_material = ITURadioMaterial(
                                name=f"mat-{material_name}",
                                itu_type=material_name,
                                thickness=thickness,
                                scattering_coefficient=scattering_coefficient)

    return scene

def load_box_knife_scene(material_name, thickness, scattering_coefficient):

    # scene = load_scene(rt.scene.box_knife, merge_shapes=False)
    scene = load_scene(rt.scene.box_knife, merge_shapes=False)
    screen = scene.get("box")

    screen.radio_material = ITURadioMaterial(
                                name=f"mat-{material_name}",
                                itu_type=material_name,
                                thickness=thickness,
                                scattering_coefficient=scattering_coefficient)

    return scene

def load_box_box_two_screens_scene(material_name, thickness, scattering_coefficient):

    scene = load_scene(rt.scene.box_two_screens, merge_shapes=False)
    screen = scene.get("box")

    screen.radio_material = ITURadioMaterial(
                                name=f"mat-{material_name}",
                                itu_type=material_name,
                                thickness=thickness,
                                scattering_coefficient=scattering_coefficient)

    return scene

############################################################
# Unit tests
############################################################

@pytest.mark.parametrize("int_type_str", [
    'specular', # Specular reflection
    'transmission', # Transmission
])
@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_specular_reflection_transmission_depth_1(int_type_str, Generator):
    """
    Tests chains of depth 1 with specular or transmission

    Input
    ------
    type: str, 'specular' or 'transmission'
        'specular': Test with only specular reflection
        'transmission': Test with only transmission
    """
    if int_type_str == 'specular':
        thickness = 1.0 # Only reflection as using metal as material
        int_type = InteractionType.SPECULAR
        expected_count = 6 # 1 per plane
    elif int_type_str == 'transmission':
        thickness = 0.0 # Only transmission
        int_type = InteractionType.REFRACTION
        expected_count = 1 # Only a single one suffices,
                           # through which primitive is not important
    else:
        raise ValueError(f"Invalid interaction type: {int_type_str}")

    source = mi.Point3f(0., 0., 1.5)
    target = mi.Point3f(1., 1., 1.)

    max_depth = 1
    samples_per_src = 100
    max_num_paths = 1000

    scene = load_box_scene("metal", thickness, scattering_coefficient=0.)
    tracer = Generator()

    paths = tracer(scene.mi_scene, source, target, samples_per_src, max_num_paths, max_depth,
                   los=True, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, diffraction=False, edge_diffraction=False, seed=1)
    paths.shrink()

    shapes = paths.shapes.numpy()[:,0]
    primitives = paths.primitives.numpy()[:,0]
    int_types = paths.interaction_types.numpy()[:,0]
    valid = paths.valid.numpy()

    # Depth should be set to 1 and max_num_path to 13
    assert paths.buffer_size == expected_count + 1
    assert paths.max_depth == 1

    # There should be one LoS, and all other interactions must be of the
    # selected type
    int_types_u, int_types_i, int_types_c = np.unique(int_types, return_counts=True, return_index=True)
    assert len(int_types_u) == 2
    assert InteractionType.NONE in int_types_u
    assert int_type in int_types_u
    assert 1 in int_types_c
    assert expected_count in int_types_c

    # Index of the LoS
    if int_types_c[0] == 1:
        los_index = int_types_i[0]
    else:
        los_index = int_types_i[1]

    # Only one shape
    assert shapes[los_index] == INVALID_SHAPE
    assert np.unique(shapes).shape[0] == 2

    # Each interaction should hit a unique primitive
    assert primitives[los_index] == INVALID_PRIMITIVE
    assert np.unique(primitives).shape[0] == expected_count + 1

    # No paths should be valid, i.e., only candidates
    assert valid[los_index], f"{valid=}, {los_index=}"
    assert np.all(np.logical_not(np.delete(valid, los_index)))


@pytest.mark.parametrize("int_type_str", [
    'specular', # Specular reflection
    'transmission', # Transmission
])
@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_specular_or_transmission_depth_1_multilink(int_type_str, Generator):
    """
    Tests chains of depth 1 with specular *or* transmission with multiple
    sources and targets

    Input
    ------
    type: str, 'specular' or 'transmission'
        'specular': Test with only specular reflection
        'transmission': Test with only transmission
    """

    if int_type_str == 'specular':
        thickness = 1.0 # Only reflection as using metal as material
        int_type = InteractionType.SPECULAR
        expected_count = 6 # 1 per plane
    elif int_type_str == 'transmission':
        thickness = 0.0 # Only transmission
        int_type = InteractionType.REFRACTION
        expected_count = 1 # Only a single one suffices,
                            # through which primitive is not important
    else:
        raise ValueError(f"Invalid interaction type: {int_type_str}")

    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 0., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    max_depth = 1
    samples_per_src = 100
    max_num_paths = 10000

    scene = load_box_scene("metal", thickness, scattering_coefficient=0.)
    tracer = Generator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=True, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, diffraction=False, edge_diffraction=False, seed=1)
    paths.shrink()

    shapes = paths.shapes.numpy()[:,0]
    primitives = paths.primitives.numpy()[:,0]
    int_types = paths.interaction_types.numpy()[:,0]
    valid = paths.valid.numpy()
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # Depth should be set to 1 and max_num_path to 12*6, as they are 12
    # primitives in the scene.mi_scene, and each of the 6 links should have all
    # 12 primitives as candidates.
    assert paths.buffer_size == (expected_count+1)*6
    assert paths.max_depth == 1

    # Extract LoS and check it

    los_indices = np.where(int_types == InteractionType.NONE)[0]
    assert los_indices.shape[0] == 6 # 1 per link

    los_shapes = shapes[los_indices]
    los_primitives = primitives[los_indices]
    los_valid = valid[los_indices]
    los_src_indices = src_indices[los_indices]
    los_tgt_indices = tgt_indices[los_indices]

    assert np.unique(los_shapes) == np.array([INVALID_SHAPE])
    assert np.unique(los_primitives) == np.array([INVALID_PRIMITIVE])
    assert np.all(los_valid)
    assert np.unique(los_src_indices).shape[0] == 2
    assert np.unique(los_tgt_indices).shape[0] == 3

    # Check NLoS paths

    shapes = np.delete(shapes, los_indices)
    primitives = np.delete(primitives, los_indices)
    valid = np.delete(valid, los_indices)
    int_types = np.delete(int_types, los_indices)
    src_indices = np.delete(src_indices, los_indices)
    tgt_indices = np.delete(tgt_indices, los_indices)

    # All interactions must be of the selected type
    assert np.unique(int_types) == np.array([int_type])

    # Only one shape
    assert np.unique(shapes).shape[0] == 1

    # Primitives each (source, target) link and each candidate should be unique
    primitives_per_link = {}
    for src_ind, tgt_ind, prim_ind in zip(src_indices, tgt_indices, primitives):
        key = (src_ind, tgt_ind)
        if key not in primitives_per_link:
            primitives_per_link[key] = []
        primitives_per_link[key].append(prim_ind)

    # There should be 6 links
    assert len(primitives_per_link) == 6
    for src_ind in range(2):
        for tgt_ind in range(3):
            key = (src_ind, tgt_ind)
            # Check that the number of primitives per link is correct
            assert np.unique(primitives_per_link[key]).shape[0] == expected_count

    # No paths should be valid, i.e., only candidates
    assert np.all(np.logical_not(valid))

@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_specular_and_transmission_depth_1(Generator):
    """
    Test single reflection (depth of 1) with both specular reflection
    and transmission
    """

    source = mi.Point3f(0., 0., 2.5)
    target = mi.Point3f(1., 1., 1.)

    max_depth = 1
    samples_per_src = 1000
    max_num_paths = 10000

    # Scattering coefficient is set to 0
    # Set material to glass with a thickness of 1cm, which lead to almost equal
    # splitting of the energy between transmission and reflection
    scene = load_box_scene("glass", 0.01, scattering_coefficient=0.)
    tracer = Generator()

    paths = tracer(scene.mi_scene, source, target, samples_per_src, max_num_paths, max_depth,
                   los=True, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, diffraction=False, edge_diffraction=False, seed=1)
    paths.shrink()

    shapes = paths.shapes.numpy()[:,0]
    primitives = paths.primitives.numpy()[:,0]
    int_types = paths.interaction_types.numpy()[:,0]
    valid = paths.valid.numpy()

    # Depth should be set to 1 and max_num_path to 8:
    # - 1 LoS
    # - 1 transmitted path (doesn't matter through which primitive)
    # - 6 specular reflections (1 per plane)
    assert paths.buffer_size == 8
    assert paths.max_depth == 1

    # Extract LoS and check it

    los_indices = np.where(int_types == InteractionType.NONE)[0]
    assert los_indices.shape[0] == 1 # Only a single LoS

    los_shape = shapes[los_indices]
    los_primitive = primitives[los_indices]
    los_valid = valid[los_indices]

    assert los_shape == INVALID_SHAPE
    assert los_primitive == INVALID_PRIMITIVE
    assert los_valid

    # Check NLoS paths

    shapes = np.delete(shapes, los_indices)
    primitives = np.delete(primitives, los_indices)
    valid = np.delete(valid, los_indices)
    int_types = np.delete(int_types, los_indices)

    # All interactions must be either specular reflection or transmission
    for int_type in int_types:
        assert int_type in (InteractionType.SPECULAR, InteractionType.REFRACTION)

    # There should be one specular reflection per plane
    spec_indices = np.where(int_types == InteractionType.SPECULAR)[0]
    assert spec_indices.shape[0] == 6

    # There should be a single transmission path
    tr_indices = np.where(int_types == InteractionType.REFRACTION)[0]
    assert tr_indices.shape[0] == 1

    # Only one shape
    assert np.unique(shapes).shape[0] == 1

    # No paths should be valid, i.e., only candidates
    assert np.all(np.logical_not(valid))

    # Check that there is no redundancy
    inter_pair = np.stack([primitives, int_types], axis=1)
    inter_pair = np.unique(inter_pair, axis=0)
    assert inter_pair.shape[0] == 7

@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_specular_and_transmission_depth_1_multilink(Generator):
    """
    Test single reflection (depth of 1) with both specular reflection
    and transmission with multiple sources and targets
    """

    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 0., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    max_depth = 1
    samples_per_src = int(1e5)
    max_num_paths = int(1e4)

    # Scattering coefficient is set to 0
    # Set material to glass with a thickness of 1cm, which lead to almost equal
    # splitting of the energy between transmission and reflection
    scene = load_box_scene("glass", 0.01, scattering_coefficient=0.)
    tracer = Generator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=True, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, diffraction=False, edge_diffraction=False, seed=1)
    paths.shrink()

    shapes = paths.shapes.numpy()[:,0]
    primitives = paths.primitives.numpy()[:,0]
    int_types = paths.interaction_types.numpy()[:,0]
    valid = paths.valid.numpy()
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # Depth should be set to 1 and max_num_path to 8*6
    # Indeed, each link should have a single LoS, a single
    # transmitted path as well as one specularly reflected
    # path per plane.
    assert paths.buffer_size == 8*6
    assert paths.max_depth == 1

    # Extract LoS and check it

    los_indices = np.where(int_types == InteractionType.NONE)[0]
    assert los_indices.shape[0] == 6 # 1 per link

    los_shapes = shapes[los_indices]
    los_primitives = primitives[los_indices]
    los_valid = valid[los_indices]
    los_src_indices = src_indices[los_indices]
    los_tgt_indices = tgt_indices[los_indices]

    assert np.unique(los_shapes) == np.array([INVALID_SHAPE])
    assert np.unique(los_primitives) == np.array([INVALID_PRIMITIVE])
    assert np.all(los_valid)
    assert np.unique(los_src_indices).shape[0] == 2
    assert np.unique(los_tgt_indices).shape[0] == 3

    # Check NLoS paths

    shapes = np.delete(shapes, los_indices)
    primitives = np.delete(primitives, los_indices)
    valid = np.delete(valid, los_indices)
    int_types = np.delete(int_types, los_indices)
    src_indices = np.delete(src_indices, los_indices)
    tgt_indices = np.delete(tgt_indices, los_indices)

    # All interactions must be either specular reflection or transmission
    for int_type in int_types:
        assert int_type in (InteractionType.SPECULAR, InteractionType.REFRACTION)

    # There should be one specular reflection per plane per link
    spec_indices = np.where(int_types == InteractionType.SPECULAR)[0]
    assert spec_indices.shape[0] == 6*6

    # There should be a single transmission path
    tr_indices = np.where(int_types == InteractionType.REFRACTION)[0]
    assert tr_indices.shape[0] == 6

    # Only one shape
    assert np.unique(shapes).shape[0] == 1

    # No paths should be valid, i.e., only candidates
    assert np.all(np.logical_not(valid))

    # Check that there is no redundancy
    inter_pair = np.stack([src_indices, tgt_indices, primitives, int_types], axis=1)
    inter_pair = np.unique(inter_pair, axis=0)
    assert inter_pair.shape[0] == 7*6

@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_los_with_obstruction_multilink(Generator):
    r"""
    In the box scene with a screen and multiple links, check that LoS that should
    be obstructed are
    """

    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [4., -4.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 3., 3.],
                         [4., 0., -4.],
                         [2.5, 1.5, 2.5])

    max_depth = 1
    samples_per_src = int(1e3)
    max_num_paths = int(1e6)

    # Set material of the screen to glass with a thickness of 1cm, which lead to
    # almost equal splitting of the energy between transmission and reflection
    # Scattering coefficient for the screen set to 0
    scene = load_box_one_screen_scene("glass", 0.01, 0.0)

    tracer = Generator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=True, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, diffraction=True, edge_diffraction=True, seed=1)
    paths.shrink()

    shapes = paths.shapes.numpy()
    primitives = paths.primitives.numpy()
    int_types = paths.interaction_types.numpy()
    valid = paths.valid.numpy()
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # Extract LoS and check it

    los_indices = np.where(int_types == InteractionType.NONE)[0]
    assert los_indices.shape[0] == 2

    los_shapes = shapes[los_indices]
    los_primitives = primitives[los_indices]
    los_valid = valid[los_indices]
    los_src_indices = src_indices[los_indices]
    los_tgt_indices = tgt_indices[los_indices]

    assert np.unique(los_shapes) == np.array([INVALID_SHAPE])
    assert np.unique(los_primitives) == np.array([INVALID_PRIMITIVE])
    assert np.all(los_valid)

    # There should be only 2 LoS:
    #   source 0 --> target 0
    #   source 1 --> target 2
    los_pairs = set(zip(los_src_indices, los_tgt_indices))
    assert los_pairs == {(0, 0), (1, 2)}

    if Generator is SBDeterministicCandidateGenerator:
        los_pairs = np.stack([los_src_indices, los_tgt_indices], axis=1)
        assert np.array_equal(los_pairs, np.array([[0, 0], [1, 2]]))



@pytest.mark.parametrize("scene_name", [
    "box_knife",
    "box_one_screen",
])
@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_specular_chains_high_depth(scene_name, Generator):
    r"""
    Test specular chains of high depth
    - No duplicates
    - No paths continuing after a None interaction
    - At most a single diffraction per path
    """

    # 2 sources
    sources = mi.Point3f([-3., -3],
                            [1., -1.],
                            [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 3., 3.],
                            [1., 0., -1.],
                            [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    max_depth = 10
    samples_per_src = int(1e3)
    max_num_paths = int(1e6)

    # Set material of the screen to glass with a thickness of 1cm, which lead to
    # almost equal splitting of the energy between transmission and reflection
    # Scattering coefficient for the screen set to 0
    if scene_name == "box_knife":
        scene = load_box_knife_scene("glass", 0.01, 0.0)
    else:
        scene = load_box_one_screen_scene("glass", 0.01, 0.0)

    tracer = Generator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                    los=False, refraction=True, specular_reflection=True,
                    diffuse_reflection=True, diffraction=True, edge_diffraction=True, seed=1)
    paths.shrink()

    shapes = paths.shapes.numpy()
    primitives = paths.primitives.numpy()
    local_edges = paths.diffracting_wedges.local_edge.numpy()
    int_types = paths.interaction_types.numpy()
    valid = paths.valid.numpy()
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # Depth should be set to `max_depth`
    assert paths.max_depth == max_depth

    # Check interaction types
    num_paths = paths.buffer_size
    paths_active = np.full([num_paths], True)
    for d in range(max_depth):
        inters = int_types[:,d]
        specular = np.equal(inters, InteractionType.SPECULAR)
        transmission = np.equal(inters, InteractionType.REFRACTION)
        diffraction = np.equal(inters, InteractionType.DIFFRACTION)
        no_int = np.equal(inters, InteractionType.NONE)

        # Interaction type is valid
        assert np.all(np.any([specular, transmission, diffraction, no_int], axis=0))

        # If paths is done, there should be only none interaction
        assert np.all(np.logical_or(paths_active, no_int))

        # Update paths state
        paths_active = np.logical_and(paths_active, np.any([specular,
                                                            transmission,
                                                            diffraction], axis=0))

    # Check that there is at most one diffraction
    num_diffraction = np.sum(int_types == InteractionType.DIFFRACTION, axis=1)
    assert np.all(num_diffraction <= 1)

    # No paths should be valid, i.e., only candidates
    assert np.all(np.logical_not(valid))

    # Check there are no duplicate candidates
    # First, aggregate all interaction for each link
    link_interactions = {}
    i = 1
    for src_ind, tgt_ind, shape_ind, prim_ind, local_edge, int_type\
        in zip(src_indices, tgt_indices, shapes, primitives, local_edges, int_types):
        key = (src_ind, tgt_ind)
        if key not in link_interactions:
            link_interactions[key] = []
        # As diffraction is only supported for first order, we
        # duplicate the local edge for each depth
        local_edge = np.full(max_depth, local_edge)
        inter = np.stack([shape_ind, prim_ind, local_edge, int_type], axis=1)
        i += 1
        link_interactions[key].append(inter)
    # Check each link
    for src_ind in range(2):
        for tgt_ind in range(3):
            key = (src_ind, tgt_ind)
            inter = link_interactions[key]
            inter = np.stack(inter, axis=0)
            inter = np.reshape(inter, [inter.shape[0], -1])
            _, counts = np.unique(inter, axis=0, return_counts=True)
            assert np.all(np.equal(counts, 1))

@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_specular_prefixes(Generator):
    """
    Thest that, for specular chains, all possible candidates are generated with
    diffraction disabled.
    """
    # 2 sources
    sources = mi.Point3f([-3., -3.5],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 0., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    max_depth = 3
    samples_per_src = 10000
    max_num_paths = int(1e5)

    # Set material to metal which leads to all the energy being reflected
    # Scattering coefficient set to 0
    scene = load_box_scene("metal", 0.01, scattering_coefficient=0.)
    tracer = Generator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                    los=False, refraction=True, specular_reflection=True,
                    diffuse_reflection=False, diffraction=False, edge_diffraction=False, seed=1)
    paths.shrink()
    # In this scene, for each link, there should be 150 individual candidates:
    # - 6 depth-1 candidates, one for each face
    # - 30 depth-2 candidates, as each depth-1 candidate can hit any of the other 5 faces
    # - 114 depth-3 candidates. The maximum number of such candidates is 126:
    #   - 6x4x4 for paths that hit adjacent faces in the first two interactions
    #   - 6x1x5 for paths that hit parallel faces in the first two interactions
    #   - In this configuration however, 12 paths are inadmissible because of the
    #     origin of the path, which leads to 114 candidates.
    assert paths.buffer_size == 6 * 150

@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_diffraction_knife(Generator):
    """
    Test diffraction in the knife scene (wedge diffraction) with max_depth = 1.
    Ensure that all candidates are found in this simple setup.
    """

    sources = mi.Point3f([-3.],
                        [-2.],
                        [3.])

    targets = mi.Point3f([-2, 3],
                        [2., 2],
                        [3., 4])

    max_depth = 1
    samples_per_src = 10000
    max_num_paths = int(1e5)

    # Set material to metal which leads to all the energy being reflected
    # Scattering coefficient set to 0
    scene = load_box_knife_scene("metal", 0.01, scattering_coefficient=0.)
    tracer = Generator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                    los=False, refraction=True, specular_reflection=True,
                    diffuse_reflection=False, diffraction=True, edge_diffraction=False, seed=1)
    paths.shrink()

    int_types = paths.interaction_types.numpy().flatten()
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # Count number of diffraction per link
    num_diffractions = {}

    for s, t, i in zip(src_indices, tgt_indices, int_types):
        k = (s,t)
        if k not in num_diffractions:
            num_diffractions[k] = 0
        if i == InteractionType.DIFFRACTION:
            num_diffractions[k] += 1

    # Number of diffraction per link should be 2
    # This is because the knife is made of two separate segments

    all_num_diffractions = np.array(list(num_diffractions.values()))
    assert np.all(all_num_diffractions == 2)

@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_diffraction_one_screen(Generator):
    """
    Test diffraction in the one screen scene (edge diffraction) with max_depth = 1.
    Ensure that all candidates are found in this simple setup.
    """

    sources = mi.Point3f([-2., -3.],
                        [-2., 3.],
                        [3., 2.])
    num_sources = sources.shape[1]

    targets = mi.Point3f([2, 3],
                        [3., -1],
                        [3., 4])
    num_targets = targets.shape[1]

    max_depth = 1
    samples_per_src = 10000
    max_num_paths = int(1e5)

    # Set material to metal which leads to all the energy being reflected
    # Scattering coefficient set to 0
    scene = load_box_one_screen_scene("metal", 0.01, scattering_coefficient=0.)
    tracer = Generator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                    los=False, refraction=True, specular_reflection=True,
                    diffuse_reflection=False, diffraction=True, edge_diffraction=True, seed=1)
    paths.shrink()

    int_types = paths.interaction_types.numpy().flatten()
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # Count number of diffraction per link
    num_diffractions = {}

    for s, t, i in zip(src_indices, tgt_indices, int_types):
        k = (s,t)
        if k not in num_diffractions:
            num_diffractions[k] = 0
        if i == InteractionType.DIFFRACTION:
            num_diffractions[k] += 1

    # Number of diffraction per link should be 2
    # This is because the knife is made of two separate segments

    all_num_diffractions = np.array(list(num_diffractions.values()))
    assert np.all(all_num_diffractions == 2)

@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_diffuse_depth_high(Generator):
    """
    Test paths made only of diffuse reflection
    """
    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 0., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    max_depth = 10
    samples_per_src = int(1e4)
    max_num_paths = int(1e6)

    # Set material to metal which leads to all the energy being reflected
    # Scattering coefficient set to 0
    scene = load_box_scene("metal", 0.01, scattering_coefficient=1.)
    tracer = Generator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=False, refraction=False, specular_reflection=False,
                   diffuse_reflection=True, diffraction=False, edge_diffraction=False, seed=1)
    paths.shrink()

    int_types = paths.interaction_types.numpy()
    valid = paths.valid.numpy()

    # Depth should be set to `max_depth`
    assert paths.max_depth == max_depth

    # Number of paths should be max_depth*samples_per_src*num_links
    # if not constraints by max_num_paths
    assert paths.buffer_size == samples_per_src*max_depth*6

    # Check interaction types
    num_paths = paths.buffer_size
    paths_active = np.full([num_paths], True)
    for d in range(max_depth):
        inters = int_types[:,d]
        diffuse = np.equal(inters, InteractionType.DIFFUSE)
        no_int = np.equal(inters, InteractionType.NONE)

        # Interaction type is diffuse or none
        assert np.all(np.logical_or(diffuse, no_int))

        # If paths is done, there should be only none interaction
        assert np.all(np.logical_or(paths_active, no_int))

        # Update paths state
        paths_active = np.logical_and(paths_active, diffuse)

    # All paths should be valid, i.e., no candidates
    assert np.all(valid)

@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_diffuse_prefixes(Generator):
    """
    Check that, in the case of the box scene for which there is no occlusion,
    all all prefixes are listed as valid paths when there is only diffuse
    reflections
    """
    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 0., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    max_depth = 3
    samples_per_src = int(100)
    max_num_paths = int(1e4)

    # Set material to metal which leads to all the energy being reflected
    # Scattering coefficient set to 0
    scene = load_box_scene("metal", 0.01, scattering_coefficient=1.)
    tracer = Generator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=False, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, diffraction=True, edge_diffraction=True, seed=1)
    paths.shrink()

    primitives = paths.primitives.numpy()
    max_depth = paths.max_depth
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # All paths should be valid
    assert np.all(paths.valid.numpy())

    # All interaction types should be diffuse or None
    assert np.all(np.logical_or(
        np.equal(paths.interaction_types.numpy(), InteractionType.DIFFUSE),
        np.equal(paths.interaction_types.numpy(), InteractionType.NONE)
    ))

    # Test that all prefixes are listed.
    # First, aggregate all interaction for each link
    link_interactions = {}
    for src_ind, tgt_ind, prim_ind in zip(src_indices, tgt_indices, primitives):
        key = (src_ind, tgt_ind)
        if key not in link_interactions:
            link_interactions[key] = []
        link_interactions[key].append(prim_ind)
    # Check the prefixes for every link
    for src_ind in range(2):
        for tgt_ind in range(3):
            key = (src_ind, tgt_ind)
            primitives = link_interactions[key]
            primitives = np.stack(primitives, axis=0)

            num_paths = primitives.shape[0]
            for i in range(num_paths):
                # Extract the sequence of primitives forming this path
                path = primitives[i]

                # Compute the path depth
                for d in range(max_depth):
                    if path[d] == INVALID_PRIMITIVE:
                        break
                d = d+1
                # Nothing to check if depth is 1
                if d == 1:
                    continue

                # Remove last interaction to get a prefix
                prefix = path[:d-1]
                prefix = np.pad(prefix, [[0,max_depth-d+1]],
                                constant_values=INVALID_PRIMITIVE)

                # Check that the prefix is part of the found paths
                found = False
                for j in range(num_paths):
                    path_2 = primitives[j]
                    eq = np.sum(np.abs(path_2-prefix))
                    found = np.equal(eq, 0.)
                    if found:
                        break
                assert found

@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_diffuse_specular(Generator):
    """
    Check paths made of mixtures of specular and diffuse
    """
    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 0., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    # To compute correctly the probability of a diffuse or reflection event,
    # we need to average over the path depth to not be biased by the fact
    # that specular paths are not duplicated.
    max_depth = 200
    samples_per_src = int(1e2)
    max_num_paths = int(1e5)

    # Probabilty of an interaction to be specular
    ps = 0.7

    # Set material to metal which leads to all the energy being reflected
    # Scattering coefficient set to 0
    scene = load_box_scene("metal", 0.01, scattering_coefficient=np.sqrt(1.-ps))
    tracer = Generator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=False, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, diffraction=True, edge_diffraction=True, seed=1)
    paths.shrink()

    int_types = paths.interaction_types.numpy()
    valid = paths.valid.numpy()
    num_paths = paths.buffer_size

    # Depth should be set to `max_depth`
    assert paths.max_depth == max_depth

    # Check interaction types
    paths_active = np.full([num_paths], True)
    for d in range(max_depth):
        inters = int_types[:,d]
        specular = np.equal(inters, InteractionType.SPECULAR)
        diffuse = np.equal(inters, InteractionType.DIFFUSE)
        no_int = np.equal(inters, InteractionType.NONE)
        bounce = np.logical_or(specular,diffuse)

        # The ray bounced or the interaction is none
        # I.e., there should be no refraction or diffraction
        assert np.all(np.logical_or(bounce, no_int))

        # If paths is done, there should be only none interaction
        assert np.all(np.logical_or(paths_active, no_int))

        # Update paths state
        paths_active = np.logical_and(paths_active, bounce)

    # Check that the ratio of specular paths matches the configured
    # scattering coefficient.
    # Only the paths with depth = max_depth are used to get an unbiased
    # estimate
    long_paths_ind = np.where(paths_active)[0]
    long_types_chains = int_types[long_paths_ind]
    num_specular = 0
    for d in range(max_depth):
        inters = long_types_chains[:,d]
        specular = np.equal(inters, InteractionType.SPECULAR)

        num_specular += np.sum(specular)
    ratio_specular = num_specular/long_types_chains.shape[0]\
                                 /long_types_chains.shape[1]
    assert np.abs(ratio_specular-ps) < 0.01

    # Ensure that paths ending by a diffuse reflection are valid, whereas those
    # ending by a specular reflection are not
    for d in range(max_depth):
        specular = np.equal(int_types[:,d], InteractionType.SPECULAR)
        diffuse = np.equal(int_types[:,d], InteractionType.DIFFUSE)
        bounce = np.logical_or(specular,diffuse)

        # Indices of paths for which the previous bounce was
        # the last one
        if d == max_depth-1:
            last_bounce = bounce
        else:
            next_is_none = np.equal(int_types[:,d+1], InteractionType.NONE)
            last_bounce = np.logical_and(bounce, next_is_none)

        diff_ind = np.where(np.logical_and(diffuse,last_bounce))[0]
        spec_ind = np.where(np.logical_and(specular,last_bounce))[0]

        assert np.all(valid[diff_ind])
        assert np.all(np.logical_not(valid[spec_ind]))

@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_edge_diffraction_flag(Generator):
    """Test flag enabling/disabling edge diffraction
    """
    source = mi.Point3f(-3., -3., 3.5)
    target = mi.Point3f(3., 3., 3.0)

    max_depth = 4
    samples_per_src = 10000
    max_num_paths = 100000

    scene = load_box_one_screen_scene("metal", 0.01, 1.0)
    tracer = Generator()

    # Edge diffraction enabled
    paths = tracer(scene.mi_scene, source, target, samples_per_src, max_num_paths, max_depth,
                los=False, refraction=False, specular_reflection=False,
                diffuse_reflection=True, diffraction=True, edge_diffraction=True, seed=1)
    paths.shrink()
    interactions = paths.interaction_types.numpy()

    has_diffraction = np.any(np.equal(interactions, InteractionType.DIFFRACTION))
    assert has_diffraction

    # Edge diffraction disabled
    paths = tracer(scene.mi_scene, source, target, samples_per_src, max_num_paths, max_depth,
                los=False, refraction=False, specular_reflection=False,
                diffuse_reflection=True, diffraction=True, edge_diffraction=False, seed=1)
    paths.shrink()
    interactions = paths.interaction_types.numpy()

    has_diffraction = np.any(np.equal(interactions, InteractionType.DIFFRACTION))
    assert not has_diffraction

@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_no_diffuse_and_diffraction(Generator):
    """Test that no path contains both diffuse and diffraction"""

    # 2 sources
    sources = mi.Point3f([4.],
                        [1.],
                        [2.])

    # 3 targets
    targets = mi.Point3f([-4.],
                        [0.],
                        [3.])
    # 2x3 = 6 links

    max_depth = 10
    samples_per_src = int(1e4)
    max_num_paths = int(1e6)

    # Set material to metal which leads to all the energy being reflected
    # Scattering coefficient set to 0
    scene = load_box_box_two_screens_scene("metal", 0.01, scattering_coefficient=0.7)
    tracer = Generator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                    los=False, refraction=True, specular_reflection=True,
                    diffuse_reflection=True, diffraction=True, edge_diffraction=True, seed=1)
    paths.shrink()

    int_types = paths.interaction_types.numpy()

    has_diffraction = np.any(int_types == InteractionType.DIFFRACTION, axis=1)
    has_diffuse = np.any(int_types == InteractionType.DIFFUSE, axis=1)

    assert np.any(has_diffuse)
    assert np.any(has_diffraction)
    assert np.all(np.logical_not(np.logical_and(has_diffraction, has_diffuse)))

@pytest.mark.parametrize("los", [True, False])
@pytest.mark.parametrize("refraction", [True, False])
@pytest.mark.parametrize("specular_reflection", [True, False])
@pytest.mark.parametrize("diffuse_reflection", [True, False])
@pytest.mark.parametrize("diffraction", [True, False])
@pytest.mark.parametrize("Generator", GENERATOR_TYPES)
def test_interaction_type_flags(los, refraction, specular_reflection,
                                diffuse_reflection, diffraction, Generator):
    """
    Test flags enabling/disabling interaction types
    """

    source = mi.Point3f(-3., -3., 3.5)
    target = mi.Point3f(3., 3., 3.0)

    max_depth = 4
    samples_per_src = 1000
    max_num_paths = 10000

    scene = load_box_knife_scene("glass", 0.01, 0.7)
    tracer = Generator()

    paths = tracer(scene.mi_scene, source, target, samples_per_src, max_num_paths, max_depth,
                los=los, refraction=refraction, specular_reflection=specular_reflection,
                diffuse_reflection=diffuse_reflection, diffraction=diffraction,
                edge_diffraction=True, seed=1)
    paths.shrink()
    interactions = paths.interaction_types.numpy()

    print(f"{paths.paths_counter=}")
    print(f"{paths.buffer_size=}")
    print(f"{interactions=}")

    has_los = np.any(np.all(np.equal(interactions, InteractionType.NONE), axis=1))
    assert np.logical_not(np.bitwise_xor(los, has_los))

    has_specular = np.any(np.equal(interactions, InteractionType.SPECULAR))
    assert np.logical_not(np.bitwise_xor(specular_reflection, has_specular))

    has_refraction = np.any(np.equal(interactions, InteractionType.REFRACTION))
    assert np.logical_not(np.bitwise_xor(refraction, has_refraction))

    has_diffuse = np.any(np.equal(interactions, InteractionType.DIFFUSE))
    assert np.logical_not(np.bitwise_xor(diffuse_reflection, has_diffuse))

    has_diffraction = np.any(np.equal(interactions, InteractionType.DIFFRACTION))
    assert np.logical_not(np.bitwise_xor(diffraction, has_diffraction))


def test_diffraction_material_probability():
    """Test the conditional material probability of forced diffraction."""

    material = RadioMaterial(name="diffraction-probability",
                             relative_permittivity=5.0)

    si = mi.SurfaceInteraction3f()
    si.n = mi.Normal3f(0.0, 0.0, 1.0)
    si.sh_frame = mi.Frame3f(si.n)
    si.wi = dr.normalize(mi.Vector3f(0.0, -0.5, -1.0))
    si.dn_du = mi.Vector3f(1.0, 0.0, 0.0)
    si.dn_dv = mi.Vector3f(0.0, 1.0, 0.0)
    flags = mi.UInt(InteractionType.DIFFRACTION)
    si.dp_du = mi.Vector3f(
        1.0, 1.0, dr.reinterpret_array(mi.Float, flags))

    ctx = mi.BSDFContext(mode=mi.TransportMode.Importance,
                         type_mask=0, component=0)
    ctx.component |= InteractionType.DIFFRACTION
    ctx.component |= NO_JONES_MATRIX

    sample, _ = material.sample(ctx, si, mi.Float(0.5),
                                mi.Point2f(0.25, 0.5), True)
    pdf = material.pdf(ctx, si, sample.wo, True)

    assert sample.sampled_component[0] == InteractionType.DIFFRACTION
    assert sample.pdf[0] == 1.0
    assert pdf[0] == 1.0


@pytest.mark.parametrize(
    "specular_reflection, expected_probability",
    [(False, 1.0),
     (True, SBCandidateGenerator.DIFFRACTION_SAMPLING_PROBABILITY)])
def test_sampled_diffraction_path_probability(specular_reflection,
                                              expected_probability):
    """Test stored diffraction probabilities for forced and mixed sampling."""

    scene = load_box_one_screen_scene("metal", 0.01, 0.0)
    source = mi.Point3f(-3.0, -3.0, 3.5)
    target = mi.Point3f(3.0, 3.0, 3.0)

    paths = SBCandidateGenerator()(
        scene.mi_scene, source, target,
        samples_per_src=10_000,
        max_num_paths_per_src=100_000,
        max_depth=1,
        los=False,
        specular_reflection=specular_reflection,
        diffuse_reflection=False,
        refraction=False,
        diffraction=True,
        edge_diffraction=True,
        seed=1)
    paths.shrink()

    interactions = paths.interaction_types.numpy()
    diffraction = interactions == InteractionType.DIFFRACTION
    assert np.any(diffraction)
    probs = paths.probs.numpy()[diffraction]
    assert np.allclose(probs, expected_probability)
    # Diffraction compensation divides by these probabilities; they must stay
    # strictly positive and finite for every diffracted sample.
    assert np.all(np.isfinite(probs))
    assert np.all(probs > 0.0)


def test_material_zero_probability_when_no_interactions_enabled():
    """No enabled interaction types yields NONE with pdf 0 and a finite Jones matrix.

    The importance-sampling weight uses ``1/sqrt(pdf)``. Without a zero-pdf
    guard that weight is infinite and would poison radio-map fields when a ray
    hits a surface with no enabled interaction.
    """

    material = RadioMaterial(name="zero-probability",
                             relative_permittivity=5.0)

    si = mi.SurfaceInteraction3f()
    si.n = mi.Normal3f(0.0, 0.0, 1.0)
    si.sh_frame = mi.Frame3f(si.n)
    si.wi = dr.normalize(mi.Vector3f(0.0, -0.5, -1.0))
    si.t = mi.Float(1.0)
    si.duv_dx = mi.Vector2f(1.0, 0.0)
    si.duv_dy = mi.Vector2f(0.0, 0.0)
    # No specular / diffuse / refraction / diffraction locally enabled.
    flags = mi.UInt(0)
    si.dp_du = mi.Vector3f(
        1.0, 1.0, dr.reinterpret_array(mi.Float, flags))

    ctx = mi.BSDFContext(mode=mi.TransportMode.Importance,
                         type_mask=0, component=0)

    sample, jones = material.sample(ctx, si, mi.Float(0.5),
                                    mi.Point2f(0.25, 0.5), True)
    pdf = material.pdf(ctx, si, sample.wo, True)

    assert sample.sampled_component[0] == InteractionType.NONE
    assert sample.pdf[0] == 0.0
    assert pdf[0] == 0.0
    assert np.all(np.isfinite(jones.numpy()))
    assert np.allclose(jones.numpy(), 0.0)
