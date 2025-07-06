#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Rendering-related utilities.
"""
from __future__ import annotations

import drjit as dr
import matplotlib
import mitsuba as mi
import numpy as np

from sionna import rt
from sionna.rt.constants import INTERACTION_TYPE_TO_COLOR,\
                                InteractionType,\
                                LOS_COLOR
from sionna.rt.camera import Camera
from sionna.rt.utils.meshes import clone_mesh


def scene_scale(scene: rt.Scene):
    """
    Returns the "scale" of the scene, i.e., the diameter of the smallest
    sphere containing all the scene objects and centered at the center
    of the scene.

    Input
    ------
    scene : rt.Scene
        Scene to be measured.

    Output
    -------
    : float
        Scene size.
    """
    bbox = scene.mi_scene.bbox()
    sc = 2. * bbox.bounding_sphere().radius
    if np.isnan(sc):
        sc = 0.0
    return sc


def make_render_sensor(
    scene: rt.Scene,
    camera: str | rt.Camera | mi.ScalarTransform4f | mi.Sensor,
    resolution: tuple[int, int], fov: float
) -> mi.Sensor:
    r"""
    Instantiates a Mitsuba sensor (camera) from the provided ``camera`` object.

    Input
    ------
    scene : :class:`~sionna.rt.Scene`
        The scene

    camera : str | :class:`~sionna.rt.Camera` | :class:`~mitsuba.Sensor`
        The name of a camera registered in the scene, or a camera object
        instance, or a "camera to world" transform matrix.

    resolution : [2], int
        Size of the rendered figure.

    fov : float
        Field of view, in degrees.

    Output
    -------
    : :class:`~mitsuba.Sensor`
        A Mitsuba sensor (camera)
    """
    props = {
        'type': 'perspective',
    }

    if isinstance(camera, str):
        if camera == 'preview':
            # Use the viewpoint from the preview.
            w = scene._preview_widget  # pylint: disable=protected-access
            if w is None:
                raise RuntimeError("Could not find an open preview widget, "
                                   "please call `scene.preview()` first.")

            cam = w.camera
            props['to_world'] = mi.ScalarTransform4f().look_at(
                origin=cam.position,
                target=w.orbit.target,
                up=(0, 0, 1),
            )
            props['near_clip'] = cam.near
            props['far_clip'] = cam.far
            # This will get overriden below if the user provided an `fov` value.
            props['fov'] = cam.fov
            props['fov_axis'] = 'y'
            del w, cam

        else:
            cam_name = camera
            camera = scene.get(cam_name)
            if not isinstance(camera, Camera):
                raise ValueError(f"The scene has no camera named '{cam_name}'")

    if isinstance(camera, Camera):
        world_transform = camera.world_transform.matrix.numpy()[:,:,0]
        props['to_world'] = mi.ScalarTransform4f(world_transform)
        props['near_clip'] = 0.1
        props['far_clip'] = 10000

    elif isinstance(camera, mi.Sensor):
        # TODO: adopt more properties from the given sensor, e.g. sensor type
        #       (ortho, thinlens, etc).
        sensor_params = mi.traverse(camera)
        world_transform = camera.world_transform().matrix.numpy()
        props['to_world'] = mi.ScalarTransform4f(world_transform)
        props['near_clip'] = sensor_params['near_clip']
        props['far_clip'] = sensor_params['far_clip']

    elif isinstance(camera, mi.ScalarTransform4f):
        props['to_world'] = camera

    elif isinstance(camera, str):
        # Do nothing as this was already handled. This is to avoid wrongly
        # raising an exception
        pass

    else:
        raise ValueError(f'Unsupported camera type: {type(camera)}')

    if fov is not None:
        props['fov'] = fov
        props['fov_axis'] = 'x'
    props['film'] = {
        'type': 'hdrfilm',
        'width': resolution[0],
        'height': resolution[1],
        'pixel_format': 'rgba',
        'rfilter': {'type': 'box'},
    }
    return mi.load_dict(props)


def paths_to_segments(paths : rt.Paths):
    r"""
    Extracts the segments corresponding to a set of ``paths``

    Input
    -----
    paths : :class:`~sionna.rt.Paths`
        Paths to plot

    Output
    -------
    starts, ends : [n,3], float
        Endpoints of the segments making the paths.
    """

    vertices = paths.vertices.numpy()
    valid = paths.valid.numpy()
    types = paths.interactions.numpy()
    max_depth = vertices.shape[0]

    num_paths = vertices.shape[-2]
    if num_paths == 0:
        return # Nothing to do

    # Build sources and targets
    src_positions, tgt_positions = paths.sources, paths.targets
    src_positions = src_positions.numpy().T
    tgt_positions = tgt_positions.numpy().T

    num_src = src_positions.shape[0]
    num_tgt = tgt_positions.shape[0]

    # Merge device and antenna dimensions if required
    if not paths.synthetic_array:
        # The dimension corresponding to the number of antenna patterns
        # is removed as it is a duplicate
        num_rx = paths.num_rx
        rx_array_size = paths.rx_array.array_size
        num_rx_patterns = len(paths.rx_array.antenna_pattern.patterns)
        #
        num_tx = paths.num_tx
        tx_array_size = paths.tx_array.array_size
        num_tx_patterns = len(paths.tx_array.antenna_pattern.patterns)
        #
        vertices = np.reshape(vertices, [max_depth,
                                            num_rx,
                                            num_rx_patterns,
                                            rx_array_size,
                                            num_tx,
                                            num_tx_patterns,
                                            tx_array_size,
                                            -1,
                                            3])
        valid = np.reshape(valid, [num_rx,
                                    num_rx_patterns,
                                    rx_array_size,
                                    num_tx,
                                    num_tx_patterns,
                                    tx_array_size,
                                    -1])
        types = np.reshape(types, [max_depth,
                                    num_rx,
                                    num_rx_patterns,
                                    rx_array_size,
                                    num_tx,
                                    num_tx_patterns,
                                    tx_array_size,
                                    -1])
        vertices = vertices[:,:,0,:,:,0,:,:,:]
        types = types[:,:,0,:,:,0,:,:]
        valid = valid[:,0,:,:,0,:,:]
        vertices = np.reshape(vertices, [max_depth,
                                            num_tgt,
                                            num_src,
                                            -1,
                                            3])
        valid = np.reshape(valid, [num_tgt,
                                    num_src,
                                    -1])
        types = np.reshape(types, [max_depth,
                                    num_tgt,
                                    num_src,
                                    -1])

    # Emit directly two lists of the beginnings and endings of line segments
    starts = []
    ends = []
    colors = []
    for rx in range(num_tgt): # For each receiver
        for tx in range(num_src): # For each transmitter
            for p in range(num_paths): # For each path
                if not valid[rx, tx, p]:
                    continue
                start = src_positions[tx]
                i = 0
                color = LOS_COLOR
                while i < max_depth:
                    t = types[i, rx, tx, p]
                    if t == InteractionType.NONE:
                        break
                    end = vertices[i, rx, tx, p]
                    starts.append(start)
                    ends.append(end)
                    colors.append(color)
                    start = end
                    color = INTERACTION_TYPE_TO_COLOR[t]
                    i += 1
                # Explicitly add the path endpoint
                starts.append(start)
                ends.append(tgt_positions[rx])
                colors.append(color)

    return starts, ends, colors


def unmultiply_alpha(arr: np.ndarray):
    """
    Un-multiply the alpha channel.

    Input
    -----
    arr : [w,h,4]
        An image

    Output
    -------
    arr : [w,h,4]
        Image with the alpha channel de-modulated (divided out).
    """
    arr = arr.copy()
    alpha = arr[:, :, 3]
    weight = 1. / np.where(alpha > 0, alpha, 1.)
    arr[:, :, :3] *= weight[:, :, None]
    return arr


def twosided_diffuse(color: mi.Color3f | list[float]) -> mi.BSDF:
    return mi.load_dict({
        "type": "twosided",
        "nested": {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": list(color)},
        }
    })


def radio_map_to_emissive_shape(radio_map: rt.RadioMap, tx: int | None,
                                db_scale: bool = True,
                                vmin: float | None = None,
                                vmax: float | None = None,
                                rm_metric: str = "path_gain",
                                viewpoint: mi.Vector3f | None = None):
    """
    Given a pre-computed Radio Map, create a Mitsuba shape associated with a
    color-mapped emitter in order to visualize it.
    """

    # Resample values from cell centers to cell corners
    rm_values = radio_map.transmitter_radio_map(metric=rm_metric, tx=tx).numpy()
    # Ensure that dBm is correctly computed for RSS
    if rm_metric=="rss" and db_scale:
        rm_values *= 1000

    if isinstance(radio_map, rt.PlanarRadioMap):
        rm_values = resample_to_corners(rm_values)

        texture, opacity = radio_map_texture(
            rm_values, db_scale=db_scale, vmin=vmin, vmax=vmax)
        bsdf = {
            'type': 'mask',
            'opacity': {
                'type': 'bitmap',
                'bitmap': mi.Bitmap(opacity.astype(np.float32)),
                "filter_type": "nearest"
            },
            'nested': {
                'type': 'diffuse',
                'reflectance': 0.,
            },
        }

        emitter = {
            'type': 'area',
            'radiance': {
                'type': 'bitmap',
                'bitmap': mi.Bitmap(texture.astype(np.float32)),
                "filter_type": "nearest"
            },
        }

        # Pose of the measurement plane (ensuring it's a CPU-side value)
        matrix_numpy = radio_map.to_world.matrix.numpy().squeeze()
        to_world = mi.ScalarTransform4f(matrix_numpy)

        flip_normal = False
        if viewpoint is not None:
            viewpoint = mi.ScalarPoint3f(viewpoint.numpy().squeeze())

            # Area emitters are single-sided, so we need to flip the rectangle's
            # normals if the camera is on the wrong side.
            p0 = to_world.transform_affine([-1, -1, 0])
            p1 = to_world.transform_affine([-1, 0, 0])
            p2 = to_world.transform_affine([0, -1, 0])
            plane_center = to_world.transform_affine([0, 0, 0])
            normal = dr.cross(p1 - p0, p2 - p0)
            flip_normal = dr.dot(plane_center - viewpoint, normal) < 0

        return {
            'type': 'rectangle',
            'flip_normals': flip_normal,
            'to_world': to_world,
            'bsdf': bsdf,
            'emitter': emitter,
        }

    elif isinstance(radio_map, rt.DemRadioMap):
        # rm_values has per-cell values that need to get applied as a texture.
        # This works because triangulate_elevation() also assigned texcoords.
        texture, opacity = radio_map_texture(
            rm_values, db_scale=db_scale, vmin=vmin, vmax=vmax)
        bsdf = {
            'type': 'mask',
            'opacity': {
                'type': 'bitmap',
                'bitmap': mi.Bitmap(opacity.astype(np.float32)),
                "filter_type": "nearest"
            },
            'nested': {
                'type': 'diffuse',
                'reflectance': 0.,
            },
        }

        emitter = {
            'type': 'area',
            'radiance': {
                'type': 'bitmap',
                'bitmap': mi.Bitmap(texture.astype(np.float32)),
                "filter_type": "nearest"
            },
        }

        props = mi.Properties()
        props['bsdf'] = mi.load_dict(bsdf)
        props['emitter'] = mi.load_dict(emitter)
        cloned_shape = clone_mesh(radio_map.measurement_surface, props=props)
        return cloned_shape

    elif isinstance(radio_map, rt.MeshRadioMap):
        # rm_values has per-triangle values for the requested tx and metric.
        texture, opacity = radio_map_texture(
            rm_values, db_scale=db_scale, vmin=vmin, vmax=vmax)

        measurement_surface = radio_map.measurement_surface
        # The measurement surface is not actually part of the scene, so we
        # let it be transparent where there's no coverage, just like for
        # planar radio maps.
        bsdf = {
            'type': 'mask',
            'opacity': {
                'type': 'mesh_attribute',
                "name": "face_opacity",
            },
            'nested': {
                'type': 'diffuse',
                'reflectance': 0.,
            },
        }

        emitter = {
            # Special two-sided area emitter because we cannot really guess
            # the orientation of the measurement surface.
            'type': 'twosided_area',
            'nested': {
                'type': 'area',
                'radiance': {
                    'type': 'mesh_attribute',
                    "name": "face_rm_values",
                },
            },
        }

        props = mi.Properties()
        props['bsdf'] = mi.load_dict(bsdf)
        props['emitter'] = mi.load_dict(emitter)
        # TODO: this doesn't preserve e.g. the `flip_normals` property :(
        cloned_shape = clone_mesh(measurement_surface, props=props)
        cloned_shape.add_attribute("face_opacity", 1, opacity)
        cloned_shape.add_attribute("face_rm_values", 3, texture.ravel())

        return cloned_shape

    else:
        raise ValueError(f"Unsupported RadioMap type: {type(radio_map)}")


def resample_to_corners(values: np.ndarray) -> np.ndarray:
    """
    Takes a 2D NumPy array of values defined at cell centers and converts it
    into an array (with one more row and one more column) with values defined at
    cell corners.
    """
    assert values.ndim == 2
    padded = np.pad(values, pad_width=((1, 1), (1, 1)), mode='edge')
    return 0.25 * (
          padded[ :-1,  :-1]
        + padded[1:  ,  :-1]
        + padded[ :-1, 1:  ]
        + padded[1:  , 1:  ]
    )


def radio_map_texture(
    rm_values: np.ndarray, db_scale: bool = True,
    vmin: float | None = None, vmax: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    # Leave zero-valued regions as transparent
    valid = rm_values > 0.
    opacity = valid.astype(np.float32)

    # Color mapping of real values
    rm_values, normalizer, color_map = radio_map_color_mapping(
        rm_values, db_scale=db_scale, vmin=vmin, vmax=vmax)
    texture = color_map(normalizer(rm_values))
    # Eliminate alpha channel
    texture = texture[..., :3]
    # Colors from the color map are gamma-compressed, go back to linear
    texture = np.power(texture, 2.2)

    # Pre-multiply alpha to avoid fringe
    texture *= opacity[..., None]

    return texture, opacity


def radio_map_color_mapping(radio_map: np.ndarray,
                            db_scale: bool = True,
                            vmin: float | None = None,
                            vmax: float | None = None):
    """
    Prepare a Matplotlib color maps and normalizing helper based on the
    requested value scale to be displayed.
    Also applies the dB scaling to a copy of the radio map, if requested.
    """
    valid = np.logical_and(radio_map > 0., np.isfinite(radio_map))
    any_valid = np.any(valid)
    radio_map = radio_map.copy()
    if db_scale:
        radio_map[valid] = 10. * np.log10(radio_map[valid])
    else:
        radio_map[valid] = radio_map[valid]

    if vmin is None:
        vmin = radio_map[valid].min() if any_valid else 0
    if vmax is None:
        vmax = radio_map[valid].max() if any_valid else 0
    normalizer = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    color_map = matplotlib.colormaps.get_cmap('viridis')
    return radio_map, normalizer, color_map
