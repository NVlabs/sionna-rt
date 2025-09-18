#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Previewer of Sionna RT"""

import drjit as dr
import mitsuba as mi
import numpy as np
from ipywidgets import widgets
from ipywidgets.embed import embed_snippet
import pythreejs as p3s
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display
import matplotlib as mpl

from .constants import InteractionType, LOS_COLOR, SPECULAR_COLOR, \
    DIFFUSE_COLOR, REFRACTION_COLOR, INTERACTION_TYPE_TO_COLOR,\
        DEFAULT_TRANSMITTER_COLOR, DEFAULT_RECEIVER_COLOR
from .utils import rotation_matrix, scene_scale


class Previewer:
    """
    Interactive preview widget using `pythreejs` for visualizing the
    scene, radio devices, paths, and coverage maps.

    Input
    ------
    scene : :class:`rt.Scene`
        Scene to preview

    resolution : [2], int
        Size of the viewer figure.
        Defaults to (655,500).

    fov : float
        Field of view, in degrees.
        Defautls to 45 degrees.

    background : str
        Background color in hex format prefixed by '#'.
        Defaults to '#87CEEB'.
    """

    def __init__(self, scene, resolution=(655,500), fov=45.,
                 background='white'):

        self._scene = scene
        self._disk_sprite = None

        # List of objects in the scene
        self._objects = []
        # Bounding box of the scene
        self._bbox = mi.ScalarBoundingBox3f()

        ####################################################
        # Setup the viewer
        ####################################################

        # Lighting
        ambient_light = p3s.AmbientLight(intensity=0.80)
        camera_light = p3s.DirectionalLight(
            position=[0, 0, 0], intensity=0.25
        )

        # Camera & controls
        self._camera = p3s.PerspectiveCamera(
            fov=fov, aspect=resolution[0]/resolution[1],
            up=[0, 0, 1], far=10000,
            children=[camera_light],
        )
        self._orbit = p3s.OrbitControls(
            controlling = self._camera
        )

        # Scene & renderer
        self._p3s_scene = p3s.Scene(
            background=background, children=[self._camera, ambient_light]
        )
        self._renderer = p3s.Renderer(
            scene=self._p3s_scene, camera=self._camera, controls=[self._orbit],
            width=resolution[0], height=resolution[1], antialias=True
        )

        ####################################################
        # Plot the scene geometry
        ####################################################
        self.plot_scene()

        # Finally, ensure the camera is looking at the scene
        self.center_view()

    def reset(self):
        """
        Removes objects that are not flagged as persistent
        """
        remaining = []
        for obj, persist in self._objects:
            if persist:
                remaining.append((obj, persist))
            else:
                self._p3s_scene.remove(obj)
        self._objects = remaining

    def redraw_scene_geometry(self):
        """
        Redraw the scene geometry
        """
        remaining = []
        for obj, persist in self._objects:
            if not persist: # Only scene objects are flagged as persistent
                remaining.append((obj, persist))
            else:
                self._p3s_scene.remove(obj)
        self._objects = remaining

        # Plot the scene geometry
        self.plot_scene()

    def center_view(self):
        """
        Automatically place the camera such that it is located at spherical
        coordinates (r = scene_scale*3/2, theta=pi/4, phi=pi/4), and oriented
        toward the center of the scene.
        """
        bbox = self._bbox if self._bbox.valid() else mi.ScalarBoundingBox3f(0.)
        center = bbox.center()

        sc = scene_scale(self._scene)
        if sc == 0.:
            sc = 1.
        r = sc*1.5
        theta = np.pi*0.25
        phi = np.pi*0.25
        position = (np.sin(theta) * np.cos(phi) * r + center.x,
                    np.sin(theta) * np.sin(phi) * r + center.y,
                    np.cos(theta) * r + center.z)
        self._camera.position = tuple(position)

        self._camera.lookAt(center)
        self._orbit.exec_three_obj_method('update')
        self._camera.exec_three_obj_method('updateProjectionMatrix')

    def plot_radio_devices(self, show_orientations=False):
        """
        Plots the radio devices

        If `show_orientations` is set to `True`, the orientation of each device
        is shown using an arrow.

        Input
        ------
        show_orientations : bool
            If set to `True`, the orientation of the radio device is shown using
            an arrow. Defaults to `False`.
        """
        scene = self._scene
        sc = scene_scale(scene)
        # If scene is empty, set the scene scale to 1
        if sc == 0.:
            sc = 1.

        tx_positions = [tx.position.numpy().T[0]
                        for tx in scene.transmitters.values()]
        rx_positions = [rx.position.numpy().T[0]
                        for rx in scene.receivers.values()]

        sources_colors = [tx.color for tx in scene.transmitters.values()]
        target_colors = [rx.color for rx in scene.receivers.values()]

        # Radio emitters, shown as points
        p = np.array(list(tx_positions) + list(rx_positions))
        # Stop here if no radio devices to plot
        if p.shape[0] == 0:
            return
        albedo = np.array(sources_colors + target_colors)

        # Expand the bounding box to include radio devices
        pmin = np.min(p, axis=0)
        pmax = np.max(p, axis=0)
        self._bbox.expand(pmin)
        self._bbox.expand(pmax)

        # Radio devices are not persistent.
        default_radius = max(0.005 * sc, 1)
        only_default = True
        radii = []
        for devices in scene.transmitters, scene.receivers:
            for rd in devices.values():
                r = rd.display_radius
                if r is not None:
                    radii.append(r)
                    only_default = False
                else:
                    radii.append(default_radius)

        if only_default:
            self._plot_points(p, persist=False, colors=albedo,
                              radius=default_radius)
        else:
            # Since we can only have one radius value per draw call,
            # we group the points to plot by radius.
            unique_radii, mapping = np.unique(radii, return_inverse=True)
            for i, r in enumerate(unique_radii):
                mask = mapping == i
                self._plot_points(p[mask], persist=False, colors=albedo[mask],
                                  radius=r)

        if show_orientations:
            line_length = 0.05 * sc
            head_length = 0.05 * line_length
            zeros = np.zeros((3,))

            for devices in [scene.transmitters.values(),
                            scene.receivers.values()]:
                if len(devices) == 0:
                    continue
                starts, ends, colors = [], [], []
                for rd in devices:
                    # Arrow line
                    color = f'rgb({", ".join([str(int(v)) for v in rd.color])})'
                    starts.append(rd.position.numpy()[:,0])
                    rot_mat = rotation_matrix(rd.orientation)
                    local_endpoint = mi.Point3f(line_length, 0.0, 0.0)
                    endpoint = rd.position + rot_mat@local_endpoint
                    endpoint = endpoint.numpy()[:,0]
                    ends.append(endpoint)
                    colors.append([rd.color[0], rd.color[1], rd.color[2]])

                    geo = p3s.CylinderGeometry(
                        radiusTop=0, radiusBottom=0.3 * head_length,
                        height=head_length, radialSegments=8,
                        heightSegments=0, openEnded=False)
                    mat = p3s.MeshLambertMaterial(color=color)
                    mesh = p3s.Mesh(geo, mat)
                    mesh.position = (endpoint[0], endpoint[1], endpoint[2])
                    angles = rd.orientation.numpy()[:,0]
                    mesh.rotateZ(angles[2] - np.pi/2)
                    mesh.rotateY(angles[0])
                    mesh.rotateX(-angles[1])
                    self._add_child(mesh, zeros, zeros, persist=False)

                self._plot_lines(np.array(starts), np.array(ends),
                                 width=2, colors=colors)

    def plot_paths(self, paths, line_width=1.0):
        """
        Plot the ``paths``.

        Input
        -----
        paths : :class:`~rt.Paths`
            Paths to plot

        line_width : float
            Width of the lines.
            Defaults to 0.8.
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

        self._plot_lines(np.vstack(starts), np.vstack(ends),
                         np.vstack(colors), line_width)

    def plot_planar_radio_map(self, radio_map, tx=0, db_scale=True,
                              vmin=None, vmax=None, metric="path_gain"):
        """
        Plot the coverage map as a textured rectangle in the scene. Regions
        where the coverage map is zero-valued are made transparent.
        """
        to_world = radio_map.to_world

        tensor = radio_map.transmitter_radio_map(metric, tx)
        tensor = tensor.numpy()

        # Mask for discarding empty cells
        non_zero_mask = tensor > 0.
        if not np.any(non_zero_mask):
            return

        # Create a rectangle from two triangles
        p00 = to_world.transform_affine([-1, -1, 0]).numpy().T[0]
        p01 = to_world.transform_affine([1, -1, 0]).numpy().T[0]
        p10 = to_world.transform_affine([-1, 1, 0]).numpy().T[0]
        p11 = to_world.transform_affine([1, 1, 0]).numpy().T[0]

        vertices = np.array([p00, p01, p10, p11])
        pmin = np.min(vertices, axis=0)
        pmax = np.max(vertices, axis=0)

        faces = np.array([
            [0, 1, 2],
            [2, 1, 3],
        ], dtype=np.uint32)

        vertex_uvs = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1]
        ], dtype=np.float32)

        geo = p3s.BufferGeometry(
            attributes={
                'position': p3s.BufferAttribute(vertices, normalized=False),
                'index': p3s.BufferAttribute(faces.ravel(), normalized=False),
                'uv': p3s.BufferAttribute(vertex_uvs, normalized=False),
            }
        )

        to_map, normalizer, color_map = self._coverage_map_color_mapping(
            tensor, db_scale=db_scale, vmin=vmin, vmax=vmax)
        texture = color_map(normalizer(to_map)).astype(np.float32)
        texture[:, :, 3] = non_zero_mask.astype(np.float32)
        # Pre-multiply alpha
        texture[:, :, :3] *= texture[:, :, 3, None]

        texture = p3s.DataTexture(
            data=(255. * texture).astype(np.uint8),
            format='RGBAFormat',
            type='UnsignedByteType',
            magFilter='NearestFilter',
            minFilter='NearestFilter',
        )

        mat = p3s.MeshLambertMaterial(
            side='DoubleSide',
            map=texture, transparent=True,
        )
        mesh = p3s.Mesh(geo, mat)

        self._add_child(mesh, pmin, pmax, persist=False)


    def plot_mesh_radio_map(self, radio_map, tx=0, db_scale=True,
                            vmin=None, vmax=None, metric="path_gain"):
        """
        Plots the mesh radio map
        """
        s = radio_map.measurement_surface

        # Radio map
        tensor = radio_map.transmitter_radio_map(metric, tx)
        tensor = tensor.numpy()

        # Mask for discarding empty cells
        non_zero_mask = tensor > 0.
        if not np.any(non_zero_mask):
            return

        # Mesh geometry
        n_vertices = s.vertex_count()
        vertices = s.vertex_position(dr.arange(mi.UInt32, n_vertices))
        vertices = np.transpose(vertices.numpy())

        faces = s.face_indices(dr.arange(mi.UInt32, s.face_count()))
        faces = np.transpose(faces.numpy())
        faces = faces[non_zero_mask]
        vertices = vertices[faces]
        vertices = np.reshape(vertices, (-1, 3))

        pmin = np.min(vertices, axis=0)
        pmax = np.max(vertices, axis=0)

        # Mesh color from the radio map
        to_map, normalizer, color_map = self._coverage_map_color_mapping(
            tensor, db_scale=db_scale, vmin=vmin, vmax=vmax)
        colors = color_map(normalizer(to_map)).astype(np.float32)
        colors = colors[:,:3]
        colors = colors[non_zero_mask]
        colors = np.repeat(colors, 3, axis=0)

        geometry = p3s.BufferGeometry(
            attributes={
                'position': p3s.BufferAttribute(vertices, normalized=False),
                'color': p3s.BufferAttribute(colors, normalized=False),
            }
        )

        material = p3s.MeshBasicMaterial(vertexColors='VertexColors',
                                         side='DoubleSide')
        mesh = p3s.Mesh(geometry, material)
        self._add_child(mesh, pmin, pmax, persist=False)

    def plot_scene(self):
        """
        Plots the meshes that make the scene
        """
        objects = self._scene.objects.values()
        n = len(objects)
        if n <= 0:
            return

        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = mi.Vector3f(0, 0, 1)

        # Shapes (e.g. buildings)
        vertices, faces, albedos = [], [], []
        f_offset = 0
        for s in objects:

            s = s.mi_mesh

            null_transmission = s.bsdf().eval_null_transmission(si).numpy()
            if np.min(null_transmission) > 0.99:
                # The BSDF for this shape was probably set to `null`, do not
                # include it in the scene preview.
                continue

            n_vertices = s.vertex_count()
            v = s.vertex_position(dr.arange(mi.UInt32, n_vertices))
            v = np.transpose(v.numpy())
            vertices.append(v)

            f = s.face_indices(dr.arange(mi.UInt32, s.face_count()))
            f = np.transpose(f.numpy())
            faces.append(f + f_offset)
            f_offset += n_vertices

            albedo = np.array(s.bsdf().radio_material.color)

            albedos.append(np.tile(albedo, (n_vertices, 1)))

        # Plot all objects as a single PyThreeJS mesh, which is must faster
        # than creating individual mesh objects in large scenes.
        vertices = np.concatenate(vertices, axis=0)
        faces = np.concatenate(faces, axis=0)
        albedos = np.concatenate(albedos, axis=0)
        self._plot_mesh(vertices,
                        faces,
                        persist=True, # The scene geometry is persistent
                        colors=albedos)

    def set_clipping_plane(self, offset, orientation):
        """
        Set a plane such that the scene preview is clipped (cut) by that plane.

        The clipping plane has normal orientation ``clip_plane_orientation`` and
        offset ``offset``. This allows, e.g., visualizing the interior of meshes
        such as buildings.

        Input
        -----
        offset : float
            Offset to position the plane

        clip_plane_orientation : tuple[float, float, float]
            Normal vector of the clipping plane
        """

        if offset is None:
            self._renderer.localClippingEnabled = False
            self._renderer.clippingPlanes = []
        else:
            self._renderer.localClippingEnabled = True
            self._renderer.clippingPlanes = [p3s.Plane(orientation, offset)]

    def show_legend(self, show_paths, show_devices):
        r"""
        Display the legend
        """

        def circular_item(color):
            r = int(color[0]*255)
            g = int(color[1]*255)
            b = int(color[2]*255)
            s = f"background-color: rgb({r},{g},{b}); width: 20px; height:" + \
                 " 20px; border-radius: 50%; display: inline-block;"
            return s

        def segment_item(color):
            r = int(color[0]*255)
            g = int(color[1]*255)
            b = int(color[2]*255)
            s = f"background-color: rgb({r},{g},{b}); width: 20px; height:" + \
                 " 2px; display: inline-block;"
            return s

        # Create a legend
        legend_items = []
        if show_paths:
            legend_items += [
                ("Line-of-sight", segment_item(LOS_COLOR)),
                ("Specular reflection", segment_item(SPECULAR_COLOR)),
                ("Diffuse reflection", segment_item(DIFFUSE_COLOR)),
                ("Refraction", segment_item(REFRACTION_COLOR))]
        if show_devices:
            legend_items += [
                ("Transmitter", circular_item(DEFAULT_TRANSMITTER_COLOR)),
                ("Receiver", circular_item(DEFAULT_RECEIVER_COLOR))]

        legend_labels = [widgets.HTML(
            value=f"<div style='{style}'></div> {label}")
                         for label, style in legend_items]
        legend = widgets.VBox(legend_labels)

        # Check if we are in a IPython interactive shell
        if InteractiveShell.initialized():
            # Display the renderer and legend together
            display(widgets.HBox([self._renderer, legend]))

    ##################################################
    # Accessors
    ##################################################

    @property
    def resolution(self) -> tuple[int, int]:
        """
        (float, float) : Rendering resolution `(width, height)`
        """
        return (self._renderer.width, self._renderer.height)

    @property
    def camera(self) -> p3s.PerspectiveCamera:
        return self._camera

    @property
    def orbit(self) -> p3s.OrbitControls:
        return self._orbit

    ##################################################
    # Internal methods
    ##################################################

    def _scene_scale(self):
        """
        Returns the size of the scene, i.e., the diameter of the smallest
        sphere containing all the scene objects and centered at the center
        of the scene

        Output
        -------
        : float
            Scene size
        """
        bbox = self._scene.mi_scene.bbox()

        sc = 2. * bbox.bounding_sphere().radius
        return sc

    def _plot_mesh(self, vertices, faces, persist, colors=None):
        """
        Plots a mesh.

        Input
        ------
        vertices : [n,3], float
            Position of the vertices

        faces : [n,3], int
            Indices of the triangles associated with ``vertices``

        persist : bool
            Flag indicating if the mesh is persistent, i.e., should not be
            erased when ``reset()`` is called.

        colors : [n,3] | [3] | None
            Colors of the vertices. If `None`, black is used.
            Defaults to `None`.
        """
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3
        n_v = vertices.shape[0]
        pmin, pmax = np.min(vertices, axis=0), np.max(vertices, axis=0)

        # Assuming per-vertex colors
        if colors is None:
            # Black is default
            colors = np.zeros((n_v, 3), dtype=np.float32)
        elif colors.ndim == 1:
            colors = np.tile(colors[None, :], (n_v, 1))
        colors = colors.astype(np.float32)
        assert ( (colors.ndim == 2)
             and (colors.shape[1] == 3)
             and (colors.shape[0] == n_v) )

        # Closer match to Mitsuba and Blender
        colors = np.power(colors, 1/1.8)

        geo = p3s.BufferGeometry(
            attributes={
                'index': p3s.BufferAttribute(faces.ravel(), normalized=False),
                'position': p3s.BufferAttribute(vertices, normalized=False),
                'color': p3s.BufferAttribute(colors, normalized=False)
            }
        )

        mat = p3s.MeshStandardMaterial(
            side='DoubleSide', metalness=0., roughness=1.0,
            vertexColors='VertexColors', flatShading=True,
        )
        mesh = p3s.Mesh(geo, mat)
        self._add_child(mesh, pmin, pmax, persist=persist)

    def _plot_points(self, points: np.ndarray, persist: bool,
                     colors: np.ndarray | None = None,
                     radius: float = 0.05):
        """
        Plots a set of `n` points.

        Input
        -------
        points : [n, 3], float
            Coordinates of the `n` points.

        persist : bool
            Indicates if the points are persistent, i.e., should not be erased
            when ``reset()`` is called.

        colors : [n, 3], float | [3], float | None
            Colors of the points.

        radius : float
            Radius of the points.
        """
        assert points.ndim == 2 and points.shape[1] == 3
        n = points.shape[0]
        pmin, pmax = np.min(points, axis=0), np.max(points, axis=0)

        # Assuming per-vertex colors
        if colors is None:
            colors = np.zeros((n, 3), dtype=np.float32)
        elif colors.ndim == 1:
            colors = np.tile(colors[None, :], (n, 1))
        colors = colors.astype(np.float32)
        assert ( (colors.ndim == 2)
             and (colors.shape[1] == 3)
             and (colors.shape[0] == n) )

        tex = p3s.DataTexture(data=self._get_disk_sprite(), format="RGBAFormat",
                              type="FloatType")

        points = points.astype(np.float32)
        geo = p3s.BufferGeometry(attributes={
            'position': p3s.BufferAttribute(points, normalized=False),
            'color': p3s.BufferAttribute(colors, normalized=False),
        })
        mat = p3s.PointsMaterial(
            size=2 * radius, sizeAttenuation=True, vertexColors='VertexColors',
            map=tex, alphaTest=0.5, transparent=True,
        )
        mesh = p3s.Points(geo, mat)
        self._add_child(mesh, pmin, pmax, persist=persist)

    def _add_child(self, obj, pmin, pmax, persist):
        """
        Adds an object for display

        Input
        ------
        obj : :class:`~pythreejs.Mesh`
            Mesh to display

        pmin : [3], float
            Lowest position for the bounding box

        pmax : [3], float
            Highest position for the bounding box

        persist : bool
            Flag that indicates if the object is persistent, i.e., if it should
            be removed from the display when `reset()` is called.
        """
        self._objects.append((obj, persist))
        self._p3s_scene.add(obj)

        self._bbox.expand(pmin)
        self._bbox.expand(pmax)

    def _plot_lines(self, starts, ends, colors, width):
        """
        Plots a set of `n` lines. This is used to plot the paths.

        Input
        ------
        starts : [n, 3], float
            Coordinates of the lines starting points

        ends : [n, 3], float
            Coordinates of the lines ending points

        color : str
            Color of the lines.

        width : float
            Width of the lines.
        """

        assert starts.ndim == 2 and starts.shape[1] == 3
        assert ends.ndim == 2 and ends.shape[1] == 3
        assert starts.shape[0] == ends.shape[0]

        segments = np.hstack((starts, ends)).astype(np.float32).reshape(-1,2,3)
        pmin = np.min(segments, axis=(0, 1))
        pmax = np.max(segments, axis=(0, 1))

        colors = np.hstack((colors, colors)).astype(np.float32).reshape(-1,2,3)
        geo = p3s.LineSegmentsGeometry(positions=segments, colors=colors)
        mat = p3s.LineMaterial(linewidth=width, vertexColors='VertexColors')
        mesh = p3s.LineSegments2(geo, mat)

        # Lines are not flagged as persistent as they correspond to paths, which
        # can changes from one display to the next.
        self._add_child(mesh, pmin, pmax, persist=False)

    def _get_disk_sprite(self):
        """
        Returns the sprite used to represent sources and targets though
        ``_plot_points()``.

        Output
        ------
        : [n,n,4], float
            Sprite
        """
        if self._disk_sprite is not None:
            return self._disk_sprite

        n = 128
        sprite = np.ones((n, n, 4))
        sprite[:, :, 3] = 0.
        # Draw a disk with an empty circle close to the edge
        ij = np.mgrid[:n, :n]
        ij = ij.reshape(2, -1)

        p = (ij + 0.5) / n - 0.5
        t = np.linalg.norm(p, axis=0).reshape((n, n))
        inside = t < 0.48
        in_band = (t < 0.45) & (t > 0.42)
        sprite[inside & (~in_band), 3] = 1.0

        sprite = sprite.astype(np.float32)
        self._disk_sprite = sprite
        return sprite

    # The following methods are required for
    # integration in Jupyter notebooks

    # pylint: disable=unused-argument
    def _repr_mimebundle_(self, **kwargs):
        # pylint: disable=protected-access,not-callable
        bundle = self._renderer._repr_mimebundle_()
        assert 'text/html' not in bundle
        bundle['text/html'] = self._repr_html_()
        return bundle

    def _repr_html_(self):
        """
        Standalone HTML display, i.e. outside of an interactive Jupyter
        notebook environment.
        """

        html = embed_snippet(self._renderer, requirejs=True)
        return html

    def _coverage_map_color_mapping(self, coverage_map, db_scale=True,
                                    vmin=None, vmax=None):
        """
        Prepare a Matplotlib color maps and normalizing helper based on the
        requested value scale to be displayed.
        Also applies the dB scaling to a copy of the coverage map, if requested.
        """
        valid = np.logical_and(coverage_map > 0., np.isfinite(coverage_map))
        coverage_map = coverage_map.copy()
        if db_scale:
            coverage_map[valid] = 10. * np.log10(coverage_map[valid])
        else:
            coverage_map[valid] = coverage_map[valid]

        if vmin is None:
            vmin = coverage_map[valid].min()
        if vmax is None:
            vmax = coverage_map[valid].max()
        normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        color_map = mpl.colormaps.get_cmap('viridis')
        return coverage_map, normalizer, color_map
