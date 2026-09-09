#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Class representing objects in the scene"""

from __future__ import annotations

import weakref

import drjit as dr
import mitsuba as mi
import os
from .utils import look_at_orientation, rotation_matrix
from .utils.meshes import clone_mesh, load_mesh, remove_mesh_duplicate_vertices, build_mesh_directed_edges
from .radio_materials import RadioMaterialBase
from . import scene as scene_module
from sionna.rt import RadioDevice


class SceneObject:
    # pylint: disable=line-too-long
    r"""
    Class implementing a scene object

    Scene objects can be either created from an existing Mitsuba shape
    or by loading a mesh from a file. In the latter case, a name
    and radio material for the scene object must be provided.

    To create a scene object from a mesh file, use the following approach:

    .. code-block:: python

        obj = SceneObject(fname=sionna.rt.scene.sphere,
                        name="sphere",
                        radio_material=ITURadioMaterial(name="sphere-material",
                                                        itu_type="metal",
                                                        thickness=0.01))

    To instantiate a scene object using a Mitsuba shape, follow these steps:

    .. code-block:: python

        mesh = load_mesh(sionna.rt.scene.sphere)
        obj = SceneObject(mi_mesh=mesh,
                          name="sphere",
                          radio_material=ITURadioMaterial(name="sphere-material",
                                                        itu_type="metal",
                                                        thickness=0.01))

    :param mi_mesh: Mitsuba shape.
        Must be provided if ``fname`` is :py:class:`None`.

    :param name: Object name.
        Must be provided if ``fname`` is not :py:class:`None`.

    :param fname: Filename of a valid mesh ( "*.ply" | "*.obj").
        Must be provided if ``mi_mesh`` is :py:class:`None`.

    :param radio_material: Radio material of the object.
        Must be provided if ``fname`` is not :py:class:`None`.
    """
    # Counter to handle objects with no name
    NO_NAME_COUNTER = 0

    def __init__(self,
                 mi_mesh: mi.Mesh | None=None,
                 name: str | None=None,
                 fname: str | None=None,
                 radio_material: RadioMaterialBase | None=None,
                 remove_duplicate_vertices: bool = False):

        if mi_mesh is not None:
            if not isinstance(mi_mesh, mi.Mesh):
                raise ValueError("`mi_mesh` must a Mitsuba Mesh object")
        elif fname:
            # Mesh type
            mesh_type = os.path.splitext(fname)[1][1:]
            if mesh_type not in ('ply', 'obj'):
                raise ValueError("Invalid mesh type."
                                 " Supported types: `ply` and `obj`")

            mi_mesh = load_mesh(fname)
        else:
            raise ValueError("Either a Mitsuba Shape (mi_mesh) or a filename"
                             " (fname) must be provided")

        # If duplicated vertices are removed, the utility ``remove_mesh_duplicate_vertices``
        # will build the directed edges. Otherwise, we need to build them manually.
        if remove_duplicate_vertices:
            remove_mesh_duplicate_vertices(mi_mesh)
        else:
            build_mesh_directed_edges(mi_mesh)

        if radio_material is not None:
            if not isinstance(radio_material, RadioMaterialBase):
                raise ValueError("The `radio_material` for the object to"
                                 " instantiate must be a RadioMaterialBase")
            mi_mesh.set_bsdf(radio_material)

        # Object naming.
        if name is not None:
            # If a `name` parameter is provided, adopt it regardless of the current
            # Mitsuba shape's ID.
            if not isinstance(name, str):
                raise ValueError("The `name` of the object to instantiate must"
                                 f" be a `str`, found `{type(name)}` instead.")
            mi_mesh.set_id(name)
        elif mi_mesh.id() in ("", "__root__"):
            # Default name.
            SceneObject.NO_NAME_COUNTER += 1
            name = f"no-name-{SceneObject.NO_NAME_COUNTER}"
            mi_mesh.set_id(name)
        else:
            # Otherwise, keep the current Mitsuba shape's ID.
            pass

        # Also use the name for the BSDF, if not already set to something else.
        bsdf = mi_mesh.bsdf()
        if (name is not None) and (bsdf is not None) and bsdf.id() in ("", "__root__"):
            bsdf.set_id("mat-" + name)
        del name

        # Set the Mitsuba shape
        self._mi_mesh = mi_mesh

        # Scene object to which the object belongs
        self._scene: weakref.ref[scene_module.Scene] = lambda: None

        # Read the ID from the Mitsuba Shape.
        # The object ID is the corresponding Mitsuba shape pointer
        # reinterpreted as an UInt32 (not the object's name).
        self._object_id = dr.reinterpret_array(mi.UInt32,
                                               mi.ShapePtr(mi_mesh))[0]

        # Increment the material counter of objects
        self.radio_material.add_object()

        # Set initial position and orientation of the object
        self._position = mi.Point3f(0, 0, 0)
        self._orientation = mi.Point3f(0, 0, 0)
        self._scaling = mi.Vector3f(1.0)

        # The object storing the velocity is only created when the user sets
        # a nonzero value. This is possible because evaluating a shape
        # attribute returns `0.f` by default when the attribute does not exist.
        self._velocity_params = None

    @property
    def scene(self):
        """
        Get/set the scene to which the object belongs. Note that the scene can
        only be set once.

        :type: :py:class:`sionna.rt.Scene`
        """
        return self._scene()

    @scene.setter
    def scene(self, scene: scene_module):
        if not isinstance(scene, scene_module.Scene):
            raise ValueError("`scene` must be an instance of Scene")
        if (self._scene() is not None) and (self._scene() is not scene):
            raise ValueError(f"This object ('{self.name}') is already used"
                             " by another scene.")
        self._scene = weakref.ref(scene)

    @staticmethod
    def shape_id_to_name(shape_id):
        name = shape_id
        if shape_id.startswith("mesh-"):
            name = shape_id[5:]
        return name

    @property
    def name(self):
        r"""Name

        :type: :py:class:`str`
        """
        return SceneObject.shape_id_to_name(self._mi_mesh.id())

    @property
    def object_id(self):
        r"""Identifier

        :type: :py:class:`int`
        """
        return self._object_id

    @property
    def mi_mesh(self):
        r"""Get/set the Mitsuba shape

        :type: :py:class:`mi.Mesh`
        """
        return self._mi_mesh

    @mi_mesh.setter
    def mi_mesh(self, v: mi.Mesh):
        self._mi_mesh = v

    @property
    def radio_material(self):
        r"""Get/set the radio material of the object. Setting can be done by
        using either an instance of :class:`~sionna.rt.RadioMaterialBase` or
        the material name (:py:class:`str`).

        :type: :class:`~sionna.rt.RadioMaterialBase`
        """
        return self._mi_mesh.bsdf()

    @radio_material.setter
    def radio_material(self, mat: str | RadioMaterialBase):

        if isinstance(mat, str) and (self.scene is not None):
            mat_obj = self.scene.get(mat)
            if (mat_obj is None) or not isinstance(mat_obj, RadioMaterialBase):
                raise TypeError(f"Unknown radio material '{mat}'")

        elif not isinstance(mat, RadioMaterialBase):
            raise TypeError("The material must be a material name (str) or an "
                            f"instance of RadioMaterialBase, found {type(mat)}")

        else:
            mat_obj = mat

        # Current radio material
        current_mat = self.radio_material

        # Add the radio material to the scene
        if self.scene is not None:
            self.scene.add(mat_obj)
            # Ensure that the object and the material belong to the same scene
            if self.scene != mat_obj.scene:
                raise ValueError("Radio material and object are not part of the"
                                 " same scene")

        # Increment the material counter of objects
        mat_obj.add_object()

        # Remove the object from the set of the currently used material, if any
        if isinstance(current_mat, RadioMaterialBase):
            current_mat.remove_object()

        # Effectively update the radio material of the Mitsuba shape
        self._mi_mesh.set_bsdf(mat_obj)

    @property
    def velocity(self):
        r"""Get/set the velocity vector [m/s]

        The velocity must be set at least once before it can be
        differentiated.

        :type: :py:class:`mi.Vector3f`
        """
        if self._velocity_params is None:
            return mi.Vector3f(0.)
        else:
            return self._velocity_params["value"]

    @velocity.setter
    def velocity(self, v: mi.Vector3f):
        v = mi.Vector3f(v)
        if dr.width(v) != 1:
            raise ValueError("Only a single velocity vector must be provided")

        # If the raw attribute was not yet instantiated, it is.
        if self._velocity_params is None:
            tex = mi.load_dict({
                "type": "rawconstant",
                "value" : mi.Float(v.x[0], v.y[0], v.z[0])
            })
            self._mi_mesh.add_texture_attribute("velocity", tex)
            # Keep parameters to update the speed
            self._velocity_params = mi.traverse(tex)
        else:
            self._velocity_params["value"] = v
            self._velocity_params.update()

    @property
    def position(self):
        r"""Get/set the position vector [m] of the center of the object. The
        center is defined as the object's axis-aligned bounding box (AABB).

        :type: :py:class:`mi.Point3f`
        """
        # Bounding box
        bbox_min = self._mi_mesh.bbox().min
        bbox_max = self._mi_mesh.bbox().max
        position = (bbox_min + bbox_max)*0.5
        return mi.Point3f(position)

    @position.setter
    def position(self, new_position: mi.Point3f):

        if self.scene is None:
            raise ValueError("Scene is not set: Object must be added to a"
                             " scene before setting its position")

        # Scene parameters
        scene_params = self.scene.mi_scene_params

        # Use the shape id, and not the object name, to access the Mitsuba
        # scene
        vp_key = self._mi_mesh.id() + ".vertex_positions"

        current_vertices = dr.unravel(mi.Point3f, scene_params[vp_key])
        translation_vector = new_position - self.position
        translated_vertices = current_vertices + translation_vector
        scene_params[vp_key] = dr.ravel(translated_vertices)

        scene_params.update()
        self.scene.scene_geometry_updated()

    @property
    def orientation(self):
        r"""Get/set the orientation [rad] specified through three angles
        :math:`(\alpha, \beta, \gamma)` corresponding to a 3D rotation as
        defined in :eq:`rotation`

        :type: :py:class:`mi.Point3f`
        """
        return self._orientation

    @orientation.setter
    def orientation(self, new_orientation: mi.Point3f):

        if self.scene is None:
            raise ValueError("Scene is not set: Object must be added to a"
                             " scene before setting its orientation")

        new_orientation = mi.Point3f(new_orientation)

        # Build the transformation corresponding to the new rotation
        new_rotation = rotation_matrix(new_orientation)

        # Invert the current orientation
        cur_rotation = rotation_matrix(self.orientation)
        inv_cur_rotation = cur_rotation.T

        # Scene parameters
        scene_params = self.scene.mi_scene_params

        # Use the shape id, and not the object name, to access the Mitsuba
        # scene
        vp_key = self._mi_mesh.id() + ".vertex_positions"

        # To rotate the object, we need to:
        # 1. Position such that its center is (0,0,0)
        # 2. Undo the current orientation (if any)
        # 3. Apply the new orientation
        # 4. Reposition the object to its current position

        current_vertices = dr.unravel(mi.Point3f, scene_params[vp_key])
        position = self.position
        rotated_vertices = current_vertices - position
        rotated_vertices = new_rotation@inv_cur_rotation@rotated_vertices
        rotated_vertices = rotated_vertices + position
        scene_params[vp_key] = dr.ravel(rotated_vertices)
        scene_params.update()

        self._orientation = new_orientation
        self.scene.scene_geometry_updated()

    @property
    def scaling(self):
        r"""Get the scaling in the coordinate system of the object.
        If a scalar value is provided, the object is uniformly scaled
        across all dimensions by that value. Alternatively, if a vector
        is provided, the object is scaled independently along the x, y,
        and z axes according to the respective components of the vector,
        within the object's coordinate system.

        :type: :py:class:`mi.Float` | :py:class:`mi.Vector3f`
        """
        return self._scaling

    @scaling.setter
    def scaling(self, new_scaling: mi.Float | mi.Vector3f):
        r"""Set the scaling value in the coordinate system of the object. If a
        scalar value is passed in, each dimension is scaled equally by
        new_scaling. Otherwise, the object is scaled by the x, y and z values
        of new_scaling in the coordinate system of the object.

        :param new_scaling: The new scaling factor in the objects coordinate
            system. Can be a scalar or a vector value
        """
        if self.scene is None:
            raise ValueError("Scene is not set: Object must be added to a"
                             " scene before setting its scaling")

        new_scaling = mi.Vector3f(new_scaling)

        if dr.any(new_scaling <= 0.0):
            raise ValueError("Scaling must be positive")

        # Scene parameters
        scene_params = self.scene.mi_scene_params

        # Use the shape id, and not the object name, to access the Mitsuba
        # scene
        vp_key = self._mi_mesh.id() + ".vertex_positions"

        # Get the current rotation and its inverse
        cur_rotation = rotation_matrix(self.orientation)
        inv_cur_rotation = cur_rotation.T

        current_vertices = dr.unravel(mi.Point3f, scene_params[vp_key])

        current_vertices -= self.position  # Undo the translation
        rotated_vertices = inv_cur_rotation @ current_vertices  # Undo the rotation

        scaled_vertices = (new_scaling / self._scaling) * rotated_vertices  # Perform the scaling

        scaled_vertices = cur_rotation @ scaled_vertices  # Redo the rotation
        scaled_vertices += self.position  # Redo the translation

        scene_params[vp_key] = dr.ravel(scaled_vertices)

        self._scaling = new_scaling

        scene_params.update()
        self.scene.scene_geometry_updated()

    def look_at(self, target: mi.Point3f | SceneObject | RadioDevice | str):
        # pylint: disable=line-too-long
        r"""
        Sets the orientation so that the x-axis points toward a position

        The world z-axis is used as up direction. For a vertical look-at
        direction, for which the rotation about the x-axis is not determined by
        the look-at direction alone, the azimuth is set to :math:`\varphi=0`.

        :param target: A position, or the name or instance of a scene object
            or radio device to point toward

        :raises ValueError: If ``target`` coincides with the object
            position, for which the look-at direction is undefined
        """

        # Get position to look at
        if isinstance(target, str):
            if self.scene is None:
                raise ValueError("Scene is not set: Object must be added to a"
                                 " scene before looking at a named target")
            item = self.scene.get(target)
            if item is None:
                raise ValueError(f"Unknown target '{target}'")
            if not isinstance(item, (SceneObject, RadioDevice)):
                raise ValueError(f"Cannot look at '{target}' of type"
                                 f" {type(item).__name__}")
            target = item.position
        elif isinstance(target, (SceneObject, RadioDevice)):
            target = target.position
        elif isinstance(target, mi.Point3f):
            pass # Nothing to do
        else:
            raise ValueError("Invalid type for `target`")

        # Compute angles relative to LCS
        self.orientation = look_at_orientation(self.position, target, "object")

    def clone(self, name: str | None = None, as_mesh=False,
              props: mi.Properties | None = None) -> SceneObject | mi.Mesh:
        r"""
        Creates a clone of the current scene object

        The clone will have the same geometry and material properties as the
        original object but will be assigned a new name.

        :param name: Name (id) of the cloned object.
            If :py:class:`None`, the clone will be named as
            ``<original_name>-clone``.

        :param as_mesh: If set to `True`, the clone will be returned as a
            :py:class:`mitsuba.Mesh` object. Otherwise, a
            :py:class:`sionna.rt.SceneObject` will be returned.

        :param props: Pre-populated properties to be used in the new Mitsuba
            shape. Allows overriding the BSDF, emitter, etc.

        :return: A clone of the current object
        """
        cloned_mesh = clone_mesh(self.mi_mesh, name=name, props=props)
        cloned_mesh.set_bsdf(self.radio_material)

        # Build scene object
        if as_mesh:
            return cloned_mesh
        else:
            return SceneObject(name=cloned_mesh.id(), mi_mesh=cloned_mesh)
