#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Base class for implementing a radio material"""

import mitsuba as mi
import numpy as np
import matplotlib
from typing import Tuple
import weakref

from sionna.rt import scene as scene_module


class RadioMaterialBase(mi.BSDF):
    # pylint: disable=line-too-long
    r"""
    Base class to implement a radio material

    This is an abstract class that cannot be instantiated, and serves as
    interface for defining radio materials.

    A radio material defines how an object scatters incident radio waves.

    :param props: Property container storing the parameters required to build the material
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)

        # Counter indicating how many objects are using `self`
        self._count_using_objects = 0

        # Scene object that uses this radio material
        self._scene: weakref.ref[scene_module.Scene] = lambda: None

        # Read color from properties, if any
        color = None
        for pname in ("color", "reflectance", "base_color"):
            if pname in props:
                color = tuple(props[pname])
                del props[pname]
                break
        if color is None:
            # If not specified, use an arbitrary value
            palette = matplotlib.colormaps.get_cmap("Pastel1")
            i = np.random.randint(low=0, high=palette.N)
            color = palette((i % palette.N + 0.5) / palette.N)[:3]
        self._color = color

    @property
    def scene(self):
        """Get/set the scene used by the radio material.
        Note that the scene can only be set once.

        :type: :class:`~sionna.rt.Scene`
        """
        return self._scene()

    @scene.setter
    def scene(self, scene: mi.Scene):
        if not isinstance(scene, scene_module.Scene):
            raise ValueError("`scene` must be an instance of Scene")
        if (self._scene() is not None) and (self._scene() is not scene):
            raise ValueError(f"Radio material ('{self.name}') already used by"
                             " another scene.")
        self._scene = weakref.ref(scene)

    @property
    def name(self):
        r"""
        (read-only) Name of the radio material

        :type: :py:class:`str`
        """
        # If a prefix 'mat-' is used, it is removed.
        # This prefix is automatically added by the Mitsuba-Blender extension
        name = self.id()
        if name.startswith('mat-'):
            name = name[4:]
        return name

    @property
    def color(self):
        r"""
        Get/set the the RGB (red, green, blue) color for the
        radio material as displayed in the previewer and renderer.
        Each RGB component must have a value within the range :math:`\in [0,1]`.

        :type: :py:class:`Tuple[float, float, float]`
        """
        return self._color

    @color.setter
    def color(self, new_color: Tuple[float, float, float]):
        if len(new_color) == 3:
            if min(new_color) < 0. or max(new_color) > 1.:
                raise ValueError("Color components must be in the range (0,1)")
        self._color = (new_color[0], new_color[1], new_color[2])

    # pylint: disable=unused-argument
    def clone(self,
              name: str | None = None,
              **overrides) -> "RadioMaterialBase":
        r"""
        Returns a new radio material, identical to this one except for any
        specified ``overrides``

        This method performs a shallow clone: non-overridden properties
        share their underlying values and arrays/tensors with the origin
        material. In differentiable ray tracing, this allows backpropagating
        gradients from interactions on cloned materials to the shared
        parameters of the origin material. Note that each clone is an
        independent object: updating a property on one instance via its
        setter (e.g., applying a gradient descent step) re-binds that
        attribute on that instance and does not implicitly re-bind attributes
        on other clones.

        Subclasses that support this operation must override this method.
        The default implementation is a no-op, consistently with the other
        methods of this interface (:meth:`sample`, :meth:`eval`, etc.):
        callers that rely on a working override are responsible for
        checking the returned value and raising an appropriate error if a
        subclass does not implement this method.

        :param name: Optional new name for the cloned material.
        :param overrides: Keyword arguments specifying properties to override.
        """
        return ...

    @property
    def is_used(self) -> bool:
        r"""
        (read-only) Return `True` if at least one object in the scene
        uses this material

        :type: :py:class:`bool`
        """
        return self._count_using_objects > 0

    def add_object(self):
        r"""
        Increment the counter indicating the number of objects using this
        material
        """
        self._count_using_objects += 1

    def remove_object(self):
        r"""
        Decreases the counter indicating the number of objects using this
        material
        """
        self._count_using_objects -= 1

    # pylint: disable=unused-argument
    def sample(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        sample1: mi.Float,
        sample2: mi.Point2f,
        active: bool | mi.Bool = True
    ) -> Tuple[mi.BSDFSample3f, mi.Spectrum]:
        # pylint: disable=line-too-long
        r"""
        Samples the radio material

        This function samples an interaction type (e.g., specular reflection,
        diffuse reflection or refraction) and direction of propagation for the
        scattered ray, and returns the corresponding radio material sample and Jones matrix.
        The returned radio material sample stores the sampled type of interaction and sampled direction
        of propagation of the scattered ray.

        The following assumptions are made on the inputs:

        - ``ctx.component`` is a binary mask that specifies the types of interaction enabled. Booleans can be obtained from this mask as follows:

        .. code-block:: python

            specular_reflection_enabled = (ctx.component & InteractionType.SPECULAR) > 0
            diffuse_reflection_enabled = (ctx.component & InteractionType.DIFFUSE) > 0
            transmission_enabled = (ctx.component & InteractionType.TRANSMISSION) > 0

        - ``si.wi`` is the direction of propagation of the incident wave in the world frame
        - ``si.sh_frame`` is the frame such that the ``sh_frame.n`` is the normal to the intersected surface in the world coordinate system
        - ``si.dn_du`` stores the real part of the S and P components of the incident electric field represented in the implicit world frame (first and second components of ``si.dn_du``)
        - ``si.dn_dv`` stores the imaginary part of the S and P components of the incident electric field represented in the implicit world frame (first and second components of ``si.dn_dv``)
        - ``si.dp_du`` stores the solid angle of the ray tube (first component of ``si.dn_du``)

        The outputs are set as follows:

        - ``bs.wo`` is the direction of propagation of the sampled scattered ray in the world frame
        - ``jones_mat`` is the Jones matrix describing the transformation incurred to the incident wave in the implicit world frame

        :param ctx: A context data structure used to specify which interaction types are enabled
        :param si: Surface interaction data structure describing the underlying surface position
        :param sample1: A uniformly distributed sample on :math:`[0,1]` used to sample the type of interaction
        :param sample2: A uniformly distributed sample on :math:`[0,1]^2` used to sample the direction of the reflected wave in the case of diffuse reflection
        :param active: Mask to specify active rays

        :return: Radio material sample and Jones matrix as a :math:`4 \times 4` real-valued matrix
        """
        return ...

    def eval(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool | mi.Bool = True
    ) -> mi.Spectrum:
        # pylint: disable=line-too-long
        r"""
        Evaluates the radio material

        This function evaluates the Jones matrix of the radio material for the scattered
        direction ``wo`` and for the interaction type stored in ``si.prim_index``.

        The following assumptions are made on the inputs:

        - ``si.wi`` is the direction of propagation of the incident wave in the world frame
        - ``si.sh_frame`` is the frame such that the ``sh_frame.n`` is the normal to the intersected surface in the world coordinate system
        - ``si.dn_du`` stores the real part of the S and P components of the incident electric field represented in the implicit world frame (first and second components of ``si.dn_du``)
        - ``si.dn_dv`` stores the imaginary part of the S and P components of the incident electric field represented in the implicit world frame (first and second components of ``si.dn_dv``)
        - ``si.dp_du`` stores the solid angle of the ray tube (first component of ``si.dn_du``)
        - ``si.prim_index`` stores the interaction type to evaluate

        :param ctx: A context data structure used to specify which interaction types are enabled
        :param si: Surface interaction data structure describing the underlying surface position
        :param wo: Direction of propagation of the scattered wave in the world frame
        :param active: Mask to specify active rays

        :return: Jones matrix as a :math:`4 \times 4` real-valued matrix
        """
        return ...

    def pdf(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool | mi.Bool = True
    ) -> mi.Float:
        # pylint: disable=line-too-long
        r"""
        Evaluates the probability of the sampled interaction type and direction of scattered ray

        This function evaluates the probability density of the radio material for the scattered
        direction ``wo`` and for the interaction type stored in ``si.prim_index``.

        The following assumptions are made on the inputs:

        - ``si.wi`` is the direction of propagation of the incident wave in the world frame
        - ``si.sh_frame`` is the frame such that the ``sh_frame.n`` is the normal to the intersected surface in the world coordinate system
        - ``si.dn_du`` stores the real part of the S and P components of the incident electric field represented in the implicit world frame (first and second components of ``si.dn_du``)
        - ``si.dn_dv`` stores the imaginary part of the S and P components of the incident electric field represented in the implicit world frame (first and second components of ``si.dn_dv``)
        - ``si.dp_du`` stores the solid angle of the ray tube (first component of ``si.dn_du``)
        - ``si.prim_index`` stores the interaction type to evaluate

        :param ctx: A context data structure used to specify which interaction types are enabled
        :param si: Surface interaction data structure describing the underlying surface position
        :param wo: Direction of propagation of the scattered wave in the world frame
        :param active: Mask to specify active rays

        :return: Probability density value
        """
        return ...

    def traverse(self, callback: mi.TraversalCallback):
        # pylint: disable=line-too-long
        r"""
        Traverse the attributes and objects of this instance

        Implementing this function is required for Mitsuba to traverse a scene graph,
        including materials, and determine the differentiable parameters.

        :param callback: Object used to traverse the scene graph
        """
        return ...

    def to_string(self) -> str:
        r"""
        Returns object description
        """
        return ...
