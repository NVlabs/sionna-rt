#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Class implementing a radio device, i.e., a transmitter or a receiver"""

import mitsuba as mi
from typing_extensions import Tuple, Self

from sionna.rt.utils import look_at_orientation

class RadioDevice:
    # pylint: disable=line-too-long
    r"""Class defining a generic radio device

    The classes :class:`~sionna.rt.Transmitter` and
    :class:`~sionna.rt.Receiver` inherit from this class and should be used.

    :param name: Name

    :param position: Position :math:`(x,y,z)` [m]

    :param orientation: Orientation specified through three angles
        :math:`(\alpha, \beta, \gamma)`
        corresponding to a 3D rotation as defined in :eq:`rotation`.
        Mutually exclusive with ``look_at``; specifying both raises
        :py:class:`ValueError`. Defaults to :math:`(0,0,0)` if both
        ``orientation`` and ``look_at`` are :py:class:`None`.

    :param look_at: A position or the instance of
        :class:`~sionna.rt.RadioDevice` to look at.
        Mutually exclusive with ``orientation``.
        If set to :py:class:`None`, then ``orientation`` is used to
        orient the device.

    :param velocity: Velocity vector of the radio device [m/s]

    :param color: Defines the RGB (red, green, blue) ``color`` parameter
        for the device as displayed in the previewer and renderer. Each RGB
        component must have a value within the range :math:`\in [0,1]`.

    :param display_radius: Defines the radius, in meters, of the sphere that
        will represent this device when displayed in the previewer and renderer.
        If not specified, the radius will be chosen automatically using a heuristic.
    """
    def __init__(self,
                 name: str,
                 position: mi.Point3f,
                 orientation: mi.Point3f | None = None,
                 look_at: mi.Point3f | Self | None = None,
                 velocity: mi.Vector3f | None = None,
                 color: Tuple[float, float, float] = (0, 0, 0),
                 display_radius: float | None = None):

        self._name = name
        self.position = position
        self.color = color
        self.display_radius = display_radius

        if (orientation is not None) and (look_at is not None):
            raise ValueError("Only one of `orientation` or `look_at` can be "
                             "specified.")
        if look_at is None:
            self.orientation = mi.Point3f(orientation) \
                               if orientation is not None \
                               else mi.Point3f(0)
        else:
            self.look_at(look_at)

        if velocity is None:
            self.velocity = mi.Vector3f(0., 0., 0.)
        else:
            self.velocity = velocity

    @property
    def name(self):
        """
        Name

        :type: :py:class:`str`
        """
        return self._name

    @property
    def position(self):
        """
        Get/set the position

        :type: :py:class:`mi.Point3f`
        """
        return self._position

    @position.setter
    def position(self, v):
        self._position = mi.Point3f(v)

    @property
    def orientation(self):
        """
        Get/set the orientation

        :type: :py:class:`mi.Point3f`
        """
        return self._orientation

    @orientation.setter
    def orientation(self, v):
        self._orientation = mi.Point3f(v)

    @property
    def velocity(self):
        """
        Get/set the velocity

        :type: :py:class:`mi.Vector3f`
        """
        return self._velocity

    @velocity.setter
    def velocity(self, v):
        self._velocity = mi.Vector3f(v)

    def look_at(self, target: mi.Point3f | Self):
        # pylint: disable=line-too-long
        r"""
        Sets the orientation so that the x-axis points toward a
        position, or radio device.

        Given a point :math:`\mathbf{x}\in\mathbb{R}^3` with spherical angles
        :math:`\theta` and :math:`\varphi`, the orientation of the radio device
        will be set equal to :math:`(\varphi, \theta-\frac{\pi}{2}, 0.0)`.
        The world z-axis is used as up direction. For a vertical
        :math:`\mathbf{x}`, for which the rotation about the local x-axis is
        not determined by the look-at direction alone, the azimuth is set to
        :math:`\varphi=0`.

        :param target: A position, or instance of a
            :class:`~sionna.rt.RadioDevice`, in the scene to look at

        :raises ValueError: If ``target`` coincides with the radio device
            position, for which the look-at direction is undefined
        """
        # Get position to look at
        from sionna.rt import SceneObject # pylint: disable=import-outside-toplevel
        if isinstance(target, (SceneObject, RadioDevice)):
            target = target.position
        else:
            target = mi.Point3f(target)

        # Compute angles relative to LCS
        self.orientation = look_at_orientation(self.position, target,
                                               "radio device")

    @property
    def color(self):
        r"""
        Get/set the RGB (red, green, blue) color for the
        device as displayed in the previewer and renderer.
        Each RGB component must have a value within the range :math:`\in [0,1]`.

        :type: :py:class:`Tuple[float, float, float]`
        """
        return self._color

    @color.setter
    def color(self, new_color):
        if len(new_color) != 3:
            raise ValueError(
                "Color must be a tuple of three RGB components in the range [0, 1]"
            )
        if min(new_color) < 0. or max(new_color) > 1.:
            raise ValueError("Color components must be in the range [0, 1]")
        self._color = (float(new_color[0]), float(new_color[1]), float(new_color[2]))

    @property
    def display_radius(self):
        r"""
        :py:class:`float | None`: Get/set the radius [m] of the sphere that
        represents this device when displayed in the previewer and renderer.
        If set to `None`, the radius will be chosen automatically using a heuristic.
        """
        return self._display_radius

    @display_radius.setter
    def display_radius(self, radius):
        if (radius is not None) and (radius < 0):
            raise ValueError("The display radius must be a float >= 0 or None.")
        self._display_radius = radius
