#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Class implementing a transmitter"""

import mitsuba as mi
from typing_extensions import Tuple, Self

from .radio_device import RadioDevice
from sionna.rt.utils import dbm_to_watt
from sionna.rt.constants import DEFAULT_TRANSMITTER_COLOR, \
                                DEFAULT_TRANSMIT_POWER_DBM


class Transmitter(RadioDevice):
    # pylint: disable=line-too-long
    r"""
    Class defining a transmitter

    :param name: Name

    :param position: Position :math:`(x,y,z)` [m]

    :param power_dbm: Transmit power [dBm]

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

    :param velocity: Velocity vector of the transmitter [m/s]

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
                 power_dbm=DEFAULT_TRANSMIT_POWER_DBM,
                 color: Tuple[float, float, float] = DEFAULT_TRANSMITTER_COLOR,
                 display_radius: float | None = None):

        super().__init__(name=name,
                         position=position,
                         orientation=orientation,
                         look_at=look_at,
                         velocity=velocity,
                         color=color,
                         display_radius=display_radius)

        self.power_dbm = power_dbm

    @property
    def power_dbm(self):
        """
        Get/set transmit power [dBm]

        :type: :py:class:`mi.ScalarFloat`
        """
        return self._power_dbm

    @power_dbm.setter
    def power_dbm(self, value):
        self._power_dbm = mi.Float(value)

    @property
    def power(self):
        """
        Get the transmit power [W]

        :type: :py:class:`mi.Float`
        """
        return dbm_to_watt(self._power_dbm)
