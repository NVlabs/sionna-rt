#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Implements classes and methods related to antenna arrays"""

from numbers import Integral

import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

from .utils import rotation_matrix
from .antenna_pattern import AntennaPattern, antenna_pattern_registry

class AntennaArray:
    # pylint: disable=line-too-long
    r"""
    Class implementing an antenna array

    An antenna array is composed of antennas which are placed at
    different positions. All antennas share the same antenna pattern,
    which can be single- or dual-polarized.

    :param antenna_pattern: Antenna pattern to be used across the array

    :param normalized_positions: Array of relative positions of each
        antenna with respect to the position of the radio device,
        normalized by the wavelength.
        Dual-polarized antennas are counted as a single antenna
        and share the same position.
    """
    def __init__(self,
                 antenna_pattern: AntennaPattern,
                 normalized_positions: mi.Point3f):
        self.antenna_pattern = antenna_pattern
        self.normalized_positions = normalized_positions

    @property
    def antenna_pattern(self):
        """
        Get/set the antenna pattern

        :type: :class:`~sionna.rt.AntennaPattern`
        """
        return self._antenna_pattern

    @antenna_pattern.setter
    def antenna_pattern(self, v):
        if not isinstance(v, AntennaPattern):
            raise TypeError("`antenna_pattern` must be an instance of type"
                            f" AntennaPattern, found type '{type(v)}'.")
        self._antenna_pattern = v

    def positions(self, wavelength: float) -> mi.Point3f:
        """
        Get the relative positions of all antennas
        (dual-polarized antennas are counted as a single antenna and share the
        same position).

        Positions are computed by scaling the normalized positions of antennas
        by the ``wavelength``.

        :param wavelength: Wavelength [m]

        :returns: Relative antenna positions :math:`(x,y,z)` [m]
        """
        return self._normalized_positions*wavelength

    @property
    def normalized_positions(self):
        r"""
        Get/set  array of relative normalized positions :math:`(x,y,z)`
        [:math:`\lambda`] of each antenna. Dual-polarized antennas are counted
        as a single antenna and share the same position.

        :type: :py:class:`mi.Point3f`
        """
        return self._normalized_positions

    @normalized_positions.setter
    def normalized_positions(self, normalized_positions):
        normalized_positions = mi.Point3f(normalized_positions)
        self._normalized_positions = normalized_positions

    @property
    def num_ant(self):
        """
        Number of linearly polarized antennas in the array. Dual-polarized
        antennas are counted as two linearly polarized antennas.

        :type: :py:class:`int`
        """
        return dr.shape(self._normalized_positions)[1]\
            *len(self.antenna_pattern.patterns)

    @property
    def array_size(self):
        """
        Number of antennas in the array. Dual-polarized antennas are counted as
        a single antenna.

        :type: :py:class:`int`
        """
        return dr.shape(self.normalized_positions)[1]

    def rotate(self,
               wavelength: float,
               orientation: mi.Point3f) -> mi.Point3f:
        r"""
        Computes the relative positions of all antennas rotated according
        to the ``orientation``

        Dual-polarized antennas are counted as a single antenna and share the
        same position.

        Positions are computed by scaling the normalized positions of antennas
        by the ``wavelength`` and rotating by ``orientation``.

        :param wavelength: Wavelength [m]

        :param orientation: Orientation [rad] specified through three angles
            corresponding to a 3D rotation as defined in :eq:`rotation`

        :returns: Rotated relative antenna positions :math:`(x,y,z)` [m]
        """
        rot_mat = rotation_matrix(orientation)
        p = self.positions(wavelength)
        rot_p = rot_mat@p
        return rot_p

class PlanarArray(AntennaArray):
    # pylint: disable=line-too-long
    r"""
    Class implementing a planar antenna array

    The antennas of a planar array are regularly spaced, located in the
    y-z plane, and numbered column-first from the top-left to
    bottom-right corner.

    :param num_rows: Number of rows (``>= 1``)

    :param num_cols: Number of columns (``>= 1``)

    :param vertical_spacing: Vertical antenna spacing
        [multiples of wavelength]

    :param horizontal_spacing: Horizontal antenna spacing
        [multiples of wavelength]

    :param pattern: Name of a registered antenna pattern factory method
        :list-registry:`sionna.rt.antenna_pattern.antenna_pattern_registry`

    Keyword Arguments
    -----------------
    polarization  : :py:class:`str`
        Name of a registered polarization
        :list-registry:`sionna.rt.antenna_pattern.polarization_registry`

    polarization_model: :py:class:`str`
        Name of a registered polarization model
        :list-registry:`sionna.rt.antenna_pattern.polarization_model_registry`.
        Defaults to "tr38901_2".

    ``**kwargs``: :py:class:`Any`
        Depending on the chosen antenna pattern, other keyword arguments
        must be provided.
        See the :ref:`Developer Guide <dev_custom_antenna_patterns>` for
        more details.

    Example
    -------
    .. code-block:: python

        from sionna.rt import PlanarArray
        array = PlanarArray(num_rows=8, num_cols=4,
                            pattern="tr38901",
                            polarization="VH")
        array.show()

    .. figure:: ../figures/antenna_array.png
        :align: center
        :scale: 100%
    """
    def __init__(self,
                 *,
                 num_rows: int,
                 num_cols: int,
                 vertical_spacing: float=0.5,
                 horizontal_spacing: float=0.5,
                 pattern: str,
                 **kwargs):

        if isinstance(num_rows, bool) or not isinstance(num_rows, Integral):
            raise TypeError("`num_rows` must be an integer")
        if isinstance(num_cols, bool) or not isinstance(num_cols, Integral):
            raise TypeError("`num_cols` must be an integer")
        if num_rows < 1:
            raise ValueError("`num_rows` must be greater than or equal to one")
        if num_cols < 1:
            raise ValueError("`num_cols` must be greater than or equal to one")

        # Create list of antennas
        array_size = num_rows*num_cols
        antenna_pattern = antenna_pattern_registry.get(pattern)(**kwargs)

        # Compute antenna positions
        d_v = vertical_spacing
        d_h = horizontal_spacing
        normalized_positions =  dr.zeros(mi.Point3f, array_size)
        ii, jj = dr.meshgrid(dr.arange(mi.UInt, num_rows), dr.arange(mi.UInt, num_cols))
        # Set Y-Z positions
        normalized_positions.y =  d_h*jj - (num_cols-1)*d_h/2
        normalized_positions.z = -d_v*ii + (num_rows-1)*d_v/2

        # Positions
        super().__init__(antenna_pattern, normalized_positions)

    def show(self):
        r"""show()

        Visualizes the planar antenna array

        Antennas are depicted by markers that are annotated with the antenna
        number. The marker is not related to the polarization of an antenna.

        Output
        ------
        : :class:`matplotlib.pyplot.Figure`
            Figure depicting the antenna array
        """
        positions = self.normalized_positions
        fig = plt.figure()
        plt.plot(positions.y.numpy(), positions.z.numpy(),
                 marker=MarkerStyle("+").get_marker(), markeredgecolor='red',
                 markerfacecolor='red', markersize="10", linestyle="None",
                 markeredgewidth="1")
        for i in range(self.array_size):
            fig.axes[0].annotate(i+1, (positions.y[i], positions.z[i]))
        plt.xlabel(r"y ($\lambda$)")
        plt.ylabel(r"z ($\lambda)$")
        plt.title("Planar Array Layout")
        return fig
