#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Planar radio map"""

import warnings
from typing import Tuple, List

import drjit as dr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import from_levels_and_colors
import mitsuba as mi
import numpy as np

from sionna.rt.utils import watt_to_dbm, log10, rotation_matrix, WedgeGeometry
from sionna.rt.scene import Scene
from sionna.rt.constants import DEFAULT_TRANSMITTER_COLOR,\
    DEFAULT_RECEIVER_COLOR
from .radio_map import RadioMap


class PlanarRadioMap(RadioMap):
    r"""
    Planar Radio Map

    A planar radio map is defined by a measurement grid, i.e., a rectangular
    grid of cells. It is computed by
    a :doc:`radio map solver <radio_map_solvers>`.

    :param scene: Scene for which the radio map is computed

    :param center: Center of the radio map :math:`(x,y,z)` [m] as
        three-dimensional vector

    :param orientation: Orientation of the radio map
        :math:`(\alpha, \beta, \gamma)` specified through three angles
        corresponding to a 3D rotation as defined in :eq:`rotation`

    :param size: Minimum size of the radio map [m]. If ``size`` is not an
        integer multiple of ``cell_size``, it is expanded to
        ``ceil(size / cell_size) * cell_size`` so that every cell has exactly
        the requested extent.

    :param cell_size: Size of a cell of the radio map [m]
    """

    def __init__(self,
                 scene: Scene,
                 cell_size: mi.Point2f,
                 center: mi.Point3f | None = None,
                 orientation: mi.Point3f | None = None,
                 size: mi.Point2f | None = None):

        super().__init__(scene)

        cell_size = mi.Point2f(cell_size)
        if (cell_size.x[0] <= 0) or (cell_size.y[0] <= 0):
            raise ValueError("`cell_size` entries must be positive")

        # Check the properties of the rectangle defining the radio map
        if ((center is None) and (size is None) and (orientation is None)):
            # Default value for center: Center of the scene
            # Default value for the scale: Just enough to cover all the scene
            # with axis-aligned edges of the rectangle
            # [min_x, min_y, min_z]
            scene_min = scene.mi_scene.bbox().min
            # In case of empty scene, bbox min is -inf
            scene_min = dr.select(dr.isinf(scene_min), -1.0, scene_min)
            # [max_x, max_y, max_z]
            scene_max = scene.mi_scene.bbox().max
            # In case of empty scene, bbox min is inf
            scene_max = dr.select(dr.isinf(scene_max), 1.0, scene_max)
            # Center and size
            center = 0.5 * (scene_min + scene_max)
            center.z = 1.5
            size = scene_max - scene_min
            size = mi.Point2f(size.x, size.y)
            # Set the orientation to default value
            orientation = mi.Point3f(0)
        elif ((center is None) or (size is None) or (orientation is None)):
            raise ValueError("If one of `center`, `orientation`," \
                             " or `size` is not None, then all of them" \
                             " must not be None.")
        else:
            center = mi.Point3f(center)
            orientation = mi.Point3f(orientation)
            size = mi.Point2f(size)

        if (size.x[0] <= 0) or (size.y[0] <= 0):
            raise ValueError("`size` entries must be positive")

        # Number of cells and snapped measurement-plane size. Expanding ``size``
        # to an integer multiple of ``cell_size`` keeps a single cell extent for
        # the rectangle, cell centers, indexing, sampling offsets, and area
        # normalization.
        cells_per_dim = mi.Point2u(dr.ceil(size / cell_size))
        size = mi.Point2f(cells_per_dim) * cell_size

        self._cells_per_dim = cells_per_dim
        self._center = mi.Point3f(center)
        self._cell_size = cell_size
        self._orientation = mi.Point3f(orientation)
        self._size = size

        self._meas_plane = mi.load_dict({
            'type': 'rectangle',
            'to_world': self._build_transform(scalar=True)
        })

        self._normalization_factor = self._compute_normalization_factor()

        # Initialize the pathgain map to zero
        self._pathgain_map = dr.zeros(
            mi.TensorXf, (self.num_tx, cells_per_dim.y[0], cells_per_dim.x[0])
        )

    @property
    def measurement_surface(self):
        r"""Mitsuba rectangle shape corresponding to the
        radio map measurement surface

        :type: :py:class:`mi.Shape`
        """
        return self._meas_plane

    @property
    def cells_count(self):
        r"""Total number of cells

        :type: :py:class:`int`
        """
        cells_per_dim = self._cells_per_dim
        return cells_per_dim.x[0] * cells_per_dim.y[0]

    @property
    def cells_per_dim(self):
        r"""Number of cells per dimension

        :type: :py:class:`mi.Point2u`
        """
        return self._cells_per_dim

    @property
    def cell_centers(self):
        r"""Positions of the centers of the cells in the global coordinate
        system

        :type: ``mi.TensorXf [cells_per_dim_y, cells_per_dim_x, 3]``
        """
        cells_per_dim = self._cells_per_dim
        cell_size = self._cell_size

        # Positions of cell centers in measurement plane coordinate system

        # [cells_per_dim_x]
        x_positions = dr.arange(mi.Float, 0, cells_per_dim.x[0])
        x_positions = (x_positions + 0.5) * cell_size.x - 0.5 * self._size.x
        # [cells_per_dim_y * cells_per_dim_x]
        x_positions = dr.tile(x_positions, cells_per_dim.y[0])

        # [cells_per_dim_y]
        y_positions = dr.arange(mi.Float, cells_per_dim.y[0])
        y_positions = (y_positions + 0.5) * cell_size.y - 0.5 * self._size.y
        # [cells_per_dim_y * cells_per_dim_x]
        y_positions = dr.repeat(y_positions, cells_per_dim.x[0])

        # [cells_per_dim_y * xcells_per_dim_x]
        cell_pos = mi.Point3f(x_positions, y_positions, 0.)

        # Rotate to world frame
        to_world = rotation_matrix(self._orientation)
        cell_pos = to_world @ cell_pos + self._center

        # To-tensor
        cell_pos = dr.ravel(cell_pos)
        cell_pos = dr.reshape(mi.TensorXf, cell_pos,
                              [cells_per_dim.y[0], cells_per_dim.x[0], 3])
        return cell_pos

    @property
    def cell_size(self):
        r"""Size of a cell of the radio map [m]

        :type: :py:class:`mi.Point2f`
        """
        return self._cell_size

    @property
    def tx_cell_indices(self):
        r"""Cell index position of each transmitter in the format
        `(column, row)`

        :type: :py:class:`mi.Point2u`
        """
        return self._global_to_cell_ind(self._tx_positions)

    @property
    def rx_cell_indices(self):
        r"""Computes and returns the cell index positions corresponding to
        receivers in the format `(column, row)`

        :type: :py:class:`mi.Point2u`
        """
        return self._global_to_cell_ind(self._rx_positions)

    @property
    def center(self):
        r"""Center of the radio map in the global coordinate system

        :type: :py:class:`mi.Point3f`
        """
        return self._center

    @property
    def orientation(self):
        r"""Orientation of the radio map :math:`(\alpha, \beta, \gamma)`
        specified through three angles corresponding to a 3D rotation as defined
        in :eq:`rotation`. An orientation of :math:`(0,0,0)` corresponds to a
        radio map that is parallel to the XY plane.

        :type: :py:class:`mi.Point3f`
        """
        return self._orientation

    @property
    def size(self):
        r"""Size of the radio map [m]

        Equal to ``cells_per_dim * cell_size``. May be larger than the size
        requested at construction when that size was not an integer multiple of
        ``cell_size``.

        :type: :py:class:`mi.Point2f`
        """
        return self._size

    @property
    def path_gain(self):
        # pylint: disable=line-too-long
        r"""Path gains across the radio map from all transmitters [unitless, linear scale]

        :type: ``mi.TensorXf [num_tx, cells_per_dim_y, cells_per_dim_x]``
        """
        return self._pathgain_map

    def add_paths(
        self,
        e_fields: List[mi.Vector4f],
        array_w: List[mi.Float],
        si: mi.SurfaceInteraction3f,
        k_world: mi.Vector3f,
        tx_indices: mi.UInt,
        active: mi.Bool,
        diffracted_paths: bool,
        solid_angle: mi.Float | None = None,
        tx_positions: mi.Point3f | None = None,
        wedges: WedgeGeometry | None = None,
        diff_point: mi.Point3f | None = None,
        wedges_samples_cnt: mi.UInt | None = None,
        diffraction_angular_measure: mi.Float | None = None):
        # pylint: disable=line-too-long
        r"""
        Adds the contribution of the paths that hit the measurement surface
        to the radio maps

        The radio maps are updated in place.

        :param e_fields: Electric fields as real-valued vectors of dimension 4
        :param array_w: Weighting used to model the effect of the transmitter
            array
        :param si: Information about the interaction with the measurement
            surface
        :param k_world: Directions of propagation of the incident paths
        :param tx_indices: Indices of the transmitters from which the rays originate
        :param active: Flags indicating if the paths should be added to the radio map
        :param diffracted_paths: Flags indicating if the paths are diffracted
        :param solid_angle: Ray tubes solid angles [sr] for non-diffracted paths.
            Not required for diffracted paths.
        :param tx_positions: Positions of the transmitters
        :param wedges: Properties of the intersected wedges.
            Not required for non-diffracted paths.
        :param diff_point: Position of the diffraction point on the wedge.
            Not required for non-diffracted paths.
        :param wedges_samples_cnt: Number of samples on the wedge for the
            sample's transmitter.
            Not required for non-diffracted paths.
        :param diffraction_angular_measure: Angular measure sampled on the
            Keller cone.
            Not required for non-diffracted paths.
        """

        # Indices of the hit cells
        cell_ind = self._local_to_cell_ind(si.uv)
        # Indices of the item in the tensor storing the radio maps
        tensor_ind = tx_indices * self.cells_count + cell_ind

        # Contribution to the path loss map
        a = dr.zeros(mi.Vector4f, 1)
        for e_field, aw in zip(e_fields, array_w):
            a += aw @ e_field
        a = dr.squared_norm(a)

        # Ray weight for non-diffracted paths
        if not diffracted_paths:
            k_local = si.to_local(k_world)
            cos_theta = dr.abs(k_local.z)
            w = solid_angle * dr.rcp(cos_theta)
        else:
            # Ray weight for diffracted paths
            tx_positions_ = dr.gather(mi.Point3f, tx_positions, tx_indices,
                                     active=active)
            w = self._diffraction_integration_weight(wedges, tx_positions_,
                                                   diff_point, k_world, si)
            # Multiply by edge length and the angular measure actually sampled
            w *= wedges.length * diffraction_angular_measure
            # Divide by the number of samples on this edge
            w /= wedges_samples_cnt

        # Apply ray weight and normalization factor
        a *= w * self._normalization_factor

        # Update the path loss map
        dr.scatter_reduce(dr.ReduceOp.Add, self._pathgain_map.array, value=a,
                          index=tensor_ind, active=active)

    @property
    def to_world(self):
        r"""Transform that maps a unit square in the X-Y plane to the rectangle
        that defines the radio map surface

        :type: :py:class:`mi.Transform4f`
        """
        return self._build_transform()

    def show(
        self,
        metric: str = "path_gain",
        tx: int | str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        show_tx: bool = True,
        show_rx: bool = False
    ) -> plt.Figure:
        r"""Visualizes a radio map

        The position of the transmitters is indicated by "+" markers.
        The positions of the receivers are indicated by "x" markers.

        :param metric: Metric to show.
            One of ``"path_gain"``, ``"rss"``, or ``"sinr"``.

        :param tx: Index or name of the transmitter for which to show the radio
            map. If `None`, the maximum value over all transmitters for each
            cell is shown.

        :param vmin: Defines the minimum value [dB] for the colormap covers.
            If set to `None`, then the minimum value across all cells is used.

        :param vmax: Defines the maximum value [dB] for the colormap covers.
            If set to `None`, then the maximum value across all cells is used.

        :param show_tx: If set to `True`, then the position of the transmitters
            are shown.

        :param show_rx: If set to `True`, then the position of the receivers are
            shown.

        :return: Figure showing the radio map
        """

        tx_cell_indices = self.tx_cell_indices
        rx_cell_indices = self.rx_cell_indices
        tensor = self.transmitter_radio_map(metric, tx)

        # Convert to dB-scale
        if metric in ["path_gain", "sinr"]:
            with warnings.catch_warnings(record=True) as _:
                # Convert the path gain to dB
                tensor = 10. * log10(tensor)
        else:
            with warnings.catch_warnings(record=True) as _:
                # Convert the signal strength to dBm
                tensor = watt_to_dbm(tensor)

        # Set label
        if metric == "path_gain":
            colorbar_label = "Path gain [dB]"
            title = "Path gain"
        elif metric == "rss":
            colorbar_label = "Received signal strength (RSS) [dBm]"
            title = 'RSS'
        else:
            colorbar_label = "Signal-to-interference-plus-noise ratio (SINR)"\
                            " [dB]"
            title = 'SINR'

        # Visualization the radio map
        fig_cm = plt.figure()
        plt.imshow(tensor.numpy(), origin='lower', vmin=vmin, vmax=vmax)

        # Set label
        if (tx is None) & (self.num_tx > 1):
            title = 'Highest ' + title + ' across all TXs'
        elif tx is not None:
            title = title + f" for TX '{tx}'"
        plt.colorbar(label=colorbar_label)
        plt.xlabel('Cell index (X-axis)')
        plt.ylabel('Cell index (Y-axis)')
        plt.title(title)

        # Show transmitter, receiver
        if show_tx:
            if tx is not None:
                fig_cm.axes[0].scatter(tx_cell_indices.x[tx],
                                    tx_cell_indices.y[tx],
                                    marker='P',
                                    color=DEFAULT_TRANSMITTER_COLOR)
            else:
                for tx_ in range(self.num_tx):
                    fig_cm.axes[0].scatter(tx_cell_indices.x[tx_],
                                        tx_cell_indices.y[tx_],
                                        marker='P',
                                        color=DEFAULT_TRANSMITTER_COLOR)

        if show_rx:
            for rx in range(self.num_rx):
                fig_cm.axes[0].scatter(rx_cell_indices.x[rx],
                                    rx_cell_indices.y[rx],
                                    marker='x',
                                    color=DEFAULT_RECEIVER_COLOR)

        return fig_cm

    def show_association(
        self,
        metric: str = "path_gain",
        show_tx: bool = True,
        show_rx: bool = False,
        color_map: str | np.ndarray | None = None,
    ) -> plt.Figure:
        r"""Visualizes cell-to-tx association for a given metric

        The positions of the transmitters and receivers are indicated
        by "+" and "x" markers, respectively.

        :param metric: Metric to show.
            One of ``"path_gain"``, ``"rss"``, or ``"sinr"``.

        :param show_tx: If set to `True`, then the position of the transmitters
            are shown.

        :param show_rx: If set to `True`, then the position of the receivers are
            shown.

        :param color_map: Either the name of a Matplotlib colormap or a NumPy
            array of shape (num_tx, 3) containing the RGB values for the colors
            of the transmitters. Listed colormaps use their discrete color
            list; continuous colormaps are sampled at ``num_tx`` locations. If
            None, a default color map with the right number of colors is
            generated.

        :return: Figure showing the cell-to-transmitter association
        """

        tx_cell_indices = self.tx_cell_indices
        rx_cell_indices = self.rx_cell_indices

        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")

        # Create the colormap and normalization
        if color_map is None:
            # Generate a colormap with enough distinct colors
            # for all transmitters
            colors = mpl.colormaps["rainbow"](np.linspace(0, 1.0, self.num_tx))
        elif isinstance(color_map, str):
            cmap = mpl.colormaps[color_map]
            # Listed colormaps expose a finite `.colors` table; continuous ones
            # do not and must be sampled as callables.
            if hasattr(cmap, "colors"):
                colors = cmap.colors
            else:
                colors = cmap(np.linspace(0, 1.0, self.num_tx))
        else:
            colors = color_map
        del color_map

        if len(colors) < self.num_tx:
            raise ValueError(f"The color map has {len(colors)} entries, but"
                             f" there are {self.num_tx} transmitters. Please"
                             " provide a color map with at least as many"
                             " entries as the number of transmitters.")

        colors = colors[:self.num_tx]

        cmap, norm = from_levels_and_colors(
            list(range(self.num_tx + 1)), colors)
        fig_tx = plt.figure()

        plt.imshow(self.tx_association(metric).numpy(),
                    origin='lower', cmap=cmap, norm=norm)
        plt.xlabel('Cell index (X-axis)')
        plt.ylabel('Cell index (Y-axis)')
        plt.title('Cell-to-TX association')
        cbar = plt.colorbar(label="TX")
        cbar.ax.get_yaxis().set_ticks([])
        for tx in range(self.num_tx):
            cbar.ax.text(.5, tx + .5, str(tx), ha='center', va='center')

        # Show transmitter, receiver
        if show_tx:
            for tx in range(self.num_tx):
                fig_tx.axes[0].scatter(tx_cell_indices.x[tx],
                                       tx_cell_indices.y[tx],
                                       marker='P',
                                       color=DEFAULT_TRANSMITTER_COLOR)

        if show_rx:
            for rx in range(self.num_rx):
                fig_tx.axes[0].scatter(rx_cell_indices.x[rx],
                                       rx_cell_indices.y[rx],
                                       marker='x',
                                       color=DEFAULT_RECEIVER_COLOR)

        return fig_tx

    def tx_association(self, metric: str = "path_gain") -> mi.TensorXi:
        r"""Computes cell-to-transmitter association

        Each cell is associated with the transmitter providing the highest
        metric, such as path gain, received signal strength (RSS), or
        SINR.

        :param metric: Metric to be used.
            One of ``"path_gain"``, ``"rss"``, or ``"sinr"``.

        :return: Cell-to-transmitter association
        """
        tx_association = super().tx_association(metric)

        # Reshape to (num_tx, cells_per_dim_y, cells_per_dim_x)
        tx_association = dr.reshape(mi.TensorXi, tx_association,
                                    [self.cells_per_dim.y[0],
                                    self.cells_per_dim.x[0]])
        return tx_association

    def sample_positions(
        self,
        num_pos: int,
        metric: str = "path_gain",
        min_val_db: float | None = None,
        max_val_db: float | None = None,
        min_dist: float | None = None,
        max_dist: float | None = None,
        tx_association: bool = True,
        center_pos: bool = False,
        seed: int = 1
    ) -> Tuple[mi.TensorXf, mi.TensorXu, mi.TensorXu]:
        # pylint: disable=line-too-long
        r"""Samples random user positions in a scene based on a radio map

        For a given radio map, ``num_pos`` random positions are sampled
        around each transmitter, such that the selected metric, e.g., SINR, is
        larger than ``min_val_db`` and/or smaller than ``max_val_db``.
        Similarly, ``min_dist`` and ``max_dist`` define the minimum and maximum
        distance of the random positions to the transmitter under consideration.
        By activating the flag ``tx_association``, only positions are sampled
        for which the selected metric is the highest across all transmitters.
        This is useful if one wants to ensure, e.g., that the sampled positions
        for each transmitter provide the highest SINR or RSS.

        If no cell satisfies the constraints for a transmitter, that
        transmitter contributes no samples: the corresponding entry of
        ``num_valid`` is zero, the matching positions are set to NaN, and the
        matching cell indices are undefined and must not be used.

        Note that due to the quantization of the radio map into cells it is
        not guaranteed that all above parameters are exactly fulfilled for a
        returned position. This stems from the fact that every
        individual cell of the radio map describes the expected *average*
        behavior of the surface within this cell. For instance, it may happen
        that half of the selected cell is shadowed and, thus, no path to the
        transmitter exists but the average path gain is still larger than the
        given threshold. Please enable the flag ``center_pos`` to sample only
        positions from the cell centers.

        .. code-block:: Python

            import numpy as np
            import sionna
            from sionna.rt import load_scene, PlanarArray, Transmitter,\
                                RadioMapSolver, Receiver

            scene = load_scene(sionna.rt.scene.munich)

            # Configure antenna array for all transmitters
            scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.7,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V")
            # Add a transmitters
            tx = Transmitter(name="tx",
                        position=[-195,-240,30],
                        orientation=[0,0,0])
            scene.add(tx)

            solver = RadioMapSolver()
            rm = solver(scene, cell_size=(1., 1.), samples_per_tx=100000000)

            positions, _, num_valid = rm.sample_positions(
                num_pos=200, min_val_db=-100., min_dist=50., max_dist=80.)
            positions = positions.numpy()
            positions = np.squeeze(positions, axis=0)

            for i,p in enumerate(positions):
                rx = Receiver(name=f"rx-{i}",
                            position=p,
                            orientation=[0,0,0])
                scene.add(rx)

            scene.preview(clip_at=10.);

        .. figure:: ../figures/rm_user_sampling.png
            :align: center

        The above example shows an example for random positions between 50m and
        80m from the transmitter and a minimum path gain of -100 dB.
        Keep in mind that the transmitter can have a different height than the
        radio map which also contributes to this distance.
        For example if the transmitter is located 20m above the surface of the
        radio map and a ``min_dist`` of 20m is selected, also positions
        directly below the transmitter are sampled.

        :param num_pos: Number of returned random positions for each transmitter

        :param metric: Metric to be considered for sampling positions.
            One of ``"path_gain"``, ``"rss"``, or ``"sinr"``.

        :param min_val_db: Minimum value for the selected metric ([dB] for path
            gain and SINR; [dBm] for RSS).
            Positions are only sampled from cells where the selected metric is
            larger than or equal to this value. Ignored if `None`.

        :param max_val_db: Maximum value for the selected metric ([dB] for path
            gain and SINR; [dBm] for RSS).
            Positions are only sampled from cells where the selected metric is
            smaller than or equal to this value.
            Ignored if `None`.

        :param min_dist:  Minimum distance [m] from transmitter for all random
            positions. Ignored if `None`.

        :param max_dist: Maximum distance [m] from transmitter for all random
            positions. Ignored if `None`.

        :param tx_association: If `True`, only positions associated with a
            transmitter are chosen, i.e., positions where the chosen metric is
            the highest among all all transmitters. Else, a user located in a
            sampled position for a specific transmitter may perceive a higher
            metric from another TX.

        :param center_pos: If `True`, all returned positions are sampled from
            the cell center (i.e., the grid of the radio map). Otherwise, the
            positions are randomly drawn from the surface of the cell.

        :return: Random positions :math:`(x,y,z)` [m]
            (shape: ``[num_tx, num_pos, 3]``). For transmitter ``n``,
            only the first ``num_valid[n]`` entries are valid; the remaining
            entries are NaN when ``num_valid[n] == 0``.

        :return: Cell indices (shape ``[num_tx, num_pos, 2]``)
            corresponding to the random positions in the format `(row, column)`.
            Entries beyond ``num_valid[n]`` are undefined.

        :return: Number of valid samples per transmitter
            (shape ``[num_tx]``)
        """

        sampled_cells, num_valid = super().sample_cells(num_pos,
                                             metric,
                                             min_val_db, max_val_db,
                                             min_dist, max_dist,
                                             tx_association,
                                             seed)

        # Per-sample validity mask: [num_tx * num_pos]
        valid = dr.repeat(mi.Bool(num_valid.array > 0), num_pos)

        # Centers of selected cells
        cell_centers_tensor = self.cell_centers
        cell_centers = mi.Point3f(cell_centers_tensor[..., 0].array,
                                  cell_centers_tensor[..., 1].array,
                                  cell_centers_tensor[..., 2].array)
        sampled_pos = dr.gather(mi.Point3f, cell_centers,
                                dr.ravel(sampled_cells), active=valid)
        sampled_pos = dr.select(valid, sampled_pos,
                                mi.Point3f(dr.nan))

        if not center_pos:
            # Directions with respect to which to apply the offset
            to_world = rotation_matrix(self._orientation)
            x_dir = to_world @ mi.Vector3f(0.5 * self._cell_size.x, 0, 0)
            y_dir = to_world @ mi.Vector3f(0, 0.5 * self._cell_size.y, 0)

            # Sample a random offset of each position
            self._sampler.seed(seed, num_pos * self.num_tx)
            offset = self._sampler.next_2d() * 2 - 1
            offset = offset.x * x_dir + offset.y * y_dir
            sampled_pos = dr.select(valid, sampled_pos + offset, sampled_pos)

        sampled_pos = dr.reshape(mi.TensorXf, sampled_pos,
                                [self.num_tx, num_pos, 3])

        # Switch to (row, column) format for cell indices
        sampled_cells_y = sampled_cells // self.cells_per_dim.x[0] # Row
        sampled_cells_x = sampled_cells % self.cells_per_dim.x[0] # Column
        sampled_cells = dr.zeros(mi.TensorXu, [self.num_tx * num_pos, 2])
        sampled_cells[...,0] = sampled_cells_y.array
        sampled_cells[...,1] = sampled_cells_x.array
        sampled_cells = dr.reshape(mi.TensorXu, sampled_cells,
                                [self.num_tx, num_pos, 2])

        return sampled_pos, sampled_cells, num_valid

    ###############################################
    # Internal methods
    ###############################################

    def _local_to_cell_ind(self, p_local: mi.Point2f) -> mi.Int:
        """
        Computes the indices of the hitted cells of the map from the local
        :math:`(x,y)` coordinates

        :param p_local: Coordinates of the intersected points in the
            measurement plane local frame

        :return: Cell indices in the flattened measurement plane
        """

        # Protect against uv == 1.0
        p_local[p_local == 1.0] = dr.one_minus_epsilon(mi.Float)

        # Size of a cell in UV space
        cell_size_uv = mi.Vector2f(self._cells_per_dim)

        # Cell indices in the 2D measurement plane
        cell_ind = mi.Point2i(dr.floor(p_local * cell_size_uv))

        # Cell indices for the flattened measurement plane
        cell_ind = cell_ind[1] * self._cells_per_dim[0] + cell_ind[0]

        return cell_ind

    def _global_to_cell_ind(self, p_global: mi.Point3f) -> mi.Point2u:
        """
        Computes the indices of the cells which includes the global
        :math:`(x,y,z)` coordinates

        :param p_global: Coordinates of the a point on the measurement plane
            in the global frame

        :return: `(x, y)` indices of the cell which contains `p_global`
        """

        if dr.width(p_global) == 0:
            return mi.Point2u()

        to_world = rotation_matrix(self._orientation)
        to_local = to_world.T

        p_local = to_local @ (p_global - self._center)

        # Discard the Z coordinate
        p_local = mi.Point2f(p_local.x, p_local.y)

        # Compute cell indices
        ind = p_local + 0.5 * self._size
        ind = mi.Point2u(dr.floor(ind / self._cell_size))

        return ind

    def _build_transform(self, scalar: bool = False) -> mi.Transform4f \
                                                        | mi.ScalarTransform4f:
        """Build the `to_world` transform for the plane based on center,
        orientation, and size properties."""

        orientation_deg = self._orientation * 180. / dr.pi
        center = self._center
        size = self._size

        if scalar:
            tp = mi.ScalarTransform4f
            orientation_deg = mi.ScalarPoint3f(orientation_deg.x[0],
                                               orientation_deg.y[0],
                                               orientation_deg.z[0])
            center = mi.ScalarPoint3f(center.x[0], center.y[0], center.z[0])
            size = mi.ScalarPoint2f(size.x[0], size.y[0])
        else:
            tp = mi.Transform4f

        return tp.translate(center) \
                 .rotate([0., 0., 1.], orientation_deg.x) \
                 .rotate([0., 1., 0.], orientation_deg.y) \
                 .rotate([1., 0., 0.], orientation_deg.z) \
                 .scale([0.5 * size.x, 0.5 * size.y, 1])

    def _compute_normalization_factor(self):
        """Computes the normalization factor for the path gain map"""

        cell_area = self._cell_size[0] * self._cell_size[1]
        return dr.square(self._wavelength * dr.rcp(4. * dr.pi)) \
               * dr.rcp(cell_area)
