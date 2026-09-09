#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Mesh radio map"""

import mitsuba as mi
import drjit as dr
from typing import List, Tuple

from sionna.rt.scene import Scene
from .radio_map import RadioMap
from sionna.rt.utils import WedgeGeometry

class MeshRadioMap(RadioMap):
    r"""
    Mesh Radio Map

    A mesh-based radio map is computed by
    a :doc:`radio map solver <radio_map_solvers>` for a measurement surface
    defined by a mesh. Each triangle of the mesh serves as a bin of the radio
    map.

    :param scene: Scene for which the radio map is computed

    :param meas_surface: Mesh to be used as the measurement surface

    :param check_cells_area: If `True`, reject early measurement surfaces that
        contain degenerate (zero-area) triangles. Default is `False`.
    """

    def __init__(self,
                 scene: Scene,
                 meas_surface: mi.Mesh,
                 check_cells_area: bool = False):

        super().__init__(scene)

        if not isinstance(meas_surface, mi.Mesh):
            raise TypeError("`meas_surface` must be a Mitsuba Mesh")
        if meas_surface.face_count() == 0:
            raise ValueError(
                "`meas_surface` must contain at least one triangle"
            )

        if check_cells_area:
            # Reject degenerate triangles early: zero area would later yield Inf
            # when path contributions are normalized by cell area.
            face_indices = meas_surface.face_indices(
                dr.arange(mi.UInt, meas_surface.face_count())
            )
            v0 = meas_surface.vertex_position(face_indices.x)
            v1 = meas_surface.vertex_position(face_indices.y)
            v2 = meas_surface.vertex_position(face_indices.z)
            e1 = v1 - v0
            e2 = v2 - v0
            cell_area = 0.5 * dr.norm(dr.cross(e1, e2))
            if bool(dr.any(cell_area <= 0)):
                raise ValueError("`meas_surface` contains degenerate triangles"
                                 " with zero area")

        self._cells_count = meas_surface.face_count()
        self._meas_surface = meas_surface

        self._normalization_factor = self._compute_normalization_factor()

        # Initialize the pathgain map to zero
        self._pathgain_map = dr.zeros(mi.TensorXf,
                                      [self.num_tx, self._cells_count])

    @property
    def measurement_surface(self):
        r"""Mitsuba shape corresponding to the
        radio map measurement surface

        :type: :py:class:`mi.Mesh`
        """
        return self._meas_surface

    @property
    def cells_count(self):
        r"""Total number of cells in the radio map

        :type: :py:class:`int`
        """
        return self._cells_count

    @property
    def cell_centers(self):
        r"""Positions of the centers of the cells in the global coordinate
        system

        :type: ``mi.Point3f [cells_count, 3]``
        """
        mesh = self.measurement_surface
        ind = mesh.face_indices(dr.arange(mi.UInt, 0, mesh.face_count()))
        v = mesh.vertex_position(dr.ravel(ind))
        v_x = dr.reshape(mi.TensorXf, v.x, [mesh.face_count(), 3])
        v_y = dr.reshape(mi.TensorXf, v.y, [mesh.face_count(), 3])
        v_z = dr.reshape(mi.TensorXf, v.z, [mesh.face_count(), 3])
        c_x = dr.mean(v_x, axis=1)
        c_y = dr.mean(v_y, axis=1)
        c_z = dr.mean(v_z, axis=1)
        c = mi.Point3f(c_x, c_y, c_z)
        return c

    @property
    def path_gain(self):
        # pylint: disable=line-too-long
        r"""Path gains across the radio map from all transmitters [unitless, linear scale]

        :type: ``mi.TensorXf [num_tx, num_primitives]``
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
        # Indices of the hit cells is the primitive ID
        tensor_ind = tx_indices * self.cells_count + si.prim_index

        # Contribution to the path loss map
        a = dr.zeros(mi.Vector4f, 1)
        for e_field, aw in zip(e_fields, array_w):
            a += aw @ e_field
        a = dr.squared_norm(a)

        # Ray weight for non-diffracted paths
        if not diffracted_paths:
            cos_theta = dr.abs(dr.dot(si.n, k_world))
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

        # Normalize by cell area
        # First, compute the primitive area
        # Primitive vertices
        meas_surface = self.measurement_surface
        prim_index = meas_surface.face_indices(si.prim_index, active=active)
        v0 = meas_surface.vertex_position(prim_index[0])
        v1 = meas_surface.vertex_position(prim_index[1])
        v2 = meas_surface.vertex_position(prim_index[2])
        # Cell area
        v1 = v1 - v0
        v2 = v2 - v0
        v1_sq_norm = dr.squared_norm(v1)
        v2_sq_norm = dr.squared_norm(v2)
        cell_area = 0.5 * dr.sqrt(v1_sq_norm * v2_sq_norm
                                  - dr.square(dr.dot(v1, v2)))
        # Skip zero-area hits so a bad mesh cannot produce Inf through ``rcp``.
        # When ``check_cells_area`` is `True`, such triangles are also rejected
        # in ``__init__``.
        active = mi.Bool(active) & (cell_area > 0)
        # Apply normalization by cell area
        w *= dr.rcp(cell_area)

        # Apply ray weight and normalization factor
        a *= w * self._normalization_factor

        # Update the path loss map
        dr.scatter_reduce(dr.ReduceOp.Add, self._pathgain_map.array, value=a,
                          index=tensor_ind, active=active)

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
            from sionna.rt import load_scene, PlanarArray, Transmitter, \
                                  RadioMapSolver, Receiver, transform_mesh

            scene = load_scene(sionna.rt.scene.san_francisco)

            # Configure antenna array for all transmitters
            scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.7,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V")
            # Add a transmitters
            tx = Transmitter(name="tx",
                        position=[15.9,121.2,25],
                        orientation=[0,0,0],
                        display_radius=2.0)
            scene.add(tx)

            # Create a measurement surface by cloning the terrain
            # and elevating it by 1.5 meters
            measurement_surface = scene.objects["Terrain"].clone(as_mesh=True)
            transform_mesh(measurement_surface,
                        translation=[0,0,1.5])

            solver = RadioMapSolver()
            rm = solver(scene,
                        cell_size=(1., 1.),
                        measurement_surface=measurement_surface,
                        samples_per_tx=100000000)

            positions, _, num_valid = rm.sample_positions(
                num_pos=200, min_val_db=-100., min_dist=50., max_dist=80.)
            positions = positions.numpy()
            positions = np.squeeze(positions, axis=0)

            for i,p in enumerate(positions):
                rx = Receiver(name=f"rx-{i}",
                            position=p,
                            orientation=[0,0,0],
                            display_radius=2.0)
                scene.add(rx)

            scene.preview();

        .. figure:: ../figures/rm_mesh_user_sample.png
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

        :return: Cell indices (shape ``[num_tx, num_pos]``)
            corresponding to the random positions. Entries beyond
            ``num_valid[n]`` are undefined.

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

        # If set to True,samples the positions from within the primitive
        if center_pos:
            cell_centers = self.cell_centers
            sampled_pos = dr.gather(mi.Point3f, cell_centers,
                                    dr.ravel(sampled_cells), active=valid)
            sampled_pos = dr.select(valid, sampled_pos,
                                    mi.Point3f(dr.nan))
            sampled_pos = dr.reshape(mi.TensorXf, sampled_pos,
                                    [self.num_tx, num_pos, 3])
        else:
            # Reset sampler
            self._sampler.seed(seed, num_pos*self.num_tx)
            #
            v_ind = self.measurement_surface.face_indices(
                dr.ravel(sampled_cells), active=valid)
            # Three vertices of the triangle
            v0 = v_ind.x
            v1 = v_ind.y
            v2 = v_ind.z
            v0 = self.measurement_surface.vertex_position(dr.ravel(v0),
                                                          active=valid)
            v1 = self.measurement_surface.vertex_position(dr.ravel(v1),
                                                          active=valid)
            v2 = self.measurement_surface.vertex_position(dr.ravel(v2),
                                                          active=valid)
            # Uniformly sample point within the primitive
            esp = self._sampler.next_2d()
            s = dr.sqrt(esp.x)
            t = esp.y
            # Barycentric coordinates
            p = (1-s)*v0 + s*(1-t)*v1 + s*t*v2
            p = dr.select(valid, p, mi.Point3f(dr.nan))
            # Reshape
            sampled_pos = dr.reshape(mi.TensorXf, p.array, [self.num_tx, num_pos, 3])

        return sampled_pos, sampled_cells, num_valid

    def _compute_normalization_factor(self):
        """Computes the normalization factor for the path gain map"""

        return dr.square(self._wavelength * dr.rcp(4. * dr.pi))
