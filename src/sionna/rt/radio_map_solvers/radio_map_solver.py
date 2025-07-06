#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Radio map solver"""

import mitsuba as mi
import drjit as dr
from typing import Tuple, Callable, List

from sionna.rt.utils import spawn_ray_from_sources, fibonacci_lattice,\
    rotation_matrix, spectrum_to_matrix_4f
from sionna.rt import Scene
from sionna.rt.antenna_pattern import antenna_pattern_to_world_implicit
from sionna.rt.constants import InteractionType
from sionna.rt.scene_object import SceneObject
from sionna.rt.scene import extend_scene_with_mesh
from .radio_map import RadioMap
from .planar_radio_map import PlanarRadioMap
from .mesh_radio_map import MeshRadioMap
from .dem_radio_map import DemRadioMap


class RadioMapSolver:
    # pylint: disable=line-too-long
    r"""
    Class that implements the radio map solver

    This solver generates a radio map for each transmitter within the scene.
    For any given transmitter, the radio map is calculated over a measurement surface that
    is segmented into flat cells. Each cell :math:`i` in the radio map is associated
    with the quantity:

    .. math::
        :label: cm_def

        g_i = \frac{1}{\lvert C_i \rvert} \int_{C_{i}} \lvert h(s) \rvert^2 ds.

    Here, :math:`\lvert h(s) \rvert^2` represents the sum of the squared amplitudes of the path coefficients
    :math:`a_i` at the position :math:`s=(x,y)`, assuming an ideal isotropic receiver.
    The integral is evaluated over the cell :math:`C_i`, with :math:`ds` being the infinitesimally
    small surface element, defined as :math:`ds=dx \cdot dy`. The value :math:`g_i` can be interpreted
    as the average :attr:`~sionna.rt.RadioMap.path_gain` across a cell.
    This solver approximates :math:`g_i` using Monte Carlo integration. For further details,
    refer to the `section on the radio map solver of the Sionna RT Technical Report <https://nvlabs.github.io/sionna/rt/tech-report/S4.html>`_.

    The path gain can be transformed into the received signal strength (:attr:`~sionna.rt.RadioMap.rss`)
    by multiplying it with the transmit :attr:`~sionna.rt.Transmitter.power`:

    .. math::

        \mathrm{RSS}_{i} = P_{tx} g_{i}.

    In scenarios with multiple transmitters, the signal-to-interference-plus-noise ratio (SINR) for a specific
    transmitter :math:`k` is calculated as follows:

    .. math::

        \mathrm{SINR}^k_{i}=\frac{\mathrm{RSS}^k_{i}}{N_0+\sum_{k'\ne k} \mathrm{RSS}^{k'}_{i}}

    where :math:`N_0` [W] represents the thermal noise power of the scene (:attr:`~sionna.rt.Scene.thermal_noise_power`), which is determined by:

    .. math::

        N_0 = B \times T \times k.

    In this equation, :math:`B` [Hz] is the bandwidth of the transmission (:attr:`~sionna.rt.Scene.bandwidth`),
    :math:`T` [K] is the temperature of the scene (:attr:`~sionna.rt.Scene.temperature`), and
    :math:`k = 1.380649 \times 10^{-23}` [J/K] is the Boltzmann constant.
    
    This solver supports arbitrary meshes as measurement surfaces, where each triangle in the mesh acts as a cell in the radio map.
    If a mesh is not provided, the solver computes the radio map over a rectangular measurement grid, which is defined by the
    parameters: ``center``, ``orientation``, ``size``, and ``cell_size``. The resulting output is then a real-valued
    matrix of dimensions ``[num_cells_y, num_cells_x]`` for each transmitter, where:

    .. math::
        \texttt{num_cells_x} = \bigg\lceil\frac{\texttt{size[0]}}{\texttt{cell_size[0]}} \bigg\rceil\\
        \texttt{num_cells_y} = \bigg\lceil \frac{\texttt{size[1]}}{\texttt{cell_size[1]}} \bigg\rceil.

    An orientation of (0,0,0) aligns the radio map parallel to the XY plane, with the surface normal directed towards
    the :math:`+z` axis. By default, the radio map is set parallel to the XY plane, spans the entire scene, and
    is positioned at an elevation of :math:`z = 1.5\text{m}`.

    For transmitters equipped with multiple antennas, transmit precoding is applied as specified by the ``precoding_vec``.
    For each path :math:`n` that intersects a radio map cell :math:`i`, the channel coefficients :math:`a_n` and the angles
    of departure (AoDs) :math:`(\theta_{\text{T},n}, \varphi_{\text{T},n})` and arrival
    (AoAs) :math:`(\theta_{\text{R},n}, \varphi_{\text{R},n})` are determined. For further information, refer to the
    `Primer on Electromagnetics <../em_primer.html>`_.

    A "synthetic" array is simulated by introducing additional phase shifts, which are dependent on the antenna's position
    relative to the transmitter and the AoDs. Let :math:`\mathbf{d}_{\text{T},k}\in\mathbb{R}^3` represent the relative
    position of the :math:`k^\text{th}` antenna (relative to the transmitter's position) for which the channel impulse
    response is calculated. This can be accessed via the antenna array's property
    :attr:`~sionna.rt.AntennaArray.positions`. Assuming a plane-wave model, the phase shift for this antenna is computed as:

    .. math::

        p_{\text{T},n,k} = \frac{2\pi}{\lambda}\hat{\mathbf{r}}(\theta_{\text{T},n}, \varphi_{\text{T},n})^\mathsf{T} \mathbf{d}_{\text{T},k}.

    The path coefficient for the :math:`k\text{th}` antenna is then expressed as:

    .. math::

        h_{n,k} =  a_n e^{j p_{\text{T}, n,k}}.

    These coefficients form the complex-valued channel vector :math:`\mathbf{h}_n` with a size of :math:`\texttt{num_tx_ant}`.

    Finally, the coefficient for the equivalent SISO channel is given by:

    .. math::
        h_n =  \mathbf{h}_n^{\textsf{H}} \mathbf{p}

    where :math:`\mathbf{p}` denotes the precoding vector ``precoding_vec``.

    Note
    -----
    This solver supports Russian roulette, which can significantly improve the
    efficiency of ray tracing by terminating rays that contribute little to the final
    result.

    The implementation of Russian roulette in this solver consists in terminating
    a ray with probability equal to the complement of its path gain (without
    the distance-based path loss). Formally,
    after the :math:`d^{\text{th}}` bounce, the ray path loss is set to:

    .. math::

        a_d \leftarrow
        \begin{cases}
            \frac{a_d}{\sqrt{\min \{ p_{c},|a_d|^2 \}}},  & \text{with probability } \min \{ p_{c},|a_d|^2 \}\\
            0, & \text{with probability } 1 - \min \{ p_{c},|a_d|^2 \}
        \end{cases}

    where :math:`a_d` is the path coefficient corresponding to the ray (without
    the distance-based pathloss) and
    :math:`p_c` the maximum probability with which to continue a path (``rr_prob``).
    The first case consists in continuing the ray, whereas the second case consists
    in terminating the ray. When the ray is continued, the scaling by
    :math:`\frac{1}{\sqrt{\min \{ p_{c},|a_d|^2 \}}}` ensures an unbiased map by accounting
    for the rays that were terminated. When a ray is terminated, it is no longer
    traced, leading to a reduction of the required computations.

    Russian roulette is by default disabled. It can be enabled by setting
    the ``rr_depth`` parameter to a positive value. ``rr_depth`` corresponds to
    the path depth, i.e., the number of bounces, from which on Russian roulette
    is enabled.

    Note
    -----
    The parameter ``stop_threshold`` can be set to deactivate (i.e., stop tracing)
    paths whose gain has dropped below this threshold (in dB).

    Example
    -------
    .. code-block:: python

        import sionna
        from sionna.rt import load_scene, PlanarArray, Transmitter, RadioMapSolver

        scene = load_scene(sionna.rt.scene.munich)
        scene.radio_materials["marble"].thickness = 0.5

        # Configure antenna array for all transmitters
        scene.tx_array = PlanarArray(num_rows=8,
                                num_cols=2,
                                vertical_spacing=0.7,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="VH")

        # Add a transmitters
        tx = Transmitter(name="tx",
                    position=[8.5,21,30],
                    orientation=[0,0,0])
        scene.add(tx)
        tx.look_at(mi.Point3f(40,80,1.5))

        solver = RadioMapSolver()
        rm = solver(scene, cell_size=(1., 1.), samples_per_tx=100000000)
        scene.preview(radio_map=rm, clip_at=15., rm_vmin=-100.)

    .. figure:: ../figures/radio_map_preview.png
        :align: center
    """

    def __init__(self):
        # Sampler
        self._sampler = mi.load_dict({'type': 'independent'})

        # Dr.Jit mode for running the loop that implement the solver.
        # Symbolic mode is the fastest mode but does not currently support
        # automatic differentiation.
        self._loop_mode = "symbolic"

    @property
    def loop_mode(self):
        # pylint: disable=line-too-long
        r"""Get/set the Dr.Jit mode used to evaluate the loop that implements
        the solver. Should be one of "evaluated" or "symbolic". Symbolic mode
        (default) is the fastest one but does not support automatic
        differentiation.
        For more details, see the `corresponding Dr.Jit documentation <https://drjit.readthedocs.io/en/latest/cflow.html#sym-eval>`_.

        :type: "evaluated" | "symbolic"
        """
        return self._loop_mode

    @loop_mode.setter
    def loop_mode(self, mode):
        if mode not in ("evaluated", "symbolic"):
            raise ValueError("Invalid loop mode. Must be either 'evaluated'"
                             " or 'symbolic'")
        self._loop_mode = mode

    def __call__(
        self,
        scene : Scene,
        center : mi.Point3f | None = None,
        orientation : mi.Point3f | None = None,
        size : mi.Point2f | None = None,
        cell_size : mi.Point2f = mi.Point2f(10, 10),
        measurement_surface : mi.Shape | SceneObject | None = None,
        digital_elevation_model : mi.TensorXf | None = None,
        precoding_vec : Tuple[mi.TensorXf, mi.TensorXf] | None = None,
        samples_per_tx : int = 1000000,
        max_depth : int = 3,
        los : bool = True,
        specular_reflection : bool = True,
        diffuse_reflection : bool = False,
        refraction : bool = True,
        seed : int = 42,
        rr_depth : int = -1,
        rr_prob : float = 0.95,
        stop_threshold : float | None = None
    ) -> RadioMap:
        # pylint: disable=line-too-long
        r"""
        Executes the solver

        :param scene: Scene for which to compute the radio map

        :param center: Center of the radio map measurement plane
            :math:`(x,y,z)` [m] as a three-dimensional vector.
            Ignored if ``measurement_surface`` is provided.
            If set to `None`, the radio map is centered on the center of the
            scene, except for the elevation :math:`z` that is set to 1.5m.
            Otherwise, ``orientation`` and ``size`` must be provided.

        :param orientation: Orientation of the radio map measurement plane
            :math:`(\alpha, \beta, \gamma)` specified through three angles
            corresponding to a 3D rotation as defined in :eq:`rotation`.
            Ignored if ``measurement_surface`` is provided.
            An orientation of :math:`(0,0,0)` or `None` corresponds to a
            radio map that is parallel to the XY plane.
            If not set to `None`, then ``center`` and ``size`` must be
            provided.

        :param size:  Size of the radio map measurement plane [m].
            Ignored if ``measurement_surface`` is provided.
            If set to `None`, then the size of the radio map is set such that
            it covers the entire scene.
            Otherwise, ``center`` and ``orientation`` must be provided.

        :param cell_size: Size of a cell of the radio map measurement plane [m].
            Ignored if ``measurement_surface`` is provided.

        :param measurement_surface: Measurement surface. If set, the
            radio map is computed for this surface, where every triangle in the
            mesh is a cell in the radio map.
            If set to `None`, then the radio map is computed for a measurement
            grid defined by ``center``, ``orientation``, ``size``, and ``cell_size``.

        :param digital_elevation_model: Digital Elevation Model (DEM). If set, 
            the provided DEM is triangulated, the radio map is computed on a 
            grid draped over the resulting mesh.
            If set to `None`, then the radio map is computed for a measurement
            grid defined by ``center``, ``orientation``, ``size``, and ``cell_size``.

        :param precoding_vec: Real and imaginary components of the
            complex-valued precoding vector.
            If set to `None`, then defaults to
            :math:`\frac{1}{\sqrt{\text{num_tx_ant}}} [1,\dots,1]^{\mathsf{T}}`.

        :param samples_per_tx: Number of samples per source

        :param max_depth: Maximum depth

        :param los: Enable line-of-sight paths

        :param specular_reflection: Enable specular reflections

        :param diffuse_reflection: Enable diffuse reflectios

        :param refraction: Enable refraction

        :param seed: Seed

        :param rr_depth: Depth from which on to start Russian roulette

        :param rr_prob: Maximum probability with which to keep a path when
            Russian roulette is enabled

        :param stop_threshold: Gain threshold [dB] below which a path is
            deactivated

        :return: Computed radio map
        """

        # Check that the scene is all set for simulations
        scene.all_set(radio_map=True)

        # Check and initialize the precoding vector
        num_tx = len(scene.transmitters)
        num_tx_ant = scene.tx_array.num_ant
        if precoding_vec is None:
            precoding_vec_real = dr.ones(mi.TensorXf, [num_tx, num_tx_ant])
            precoding_vec_real /= dr.sqrt(scene.tx_array.num_ant)
            precoding_vec_imag = dr.zeros(mi.TensorXf, [num_tx, num_tx_ant])
            precoding_vec = (precoding_vec_real, precoding_vec_imag)
        else:
            precoding_vec_real, precoding_vec_imag = precoding_vec
            if not isinstance(precoding_vec_real, type(precoding_vec_imag)):
                raise TypeError("The real and imaginary components of "\
                                "`precoding_vec` must be of the same type")
            # If a single precoding vector is provided, then it is used by
            # all transmitters
            if ( isinstance(precoding_vec_real, mi.Float) or
                (isinstance(precoding_vec_real, mi.TensorXf)
                 and len(dr.shape(precoding_vec_real)) == 1) ):
                precoding_vec_real = mi.Float(precoding_vec_real)
                precoding_vec_imag = mi.Float(precoding_vec_imag)

                precoding_vec_real = dr.tile(precoding_vec_real, num_tx)
                precoding_vec_imag = dr.tile(precoding_vec_imag, num_tx)
                #
                precoding_vec_real = dr.reshape(mi.TensorXf, precoding_vec_real,
                                                [num_tx, num_tx_ant])
                precoding_vec_imag = dr.reshape(mi.TensorXf, precoding_vec_imag,
                                                [num_tx, num_tx_ant])
                precoding_vec = (precoding_vec_real, precoding_vec_imag)

        # Transmitter configurations
        # Generates sources positions and orientations
        tx_positions, tx_orientations, rel_ant_positions_tx, _ = \
            scene.sources(True, False)
        dr.make_opaque(tx_positions, tx_orientations)

        # Trace paths and compute channel impulse responses
        tx_antenna_patterns = scene.tx_array.antenna_pattern.patterns

        num_tx = dr.shape(tx_positions)[1]
        num_samples = samples_per_tx * num_tx

        # If the Russian roulette threshold depth is set to -1, disable Russian
        # roulette by setting the threshold depth to a value higher than
        # `max_depth`
        if rr_depth == -1:
            rr_depth = max_depth + 1

        # If a threshold for the path gain is set below which paths are
        # deactivated, then convert it to linear scale
        if stop_threshold is not None:
            stop_threshold = dr.power(10., stop_threshold / 10.)

        # Set the seed of the sampler
        self._sampler.seed(seed, num_samples)

        # Build Radio Map instance
        # Modify the scene for non-planar radio maps.
        if measurement_surface is not None and digital_elevation_model is not None:
            raise ValueError("At most one of `measurement_surface` and `digital_elevation_model` can be provided.")
        elif measurement_surface is not None:
            if isinstance(measurement_surface, SceneObject):
                measurement_surface = measurement_surface.mi_mesh
            modified_scene = extend_scene_with_mesh(scene.mi_scene, measurement_surface)
            radio_map = MeshRadioMap(scene, measurement_surface)
        elif digital_elevation_model is not None:
            radio_map = DemRadioMap(scene=scene,
                                    elevation=digital_elevation_model,
                                    cell_size=cell_size,
                                    center=center,
                                    size=size)
            measurement_surface = radio_map.measurement_surface
            modified_scene = extend_scene_with_mesh(scene.mi_scene, measurement_surface)
        else:
            modified_scene = scene.mi_scene
            radio_map = PlanarRadioMap(scene, cell_size, center, orientation, size)

        # Computes the pathloss map
        # `radio_map` is updated in-place
        with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, False):
            with scene.use_mi_scene(modified_scene):
                self._shoot_and_bounce(scene, radio_map, self._sampler,
                                       tx_positions, tx_orientations,
                                       tx_antenna_patterns,
                                       precoding_vec, rel_ant_positions_tx,
                                       samples_per_tx, max_depth,
                                       los, specular_reflection, diffuse_reflection,
                                       refraction, rr_depth, rr_prob,
                                       stop_threshold)

        return radio_map

    # pylint: disable=line-too-long
    @dr.syntax
    def _shoot_and_bounce(
        self,
        scene : Scene,
        radio_map : RadioMap,
        sampler : mi.Sampler,
        tx_positions : mi.Point3f,
        tx_orientations : mi.Point3f,
        tx_antenna_patterns : List[Callable[[mi.Float, mi.Float],
                                            Tuple[mi.Complex2f, mi.Complex2f]]],
        precoding_vec : Tuple[mi.TensorXf, mi.TensorXf],
        rel_ant_positions_tx : mi.Point3f,
        samples_per_tx : int,
        max_depth : int,
        los : bool,
        specular_reflection : bool,
        diffuse_reflection : bool,
        refraction : bool,
        rr_depth : int,
        rr_prob : float,
        stop_threshold : float | None,
    ) -> None:
        r"""
        Executes the shoot-and-bounce loop

        The ``radio_map`` is updated in-place.

        :param scene: Scene for which to compute the radio map
        :param radio_map: Radio map
        :param sampler: Sampler used to generate random numbers and vectors
        :param tx_positions: Positions of the transmitters
        :param tx_orientations: Orientations of the transmitters
        :param tx_antenna_patterns: Antenna pattern of the transmitters
        :param precoding_vec: Real and imaginary components of the
            complex-valued precoding vector
        :param rel_ant_positions_tx: Positions of the antenna elements relative
            to the transmitters positions
        :param samples_per_tx: Number of samples per source
        :param max_depth: Maximum depth
        :param los: If set to `True`, then the line-of-sight paths are computed
        :param specular_reflection: If set to `True`, then the specularly
            reflected paths are computed
        :param diffuse_reflection: If set to `True`, then the diffusely
            reflected paths are computed
        :param refraction: If set to `True`, then the refracted paths are
            computed
        :param rr_depth: Depth from which to start Russian roulette
        :param rr_prob: Minimum probability with which to keep a path when
            Russian roulette is enabled
        :param stop_threshold: Gain threshold (linear scale) below which a path
            is deactivated
        """

        num_txs = dr.shape(tx_positions)[1]
        num_samples = samples_per_tx * num_txs
        num_tx_ant_patterns = len(tx_antenna_patterns)

        tx_indices = dr.repeat(dr.arange(mi.UInt, num_txs), samples_per_tx)

        # Spawn rays from the transmitters
        ray = spawn_ray_from_sources(fibonacci_lattice, samples_per_tx,
                                     tx_positions)

        # Weights to account for the antenna array and precoding
        array_w = self._synthetic_array_weighting(scene, ray.d,
                                                  rel_ant_positions_tx,
                                                  precoding_vec)

        # Mask indicating which rays are active
        active = dr.full(dr.mask_t(mi.Float), True, num_samples)

        # Length of the ray tube.
        # Only required if a threshold is set to deactivate path based on their
        # gain
        if stop_threshold is not None:
            ray_tube_length = dr.zeros(mi.Float, num_samples)
        else:
            ray_tube_length = None

        # Solid angle of the ray tube.
        # It is required to compute the diffusely reflected field.
        # Initialized assuming that all the rays initially spawn from the source
        # share the unit sphere equally, i.e., initialized to
        # 4*PI/samples_per_src
        solid_angle = dr.full(mi.Float, 4.0 * dr.pi * dr.rcp(samples_per_tx),
                              num_samples)
        # Initialize the electric field
        # Orientation of sources and targets and correspond to-world transforms
        sample_tx_orientation = dr.repeat(tx_orientations, samples_per_tx)
        tx_to_world = rotation_matrix(sample_tx_orientation)
        # The following loop transports and update the electric field.
        # The electric field is initialized by the source antenna pattern
        e_fields = [antenna_pattern_to_world_implicit(src_antenna_pattern,
                                                      tx_to_world, ray.d,
                                                      direction="out")
                    for src_antenna_pattern in tx_antenna_patterns]

        depth = mi.UInt(0)
        while dr.hint(active, mode=self.loop_mode, exclude=[array_w]):

            # Test intersection with the scene
            si_scene = scene.mi_scene.ray_intersect(ray, active=active)

            # Each interaction processes either an interaction with the scene
            # or an interaction with the measurement plane.
            # `mp_int` is set to `True` if the interaction is with the measurement
            # plane.
            # `scene_int` is set to `True` if the interaction is with the scene.
            # If the radio map is planar, then test interaction with the measurement
            # plane. Otherwise, the measurement surface is assumed to be part of
            # the scene and we only need to check that it has hit the measurement surface.
            si_mp = None
            if isinstance(radio_map, PlanarRadioMap):
                si_mp = radio_map.measurement_surface.ray_intersect(ray,
                                                                active=active)
                # An intersection with the measurement plane is valid only if
                # (i) it was not obstructed by the scene, and (ii) the intersection
                # is valid.
                mp_int = active & (si_mp.t < si_scene.t) & si_mp.is_valid()
                # Ignore interaction with measurement plane if LoS is disabled
                mp_int &= ((depth > 0) | los)
            else:
                # The measurement surface is part of the scene
                si_mp = si_scene
                mp_int = si_mp.shape == radio_map.measurement_surface

            # Update the radio map
            # The radio map is not updated if LoS is disabled and it is a
            # LoS interaction with the measurement surface
            # Note that we need to redo the LoS test to handle the case where
            # the measurement surface is part of the scene
            update_radio_map = mp_int & ((depth > 0) | los)
            radio_map.add(e_fields, solid_angle, array_w, si_mp, ray.d,
                          tx_indices, update_radio_map)

            # Update the active state of rays
            # Active if has hit the scene (which includes the measurement surface
            # if it is part of the scene)
            active &= si_scene.is_valid()

            # It is an interaction with the scene if (i) the ray has hit the scene
            # and (ii) it has not hit the measurement plane
            scene_int = active & ~mp_int

            # Sample the BSDF
            # Sample random numbers to sample the BSDF
            sample1 = sampler.next_1d()
            sample2 = sampler.next_2d()
            s, e_fields = self._sample_radio_material(
                si_scene, ray.d, e_fields,
                solid_angle, sample1, sample2, specular_reflection,
                diffuse_reflection, refraction, scene_int
            )
            interaction_type = dr.select(scene_int, s.sampled_component,
                                         InteractionType.NONE)
            diffuse = scene_int & (interaction_type == InteractionType.DIFFUSE)
            # Update the solid angle
            # If a diffuse reflection is sampled, then it is set to 2PI.
            # Otherwise it is left unchanged
            solid_angle = dr.select(diffuse, dr.two_pi, solid_angle)

            # Spawn rays for next iteration
            ray_scene_int = si_scene.spawn_ray(d=s.wo)
            ray_mp_int = si_mp.spawn_ray(d=ray.d)
            ray = dr.select(scene_int, ray_scene_int, ray_mp_int)
            depth = dr.select(scene_int, depth + 1, depth)
            active &= (depth <= max_depth)
            active &= mp_int | (interaction_type != InteractionType.NONE)

            ## Russian roulette and gain threshold deactivation

            # Compute the path gain (without applying the spreading factor)
            gain = dr.sum([dr.squared_norm(e_field) for e_field in e_fields])
            gain /= num_tx_ant_patterns

            # Update the ray tube length
            if stop_threshold is not None:
                ray_tube_length = dr.select(diffuse, 0.0, ray_tube_length)
                ray_tube_length += dr.select(scene_int, si_scene.t, si_mp.t)

            # Is the depth threshold to activate Russian roulette reached?
            rr_inactive = depth < rr_depth
            # User specify a maximum probability of continuing tracing
            rr_continue_prob = dr.minimum(gain, rr_prob)
            # Randomly stop tracing of rays
            rr_continue = scene_int & (sampler.next_1d() < rr_continue_prob)
            active &= (rr_inactive | rr_continue)

            # Scale the remaining rays accordingly to ensure an unbiased result
            for i in range(num_tx_ant_patterns):
                e_fields[i] = dr.select(rr_inactive, e_fields[i],
                                        e_fields[i] * dr.rsqrt(rr_continue_prob))

            # Deactivate rays with gains below the set threshold
            if stop_threshold is not None:
                gain_pl = gain*dr.square(scene.wavelength
                                         * dr.rcp(4. * dr.pi * ray_tube_length))
                th_continue = scene_int & (gain_pl > stop_threshold)
                active &= th_continue

        # Finalizes the computation of the radio maps
        radio_map.finalize()

    @dr.syntax
    def _synthetic_array_weighting(
        self,
        scene : Scene,
        k_tx : mi.Vector3f,
        rel_ant_positions_tx : mi.Point3f,
        precoding_vec : Tuple[mi.TensorXf, mi.TensorXf],
    ) -> List[mi.Float]:
        r"""
        Computes the weighting to apply to the electric field to synthetically
        model the transmitter array

        :param scene: Scene for which to compute the radio map
        :param k_tx: Directions of departures of paths
        :param rel_ant_positions_tx: Positions of the antenna elements relative
            to the transmitters positions
        :param precoding_vec: Real and imaginary components of the
            complex-valued precoding vector

        :return: Weightings
        """

        precoding_vec_real, precoding_vec_imag = precoding_vec
        array_size = scene.tx_array.array_size
        num_patterns = len(scene.tx_array.antenna_pattern.patterns)
        num_tx = len(scene.transmitters)
        samples_per_tx = dr.shape(k_tx)[-1] // num_tx

        # Reshape to split transmitters and array samples per tx
        # Note: Split the x,y,z coordinates to handle large number of samples,
        # as the maximum size allowed for one array is 2^32
        # [num_tx, samples_per_tx]
        k_tx_x = dr.reshape(mi.TensorXf, k_tx.x, [num_tx, samples_per_tx])
        k_tx_y = dr.reshape(mi.TensorXf, k_tx.y, [num_tx, samples_per_tx])
        k_tx_z = dr.reshape(mi.TensorXf, k_tx.z, [num_tx, samples_per_tx])

        # Reshape relative antenna positions
        # Add a dimension to broadcast with samples per tx
        # Note: Split the x,y,z coordinates to handle large number of samples,
        # as the maximum size allowed for one array is 2^32
        # [num_tx, 1, array_size]
        rel_ant_positions_tx_x = dr.reshape(mi.TensorXf, rel_ant_positions_tx.x,
                                            [num_tx, 1, array_size])
        rel_ant_positions_tx_y = dr.reshape(mi.TensorXf, rel_ant_positions_tx.y,
                                            [num_tx, 1, array_size])
        rel_ant_positions_tx_z = dr.reshape(mi.TensorXf, rel_ant_positions_tx.z,
                                            [num_tx, 1, array_size])

        # Reshape precoding vector
        # Add dimension to broadcast with number of samples, and split patterns
        # from array
        # [num_tx, 1, num_patterns, array_size]
        precoding_vec_real_ = dr.reshape(mi.TensorXf, precoding_vec_real.array,
                                         [num_tx, 1, num_patterns, array_size])
        precoding_vec_imag_ = dr.reshape(mi.TensorXf, precoding_vec_imag.array,
                                         [num_tx, 1, num_patterns, array_size])
        precoding_vec_real = []
        precoding_vec_imag = []
        # [num_tx, 1, array_size]
        for i in range(num_patterns):
            precoding_vec_real.append(precoding_vec_real_[...,i,:])
            precoding_vec_imag.append(precoding_vec_imag_[...,i,:])

        # To reduce the memory footprint, iterate over the number of transmitter
        # antennas. This avoids allocating tensors with shape
        #                                           [num_samples, num_ant, ...]
        # Weights for each antenna pattern
        w_real = []
        w_imag = []
        for i in range(num_patterns):
            w_real.append(dr.zeros(mi.TensorXf, [num_tx, samples_per_tx]))
            w_imag.append(dr.zeros(mi.TensorXf, [num_tx, samples_per_tx]))
        n = 0
        while n < array_size:
            # Extract the relative position of antenna
            # [num_tx, 1]
            ant_pos_x = rel_ant_positions_tx_x[...,n]
            ant_pos_y = rel_ant_positions_tx_y[...,n]
            ant_pos_z = rel_ant_positions_tx_z[...,n]
            # [num_tx, samples_per_tx]
            tx_phase_shifts = ant_pos_x*k_tx_x + ant_pos_y*k_tx_y\
                                + ant_pos_z*k_tx_z
            tx_phase_shifts *= dr.two_pi/scene.wavelength
            array_vec_imag, array_vec_real = dr.sincos(tx_phase_shifts)
            for i in range(num_patterns):
                # Dot product with precoding vector iteratively computed
                # [num_tx, 1]
                prec_real = precoding_vec_real[i][...,n]
                prec_imag = precoding_vec_imag[i][...,n]
                # [num_tx, samples_per_tx, num_patterns]
                w_real[i] += array_vec_real * prec_real\
                            - array_vec_imag * prec_imag
                w_imag[i] += array_vec_real * prec_imag\
                            + array_vec_imag * prec_real
            #
            n += 1

        # Reshape to fit total number of samples
        w = []
        for i in range(num_patterns):
            w_real_ = dr.reshape(mi.Float, w_real[i], [num_tx*samples_per_tx])
            w_imag_ = dr.reshape(mi.Float, w_imag[i], [num_tx*samples_per_tx])
            w_ = mi.Matrix4f(w_real_,     0.0,    -w_imag_,        0.0,
                             0.0,     w_real_,        0.0,    -w_imag_,
                             w_imag_,     0.0,    w_real_,         0.0,
                             0.0,     w_imag_,        0.0,     w_real_)
            w.append(w_)

        return w

    def _sample_radio_material(
        self,
        si : mi.SurfaceInteraction3f,
        k_world : mi.Vector3f,
        e_fields : mi.Vector4f,
        solid_angle : mi.Float,
        sample1 : mi.Float,
        sample2 : mi.Point2f,
        specular_reflection : bool,
        diffuse_reflection : bool,
        refraction : bool,
        active : mi.Bool
    ) -> Tuple[mi.BSDFSample3f, mi.Vector4f]:
        r"""
        Evaluates the radio material and updates the electric field accordingly

        :param si: Information about the interaction of the rays with a surface
            of the scene
        :param k_world: Direction of propagation of the incident wave in the
            world frame
        :param e_fields: Electric field Jones vector as a 4D real-valued vector
        :param solid_angle: Ray tube solid angle [sr]
        :param sample1: Random float uniformly distributed in :math:`[0,1]`.
            Used to sample the interaction type.
        :param sample2: Random 2D point uniformly distributed in
            :math:`[0,1]^2`. Used to sample the direction of diffusely reflected
            waves.
        :param specular_reflection: If set to `True`, then the specularly
            reflected paths are computed
        :param diffuse_reflection: If set to `True`, then the diffusely
            reflected paths are computed
        :param refraction: If set to `True`, then the refracted paths are
            computed
        :param active: Mask to specify active rays

        :return: Updated electric field and sampling record
        """

        # Ensure the normal is oriented in the opposite of the direction of
        # propagation of the incident wave
        normal_world = si.n*dr.sign(dr.dot(si.n, -k_world))
        si.sh_frame.n = normal_world
        si.initialize_sh_frame()
        si.n = normal_world

        # Set `si.wi` to the local direction of propagation of the incident wave
        si.wi = si.to_local(k_world)

        # Context.
        # Specify the components that are required
        component = 0
        if specular_reflection:
            component |= InteractionType.SPECULAR
        if diffuse_reflection:
            component |= InteractionType.DIFFUSE
        if refraction:
            component |= InteractionType.REFRACTION
        ctx = mi.BSDFContext(mode=mi.TransportMode.Importance,
                            type_mask=0,
                            component=component)

        # Samples and evaluate the radio material
        for i, e_field in enumerate(e_fields):
            # We use:
            #  `si.dn_du` to store the real components of the S and P
            #       coefficients of the incident field
            #  `si.dn_dv` to store the imaginary components of the S and P
            #       coefficients of the incident field
            #  `si.dp_du` to store the solid angle
            # Note that the incident field is represented in the implicit world
            #   frame
            # Real components
            si.dn_du = mi.Vector3f(e_field.x, # S
                                   e_field.y, # P
                                   0.)
            # Imag components
            si.dn_dv = mi.Vector3f(e_field.z, # S
                                   e_field.w, # P
                                   0.)
            # Solid angle
            si.dp_du = mi.Vector3f(solid_angle, 0., 0.)
            # Sample and evaluate the radio material
            sample, jones_mat = si.bsdf().sample(ctx, si, sample1, sample2,
                                                 active)
            jones_mat = spectrum_to_matrix_4f(jones_mat)
            # Update the field by applying the Jones matrix
            e_fields[i] = dr.select(active,
                                    mi.Vector4f(jones_mat@e_field),
                                    e_fields[i])

        return sample, e_fields
