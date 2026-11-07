#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Radio map solver"""

from numbers import Integral, Real

import mitsuba as mi
import drjit as dr
from typing import Tuple, Callable, List, Sequence

from sionna.rt.utils import spawn_ray_from_sources, fibonacci_lattice,\
    rotation_matrix, spectrum_to_matrix_4f, WedgeGeometry,\
    sample_wedge_diffraction_point, spawn_ray_towards,\
    sample_keller_cone_with_measure, spawn_ray_to
from sionna.rt import Scene
from sionna.rt.antenna_pattern import antenna_pattern_to_world_implicit
from sionna.rt.constants import InteractionType, MIN_SEGMENT_LENGTH
from sionna.rt.scene_object import SceneObject
from sionna.rt.scene_utils import extend_scene_with_mesh
from .radio_map import RadioMap
from .planar_radio_map import PlanarRadioMap
from .mesh_radio_map import MeshRadioMap
from sionna.rt.utils.hashing import EdgeHasher


# Real and imaginary components of a complex-valued precoding vector
PrecodingVec = Tuple[mi.TensorXf | mi.Float, mi.TensorXf | mi.Float]


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
        \texttt{num\_cells\_x} = \bigg\lceil\frac{\texttt{size[0]}}{\texttt{cell\_size[0]}} \bigg\rceil\\
        \texttt{num\_cells\_y} = \bigg\lceil \frac{\texttt{size[1]}}{\texttt{cell\_size[1]}} \bigg\rceil.

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

    These coefficients form the complex-valued channel vector :math:`\mathbf{h}_n` with a size of :math:`\texttt{num\_tx\_ant}`.

    Finally, the coefficient for the equivalent SISO channel is given by:

    .. math::
        h_n =  \mathbf{h}_n^{\mathsf{T}} \mathbf{p}

    where :math:`\mathbf{p}` denotes the precoding vector ``precoding_vec``.
    The product is a transpose (not a Hermitian) inner product: ``precoding_vec``
    is applied without conjugation.

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

    # Size of the hash table for the edge intersections
    _WEDGE_INTERSECTIONS_SIZE = 10000

    # Spacing between diffraction sample counters to reduce contention between
    # atomic writes to adjacent transmitter-wedge entries.
    _SAMPLE_COUNT_SPREAD_FACTOR = 8

    def __init__(self):
        # Sampler
        self._sampler = mi.load_dict({'type': 'independent'})

        # Dr.Jit mode for running the loop that implement the solver.
        # Symbolic mode is the fastest mode but does not currently support
        # automatic differentiation.
        self._loop_mode = "symbolic"

        self.edge_hash_functions = [EdgeHasher(op='round'), EdgeHasher(op='floor')]

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
        scene: Scene,
        center: mi.Point3f | None = None,
        orientation: mi.Point3f | None = None,
        size: mi.Point2f | None = None,
        cell_size: mi.Point2f = mi.Point2f(10, 10),
        measurement_surface: mi.Shape | SceneObject | None = None,
        precoding_vec: PrecodingVec | Sequence[PrecodingVec] | None = None,
        samples_per_tx: int = 1000000,
        max_depth: int = 3,
        los: bool = True,
        specular_reflection: bool = True,
        diffuse_reflection: bool = False,
        refraction: bool = True,
        diffraction: bool = False,
        edge_diffraction: bool = False,
        diffraction_lit_region: bool = True,
        seed: int = 42,
        rr_depth: int = -1,
        rr_prob: float = 0.95,
        stop_threshold: float | None = None,
        check_cells_area: bool = False,
        modified_scene: mi.Scene | None = None
    ) -> RadioMap | List[RadioMap]:
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
        :param size: Size of the radio map measurement plane [m].
            Ignored if ``measurement_surface`` is provided.
            If set to `None`, then the size of the radio map is set such that
            it covers the entire scene.
            Otherwise, ``center`` and ``orientation`` must be provided.
            If ``size`` is not an integer multiple of ``cell_size``, it is
            expanded to ``ceil(size / cell_size) * cell_size``.
        :param cell_size: Size of a cell of the radio map measurement plane [m].
            Ignored if ``measurement_surface`` is provided.
        :param measurement_surface: Measurement surface. If set, the
            radio map is computed for this surface, where every triangle in the
            mesh is a cell in the radio map.
            If set to `None`, then the radio map is computed for a measurement
            grid defined by ``center``, ``orientation``, ``size``, and ``cell_size``.
        :param precoding_vec: Real and imaginary components of the
            complex-valued precoding vector(s) as a pair ``(real, imag)`` (or
            ``[real, imag]``). Each component can have shape
            ``[num_tx_ant]`` (same precoding vector for all transmitters),
            ``[num_tx, num_tx_ant]``, or ``[num_precoding_vecs, num_tx,
            num_tx_ant]`` to compute radio maps for several precoding vectors,
            e.g., the beams of a codebook. Several precoding vectors can also
            be provided as a list of pairs
            ``[(real_0, imag_0), (real_1, imag_1), ...]``.
            When several precoding vectors are provided, rays are traced only
            once and all precoding vectors are applied to the same rays, which
            is significantly faster than calling the solver once per precoding
            vector. A list of radio maps, one per precoding vector and in the
            same order, is then returned.
            If set to `None`, then defaults to
            :math:`\frac{1}{\sqrt{\text{num\_tx\_ant}}} [1,\dots,1]^{\mathsf{T}}`.
        :param samples_per_tx: Number of samples per source (``>= 1``)
        :param max_depth: Maximum depth (``>= 0``)
        :param los: Enable line-of-sight paths
        :param specular_reflection: Enable specular reflections
        :param diffuse_reflection: Enable diffuse reflections
        :param refraction: Enable refraction
        :param diffraction: Enable diffraction
        :param edge_diffraction: Enable diffraction on free floating edges
        :param diffraction_lit_region: Enable diffraction in the lit region
        :param seed: Non-negative seed
        :param rr_depth: Depth from which on to start Russian roulette.
            Use ``-1`` to disable Russian roulette.
        :param rr_prob: Maximum probability with which to keep a path when
            Russian roulette is enabled. Must lie in ``(0, 1]``.
        :param stop_threshold: Gain threshold [dB] below which a path is
            deactivated
        :param check_cells_area: If `True` and ``measurement_surface`` is a mesh,
            reject early surfaces that contain degenerate (zero-area) triangles.
            Ignored for planar radio maps. Default is `False`.
        :param modified_scene: Advanced parameter. Can be used to provide a
            Mitsuba scene instance that was already extended to contain the
            measurement surface. This can be useful when calling the solver
            in a loop with the same scene and measurement surface.
            Not applicable to planar radio maps.

        :return: Computed radio map, or list of radio maps (one per precoding
            vector) if several precoding vectors are provided
        """

        # Validate public arguments before any allocation or sampling
        if not isinstance(scene, Scene):
            raise TypeError("`scene` must be an instance of Scene")
        if (isinstance(samples_per_tx, bool)
                or not isinstance(samples_per_tx, Integral)):
            raise TypeError("`samples_per_tx` must be an integer")
        if samples_per_tx < 1:
            raise ValueError(
                "`samples_per_tx` must be greater than or equal to one"
            )
        if isinstance(max_depth, bool) or not isinstance(max_depth, Integral):
            raise TypeError("`max_depth` must be an integer")
        if max_depth < 0:
            raise ValueError("`max_depth` must be greater than or equal to zero")
        if not isinstance(los, bool):
            raise TypeError("`los` must be a bool")
        if not isinstance(specular_reflection, bool):
            raise TypeError("`specular_reflection` must be a bool")
        if not isinstance(diffuse_reflection, bool):
            raise TypeError("`diffuse_reflection` must be a bool")
        if not isinstance(refraction, bool):
            raise TypeError("`refraction` must be a bool")
        if not isinstance(diffraction, bool):
            raise TypeError("`diffraction` must be a bool")
        if not isinstance(edge_diffraction, bool):
            raise TypeError("`edge_diffraction` must be a bool")
        if not isinstance(diffraction_lit_region, bool):
            raise TypeError("`diffraction_lit_region` must be a bool")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("`seed` must be an integer")
        if seed < 0:
            raise ValueError("`seed` must be greater than or equal to zero")
        if isinstance(rr_depth, bool) or not isinstance(rr_depth, Integral):
            raise TypeError("`rr_depth` must be an integer")
        if rr_depth < -1:
            raise ValueError(
                "`rr_depth` must be -1 (disabled) or greater than or equal to "
                "zero"
            )
        if isinstance(rr_prob, bool) or not isinstance(rr_prob, Real):
            raise TypeError("`rr_prob` must be a real number")
        if not (0.0 < float(rr_prob) <= 1.0):  # pylint: disable=superfluous-parens
            raise ValueError("`rr_prob` must lie in the interval (0, 1]")
        if stop_threshold is not None:
            if (isinstance(stop_threshold, bool)
                    or not isinstance(stop_threshold, Real)):
                raise TypeError(
                    "`stop_threshold` must be a real number or None"
                )
        if not isinstance(check_cells_area, bool):
            raise TypeError("`check_cells_area` must be a bool")
        if measurement_surface is not None:
            if isinstance(measurement_surface, SceneObject):
                measurement_surface = measurement_surface.mi_mesh
            if not isinstance(measurement_surface, mi.Mesh):
                raise TypeError(
                    "`measurement_surface` must be a Mitsuba Mesh or a "
                    "SceneObject with a mesh"
                )
        if modified_scene is not None:
            if not isinstance(modified_scene, mi.Scene):
                raise TypeError("`modified_scene` must be a Mitsuba Scene")
            if measurement_surface is None:
                raise ValueError(
                    "Parameter `modified_scene` should be omitted (or set to "
                    "None) when using planar radio maps (i.e., "
                    "`measurement_surface` is None)."
                )
            ms_id = measurement_surface.id()
            if not any(s.id() == ms_id for s in modified_scene.shapes()):
                raise ValueError(
                    "`modified_scene` does not contain the provided "
                    "`measurement_surface`"
                )

        # Check that the scene is all set for simulations
        scene.all_set(radio_map=True)

        # Check and initialize the precoding vector(s)
        # Each component has shape [num_precoding_vecs, num_tx, num_tx_ant]
        num_tx = len(scene.transmitters)
        num_tx_ant = scene.tx_array.num_ant
        precoding_vecs, is_batch = self._prepare_precoding_vecs(
            precoding_vec, num_tx, num_tx_ant)
        num_precoding_vecs = dr.shape(precoding_vecs[0])[0]

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

        # Build Radio Map instances, one per precoding vector
        # If a measurement surface is provided, then it is a mesh radio map.
        # Moreover, in this case, the Mitsuba scene is modified by adding the
        # measurement surface to the scene.
        if measurement_surface is not None:
            if modified_scene is None:
                modified_scene = extend_scene_with_mesh(scene.mi_scene,
                                                        measurement_surface)
            radio_maps = [MeshRadioMap(scene, measurement_surface,
                                       check_cells_area=check_cells_area)
                          for _ in range(num_precoding_vecs)]
        else:
            modified_scene = scene.mi_scene
            radio_maps = [PlanarRadioMap(scene, cell_size, center, orientation,
                                         size)
                          for _ in range(num_precoding_vecs)]

        # Computes the pathloss map (except for diffraction)
        # The radio maps are updated in-place.
        # This flag is set to avoid tracing the loops twice, which adds
        # a fairly significant CPU overhead.
        with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, False):
            with scene.use_mi_scene(modified_scene):
                wedges = self._shoot_and_bounce(
                    scene, radio_maps,
                    self._sampler,
                    tx_positions, tx_orientations,
                    tx_antenna_patterns,
                    precoding_vecs, rel_ant_positions_tx,
                    samples_per_tx, max_depth,
                    los,
                    specular_reflection,
                    diffuse_reflection,
                    refraction,
                    diffraction,
                    edge_diffraction,
                    rr_depth, rr_prob, stop_threshold
                )

                if diffraction:
                    # Compact the hash table. As `dr.compress` returns occupied
                    # entries in increasing index order, the resulting wedge order
                    # is deterministic. Note that this introduces a sync point.
                    wedge_indices = dr.compress(wedges.length > 0)
                    wedges = dr.gather(WedgeGeometry, wedges, wedge_indices)
                    num_wedges = dr.width(wedge_indices)
                else:
                    num_wedges = 0

                # Add the contribution of diffracted paths to
                # the radio maps
                if diffraction and (max_depth > 0) and (num_wedges > 0):
                    # Reseting the seed of the sampler to ensure its state
                    # was scheduled
                    self._sampler.seed(seed + 1, num_samples)
                    self._evaluate_first_order_diffraction(
                        scene,
                        radio_maps,
                        self._sampler,
                        tx_positions,
                        tx_orientations,
                        tx_antenna_patterns,
                        precoding_vecs,
                        rel_ant_positions_tx,
                        samples_per_tx,
                        wedges,
                        diffraction_lit_region
                    )

        if is_batch:
            return radio_maps
        return radio_maps[0]

    # pylint: disable=line-too-long
    @dr.syntax
    def _shoot_and_bounce(
        self,
        scene: Scene,
        radio_maps: List[RadioMap],
        sampler: mi.Sampler,
        tx_positions: mi.Point3f,
        tx_orientations: mi.Point3f,
        tx_antenna_patterns: List[Callable[[mi.Float, mi.Float],
                                            Tuple[mi.Complex2f, mi.Complex2f]]],
        precoding_vecs: Tuple[mi.TensorXf, mi.TensorXf],
        rel_ant_positions_tx: mi.Point3f,
        samples_per_tx: int,
        max_depth: int,
        los_enabled: bool,
        specular_reflection_enabled: bool,
        diffuse_reflection_enabled: bool,
        refraction_enabled: bool,
        diffraction_enabled: bool,
        edge_diffraction_enabled: bool,
        rr_depth: int,
        rr_prob: float,
        stop_threshold: float | None,
    ) -> WedgeGeometry | None:
        r"""
        Executes the shoot-and-bounce loop

        The ``radio_maps`` are updated in-place.

        :param scene: Scene for which to compute the radio map
        :param radio_maps: Radio maps, one per precoding vector
        :param sampler: Sampler used to generate random numbers and vectors
        :param tx_positions: Positions of the transmitters
        :param tx_orientations: Orientations of the transmitters
        :param tx_antenna_patterns: Antenna pattern of the transmitters
        :param precoding_vecs: Real and imaginary components of the
            complex-valued precoding vectors, each with shape
            `[num_precoding_vecs, num_tx, num_tx_ant]`
        :param rel_ant_positions_tx: Positions of the antenna elements relative
            to the transmitters positions
        :param samples_per_tx: Number of samples per source
        :param max_depth: Maximum depth
        :param los_enabled: If set to `True`, then the line-of-sight paths are computed
        :param specular_reflection_enabled: If set to `True`, then the specularly
            reflected paths are computed
        :param diffuse_reflection_enabled: If set to `True`, then the diffusely
            reflected paths are computed
        :param refraction_enabled: If set to `True`, then the refracted paths are
            computed
        :param diffraction_enabled: If set to `True`, then the diffracted paths are
            computed
        :param edge_diffraction_enabled: If set to `True`, then the edge diffraction is computed
        :param rr_depth: Depth from which to start Russian roulette
        :param rr_prob: Minimum probability with which to keep a path when
            Russian roulette is enabled
        :param stop_threshold: Gain threshold (linear scale) below which a path
            is deactivated

        :return: Wedge geometry for diffraction, or `None` if diffraction is disabled.
        """

        num_txs = dr.shape(tx_positions)[1]
        num_samples = samples_per_tx * num_txs
        num_tx_ant_patterns = len(tx_antenna_patterns)

        # All radio maps share the same measurement surface
        ref_radio_map = radio_maps[0]

        tx_indices = dr.repeat(dr.arange(mi.UInt, num_txs), samples_per_tx)

        # Spawn rays from the transmitters
        ray = spawn_ray_from_sources(fibonacci_lattice, samples_per_tx,
                                     tx_positions)

        # Weights to account for the antenna array and precoding, one set per
        # precoding vector
        array_weights = self._synthetic_array_weights(scene, ray.d,
                                                      rel_ant_positions_tx,
                                                      precoding_vecs)

        # Mask indicating which rays are active
        active = dr.full(dr.mask_t(mi.Float), True, num_samples)

        # Flag storing which type of interactions are enabled for the
        # current interaction
        enabled_interactions = dr.full(mi.UInt, 0, num_samples)
        if specular_reflection_enabled:
            enabled_interactions |= InteractionType.SPECULAR
        if diffuse_reflection_enabled:
            enabled_interactions |= InteractionType.DIFFUSE
        if refraction_enabled:
            enabled_interactions |= InteractionType.REFRACTION

        # Length of the ray tube.
        # Only required if a threshold is set to deactivate path based on their
        # gain
        if stop_threshold is not None:
            ray_tube_length = dr.zeros(mi.Float, num_samples)
        else:
            ray_tube_length = None

        # Hash table storing intersection counts for each edge used in diffraction.
        # This avoids allocating memory proportional to the total number of edges
        # in the scene, which is efficient since typically only a small subset of
        # edges participate in diffraction, particularly for large scenes, and as
        # only first-order diffraction is computed.
        num_hashes = len(self.edge_hash_functions)
        if diffraction_enabled:
            wedges_counters = [
                dr.zeros(mi.UInt, RadioMapSolver._WEDGE_INTERSECTIONS_SIZE)
                for _ in range(num_hashes)
            ]
            wedges = WedgeGeometry.build_with_size(
                RadioMapSolver._WEDGE_INTERSECTIONS_SIZE)
        else:
            wedges_counters = None
            wedges = None

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
        while dr.hint(active, mode=self.loop_mode, exclude=[
                          array_weights, tx_positions,
                          edge_diffraction_enabled,
                          enabled_interactions, los_enabled, max_depth,
                          num_hashes, num_tx_ant_patterns, radio_maps,
                          ref_radio_map, rr_depth, rr_prob, scene, self,
                          stop_threshold, tx_indices, wedges, wedges_counters
                      ]):

            # Test intersection with the scene
            si_scene = scene.mi_scene.ray_intersect(ray, active=active)

            # For first-order diffraction: project intersection points onto wedges
            # and store wedge geometry for subsequent diffraction calculations
            if dr.hint(diffraction_enabled, mode="scalar"):
                store_wedges = active & si_scene.is_valid() & (depth == 0) \
                    & (si_scene.shape != ref_radio_map.measurement_surface)
                # Sample diffraction point on wedges
                valid_wedge_, wedges_, _ = sample_wedge_diffraction_point(
                    si_scene, ray.o, ray.d, sampler.next_1d(), edge_diffraction_enabled,
                    store_wedges)
                store_wedges &= valid_wedge_

                # Store wedges
                for i in range(num_hashes):
                    hash_value = self.edge_hash_functions[i](
                        wedges_.o,
                        wedges_.o + wedges_.e_hat * wedges_.length
                    )
                    counter_ind = hash_value % RadioMapSolver._WEDGE_INTERSECTIONS_SIZE
                    sample_counter = dr.scatter_inc(wedges_counters[i],
                                                    counter_ind, store_wedges)
                    store_wedges &= sample_counter == 0

                # Store wedge directly at index corresponding to the last hash.
                dr.scatter(wedges, wedges_, counter_ind, store_wedges)

            # Each interaction processes either an interaction with the scene
            # or an interaction with the measurement plane.
            # `mp_int` is set to `True` if the interaction is with the measurement
            # plane.
            # `scene_int` is set to `True` if the interaction is with the scene.
            # If the radio map is planar, then test interaction with the measurement
            # plane. Otherwise, the measurement surface is assumed to be part of
            # the scene and we only need to check that it has hit the measurement surface.
            si_mp = None
            if dr.hint(isinstance(ref_radio_map, PlanarRadioMap), mode="scalar"):
                si_mp = ref_radio_map.measurement_surface.ray_intersect(
                    ray, active=active
                )
                # An intersection with the measurement plane is valid only if
                # (i) it was not obstructed by the scene, and (ii) the intersection
                # is valid.
                mp_int = active & si_mp.is_valid() & (si_mp.t < si_scene.t)
                # Ignore interaction with measurement plane if LoS is disabled
                mp_int &= ((depth > 0) | los_enabled)
            else:
                # The measurement surface is part of the scene
                si_mp = si_scene
                mp_int = active & si_mp.is_valid() \
                         & (si_mp.shape == ref_radio_map.measurement_surface)

            # Update the radio maps
            # The radio maps are not updated if LoS is disabled and it is a
            # LoS interaction with the measurement surface
            # Note that we need to redo the LoS test to handle the case where
            # the measurement surface is part of the scene
            update_radio_map = mp_int & ((depth > 0) | los_enabled)
            self._add_paths(radio_maps, e_fields, array_weights, si_mp, ray.d,
                            tx_indices, update_radio_map,
                            diffracted_paths=False, solid_angle=solid_angle)

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
                solid_angle, sample1, sample2, enabled_interactions, scene_int
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
            if dr.hint(stop_threshold is not None, mode="scalar"):
                ray_tube_length = dr.select(diffuse, 0.0, ray_tube_length)
                ray_tube_length += dr.select(scene_int, si_scene.t, si_mp.t)

            # Is the depth threshold to activate Russian roulette reached?
            # Apply Russian roulette only at scene interactions. Measurement-
            # surface crossings are pass-through and must neither terminate nor
            # reweight the ray.
            rr_inactive = mp_int | (depth < rr_depth)
            # User specify a maximum probability of continuing tracing
            rr_continue_prob = dr.minimum(gain, rr_prob)
            rr_continue = sampler.next_1d() < rr_continue_prob
            active &= (rr_inactive | rr_continue)

            # Scale the remaining rays accordingly to ensure an unbiased result
            for i in range(num_tx_ant_patterns):
                e_fields[i] = dr.select(rr_inactive, e_fields[i],
                                        e_fields[i] * dr.rsqrt(rr_continue_prob))

            # Deactivate rays with gains below the set threshold
            if dr.hint(stop_threshold is not None, mode="scalar"):
                gain_pl = gain * dr.square(scene.wavelength
                                           * dr.rcp(4. * dr.pi * ray_tube_length))
                active &= mp_int | (gain_pl > stop_threshold)

        return wedges

    @dr.syntax
    def _evaluate_first_order_diffraction(
        self,
        scene: Scene,
        radio_maps: List[RadioMap],
        sampler: mi.Sampler,
        tx_positions: mi.Point3f,
        tx_orientations: mi.Point3f,
        tx_antenna_patterns: List[Callable[[mi.Float, mi.Float],
                                            Tuple[mi.Complex2f, mi.Complex2f]]],
        precoding_vecs: Tuple[mi.TensorXf, mi.TensorXf],
        rel_ant_positions_tx: mi.Point3f,
        samples_per_tx: int,
        wedges: WedgeGeometry,
        diffraction_lit_region: bool):
        """
        Traces first-order diffracted paths for a set of `wedges` and adds their
        contributions to the radio maps

        This method computes the electric field contributions from rays that undergo
        first-order diffraction at wedges in the scene. It samples diffraction points
        along the wedges and evaluates the diffracted field.
        If paths intersect with the measurement surface, then their contributions
        are added to the radio maps. Wedges are sampled proportionally to their length.

        :param scene: Scene containing the geometry and material properties
        :param radio_maps: Radio maps to be updated with diffraction
            contributions, one per precoding vector
        :param sampler: Sampler for generating random numbers
        :param tx_positions: Positions of all transmitters in the scene
        :param tx_orientations: Orientations of all transmitters in the scene
        :param tx_antenna_patterns: List of antenna pattern functions for each transmitter
        :param precoding_vecs: Real and imaginary components of the
            precoding vectors, each with shape
            `[num_precoding_vecs, num_tx, num_tx_ant]`
        :param rel_ant_positions_tx: Relative positions of antenna elements in transmitter arrays
        :param samples_per_tx: Number of samples to generate per transmitter
        :param wedges: Geometry information for all wedges in the scene
        :param diffraction_lit_region: Enable diffraction in the lit region
        """
        num_txs = dr.shape(tx_positions)[1]
        num_samples = num_txs * samples_per_tx
        # All radio maps share the same measurement surface
        ref_radio_map = radio_maps[0]
        is_planar_radio_map = isinstance(ref_radio_map, PlanarRadioMap)

        #############################################################
        # Sample diffraction point on the wedges
        #############################################################

        # Sample wedges
        # Wedges are sampled proportionally to their length
        total_length = dr.sum(wedges.length)
        wedges_sample_prob = wedges.length / total_length
        dist = mi.DiscreteDistribution(wedges_sample_prob)
        wedges_index = dist.sample(sampler.next_1d())
        sampled_wedges = dr.gather(WedgeGeometry, wedges, wedges_index)

        tx_indices = dr.repeat(dr.arange(mi.UInt, num_txs), samples_per_tx)
        samples_per_tx_wedge = self._samples_per_tx_wedge(
            tx_indices, wedges_index, num_txs, dr.width(wedges.length))
        tx_positions_ = dr.gather(mi.Point3f, tx_positions, tx_indices)

        # Ensure that we sample consistently, regardless of which direction
        # of the edge was stored in `wedges`.
        p1 = sampled_wedges.o
        v2 = sampled_wedges.length * sampled_wedges.e_hat
        swap_wedges, _, _ = self.edge_hash_functions[-1].should_swap(
            p1, sampled_wedges.o + v2, apply_quantization=False
        )
        # Sample points on the wedges
        sample1 = sampler.next_1d()
        diff_point = p1 + dr.select(swap_wedges, 1 - sample1, sample1) * v2
        del p1, v2, sample1

        # Direction of incident wave
        ki = dr.normalize(diff_point - tx_positions_)

        # To avoid leakage, we offset the diffraction point for shooting rays
        # to test for interaction with a surface that would overlap with one
        # of the wedge face.
        # Note that this is only done for visibility test. Computation of the
        # diffraction field is done with the original diffraction point.
        edge_diffraction = sampled_wedges.primn == sampled_wedges.prim0
        ne = dr.select(edge_diffraction, mi.Vector3f(0),
                  dr.normalize(sampled_wedges.n0 + sampled_wedges.nn))
        diff_point_offset = diff_point + 5e-2 * ne

        #############################################################
        # Initialize the electric field and array weighting
        #############################################################

        # Weights to account for the antenna array and precoding, one set per
        # precoding vector
        array_weights = self._synthetic_array_weights(scene, ki,
                                                      rel_ant_positions_tx,
                                                      precoding_vecs)

        # Initialize the electric field
        # Orientation of sources and targets and correspond to-world transforms
        sample_tx_orientation = dr.repeat(tx_orientations, samples_per_tx)
        tx_to_world = rotation_matrix(sample_tx_orientation)
        # The following loop transports and update the electric field.
        # The electric field is initialized by the source antenna pattern
        e_fields = [antenna_pattern_to_world_implicit(src_antenna_pattern,
                                                      tx_to_world, ki,
                                                      direction="out")
                    for src_antenna_pattern in tx_antenna_patterns]

        #############################################################
        # Test occlusion between source and diffraction point
        #############################################################

        ray_occ = spawn_ray_to(tx_positions_, diff_point)
        vis_ray1 = spawn_ray_to(tx_positions_, diff_point_offset)
        # In the case of mesh-based radio maps, we must discard
        # intersections with the measurement surface
        active_s1 = dr.full(mi.Bool, True, num_samples)
        hit_scene = dr.full(mi.Bool, False, num_samples)
        while dr.hint(active_s1, mode=self.loop_mode):
            # Mark intersections with the scene
            si = scene.mi_scene.ray_intersect(ray_occ, active=active_s1)
            hit_scene |= si.is_valid() &\
                (si.shape != ref_radio_map.measurement_surface)
            # Test for visibility with an offset diffraction point
            # to avoid leakage
            si_vis = scene.mi_scene.ray_intersect(vis_ray1, active=active_s1)
            hit_scene |= si_vis.is_valid() &\
                (si_vis.shape != ref_radio_map.measurement_surface)
            # Deactivate ray if it hit the scene (occluded) or if it didn't hit
            # anything (reached the diffraction point without obstruction)
            active_s1 &= ~hit_scene & si.is_valid()
            ray_occ = si.spawn_ray_to(diff_point)
            vis_ray1 = si_vis.spawn_ray_to(diff_point_offset)
        valid_path = ~hit_scene
        del vis_ray1, ray_occ

        #############################################################
        # Sample Keller cone
        #############################################################

        sample1 = sampler.next_1d()
        ko, diffraction_angular_measure = sample_keller_cone_with_measure(
            sampled_wedges.e_hat, sampled_wedges.n0, sampled_wedges.nn,
            dr.select(swap_wedges, 1 - sample1, sample1),
            ki, diffraction_lit_region)

        #############################################################
        # Add path to radio map
        #############################################################

        ray_mp = spawn_ray_towards(diff_point, diff_point + ko, ne)
        vis_ray2 = spawn_ray_to(diff_point_offset, diff_point_offset + ko)
        # For mesh-based radio maps, handle potential multiple
        # intersections with the measurement surface
        while dr.hint(valid_path, mode=self.loop_mode,
                      exclude=[array_weights, tx_positions, radio_maps,
                               ref_radio_map]):

            # Test intersection with the measurement
            # surface, and ensure that the segment from the
            # diffraction to the measurement surface is not
            # occluded.
            si_mp = None
            si_vis = None
            if dr.hint(is_planar_radio_map, mode='scalar'):
                si_mp = ref_radio_map.measurement_surface.ray_intersect(
                    ray_mp,
                    active=valid_path
                )
                valid_path &= si_mp.is_valid()
                # Ensure that the segment from the diffraction to the
                # measurement surface is not occluded by the scene
                ray_to_plane = spawn_ray_to(si_mp.p, diff_point)
                valid_path &= ~scene.mi_scene.ray_test(
                    ray_to_plane, coherent=False, active=valid_path
                )
                del ray_to_plane
                # Test for visibility with an offset diffraction point
                # to avoid leakage
                vis_ray3 = spawn_ray_to(si_mp.p, diff_point_offset)
                valid_path &= ~scene.mi_scene.ray_test(
                    vis_ray3, coherent=False, active=valid_path
                )
                del vis_ray3
                si_vis = si_mp
            else:
                si_mp = scene.mi_scene.ray_intersect(
                    ray_mp, ray_flags=mi.RayFlags.Minimal, coherent=False,
                    active=valid_path
                )
                valid_path &= si_mp.is_valid() & \
                    (si_mp.shape == ref_radio_map.measurement_surface)
                # Test for visibility with an offset diffraction point
                # to avoid leakage
                si_vis = scene.mi_scene.ray_intersect(
                    vis_ray2, ray_flags=mi.RayFlags.Minimal, coherent=False,
                    active=valid_path
                )
                valid_path &= si_vis.is_valid() &\
                    (si_vis.shape == ref_radio_map.measurement_surface)

            # Discard paths where either segment is too short
            s_prime = dr.norm(diff_point - tx_positions_)
            s = dr.norm(diff_point - si_mp.p)
            add_to_rm = valid_path & (s > MIN_SEGMENT_LENGTH) \
                        & (s_prime > MIN_SEGMENT_LENGTH)

            # Compute the diffracted field
            shape = mi.ShapePtr(sampled_wedges.shape)
            diffracted_e_fields = self._evaluate_radio_material_diffraction(
                shape, sampled_wedges, diff_point, ki, ko, e_fields,
                s, s_prime, add_to_rm
            )

            # Add contribution to the radio maps
            self._add_paths(radio_maps, diffracted_e_fields, array_weights,
                            si_mp, ko, tx_indices, add_to_rm,
                            diffracted_paths=True,
                            tx_positions=tx_positions,
                            wedges=sampled_wedges,
                            diff_point=diff_point,
                            wedges_samples_cnt=samples_per_tx_wedge,
                            diffraction_angular_measure=diffraction_angular_measure)

            # Spawn new rays
            ray_mp = si_mp.spawn_ray(ko)
            vis_ray2 = si_vis.spawn_ray(ko)

            del si_mp, si_vis

    @staticmethod
    def _samples_per_tx_wedge(
        tx_indices: mi.UInt,
        wedges_index: mi.UInt,
        num_txs: int,
        num_wedges: int,
    ) -> mi.UInt:
        """
        Counts samples independently for each transmitter and wedge.

        :param tx_indices: Transmitter index of every sample
        :param wedges_index: Wedge index of every sample
        :param num_txs: Number of transmitters
        :param num_wedges: Number of wedges
        :return: Number of samples associated with the same transmitter and
            wedge as each input sample
        """
        tx_wedge_index = tx_indices * dr.opaque(mi.UInt, num_wedges) + wedges_index
        count_index = (RadioMapSolver._SAMPLE_COUNT_SPREAD_FACTOR
                       * tx_wedge_index)
        counts = dr.zeros(
            mi.UInt,
            RadioMapSolver._SAMPLE_COUNT_SPREAD_FACTOR * num_txs * num_wedges,
        )
        dr.scatter_add(counts, 1, count_index)
        return dr.gather(mi.UInt, counts, count_index)

    @staticmethod
    def _prepare_precoding_vecs(
        precoding_vec: PrecodingVec | Sequence[PrecodingVec] | None,
        num_tx: int,
        num_tx_ant: int
    ) -> Tuple[Tuple[mi.TensorXf, mi.TensorXf], bool]:
        r"""
        Validates the precoding vector(s) provided by the user and brings them
        to a common shape

        :param precoding_vec: Precoding vector(s) as provided to
            :meth:`__call__`
        :param num_tx: Number of transmitters
        :param num_tx_ant: Number of transmit antennas

        :return: Real and imaginary components of the precoding vectors, each
            with shape `[num_precoding_vecs, num_tx, num_tx_ant]`, and a flag
            set to `True` if several precoding vectors were provided (in which
            case a list of radio maps is returned)
        """
        if precoding_vec is None:
            real = dr.ones(mi.TensorXf, [1, num_tx, num_tx_ant])
            real /= dr.sqrt(num_tx_ant)
            imag = dr.zeros(mi.TensorXf, [1, num_tx, num_tx_ant])
            return (real, imag), False

        if not isinstance(precoding_vec, (tuple, list)):
            raise TypeError("`precoding_vec` must be a tuple or list of "
                            "(real, imag) components, or a list of such "
                            "pairs")
        if len(precoding_vec) == 0:
            raise ValueError("`precoding_vec` must not be empty")

        # A single pair `(real, imag)` is distinguished from a list of pairs by
        # the type of its items, not by the type of the outer container, such
        # that `[real, imag]` remains a single precoding vector.
        is_pair = [isinstance(pv, (tuple, list)) for pv in precoding_vec]
        if all(is_pair):
            # List of precoding vectors
            pairs = [RadioMapSolver._normalize_precoding_vec(pv, num_tx,
                                                             num_tx_ant)
                     for pv in precoding_vec]
            real = dr.concat([pv[0] for pv in pairs], axis=0)
            imag = dr.concat([pv[1] for pv in pairs], axis=0)
            return (real, imag), True
        if any(is_pair) or len(precoding_vec) != 2:
            raise TypeError("`precoding_vec` must be a tuple or list of "
                            "(real, imag) components, or a list of such "
                            "pairs")

        real, imag = precoding_vec
        if isinstance(real, mi.TensorXf) and real.ndim == 3:
            # Batch of precoding vectors given as tensors
            expected_shape = (dr.shape(real)[0], num_tx, num_tx_ant)
            if (not isinstance(imag, mi.TensorXf)
                    or tuple(dr.shape(real)) != expected_shape
                    or tuple(dr.shape(imag)) != expected_shape
                    or expected_shape[0] == 0):
                raise ValueError(
                    "Batched `precoding_vec` components must have shape "
                    "[num_precoding_vecs, num_tx, num_tx_ant] = "
                    f"[num_precoding_vecs, {num_tx}, {num_tx_ant}]"
                )
            return (real, imag), True

        return RadioMapSolver._normalize_precoding_vec(
            precoding_vec, num_tx, num_tx_ant), False

    @staticmethod
    def _normalize_precoding_vec(
        precoding_vec: PrecodingVec,
        num_tx: int,
        num_tx_ant: int
    ) -> Tuple[mi.TensorXf, mi.TensorXf]:
        r"""
        Validates a single precoding vector and broadcasts it to all
        transmitters if required

        :param precoding_vec: Real and imaginary components of the
            complex-valued precoding vector, each with shape `[num_tx_ant]` or
            `[num_tx, num_tx_ant]`
        :param num_tx: Number of transmitters
        :param num_tx_ant: Number of transmit antennas

        :return: Real and imaginary components of the precoding vector, each
            with shape `[1, num_tx, num_tx_ant]`
        """
        if len(precoding_vec) != 2:
            raise ValueError("Each precoding vector must be a pair "
                             "(real, imag)")
        real, imag = precoding_vec
        if not isinstance(real, type(imag)):
            raise TypeError("The real and imaginary components of "
                            "`precoding_vec` must be of the same type")

        if (isinstance(real, mi.Float)
                or (isinstance(real, mi.TensorXf) and real.ndim == 1)):
            # A single precoding vector is used by all transmitters.
            # The flat array of tensors is used as `mi.Float(tensor)` moves the
            # storage out of the tensor, which would leave the caller's tensor
            # empty.
            if isinstance(real, mi.TensorXf):
                real, imag = real.array, imag.array
            if dr.width(real) != num_tx_ant or dr.width(imag) != num_tx_ant:
                raise ValueError(
                    "A shared `precoding_vec` must have length "
                    f"`num_tx_ant` ({num_tx_ant})"
                )
            real = dr.tile(real, num_tx)
            imag = dr.tile(imag, num_tx)
        elif not isinstance(real, mi.TensorXf):
            raise TypeError(
                "The components of `precoding_vec` must be `mi.Float` "
                "or `mi.TensorXf`"
            )
        else:
            expected_shape = (num_tx, num_tx_ant)
            if (tuple(dr.shape(real)) != expected_shape
                    or tuple(dr.shape(imag)) != expected_shape):
                raise ValueError(
                    "`precoding_vec` components must have shape [num_tx_ant] "
                    f"or [num_tx, num_tx_ant] = [{num_tx}, {num_tx_ant}]"
                )
            real, imag = real.array, imag.array

        real = dr.reshape(mi.TensorXf, real, [1, num_tx, num_tx_ant])
        imag = dr.reshape(mi.TensorXf, imag, [1, num_tx, num_tx_ant])
        return real, imag

    def _add_paths(
        self,
        radio_maps: List[RadioMap],
        e_fields: List[mi.Vector4f],
        array_weights: List[List[mi.Float]],
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
        # pylint: disable=protected-access
        r"""
        Adds the contribution of the paths that hit the measurement surface
        to the radio maps of all precoding vectors

        The quantities that do not depend on the precoding (hit cells, ray
        weights, and, for single-polarized arrays, the power of the electric
        field) are computed only once and shared by all radio maps.

        :param radio_maps: Radio maps, one per precoding vector
        :param e_fields: Electric fields as real-valued vectors of dimension 4,
            one per transmit antenna pattern
        :param array_weights: Weights modeling the transmit array and
            precoding, as returned by :meth:`_synthetic_array_weights`
        :param si: Information about the interaction with the measurement
            surface

        See :meth:`~sionna.rt.RadioMap._path_weight` for the remaining
        parameters.
        """
        tensor_ind, w, active = radio_maps[0]._path_weight(
            si, k_world, tx_indices, active, diffracted_paths, solid_angle,
            tx_positions, wedges, diff_point, wedges_samples_cnt,
            diffraction_angular_measure)

        if len(e_fields) == 1:
            # Single antenna pattern: the weights reduce to a power gain that
            # scales the power of the field, which is shared by all precoding
            # vectors
            e_power = dr.squared_norm(e_fields[0]) * w
            for radio_map, (gain,) in zip(radio_maps, array_weights):
                radio_map._scatter_path_gain(gain * e_power, tensor_ind,
                                             active)
        else:
            for radio_map, weights in zip(radio_maps, array_weights):
                a = self._precoded_power(weights, e_fields)
                radio_map._scatter_path_gain(a * w, tensor_ind, active)

    @staticmethod
    def _precoded_power(
        weights: List[mi.Float],
        e_fields: List[mi.Vector4f]
    ) -> mi.Float:
        r"""
        Computes the squared norm of the precoded field
        :math:`\lVert\sum_p w_p \mathbf{E}_p\rVert^2`

        The fields are combined before computing the norm. Expanding the norm
        into self and cross terms of the patterns would be cheaper, but it
        suffers from catastrophic cancellation in single precision when the
        fields of the patterns nearly cancel each other.

        A vector :math:`(x_0, x_1, x_2, x_3)` represents the complex-valued
        field :math:`\mathbf{E} = (x_0 + jx_2, x_1 + jx_3)`.

        :param weights: Real and imaginary parts of the weights :math:`w_p`
            of one precoding vector, as returned by
            :meth:`_synthetic_array_weights`
        :param e_fields: Electric fields, one per transmit antenna pattern

        :return: Squared norm of the precoded field
        """
        s = dr.zeros(mi.Vector4f)
        for i, e in enumerate(e_fields):
            wr = weights[2*i]
            wi = weights[2*i + 1]
            s += mi.Vector4f(wr * e.x - wi * e.z,
                             wr * e.y - wi * e.w,
                             wr * e.z + wi * e.x,
                             wr * e.w + wi * e.y)
        return dr.squared_norm(s)

    @dr.syntax
    def _synthetic_array_weights(
        self,
        scene: Scene,
        k_tx: mi.Vector3f,
        rel_ant_positions_tx: mi.Point3f,
        precoding_vecs: Tuple[mi.TensorXf, mi.TensorXf],
    ) -> List[List[mi.Float]]:
        r"""
        Computes the weights to apply to the electric field to synthetically
        model the transmitter array and the precoding

        For every precoding vector :math:`\mathbf{p}` and transmit antenna
        pattern :math:`p`, the array weight of a path with direction of
        departure :math:`\mathbf{k}` is
        :math:`w_p = \sum_n p_{p,n} e^{j\frac{2\pi}{\lambda}\mathbf{k}^{\textsf{T}}\mathbf{r}_n}`,
        where :math:`\mathbf{r}_n` is the position of the :math:`n^{th}`
        antenna relative to the transmitter. The returned weights are the
        real and imaginary parts of :math:`w_p` for all patterns. As only the
        power of the precoded field is required, a single power gain
        :math:`\lvert w_0\rvert^2` is returned instead for single-polarized
        arrays.

        :param scene: Scene for which to compute the radio map
        :param k_tx: Directions of departures of paths
        :param rel_ant_positions_tx: Positions of the antenna elements relative
            to the transmitters positions
        :param precoding_vecs: Real and imaginary components of the
            complex-valued precoding vectors, each with shape
            `[num_precoding_vecs, num_tx, num_tx_ant]`

        :return: Weights, one list per precoding vector
        """

        num_patterns = len(scene.tx_array.antenna_pattern.patterns)
        array_size = scene.tx_array.array_size
        num_samples = dr.shape(k_tx)[-1]
        num_tx = len(scene.transmitters)
        samples_per_tx = num_samples // num_tx
        # k_tx : 3*[num_samples = num_tx*samples_per_tx]
        # rel_ant_positions_tx : 3*[num_tx, array_size]
        # precoding_vecs : 2*[num_precoding_vecs, num_tx,
        #                     num_patterns*array_size]
        num_precoding_vecs = dr.shape(precoding_vecs[0])[0]
        precoding_real = precoding_vecs[0].array
        precoding_imag = precoding_vecs[1].array
        # Offset between two consecutive precoding vectors
        vec_stride = num_tx * num_patterns * array_size

        # Array weights for each precoding vector and antenna pattern
        w_real = [[dr.zeros(mi.Float, num_samples) for _ in range(num_patterns)]
                  for _ in range(num_precoding_vecs)]
        w_imag = [[dr.zeros(mi.Float, num_samples) for _ in range(num_patterns)]
                  for _ in range(num_precoding_vecs)]

        ##############################################################
        # Iterate over the antennas to compute the weighting
        ##############################################################

        tx_indices = dr.repeat(dr.arange(mi.UInt, num_tx), samples_per_tx)
        ant_gather_indices = tx_indices * array_size
        prec_gather_indices = tx_indices * (num_patterns * array_size)

        n = mi.UInt(0)
        while dr.hint(n < array_size, mode=self.loop_mode):

            # Extract the relative position of antenna
            # [num_samples, 3]
            ant_pos = dr.gather(mi.Point3f, rel_ant_positions_tx,
                                ant_gather_indices)

            # Compute the phase shifts, shared by all precoding vectors
            # [num_samples]
            tx_phase_shifts = dr.dot(ant_pos, k_tx)
            tx_phase_shifts *= dr.two_pi / scene.wavelength
            array_vec_imag, array_vec_real = dr.sincos(tx_phase_shifts)

            for v in range(num_precoding_vecs):
                for i in range(num_patterns):
                    # Dot product with precoding vector iteratively computed
                    # [num_samples]
                    ind = prec_gather_indices + (v * vec_stride
                                                 + i * array_size)
                    prec_real = dr.gather(mi.Float, precoding_real, ind)
                    prec_imag = dr.gather(mi.Float, precoding_imag, ind)
                    w_real[v][i] += array_vec_real * prec_real \
                                    - array_vec_imag * prec_imag
                    w_imag[v][i] += array_vec_real * prec_imag \
                                    + array_vec_imag * prec_real

            # Prepare for next antenna
            ant_gather_indices += 1
            prec_gather_indices += 1
            n += 1

        weights = []
        for v in range(num_precoding_vecs):
            if num_patterns == 1:
                # Only the power gain of the array is required
                weights.append([dr.square(w_real[v][0])
                                + dr.square(w_imag[v][0])])
            else:
                weights.append([w for i in range(num_patterns)
                                for w in (w_real[v][i], w_imag[v][i])])

        return weights

    def _sample_radio_material(
        self,
        si: mi.SurfaceInteraction3f,
        k_world: mi.Vector3f,
        e_fields: List[mi.Vector4f],
        solid_angle: mi.Float,
        sample1: mi.Float,
        sample2: mi.Point2f,
        enabled_interactions: mi.UInt,
        active: mi.Bool
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
        :param enabled_interactions: Flags indicating the enabled interactions
        :param active: Mask to specify active rays

        :return: Updated electric field and sampling record
        """

        # Ensure the normal is oriented in the opposite of the direction of
        # propagation of the incident wave
        normal_world = si.n * dr.sign(dr.dot(si.n, -k_world))
        si.sh_frame.n = normal_world
        si.initialize_sh_frame()
        si.n = normal_world

        # Set `si.wi` to the local direction of propagation of the incident wave
        si.wi = si.to_local(k_world)

        # `si.dp_du.z` stores the flags indicating the enabled interactions
        si.dp_du = mi.Vector3f(0., 0., 0.)
        si.dp_du.z = dr.reinterpret_array(mi.Float,
                                          enabled_interactions)

        # Specify the components that are required
        ctx = mi.BSDFContext(mode=mi.TransportMode.Importance,
                             type_mask=0, component=0)

        # Samples and evaluate the radio material
        for i, e_field in enumerate(e_fields):
            # `si.duv_dx` and `si.duv_dy` stores the incident field
            si.duv_dx = mi.Vector2f(e_field.x, # S
                                    e_field.y) # P
            si.duv_dy = mi.Vector2f(e_field.z, # S
                                    e_field.w) # P
            # `si.t` stores the solid angle
            si.t = solid_angle
            # Sample and evaluate the radio material
            bs_sample, jones_mat = si.bsdf().sample(ctx, si, sample1, sample2,
                                                    active)
            jones_mat = spectrum_to_matrix_4f(jones_mat)
            # Update the field by applying the Jones matrix
            e_fields[i] = dr.select(
                active, mi.Vector4f(jones_mat @ e_field), e_fields[i]
            )

        # Note: we assume that the `bs_sample` returned is the same for all
        # `e_field` values, so we just return the last one.
        return bs_sample, e_fields

    def _evaluate_radio_material_diffraction(self,
        shape: mi.ShapePtr,
        wedges: WedgeGeometry,
        diff_point: mi.Point3f,
        ki_world: mi.Vector3f,
        ko_world: mi.Vector3f,
        e_fields: List[mi.Vector4f],
        s: mi.Float,
        s_prime: mi.Float,
        active: mi.Bool
        ) -> Tuple[mi.Vector4f]:
        # pylint: disable=line-too-long
        r"""
        Evaluates the radio material for diffraction and returns the resulting
        electric field

        :param shape: Intersected shape
        :param wedges: Geometry of the intersected wedges
        :param diff_point: Position of the diffraction point on the wedge
        :param ki_world: Directions of propagation of the incident waves in the world frame
        :param ko_world: Directions of propagation of the scattered waves in the world frame
        :param e_fields: Jones vector representing the electric field as a 4D real-valued vector
        :param s: Distance parameter for diffraction calculations
        :param s_prime: Second distance parameter for diffraction calculations
        :param active: Mask to specify active rays

        :return: Updated electric field
        """

        # Radio material of the intersected shape
        rm = shape.bsdf()

        # Build a surface interaction object and context object to call the
        # radio material
        si = dr.zeros(mi.SurfaceInteraction3f, shape.shape[0])
        ctx = mi.BSDFContext(mode=mi.TransportMode.Importance,
                             type_mask=0, component=0)
        # If diffraction is globally disabled, we can avoid running the related code to
        # speed up the computation
        ctx.component |= InteractionType.DIFFRACTION

        # Ensure the normal is oriented in the opposite of the direction of
        # propagation of the incident wave
        normal_world = wedges.n0 * dr.sign(dr.dot(wedges.n0, -ki_world))
        si.n = normal_world
        si.sh_frame.n = normal_world
        si.initialize_sh_frame()

        # Set `si.wi` to the local direction of propagation of the incident wave
        si.wi = si.to_local(ki_world)
        # Interaction point
        si.p = diff_point
        # Intersected shape
        si.shape = shape
        # Intersected primitive
        si.prim_index = wedges.prim0

        # `si.dn_du` stores the edge vector in the local frame
        si.dn_du = si.to_local(wedges.e_hat)
        # `si.dn_dv` stores the normal to the n-face in the local frame
        si.dn_dv = si.to_local(wedges.nn)
        # `si.dp_du` stores the path length from the diffraction point to the
        # source, target, and the interaction type.
        enabled_interactions = mi.UInt(InteractionType.DIFFRACTION)
        si.dp_du = mi.Vector3f(s,
                               s_prime,
                               dr.reinterpret_array(mi.Float, enabled_interactions))

        # Compute the diffracted fields
        diffracted_e_fields = []
        # Spreading factor
        sf = dr.rsqrt(s*s_prime*(s + s_prime))
        for e_field in e_fields:
            # `si.duv_dx` and `si.duv_dy` stores the incident field
            si.duv_dx = mi.Vector2f(e_field.x, # S
                                    e_field.y) # P
            si.duv_dy = mi.Vector2f(e_field.z, # S
                                    e_field.w) # P

            # Evaluate the radio material
            jones_mat = rm.eval(ctx, si, ko_world, active)
            jones_mat = spectrum_to_matrix_4f(jones_mat)
            jones_mat *= sf
            # Apply the Jones matrix
            diffracted_e_fields.append(
                dr.select(active, jones_mat@e_field, e_field))

        return diffracted_e_fields
