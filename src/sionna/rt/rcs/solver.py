#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""RCS solver: Compute propagation paths scattered by sensing targets"""

from numbers import Integral
from typing import Tuple

import drjit as dr
import mitsuba as mi
import numpy as np

from sionna.rt import Scene
from sionna.rt.constants import InteractionType
from sionna.rt.path_solvers.field_calculator import FieldCalculator
from sionna.rt.path_solvers.image_method import ImageMethod
from sionna.rt.path_solvers.paths import Paths
from sionna.rt.path_solvers.paths_buffer import PathsBuffer
from sionna.rt.path_solvers.sb_candidate_generator import SBCandidateGenerator
from sionna.rt.path_solvers.sb_deterministic import \
    SBDeterministicCandidateGenerator
from sionna.rt.utils import box_contains, concat_points, \
    jones_matrix_from_real_imag, SourceExclusionBoxes

from .scattering_points import ScatteringPoints


class RCSSolver:
    # pylint: disable=line-too-long
    r"""
    Class implementing a radar cross-section (RCS) solver

    This solver computes the propagation paths that connect the antennas of all
    transmitters to the antennas of all receivers of a scene through the
    scattering points of its sensing targets
    (:class:`~sionna.rt.rcs.SensingTarget`). Every computed path therefore
    consists of a transmitter-to-scattering-point leg, a scattering event, and
    a scattering-point-to-receiver leg:

    .. math::
        \text{TX} \rightarrow \dots \rightarrow \text{SP} \rightarrow \dots \rightarrow \text{RX}

    Both legs can undergo line-of-sight propagation, specular reflection, and
    refraction, i.e., the same interaction types as
    :class:`~sionna.rt.PathSolver` except for diffuse reflection and
    diffraction which are not supported. The scattering event is described by
    the radar cross-section (RCS) and the cross-polarization matrix (CPM) of
    the scattering point, evaluated for the incident and scattered directions
    of the path.

    Paths that do not interact with a sensing target are not computed by this
    solver; use :class:`~sionna.rt.PathSolver` to compute them.

    The sensing targets are part of the environment through which the paths are
    traced, and are seen as absorbers by the :class:`~sionna.rt.PathSolver`:
    a target shadows the scattering points of the other targets and
    blocks the legs of the computed paths. A target does however not occlude
    the scattering points it contains, i.e., the ones which lie within its
    bounding box. This is because the scattering response of a target is
    entirely described
    by its scattering model: a leg which starts from such a point is only
    tested for occlusion once it has left that bounding box. A scattering
    point placed outside of the bounding box of its target, on the other hand,
    is occluded by that target as it is by any other object of the scene.

    As with :class:`~sionna.rt.PathSolver`, if synthetic arrays are used
    (``synthetic_array`` is `True`), transmitters and receivers are modelled as
    if they had a single antenna located at their
    :attr:`~sionna.rt.RadioDevice.position`, and the channel responses of the
    individual antennas are computed "synthetically" by applying appropriate
    phase shifts.

    The Doppler shifts of the computed paths account for the mobility of the
    transmitters, receivers, scene objects, and sensing targets. Only the
    rigid translation of a sensing target is modelled: all its scattering
    points move with its :attr:`~sionna.rt.SceneObject.velocity`, so the
    Doppler shifts do not reflect the rotation of a target.

    Example
    -------
    .. code-block:: python

        import drjit as dr
        import mitsuba as mi
        import sionna
        from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
        from sionna.rt.rcs import (ConstantCPM, RCSSolver, ScatteringModel,
                                   SensingTarget)

        # Load example scene
        scene = load_scene(sionna.rt.scene.simple_street_canyon)

        # Configure antenna arrays for all transmitters and receivers
        scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                     polarization="V")
        scene.rx_array = scene.tx_array

        scene.add(Transmitter(name="tx", position=[-32,-9,25]))
        scene.add(Receiver(name="rx", position=[-32,11,31]))

        # Cross-section of 1 square meter, independent of the incident and
        # scattered directions. The seed is only used by the models with
        # random components.
        def rcs(k_i, k_s, seed):
            return dr.ones(mi.Float, dr.width(k_i))

        # Sensing target shaped as a cuboid, with a single scattering point
        # 2m below it which does not depolarize
        model = ScatteringModel([0,0,-2], rcs=rcs,
                                cpm=ConstantCPM())
        target = SensingTarget(name="st", scattering_model=model,
                               length=4., width=2., height=1.5,
                               position=[-16,-10,60])
        scene.add(target)

        # Compute paths
        solver = RCSSolver()
        paths = solver(scene)

        # Open preview showing paths
        scene.preview(paths=paths)
    """

    def __init__(self, deterministic: bool = False):
        """
        Instantiates the RCS solver.

        :param deterministic: Enable deterministic path generation. There should
                              be little to no effect on latency, however memory
                              usage will be increased.
        """

        # Instantiate the Candidate Generator
        if deterministic:
            self._candidate_generator = SBDeterministicCandidateGenerator()
        else:
            self._candidate_generator = SBCandidateGenerator()
        # Instantiate the Image Method solver
        self._image_method = ImageMethod()
        # Instantiate the Field Calculator
        self._field_calculator = FieldCalculator()

    @property
    def loop_mode(self):
        # pylint: disable=line-too-long
        r"""Get/set the Dr.Jit mode used to evaluate the loops that implement
        the solver. Should be one of "evaluated" or "symbolic". Symbolic mode
        (default) is the fastest one but does not support automatic
        differentiation.
        For more details, see the `corresponding Dr.Jit documentation <https://drjit.readthedocs.io/en/latest/cflow.html#sym-eval>`_.

        :type: "evaluated" | "symbolic"
        """
        return self._field_calculator.loop_mode

    @loop_mode.setter
    def loop_mode(self, mode):
        if mode not in ("evaluated", "symbolic"):
            raise ValueError(
                "Invalid loop mode. Must be either 'evaluated'" " or 'symbolic'"
            )
        self._image_method.loop_mode = mode
        self._field_calculator.loop_mode = mode

    def __call__(
        self,
        scene: Scene,
        max_depth: int = 3,
        buffer_size_per_sp: int = 1000000,
        samples_per_sp: int = 1000000,
        synthetic_array: bool = True,
        los: bool = True,
        specular_reflection: bool = True,
        refraction: bool = True,
        seed: int | None = None,
    ) -> Paths:
        # pylint: disable=line-too-long
        r"""
        Executes the solver

        Paths are traced from the scattering points of the sensing targets,
        which are therefore the sources of the underlying path tracing. The
        ``buffer_size_per_sp`` and ``samples_per_sp`` parameters consequently
        apply to each scattering point, and the legs of a scattering point
        towards the transmitters and towards the receivers share these
        budgets.

        :param scene: Scene for which to compute paths
        :param max_depth: Maximum depth of the paths, i.e., maximum
            total number of interactions including the scattering event on the
            sensing target. Each of the two legs can therefore undergo at most
            ``max_depth - 1`` interactions with the scene, and a value of
            ``1`` restricts the computed paths to those for which both legs
            are unobstructed.
        :param buffer_size_per_sp: Maximum number of legs stored per scattering point
        :param samples_per_sp: Number of samples per scattering point
        :param synthetic_array: If set to `True` (default), then the antenna arrays are applied synthetically
        :param los: Enable unobstructed legs, i.e., legs without any interaction
            with the scene
        :param specular_reflection: Enables specular reflection
        :param refraction: Enables refraction
        :param seed: Non-negative seed. If set to :py:class:`None` (default),
                     a seed is drawn at random, so that every call draws new
                     random components. Reproducing a call therefore requires
                     passing an explicit seed. A fixed seed does not guarantee
                     deterministic results unless the solver is constructed with
                     ``deterministic=True``. The seed is also given to the RCS
                     and CPM of the scattering points, which the models with
                     random components use to draw them, e.g.
                     :class:`~sionna.rt.rcs.TR38901RCS` and
                     :class:`~sionna.rt.rcs.TR38901CPM`.

        :return: Computed paths, as an instance of :class:`~sionna.rt.Paths`
        """

        # Validate public arguments before any allocation or sampling
        if not isinstance(scene, Scene):
            raise TypeError("`scene` must be an instance of Scene")
        if isinstance(max_depth, bool) or not isinstance(max_depth, Integral):
            raise TypeError("`max_depth` must be an integer")
        if max_depth < 1:
            raise ValueError("`max_depth` must be greater than or equal to one")
        if (isinstance(buffer_size_per_sp, bool)
                or not isinstance(buffer_size_per_sp, Integral)):
            raise TypeError("`buffer_size_per_sp` must be an integer")
        if buffer_size_per_sp < 1:
            raise ValueError(
                "`buffer_size_per_sp` must be greater than or equal to one"
            )
        if (isinstance(samples_per_sp, bool)
                or not isinstance(samples_per_sp, Integral)):
            raise TypeError("`samples_per_sp` must be an integer")
        if samples_per_sp < 1:
            raise ValueError(
                "`samples_per_sp` must be greater than or equal to one"
            )
        if not isinstance(synthetic_array, bool):
            raise TypeError("`synthetic_array` must be a bool")
        if not isinstance(los, bool):
            raise TypeError("`los` must be a bool")
        if not isinstance(specular_reflection, bool):
            raise TypeError("`specular_reflection` must be a bool")
        if not isinstance(refraction, bool):
            raise TypeError("`refraction` must be a bool")
        if seed is None:
            # Drawn from the numpy global generator, so that seeding it makes
            # the solver reproducible as well
            seed = int(np.random.randint(1 << 31))
        elif isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("`seed` must be an integer")
        elif seed < 0:
            raise ValueError("`seed` must be greater than or equal to zero")

        # Check that the scene is all set for simulations
        scene.all_set(radio_map=False)
        if len(scene.sensing_targets) == 0:
            raise ValueError("Scene has no sensing targets")

        # Generates sources and targets positions and orientations.
        # Note that the sources and targets of the traced paths are the
        # scattering points and the radio devices, respectively, as paths are
        # traced from the scattering points towards the radio devices.
        src_positions, src_orientations, rel_ant_positions_tx, tx_velocities = (
            scene.sources(synthetic_array, True)
        )
        tgt_positions, tgt_orientations, rel_ant_positions_rx, rx_velocities = (
            scene.targets(synthetic_array, True)
        )
        num_tx = dr.width(src_positions)

        src_antenna_patterns = scene.tx_array.antenna_pattern.patterns
        tgt_antenna_patterns = scene.rx_array.antenna_pattern.patterns

        # Scattering points of all the sensing targets, gathered in a single
        # collection
        spst, st_orientations, st_velocities, spst_st_indices, \
            spst_local_indices, spst_positions, spst_boxes = \
            self._sensing_geometry(scene)

        # Transmitters and receivers are the targets of the traced paths.
        # The first `num_tx` targets are the transmitters, the remaining ones
        # the receivers.
        src_tgt_positions = concat_points([src_positions, tgt_positions])

        dr.make_opaque(src_positions, tgt_positions, src_orientations,
                       tgt_orientations, src_tgt_positions, spst_positions,
                       st_orientations, st_velocities, spst_boxes.centers,
                       spst_boxes.half_extents, spst_boxes.orientations,
                       spst_boxes.enabled)

        # Generate candidates.
        # The scattering event on the sensing target is one of the
        # interactions counted by `max_depth`, leaving `max_depth - 1` for
        # each leg.
        paths_buffer = self._candidate_generator(
            mi_scene=scene.mi_scene,
            src_positions=spst_positions,
            tgt_positions=src_tgt_positions,
            samples_per_src=samples_per_sp,
            max_num_paths_per_src=buffer_size_per_sp,
            max_depth=max_depth - 1,
            los=los,
            specular_reflection=specular_reflection,
            diffuse_reflection=False,
            refraction=refraction,
            diffraction=False,
            edge_diffraction=False,
            seed=seed,
            src_boxes=spst_boxes,
        )

        paths_buffer.schedule()
        dr.eval()

        # Shrink the paths buffer to fit the number of paths effectively found
        paths_buffer.shrink()

        # Detach the paths geometry to avoid differentiation through the
        # candidate generator
        paths_buffer.detach_geometry()

        # Solve specular chains and suffixes
        paths_buffer = self._image_method(
            scene=scene.mi_scene,
            paths=paths_buffer,
            diffraction=False,
            diffraction_lit_region=False,
            src_positions=spst_positions,
            tgt_positions=src_tgt_positions,
            src_boxes=spst_boxes,
        )

        # Discard invalid paths
        paths_buffer.discard_invalid()

        # Split the paths buffer into the legs that connect the transmitters to
        # the scattering points and the ones that connect the scattering points
        # to the receivers
        from_tx_paths, to_rx_paths = self._split_at_devices(paths_buffer,
                                                            num_tx)

        # Every path is the concatenation of a leg from a transmitter and a leg
        # to a receiver that share the same scattering point
        from_tx_ind, to_rx_ind = self._pair_legs(from_tx_paths, to_rx_paths,
                                                 max_depth)

        # Stop here if no path was found
        if dr.width(from_tx_ind) == 0:
            return Paths(scene, src_positions, tgt_positions, tx_velocities,
                         rx_velocities, synthetic_array,
                         PathsBuffer(0, max_depth, False),
                         rel_ant_positions_tx, rel_ant_positions_rx)

        field_calculator = self._field_calculator

        # Compute the channel impulse response
        with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, False):

            # Initialize the electric field.
            # The targets of these legs are the scattering points.
            e_field = field_calculator.evaluate_transmitter_antenna_patterns(
                from_tx_paths,
                src_positions,
                spst_positions,
                src_orientations,
                src_antenna_patterns,
            )

            # Transport the electric field from the transmitters to the
            # scattering points
            e_field, tau_from_tx, doppler_from_tx, ki_world_spst = \
                field_calculator.transport_electric_field(
                    e_field,
                    scene.wavelength,
                    from_tx_paths,
                    samples_per_sp,
                    diffraction_enabled=False,
                    src_positions=src_positions,
                    tgt_positions=spst_positions,
                )

            # Direction in which the legs leave the scattering points
            kr_world_spst, *_ = field_calculator.first_segment(
                to_rx_paths,
                dr.gather(mi.Point3f, spst_positions,
                          to_rx_paths.source_indices),
                dr.gather(mi.Point3f, tgt_positions,
                          to_rx_paths.target_indices),
            )

            # Pair the legs, i.e., build one sample per computed path
            from_tx_paths = from_tx_paths.gather(from_tx_ind)
            to_rx_paths = to_rx_paths.gather(to_rx_ind)
            e_field = [dr.gather(mi.Vector4f, e, from_tx_ind) for e in e_field]
            tau_from_tx = dr.gather(mi.Float, tau_from_tx, from_tx_ind)
            doppler_from_tx = dr.gather(mi.Float, doppler_from_tx, from_tx_ind)
            ki_world_spst = dr.gather(mi.Vector3f, ki_world_spst, from_tx_ind)
            kr_world_spst = dr.gather(mi.Vector3f, kr_world_spst, to_rx_ind)

            # Sensing target and scattering point of every path
            spst_indices = from_tx_paths.target_indices
            st_indices = dr.gather(mi.UInt, spst_st_indices, spst_indices)

            # Evaluate and apply the Jones matrices of the scattering points
            jones_real, jones_imag = spst.eval_jones_matrix(ki_world_spst,
                                                            kr_world_spst,
                                                            st_orientations,
                                                            st_indices,
                                                            spst_indices,
                                                            seed)
            jones_mat = jones_matrix_from_real_imag(jones_real, jones_imag)
            e_field = [jones_mat@e for e in e_field]

            # Doppler shift due to the mobility of the sensing targets.
            # The scattering event contributes the velocity of the target
            # projected on the difference between the scattered and the
            # incident directions of propagation.
            v_st = dr.gather(mi.Vector3f, st_velocities, st_indices)
            doppler_sts = dr.dot(kr_world_spst - ki_world_spst, v_st) \
                          /scene.wavelength

            # Transport the electric field from the scattering points to the
            # receivers
            e_field, tau_to_rx, doppler_to_rx, ki_world_rx = \
                field_calculator.transport_electric_field(
                    e_field,
                    scene.wavelength,
                    to_rx_paths,
                    samples_per_sp,
                    diffraction_enabled=False,
                    src_positions=spst_positions,
                    tgt_positions=tgt_positions,
                )

            # Chain the two legs of every path.
            # First, build a paths buffer storing only the sensing
            # interactions, as the endpoints shared by two chained buffers are
            # not recorded as interactions.
            sensing_interactions = self._build_sensing_paths_buffer(
                dr.gather(mi.Point3f, spst_positions, spst_indices),
                st_indices,
                dr.gather(mi.UInt, spst_local_indices, spst_indices),
            )
            # The paired legs were selected to form paths of depth at most
            # `max_depth`, so this depth fits their interactions.
            paths_buffer = from_tx_paths.chain([sensing_interactions,
                                                to_rx_paths],
                                               max_depth=max_depth)

            # Compute the paths coefficients
            a, valid_a = field_calculator.compute_channel_coefficients(
                e_field,
                scene.wavelength,
                ki_world_rx,
                paths_buffer,
                tgt_orientations,
                tgt_antenna_patterns,
            )

            # Disable paths with 0 contribution
            paths_buffer.valid &= valid_a

            paths_buffer.a = a
            # The delay of a path is the sum of the contributions of its two
            # legs, and its Doppler shift also includes the contribution of
            # the scattering event on the sensing target.
            paths_buffer.tau = tau_from_tx + tau_to_rx
            paths_buffer.doppler = doppler_from_tx + doppler_to_rx \
                                   + doppler_sts

        # Discard invalid paths
        paths_buffer.discard_invalid()

        # Build the path object
        paths = Paths(
            scene,
            src_positions,
            tgt_positions,
            tx_velocities,
            rx_velocities,
            synthetic_array,
            paths_buffer,
            rel_ant_positions_tx,
            rel_ant_positions_rx,
        )

        return paths

    ##################################################
    # Internal methods
    ##################################################

    def _sensing_geometry(self, scene: Scene) -> Tuple[ScatteringPoints,
                                                       mi.Point3f,
                                                       mi.Vector3f, mi.UInt,
                                                       mi.UInt, mi.Point3f,
                                                       SourceExclusionBoxes]:
        r"""
        Gathers the scattering points of all the sensing targets of the scene
        in a single collection

        :param scene: Scene from which to read the sensing targets

        :return: Scattering points of all the sensing targets, orientations of
            the sensing targets, velocities of the sensing targets [m/s],
            index of the sensing target of every scattering point, index of
            every scattering point within its sensing target, positions of
            the scattering points in the global coordinate system [m], and
            bounding box of the sensing target of every scattering point
        """

        sensing_targets = list(scene.sensing_targets.values())

        st_positions = concat_points([st.position for st in sensing_targets])
        st_orientations = concat_points([st.orientation
                                         for st in sensing_targets])
        st_velocities = mi.Vector3f(concat_points([st.velocity
                                                   for st in sensing_targets]))
        st_scalings = mi.Vector3f(concat_points([st.scaling
                                                 for st in sensing_targets]))
        st_half_extents = mi.Vector3f(
            concat_points([st.lcs_half_extents for st in sensing_targets]))

        # Gathering the scattering points of all the sensing targets in a
        # single collection enables evaluating all the Jones matrices at once
        spst = ScatteringPoints().concat([st.scattering_model.spst
                                          for st in sensing_targets])
        if dr.width(spst.lcs_positions) == 0:
            raise ValueError("The sensing targets of the scene have no"
                             " scattering points")

        # Sensing target of every scattering point, and index of every
        # scattering point within its sensing target
        num_spst = [dr.width(st.scattering_model.spst.lcs_positions)
                    for st in sensing_targets]
        spst_st_indices = mi.UInt(np.repeat(np.arange(len(sensing_targets)),
                                            num_spst))
        spst_local_indices = mi.UInt(np.concatenate([np.arange(n)
                                                     for n in num_spst]))

        # Scattering points are positioned in the local frame of their sensing
        # target, and are scaled along with it
        spst_lcs_positions = spst.scaled_lcs_positions(st_scalings,
                                                       spst_st_indices)
        spst_positions = spst.gcs_positions(st_positions, st_orientations,
                                            spst_st_indices,
                                            st_scalings=st_scalings)

        # A sensing target does not occlude the scattering points it contains,
        # as its scattering response is entirely described by its scattering
        # model. The box of a point lying outside of its target is disabled, so
        # that the target shadows it as any other geometry of the scene would.
        # The half extents are scaled, so the positions they are tested against
        # must be the scaled ones.
        spst_half_extents = dr.gather(mi.Vector3f, st_half_extents,
                                      spst_st_indices)
        spst_boxes = SourceExclusionBoxes(
            centers=dr.gather(mi.Point3f, st_positions, spst_st_indices),
            half_extents=spst_half_extents,
            orientations=dr.gather(mi.Point3f, st_orientations,
                                   spst_st_indices),
            enabled=box_contains(spst_lcs_positions, spst_half_extents))

        return (spst, st_orientations, st_velocities, spst_st_indices,
                spst_local_indices, spst_positions, spst_boxes)

    def _split_at_devices(self,
                          paths: PathsBuffer,
                          num_tx: int) -> Tuple[PathsBuffer, PathsBuffer]:
        # pylint: disable=protected-access
        r"""
        Splits the traced paths into the legs that connect the transmitters to
        the scattering points and the legs that connect the scattering points
        to the receivers

        Paths are traced from the scattering points, so the legs that end on a
        transmitter are reversed to make the transmitter their source. The
        target indices of the legs that end on a receiver are rebased to index
        the receivers only.

        :param paths: Traced paths
        :param num_tx: Number of transmitters or transmit antennas

        :return: Legs from the transmitters to the scattering points, and legs
            from the scattering points to the receivers
        """

        if paths.buffer_size == 0:
            # No leg was found, so both sets of legs are empty
            return paths, paths

        # The targets with indices `0 ... num_tx-1` are the transmitters, and
        # the ones with indices `num_tx ...` the receivers
        to_tx = paths.target_indices < num_tx
        to_rx = paths.target_indices >= num_tx

        from_tx_paths = paths.gather(dr.compress(to_tx)).reverse()
        to_rx_paths = paths.gather(dr.compress(to_rx))
        # Rebasing is skipped if no leg reaches a receiver, as Dr.Jit does not
        # broadcast a scalar to an empty array
        if to_rx_paths.buffer_size > 0:
            to_rx_paths._tgt_indices = to_rx_paths.target_indices - num_tx

        return from_tx_paths, to_rx_paths

    def _pair_legs(self,
                   from_tx_paths: PathsBuffer,
                   to_rx_paths: PathsBuffer,
                   max_depth: int) -> Tuple[mi.UInt, mi.UInt]:
        r"""
        Pairs every leg from a transmitter with every leg to a receiver that
        shares the same scattering point and forms a path of depth at most
        ``max_depth``

        :param from_tx_paths: Legs from the transmitters to the scattering
            points
        :param to_rx_paths: Legs from the scattering points to the receivers
        :param max_depth: Maximum number of interactions of a path, including
            the scattering event on the sensing target

        :return: Indices of the legs from the transmitters and indices of the
            legs to the receivers forming the pairs
        """

        num_from_tx = from_tx_paths.buffer_size
        num_to_rx = to_rx_paths.buffer_size
        if (num_from_tx == 0) or (num_to_rx == 0):
            return dr.zeros(mi.UInt, 0), dr.zeros(mi.UInt, 0)

        num_inter_from_tx = from_tx_paths.num_interactions()
        num_inter_to_rx = to_rx_paths.num_interactions()

        # Candidate pairs, i.e., the Cartesian product of the two sets of legs
        pair = dr.arange(mi.UInt, num_from_tx*num_to_rx)
        from_tx_ind = pair % num_from_tx
        to_rx_ind = pair // num_from_tx

        # Only pairs of legs that share the same scattering point form a path.
        # The scattering points are the targets of the reversed legs from the
        # transmitters and the sources of the legs to the receivers.
        same_spst = \
            (dr.gather(mi.UInt, from_tx_paths.target_indices, from_tx_ind)
             == dr.gather(mi.UInt, to_rx_paths.source_indices, to_rx_ind))

        # Depth of the paired paths. The scattering event on the sensing target
        # adds one interaction to the ones of the two legs.
        depth = (dr.gather(mi.UInt, num_inter_from_tx, from_tx_ind)
                 + dr.gather(mi.UInt, num_inter_to_rx, to_rx_ind)
                 + 1)

        pair = dr.compress(same_spst & (depth <= max_depth))

        return (dr.gather(mi.UInt, from_tx_ind, pair),
                dr.gather(mi.UInt, to_rx_ind, pair))

    def _build_sensing_paths_buffer(self, spst_positions: mi.Point3f,
                                    st_indices: mi.UInt,
                                    spst_indices: mi.UInt) -> PathsBuffer:
        r"""
        Builds a paths buffer storing a single sensing interaction per path

        This buffer is intended to be chained with the legs of the paths, as the
        endpoints shared by two chained buffers are not recorded as interactions.

        :param spst_positions: Positions of the scattering points in the global
            coordinate system [m]
        :param st_indices: Index of the sensing target of every path
        :param spst_indices: Index of the scattering point of every path within its
            sensing target

        :return: Paths buffer storing one sensing interaction per path
        """

        num_paths = dr.width(st_indices)

        # One interaction per path, no diffracting wedges
        buffer = PathsBuffer(num_paths, max_depth=1, diffraction=False)
        active = dr.full(mi.Bool, True, num_paths)
        depth = dr.ones(mi.UInt, num_paths)
        buffer.valid = active
        buffer.set_interaction_type(
            depth, dr.full(mi.UInt, InteractionType.SENSING, num_paths), active)
        buffer.set_vertex(depth, spst_positions, active)
        buffer.set_prob(depth, dr.ones(mi.Float, num_paths), active)
        # The sensing target and sensing point indices are stored in the shapes and
        # primitives, respectively
        buffer.set_shape(depth, st_indices, active)
        buffer.set_primitive(depth, spst_indices, active)
        buffer.advance_paths_counter(num_paths)
        return buffer
