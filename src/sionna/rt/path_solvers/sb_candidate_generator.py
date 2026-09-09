#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Generator of paths candidates using shooting and bouncing of rays"""

import mitsuba as mi
import drjit as dr
from typing import Tuple

from sionna.rt.constants import InteractionType, MIN_SEGMENT_LENGTH,\
    NO_JONES_MATRIX
from sionna.rt.utils import spawn_ray_from_sources, fibonacci_lattice,\
    spawn_ray_to, sample_wedge_diffraction_point, WedgeGeometry, hash_fnv1a,\
    PlaneHasher, EdgeHasher
from .paths_buffer import PathsBuffer
from .sample_data import SampleData


class SBCandidateGenerator:
    """
    Generates path candidates using shooting-and-bouncing of rays

    This is a callable object that returns the candidates as a
    :class:`~sionna.rt.PathsBuffer` object.

    Paths flagged as valid, i.e., for which the entry in
    ``PathsBuffer.valid`` is set to `True`, are finalized: They connect a source
    to a target. Note that only paths ending with a diffuse reflection can
    be finalized through shooting-and-bouncing of rays and are therefore flagged
    as valid.

    Specular chains, which consist only of specular reflections and/or
    refractions, are only candidates. They require additional processing,
    e.g., using the image method, to either refine them to valid paths or to
    discard them.

    Similarly, paths ending by a specular suffix also require further
    processing. Neither specular chains nor paths ending by a specular suffix
    are flagged as valid.

    This generator ensures that all specular chain candidates are uniquely
    present in the returned buffer. This uniqueness is ensured through hashing
    of the paths. Note that this can cause loss of candidates due to hash
    collisions.
    """

    # Probability of selecting a diffraction event from a valid wedge
    DIFFRACTION_SAMPLING_PROBABILITY = 0.2

    # Specular chains are considered identical if they share the same
    # hash. An array is used to store the number of times that a hash has
    # been observed. If the counter is > 0, then the specular chain is not
    # considered as new and not saved as a candidate. This array is indexed by
    # taking the hash modulo the size of this array. The size of this array
    # needs therefore to be large enough to ensure that the number of collisions
    # stays low and that candidates are not discarded due to collisions. The
    # following constant gives the minimum size of this array per source.
    MIN_SPEC_COUNT_SIZE = int(1e6)

    def __init__(self):

        # Sampler for generating random numbers
        self._sampler = mi.load_dict({'type': 'independent'})
        self.plane_hash_functions = [PlaneHasher(op='round'),
                                     PlaneHasher(op='floor')]
        self.edge_hash_functions = [EdgeHasher(op='round'),
                                    EdgeHasher(op='floor')]

    def __call__(self,
                 mi_scene: mi.Scene,
                 src_positions: mi.Point3f,
                 tgt_positions: mi.Point3f,
                 samples_per_src: int,
                 max_num_paths_per_src: int,
                 max_depth: int,
                 los: bool,
                 specular_reflection: bool,
                 diffuse_reflection: bool,
                 refraction: bool,
                 diffraction: bool,
                 edge_diffraction: bool,
                 seed: int = 1) -> PathsBuffer:
        # pylint: disable=line-too-long
        r"""
        Instantiates the paths buffer and runs the candidate generator

        :param mi_scene: Mitsuba scene
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets
        :param samples_per_src: Number of samples to spawn per source
        :param max_num_paths_per_src: Maximum number of candidates per source
        :param max_depth: Maximum path depths
        :param los: If set to `True`, then the LoS paths are computed
        :param specular_reflection: If set to `True`, then the specularly reflected paths are computed
        :param diffuse_reflection: If set to `True`, then the diffusely reflected paths are computed
        :param refraction: If set to `True`, then the refracted paths are computed
        :param diffraction: If set to `True`, then the diffracted paths are computed
        :param edge_diffraction: If set to `True`, then the diffraction on free floating edges is computed
        :param seed: Seed for the sampler. Defaults to 1.

        :return: Candidate paths
        """

        num_sources = dr.shape(src_positions)[1]
        num_samples = samples_per_src*num_sources
        max_num_paths = max_num_paths_per_src*num_sources

        # Set the seed of the sampler
        self._sampler.seed(seed, num_samples)

        # Allocate memory for `max_num_paths` paths.
        # After the shoot-and-bounce process, if the number of paths found is
        # below `max_num_paths`, then the tensors are shrunk.
        paths = PathsBuffer(max_num_paths, max_depth, diffraction)

        # Counter indicating how many paths were found for each source.
        # To ensure that the path buffer is not filled by a single or a few
        # sources, we count the number of paths traced for each source to ensure
        # that no more than `max_num_paths_per_src` are stored.
        # This is a way to ensure that the buffer is equally allocated to all
        # sources.
        paths_counter_per_source = dr.zeros(mi.UInt, num_sources)

        # Test LoS and add valid LoS paths to `paths`
        if los:
            self._los(mi_scene, src_positions, tgt_positions, paths,
                      paths_counter_per_source)

        # Run Shooting-and-bouncing of rays if required, i.e., if max_depth > 0
        if max_depth > 0:
            self._shoot_and_bounce(mi_scene, src_positions, tgt_positions,
                    paths, samples_per_src, max_num_paths_per_src, max_depth,
                    paths_counter_per_source, specular_reflection,
                    diffuse_reflection, refraction, diffraction,
                    edge_diffraction, seed)

        return paths

    ##################################################
    # Internal methods
    ##################################################

    @dr.syntax
    def _los(self,
             mi_scene: mi.Scene,
             src_positions: mi.Point3f,
             tgt_positions: mi.Point3f,
             paths: PathsBuffer,
             paths_counter_per_source: mi.UInt):
        # pylint: disable=line-too-long
        r"""
        Tests line-of-sight (LoS) paths and add non-obstructed ones to the
        buffer

        The buffer ``paths`` is updated in-place.

        :param mi_scene: Mitsuba scene
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets
        :param paths: Paths buffer. Updated in-place.
        :param paths_counter_per_source: Counts the number of paths found for each source
        """

        num_src = dr.width(src_positions)
        num_tgt = dr.width(tgt_positions)

        # Sample data
        samples_data = SampleData(num_src, num_tgt, 0, diffraction=False)

        # Target indices
        tgt_indices = dr.arange(mi.UInt, num_tgt)
        tgt_indices = dr.tile(tgt_indices, num_src)

        # Rays origins and targets
        origins = dr.repeat(src_positions, num_tgt)
        targets = dr.tile(tgt_positions, num_src)

        # Discard LoS paths when sources and targets overlap
        length = dr.norm(targets - origins)
        valid = length > MIN_SEGMENT_LENGTH

        # Test LoS
        rays = spawn_ray_to(origins, targets)
        valid &= ~mi_scene.ray_test(rays, active=valid)

        # Store the paths
        # Index where to store the current path.
        path_ind = dr.scatter_inc(paths.paths_counter, mi.UInt(0), valid)
        paths.add_paths(mi.UInt(0), path_ind, samples_data, valid, tgt_indices,
                        rays.d, -rays.d, valid)
        # Increment the sources counters
        dr.scatter_inc(paths_counter_per_source, samples_data.src_indices,
                       valid)

    def _shoot_and_bounce(self,
                          mi_scene: mi.Scene,
                          src_positions: mi.Point3f,
                          tgt_positions: mi.Point3f,
                          paths: PathsBuffer,
                          samples_per_src: int,
                          max_num_paths_per_src: int,
                          max_depth: int,
                          paths_counter_per_source: mi.UInt,
                          specular_reflection: bool,
                          diffuse_reflection: bool,
                          refraction: bool,
                          diffraction: bool,
                          edge_diffraction: bool,
                          seed: int):  # pylint: disable=unused-argument
        # pylint: disable=line-too-long
        r"""
        Executes shooting-and-bouncing of rays

        The paths buffer ``path`` is updated in-place.

        :param mi_scene: Mitsuba scene
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets
        :param paths: Buffer storing the candidate paths. Updated in-place.
        :param samples_per_src: Number of samples spawned per source
        :param max_num_paths_per_src: Maximum number of candidates per source
        :param max_depth:  Maximum path depths
        :param paths_counter_per_source: Counts the number of paths found for each source
        :param specular_reflection: If set to `True`, then the specularly reflected paths are computed
        :param diffuse_reflection: If set to `True`, then the diffusely reflected paths are computed
        :param refraction: If set to `True`, then the refracted paths are computed
        :param diffraction: If set to `True`, then the diffracted paths are computed
        :param edge_diffraction: If set to `True`, then the diffraction on free floating edges is computed
        """

        # Runs the shooting-and-bouncing of rays loop
        with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, False):
            self._shoot_and_bounce_loop(mi_scene, src_positions, tgt_positions,
                samples_per_src, max_num_paths_per_src, max_depth, paths,
                paths_counter_per_source, specular_reflection, diffuse_reflection,
                refraction, diffraction, edge_diffraction, self._sampler)

    @dr.syntax
    def _shoot_and_bounce_loop(self,
                               mi_scene: mi.Scene,
                               src_positions: mi.Point3f,
                               tgt_positions: mi.Point3f,
                               samples_per_src: int,
                               max_num_paths_per_src: int,
                               max_depth: int,
                               paths: PathsBuffer,
                               paths_counter_per_source: mi.UInt,
                               specular_reflection_enabled: bool,
                               diffuse_reflection_enabled: bool,
                               refraction_enabled: bool,
                               diffraction_enabled: bool,
                               edge_diffraction_enabled: bool,
                               sampler: mi.Sampler):
        # pylint: disable=line-too-long
        """
        Executes shooting-and-bouncing of rays

        The paths buffer ``path`` is updated in-place.

        :param mi_scene: Mitsuba scene
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets
        :param samples_per_src: Number of samples spawned per source
        :param max_num_paths_per_src: Maximum number of candidates per source
        :param max_depth:  Maximum path depths
        :param paths: Buffer storing the candidate paths. Updated in-place.
        :param paths_counter_per_source: Counts the number of paths found for each source
        :param specular_reflection_enabled: If set to `True`, then the specularly reflected paths are computed
        :param diffuse_reflection_enabled: If set to `True`, then the diffusely reflected paths are computed
        :param refraction_enabled: If set to `True`, then the refracted paths are computed
        :param diffraction_enabled: If set to `True`, then the diffracted paths are computed
        :param edge_diffraction_enabled: If set to `True`, then the diffraction on free floating edges is computed
        :param sampler: Sampler used to generate pseudo-random numbers
        """

        num_sources = dr.shape(src_positions)[1]
        num_targets = dr.shape(tgt_positions)[1]
        num_samples = samples_per_src*num_sources

        # Structure storing the sample data, which is used to build the paths
        samples_data = SampleData(num_sources, samples_per_src, max_depth,
                                  diffraction_enabled)

        # Rays
        ray = spawn_ray_from_sources(fibonacci_lattice, samples_per_src,
                                     src_positions)

        # Store direction of departure
        k_tx = dr.copy(ray.d)

        # Boolean indicating if the sample is a specular chain, i.e., if it
        # consists only of specular chains.
        specular_chain = dr.full(mi.Bool, True, num_samples)

        # Hash of the paths.
        # It is computed only for specular chains, and used to not duplicate
        # specular chain candidates.
        # 64bit integer is used for hashing.
        # Multiple hash functions are used to mitigate the risk of different
        # paths having the same hash due to quantization.
        num_hashes = len(self.plane_hash_functions)
        hashes = [dr.zeros(mi.UInt64, num_samples) for _ in range(num_hashes)]

        # Counter indicating how many occurrences of a specular chain was found.
        # Specular chains are considered identical if they share the same
        # hash. Taking the hash modulo the size of the following array is used
        # to index this array and increment the counter. If the counter is > 0,
        # then the specular chain is not considered as new and not stored.
        # The size of the following array needs therefore to be large enough
        # to ensure that the number of collisions stays low and that candidates
        # are not discarded due to collisions.
        spec_counter_size = dr.maximum(max_num_paths_per_src,
                                       SBCandidateGenerator.MIN_SPEC_COUNT_SIZE)
        specular_chain_counters = [dr.zeros(mi.UInt, spec_counter_size*num_sources)
                                   for _ in range(num_hashes)]

        # Current depth
        depth = dr.full(mi.UInt, 1, num_samples)

        # Mask indicating which rays are active
        active = dr.full(mi.Mask, True, num_samples)

        # Flag storing which types of interactions are locally enabled.
        # The booleans specular_reflection_enabled, diffuse_reflection_enabled,
        # etc. enable or disable interaction types globally.
        # Globally enabled interaction types can however be locally disabled,
        # i.e., disabled at the scale of a single path or intersection point.
        # For example, diffuse reflections are disabled for a path if diffraction
        # occurs. The following flag stores which interaction types are locally
        # enabled. It is initialized using the globally enabled interaction types.
        loc_en_inter = dr.full(mi.UInt, 0, num_samples)
        if specular_reflection_enabled:
            loc_en_inter |= InteractionType.SPECULAR
        if diffuse_reflection_enabled:
            loc_en_inter |= InteractionType.DIFFUSE
        if refraction_enabled:
            loc_en_inter |= InteractionType.REFRACTION
        if diffraction_enabled:
            loc_en_inter |= InteractionType.DIFFRACTION

        # Note: here and in the inner loop, we explicitly exclude some non-state
        # variables from the loop state so that DrJit doesn't have to trace
        # the loop body twice to figure it out.
        while dr.hint(active, label="shoot_and_bounce", exclude=[
            specular_chain_counters,
            paths_counter_per_source,
        ]):

            ########################################################
            # Test intersection with the scene and evaluate the
            # intersection
            ########################################################

            # Test intersection with the scene
            si_scene = mi_scene.ray_intersect(ray, coherent=True,
                                              ray_flags=mi.RayFlags.Minimal,
                                              active=active)

            # Deactivate rays that didn't hit the scene, i.e., that bounce-out
            # of the scene
            active &= si_scene.is_valid()

            # Samples the radio material
            sample1 = sampler.next_1d()
            sample2 = sampler.next_2d()
            sample2_diffraction = sampler.next_2d()
            s, n, wedges = self._sample_radio_material(si_scene, ray.o, ray.d,
                                                       sample1, sample2,
                                                       sample2_diffraction,
                                                       loc_en_inter,
                                                       diffraction_enabled,
                                                       edge_diffraction_enabled,
                                                       active)
            # Direction of propagation of scattered wave in implicit world
            # frame
            k_world = s.wo
            # Interaction type
            int_type = dr.select(active, s.sampled_component,
                                 InteractionType.NONE)
            # Disable paths if a NONE interaction was sampled.
            # This happens if no interaction type is enabled
            active &= (int_type != InteractionType.NONE)

            # Flag indicating if the interaction is a specular reflection
            specular = int_type == InteractionType.SPECULAR

            # Flag indicating if the interaction is a transmission
            transmission = int_type == InteractionType.REFRACTION

            # Flag indicating if the interaction is a diffuse reflection
            diffuse = int_type == InteractionType.DIFFUSE

            # Flag indicating if the interaction is a diffraction
            diffraction = int_type == InteractionType.DIFFRACTION

            # Only first order diffraction is supported.
            # Therefore, we disable diffraction for future interactions
            # if diffraction is sampled.
            # Diffraction displaces the interaction point to an edge, which can
            # invalidate specular reflections, transmissions, and diffractions
            # that previously occurred. To avoid having to post-process path
            # segments in addition to the specular suffix (e.g., using the
            # image method), we disable paths that contain both diffuse and
            # diffraction.
            loc_en_inter[diffraction] &= ~(mi.UInt(InteractionType.DIFFUSE
                                                   + InteractionType.DIFFRACTION))
            loc_en_inter[diffuse] &= ~mi.UInt(InteractionType.DIFFRACTION)

            # Is the sample a specular chain?
            # A specular chain consists only of specular reflections,
            # or transmissions or diffractions
            specular_chain &= active & (specular | transmission | diffraction)

            ########################################################
            # Update samples data
            ########################################################

            # Update the samples data
            samples_data.insert(depth, int_type, si_scene.shape,
                                si_scene.prim_index, wedges,
                                si_scene.p, s.pdf, active)

            ########################################################
            # Store the paths.
            # A path is stored if:
            # - It is a new specular chain
            # - It is valid, i.e., it connects to a target
            ########################################################

            # Encode the current plane as a 3D vector
            for i in range(num_hashes):
                plane_hash = self.plane_hash_functions[i](si_scene.n, si_scene.p)
                edge_hash = self.edge_hash_functions[i](wedges.o,
                                                        wedges.o + wedges.e_hat*wedges.length)
                inter_hash = dr.select(int_type == InteractionType.SPECULAR, plane_hash, int_type)
                inter_hash = dr.select(int_type == InteractionType.DIFFRACTION, edge_hash, inter_hash)
                hashes[i] = hash_fnv1a(inter_hash, h=hashes[i])

            # Loop over all targets.

            # Target index
            t = mi.UInt(0)
            while dr.hint(t < num_targets, label="shoot_and_bounce_inner",
                          exclude=[specular_chain_counters,
                                   paths_counter_per_source]):
                # Position of the target with index t
                tgt_position = dr.gather(mi.Point3f, tgt_positions, t)

                # Test line-of-sight with the target from the current
                # interaction point
                los_ray = si_scene.spawn_ray_to(tgt_position)
                los_blocked = mi_scene.ray_test(los_ray, active=active)
                los_visible = ~los_blocked

                # If the interaction is valid and if the target is visible from
                # the intersection point, then the path is marked as valid.

                # It is also required that the target is on the same side of
                # the intersected surface than the incident wave.
                # `n`` is the normal to the intersected surface oriented towards
                # the incident half-space
                # `los_ray.d` is from the intersection point to the target
                target_incident_side = dr.dot(n, los_ray.d) > 0.
                valid = los_visible & diffuse & target_incident_side

                # If this path is a specular chain, then we hash it to ensure
                # that it is a new paths.
                # A specular chain is only considered as candidate if the
                # intersection point is in LoS with the target. This condition
                # is used as an heuristic to reduce the number of candidates.
                # It also helps to reduce the number of access to the hash table
                # storing the specular chain counter, and therefore reduces the
                # number of collisions.
                new_specular = specular_chain & los_visible

                # A specular chain is considered as new, and therefore should
                # be stored in the `path` structure, if its hash has not been
                # already observed. To ensure that the path has not
                # already been stored for the target `t`, we combine it with
                # the path hash.
                # We use several hash functions to mitigate the risk of slightly
                # different paths having a different hash due to numerical precision issues.
                # A path is considered as new if all hashes are new.
                for i in range(num_hashes):
                    path_target_hash = hash_fnv1a(t, h=hashes[i])
                    counter_ind = path_target_hash % spec_counter_size
                    counter_ind += spec_counter_size*samples_data.src_indices
                    # If the sample is a specular chain, then the counter
                    # corresponding to its hash is increased, and we check that the
                    # counter value previous to its increment equals 0.
                    # To avoid race conditions where 2 threads increment first one of
                    # the counters and therefore neither is considered as new, it is
                    # important to use `new_specular` as the active mask for this
                    # operation. This way, only threads that are considered as new
                    # *so far* will increment the next counter.
                    samples_counter = dr.scatter_inc(specular_chain_counters[i],
                                                     counter_ind, new_specular)
                    new_specular &= samples_counter == 0

                # Store the paths
                store = active & (valid | new_specular)

                # Increment the per source path counter
                num_path_per_src = dr.scatter_inc(paths_counter_per_source,
                                                  samples_data.src_indices,
                                                  store)
                # If we exceeded the specified maximum number of paths,
                # then paths are discarded
                store &= num_path_per_src < max_num_paths_per_src

                # Index where to store the current path.
                path_ind = dr.scatter_inc(paths.paths_counter, mi.UInt(0),
                                          store)

                # Store the paths
                paths.add_paths(depth, path_ind, samples_data, valid, t, k_tx,
                                -los_ray.d, store)

                t += 1

            ####################################
            # Prepare next iteration
            ####################################

            # Deactivate rays if the maximum depth is reached and
            # the ones set as valid
            depth += 1
            active &= (depth <= max_depth)

            # Spawn rays for next iteration
            ray = si_scene.spawn_ray(d=k_world)

            # Reset the value of specular_chain in case of a diffuse reflection
            specular_chain |= diffuse

    def _sample_radio_material(self,
                               si: mi.SurfaceInteraction3f,
                               ray_origin: mi.Point3f,
                               k_world: mi.Vector3f,
                               sample1: mi.Float,
                               sample2: mi.Point2f,
                               sample2_diffraction: mi.Point2f,
                               loc_en_inter: mi.UInt,
                               diffraction_enabled: bool,
                               edge_diffraction_enabled: bool,
                               active: mi.Bool
        ) -> Tuple[mi.BSDFSample3f, mi.Normal3f, WedgeGeometry]:
        # pylint: disable=line-too-long
        r"""
        Samples the radio material of the intersected object

        This function prepares the inputs of the ``sample()`` method of the
        radio material, calls it, and then returns its output.

        :param si: Information about the interaction
        :param ray_origin: Origin of the ray
        :param k_world: Direction of propagation of the incident wave in the world frame
        :param sample1: Random float uniformly distributed in :math:`[0,1]`.
            Used to sample the interaction type.
        :param sample2: Random 2D point uniformly distributed in :math:`[0,1]^2`.
            Used to sample the direction of diffusely reflected waves.
        :param sample2_diffraction: Random 2D point uniformly distributed in :math:`[0,1]^2`.
            Used to project interaction point on an edge for diffraction and to sample diffraction
            interaction type.
        :param loc_en_inter: Bitmask indicating which interaction types are locally enabled
        :param diffraction_enabled: If set to `True`, then the diffraction is enabled
        :param edge_diffraction_enabled: If set to `True`, then the diffraction on free floating edges is computed
        :param active: Mask to specify active rays

        :return: Tuple containing:
            - Sampling record
            - Normal to the incident surface in the world frame
            - Wedge geometry containing edge information
        """

        # Ensure the normal is oriented in the opposite of the direction of
        # propagation of the incident wave
        normal_world = si.n*dr.sign(dr.dot(si.n, -k_world))
        si.sh_frame.n = normal_world
        si.initialize_sh_frame()
        si.n = normal_world

        # Specify the components that are required
        ctx = mi.BSDFContext(mode=mi.TransportMode.Importance,
                             type_mask=0, component=0)
        # Computation of the Jones matrix is not needed
        ctx.component |= NO_JONES_MATRIX
        # If diffraction is globally disabled, we can avoid running the related code
        # to save computation time
        if diffraction_enabled:
            ctx.component |= InteractionType.DIFFRACTION

        # If diffraction is enabled, the intersection point is projected on
        # the silhouette to identify valid wedges for diffraction. Only valid
        # wedges should be considered for diffraction by the radio material.
        diffraction = mi.Bool(False)
        probs = mi.Float(1)
        diffr_point = mi.Point3f(0.0, 0.0, 0.0)
        wedges = WedgeGeometry.build_with_size(1)
        if diffraction_enabled:
            loc_en_diffr = (loc_en_inter & InteractionType.DIFFRACTION) > 0
            diffraction, probs, diffr_point, wedges = \
                self._sample_diffraction_point(si, loc_en_inter,
                                               ray_origin, k_world,
                                               sample2_diffraction,
                                               edge_diffraction_enabled,
                                               active & loc_en_diffr)

            # If diffraction is sampled, then update the interaction points
            # to the diffraction points and the intersected primitives to the
            # ones carrying the diffracting edges
            si.p = dr.select(diffraction, diffr_point, si.p)
            si.prim_index = dr.select(diffraction, wedges.prim0, si.prim_index)

            # Set `si.wi` to the direction of propagation of the incident wave in
            # the local frame
            k_world = dr.select(diffraction,
                                dr.normalize(diffr_point - ray_origin),
                                k_world)

            # Update locally enabled interactions according to the diffraction
            # flag
            loc_en_inter = dr.select(diffraction,
                                    mi.UInt(InteractionType.DIFFRACTION),
                                    loc_en_inter & ~mi.UInt(InteractionType.DIFFRACTION))

            # `si.dn_du` stores the edge vector in the local frame
            si.dn_du = si.to_local(wedges.e_hat)
            # `si.dn_dv` stores the normal to the n-face in the local frame
            si.dn_dv = si.to_local(wedges.nn)
        else:
            loc_en_inter &= ~mi.UInt(InteractionType.DIFFRACTION)

        si.wi = si.to_local(k_world)

        # `si.dp_du` stores the path length from the diffraction point to the
        # source, target, and the sampled local edge index. As the electric field
        # is not needed here, we set the distances to the source and target to 1.
        # It also stores the flags indicating the enabled interactions.
        si.dp_du = mi.Vector3f(1.,
                               1.,
                               dr.reinterpret_array(mi.Float, loc_en_inter))

        # Samples the radio material
        sample, _ = si.bsdf().sample(ctx, si, sample1, sample2, active)

        # Update the probability of the event to be sampled
        sample.pdf *= probs

        return sample, normal_world, wedges

    def _sample_diffraction_point(
        self,
        si: mi.SurfaceInteraction3f,
        loc_en_inter: mi.UInt,
        ray_origin: mi.Point3f,
        ki_world: mi.Vector3f,
        sample2: mi.Point2f,
        edge_diffraction: bool,
        active: bool | mi.Bool
    ) -> tuple[mi.Bool, mi.Float, mi.Point3f, WedgeGeometry]:
        # pylint: disable=line-too-long
        r"""
        Samples a diffraction point on the silhouette edge of a mesh from a surface interaction

        This method identifies potential diffraction points by sampling along the silhouette
        of the mesh.

        :param si: Surface interaction
        :param loc_en_inter: Bitmask indicating which interaction types are enabled
        :param ray_origin: Origin of the ray
        :param ki_world: Direction of propagation of the incident wave in the world frame
        :param sample2: A pair of random numbers in :math:`[0,1]^2` used to project the interaction
            point on an edge and to sample the diffraction interaction type
        :param edge_diffraction: If set to `True`, then diffraction on free floating edges is computed
        :param active: Mask indicating which rays are active

        :return: A tuple containing:
            - diffraction: Boolean mask indicating if a diffraction is sampled
            - probs: Probability of selecting diffraction or other interactions
            - diff_point: 3D position of the diffraction point on an edge
            - wedges: Wedge geometry containing edge information
        """

        valid_wedge, wedges, diff_point = sample_wedge_diffraction_point(si,
                                                                         ray_origin,
                                                                         ki_world,
                                                                         sample2.x,
                                                                         edge_diffraction,
                                                                         active)

        # Only a predefined fixed ratio of valid wedges result in diffraction
        # interactions. The remaining paths with valid wedges are associated
        # with other interaction types.
        select_threshold = dr.select(loc_en_inter == InteractionType.DIFFRACTION,
                                     1.0, self.DIFFRACTION_SAMPLING_PROBABILITY)
        diffraction = valid_wedge & (sample2.y < select_threshold)

        # Probability of selecting diffraction or other interactions
        probs = dr.select(diffraction, select_threshold,
                          1. - select_threshold)
        probs = dr.select(valid_wedge, probs, 1.0)

        return diffraction, probs, diff_point, wedges
