#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Custom Mitsuba integrator to render scenes with a cut view (slice plane)."""

from __future__ import annotations

import drjit as dr
import mitsuba as mi

from mitsuba.python.ad.integrators.common import RBIntegrator, mis_weight


class SlicedPathIntegrator(RBIntegrator):
    """
    Path tracer which skips interactions with shapes encountered before a
    specified cut plane.
    """
    def __init__(self, props):
        super().__init__(props)

        # Slice planes should be specified as nested `rectangle` shapes.
        self.slice_planes: list[mi.Shape] = []
        for k in props.unqueried():
            v = props[k]
            if not isinstance(v, mi.Shape):
                raise ValueError(f"SlicedPathIntegrator: unexpected parameter "
                                 f"\"{k}\" of type: {type(v)}")
            elif v.shape_type() != mi.ShapeType.Rectangle:
                raise ValueError(
                    f"SlicedPathIntegrator: all nested shapes must be analytic"
                    f" `rectangle` plugins, but found \"{k}\" of shape type:"
                    f" {mi.ShapeType(v.shape_type())}"
                )

            self.slice_planes.append(v)


    def as_depth_integrator(self) -> SlicedPathIntegrator:
        """
        Creates another instance of this integrator, with the same slice planes,
        but set up to render depth instead of radiance.
        """
        d = {
            "type": "sliced_depth",
            "max_depth": 1,
        }
        for rect in self.slice_planes:
            d[rect.id()] = rect

        return mi.load_dict(d)


    @dr.syntax
    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: mi.Spectrum | None, # pylint: disable=non-ascii-name,invalid-name
               state_in: mi.Spectrum | None,
               active: mi.Bool,
               **_ # Absorbs unused arguments
    ) -> tuple[mi.Spectrum, mi.Bool, list[mi.Float], mi.Spectrum]:
        """Adapted from `mitsuba.python.ad.integrators.prb.PRBIntegrator`."""

        diff_radiance = δL

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal
        if not primal:
            raise NotImplementedError(
                "Not implemented: SlicedPathIntegrator non-primal modes."
            )

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        # Depth of current vertex
        depth = mi.UInt32(0)
        # Radiance accumulator
        radiance = mi.Spectrum(0 if primal else state_in)
        # Differential/adjoint radiance
        diff_radiance = mi.Spectrum(diff_radiance
                                    if (diff_radiance is not None) else 0)
        # Path throughput weight
        throughput = mi.Spectrum(1)
        # Index of refraction
        eta = mi.Float(1)
        # Active SIMD lanes
        active = mi.Bool(active)

        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        while dr.hint(active,
                      max_iterations=self.max_depth,
                      label=f"Path Replay Backpropagation ({mode.name})"):
            active_next = mi.Bool(active)

            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.
            si = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.AllNonDifferentiable,
                coherent=(depth == 0), active=active
            )
            self.advance_past_slice_planes(scene, si, ray, active)

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            # Hide the environment emitter if necessary
            if dr.hint(self.hide_emitters, mode='scalar'):
                active_next &= ~((depth == 0) & ~si.is_valid())

            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            mis = mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )

            emitted = throughput * mis * ds.emitter.eval(si, active_next)

            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next &= (depth + 1 < self.max_depth) & si.is_valid()

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(),
                                                  mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), test_visibility=False, active=active_em)
            # Custom visibility test, taking into account the slice planes
            occluded = self.ray_test_with_slice_planes(
                scene, si.spawn_ray_to(ds.p), active=active_em
            )
            ds.pdf[occluded] = 0.0
            # em_weight += mi.Spectrum(0.2, 0, 0)
            active_em &= (ds.pdf != 0.0)

            # Evaluate BSDF * cos(theta)
            wo = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo,
                                                       active_em)
            mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
            reflected = throughput * mis_em * bsdf_value_em * em_weight

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)

            # ---- Update loop variables based on current interaction -----

            radiance = radiance + emitted + reflected
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            eta *= bsdf_sample.eta
            throughput *= bsdf_weight

            # Information about the current vertex needed by the next iteration

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type,
                                          mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            throughput_max = dr.max(throughput)
            active_next &= (throughput_max != 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(throughput_max * eta**2, .95)

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            throughput[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            depth[si.is_valid()] += 1
            active = active_next

        return (
            # Radiance/differential radiance
            radiance if primal else diff_radiance,
            # Ray validity flag for alpha blending
            (depth != 0),
            # Empty tuple of AOVs
            [],
            # State for the differential phase
            radiance
        )


    def advance_past_slice_planes(
        self, scene: mi.Scene, si: mi.SurfaceInteraction3f, ray: mi.Ray3f,
        active: mi.Mask, return_depth: bool = False
    ) -> None:
        if len(self.slice_planes) > 1:
            raise NotImplementedError(
                "Untested w/ multiple slice planes")

        total_si_t = None
        if return_depth:
            total_si_t = mi.Float(si.t)

        for rect in self.slice_planes:
            si_rect = rect.ray_intersect(ray, ray_flags=mi.RayFlags.Minimal,
                                         active=active)

            # If we hit the slice plane "from above", advance ray to the other
            # side of the slice plane and intersect the scene again.
            hit_rect = active & si_rect.is_valid()
            from_above = dr.dot(ray.d, si_rect.n) < 0
            advance = hit_rect & from_above
            ray[advance] = si_rect.spawn_ray(ray.d)
            si[advance] = scene.ray_intersect(
                ray, ray_flags=int(mi.RayFlags.AllNonDifferentiable),
                coherent=False, active=advance
            )

            # If we hit the slice plane "from below", and we didn't hit the real
            # scene before that, then escape to infinity.
            escaped = hit_rect & (~from_above) & (si_rect.t < si.t)
            si.t[escaped] = dr.inf

            if return_depth:
                # If we advanced the ray (= changed its origin), the updated
                # value of `si.t` is now missing the distance to the slice plane
                total_si_t[advance | escaped] = si_rect.t + si.t

        return total_si_t

    def ray_test_with_slice_planes(self, scene: mi.Scene, shadow_ray: mi.Ray3f,
                                   active: mi.Mask) -> mi.Mask:
        # If we can hit either the target distance (`shadow_ray.maxt`) or the
        # slice plane before hitting the scene, then we're not occluded.
        for rect in self.slice_planes:
            si_rect = rect.ray_intersect(
                shadow_ray, ray_flags=mi.RayFlags.Minimal, active=active
            )
            shadow_ray.maxt = dr.minimum(shadow_ray.maxt, si_rect.t)

        return scene.ray_test(shadow_ray, coherent=False, active=active)


    def to_string(self):
        return f"""SlicedPathIntegrator[
    slice_planes = [{self.slice_planes}],
    max_depth = [{self.max_depth}]
]"""


class SlicedDepthIntegrator(SlicedPathIntegrator):
    """
    Depth integrator which skips interactions with shapes encountered before a
    specified cut plane.
    """
    def __init__(self, props):
        super().__init__(props)

        # Re-set max_depth since we need a different default value.
        self.max_depth = props.get("max_depth", 1)
        if self.max_depth != 1:
            raise ValueError("Depth integrator only support max_depth = 1,"
                             f" but found max_depth = {self.max_depth}.")

    @dr.syntax
    def sample(self,
        mode: dr.ADMode,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        δL: mi.Spectrum | None, # pylint: disable=non-ascii-name
        state_in: mi.Spectrum | None,
        active: mi.Bool,
        **_ # Absorbs unused arguments
    ) -> tuple[mi.Spectrum, mi.Bool, list[mi.Float], mi.Spectrum]:

        if mode != dr.ADMode.Primal:
            raise NotImplementedError("Not implemented: SlicedDepthIntegrator"
                                      " non-primal modes.")

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        si = dr.zeros(mi.SurfaceInteraction3f)
        active = mi.Bool(active)
        n_interactions = mi.UInt32(0)
        max_interactions = mi.UInt32(25)  # For safety

        # Keep tracing until we reach the first real interaction or escape
        acc_depth = mi.Float(0)
        while dr.hint(active, label="Depth tracing with slice planes"):

            si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.Minimal,
                                     coherent=True, active=active)
            depth = self.advance_past_slice_planes(scene, si, ray, active,
                                                   return_depth=True)
            acc_depth += depth

            active &= False

            n_interactions += 1
            active &= (n_interactions < max_interactions)

        return (acc_depth & si.is_valid()), si.is_valid(), [], None


    def to_string(self):
        return f"""SlicedDepthIntegrator[
    slice_planes = [{self.slice_planes}],
]"""



# Register this custom Integrator plugin
mi.register_integrator("sliced_path",
                       lambda props: SlicedPathIntegrator(props=props))
mi.register_integrator("sliced_depth",
                       lambda props: SlicedDepthIntegrator(props=props))
