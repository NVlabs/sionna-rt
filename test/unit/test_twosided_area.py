#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for TwosidedAreaEmitter sample-domain mapping and eval/PDF."""

import numpy as np
import mitsuba as mi
import pytest

import sionna.rt  # registers twosided_area


def _scene_with_emitter(emitter_dict):
    """Build a scene and keep it alive (nested area emitters hold raw shape ptrs)."""
    scene = mi.load_dict({
        "type": "scene",
        "rect": {
            "type": "rectangle",
            "emitter": emitter_dict,
        },
    })
    shape = list(scene.shapes())[0]
    return scene, shape, shape.emitter()


def _twosided_dict():
    return {
        "type": "twosided_area",
        "nested": {
            "type": "area",
            "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
        },
    }


def _onesided_dict():
    return {
        "type": "area",
        "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
    }


def _ray_direction(emitter, sample1):
    ray, weight = emitter.sample_ray(
        mi.Float(0.0),
        mi.Float(sample1),
        mi.Point2f(0.5, 0.5),
        mi.Point2f(0.5, 0.5),
        mi.Bool(True),
    )
    d = np.array(ray.d).reshape(-1)[:3]
    w = float(np.array(weight).ravel()[0])
    return d, w


def test_twosided_area_rejects_missing_nested():
    with pytest.raises((ValueError, RuntimeError), match="name=\"nested\""):
        mi.load_dict({"type": "twosided_area"})


def test_sample_ray_side_selection_and_domain_remap():
    """sample1 ≤ 0.5 keeps the nested direction; sample1 > 0.5 flips it.

    Both 0.25 and 0.75 remap onto nested sample1 = 0.5, so the absolute
    directions must match a one-sided sample at 0.5 (up to the flip).
    """
    # Keep both scenes alive for the duration of the test: nested area
    # emitters store a raw shape pointer (Mitsuba lifetime caveat).
    scene_two, _, two = _scene_with_emitter(_twosided_dict())
    scene_one, _, one = _scene_with_emitter(_onesided_dict())

    d_front, w_front = _ray_direction(two, 0.25)
    d_back, w_back = _ray_direction(two, 0.75)
    d_one, w_one = _ray_direction(one, 0.5)

    assert np.allclose(d_front, d_one, atol=1e-5)
    assert np.allclose(d_back, -d_one, atol=1e-5)
    assert np.allclose(d_front, -d_back, atol=1e-5)

    # Nested weight assumes a one-sided PDF; the 50/50 side choice is
    # compensated by a factor of two so that weight * (0.5 * pdf_nested)
    # recovers the emitted radiance.
    assert np.isclose(w_front, 2.0 * w_one, rtol=1e-5)
    assert np.isclose(w_back, 2.0 * w_one, rtol=1e-5)

    # Retain references until assertions complete.
    assert scene_two is not None and scene_one is not None


def test_eval_matches_on_both_sides():
    """eval returns the nested radiance for both hemispheres."""
    scene_two, shape, em = _scene_with_emitter(_twosided_dict())
    scene_one, shape_one, one = _scene_with_emitter(_onesided_dict())

    si = shape.eval_parameterization(mi.Point2f(0.5, 0.5))
    si.wi = mi.Vector3f(0.0, 0.0, 1.0)
    l_front = float(np.array(em.eval(si, True)).ravel()[0])
    si.wi = mi.Vector3f(0.0, 0.0, -1.0)
    l_back = float(np.array(em.eval(si, True)).ravel()[0])

    si_one = shape_one.eval_parameterization(mi.Point2f(0.5, 0.5))
    si_one.wi = mi.Vector3f(0.0, 0.0, 1.0)
    l_one_front = float(np.array(one.eval(si_one, True)).ravel()[0])
    si_one.wi = mi.Vector3f(0.0, 0.0, -1.0)
    l_one_back = float(np.array(one.eval(si_one, True)).ravel()[0])

    assert np.isclose(l_front, l_one_front, rtol=1e-5)
    assert np.isclose(l_back, l_one_front, rtol=1e-5)
    assert np.isclose(l_one_back, 0.0, atol=1e-6)
    assert scene_two is not None and scene_one is not None


def test_pdf_direction_matches_nested_after_side_flip():
    """pdf_direction mirrors the query onto the front side before nesting."""
    scene_two, _, em = _scene_with_emitter(_twosided_dict())
    scene_one, _, one = _scene_with_emitter(_onesided_dict())

    ref = mi.Interaction3f()
    ref.p = mi.Point3f(0.0, 0.0, 1.0)
    ref.time = mi.Float(0.0)

    ds, _ = one.sample_direction(ref, mi.Point2f(0.3, 0.7), True)
    pdf_one = float(np.asarray(one.pdf_direction(ref, ds, True)).ravel()[0])

    # Same geometric sample with the connection direction reversed — the
    # wrapper must flip it back before consulting the nested emitter.
    ds_back = mi.DirectionSample3f()
    ds_back.p = ds.p
    ds_back.n = ds.n
    ds_back.uv = ds.uv
    ds_back.time = ds.time
    ds_back.pdf = ds.pdf
    ds_back.delta = ds.delta
    ds_back.emitter = ds.emitter
    ds_back.d = -ds.d
    ds_back.dist = ds.dist

    pdf_two = float(np.asarray(em.pdf_direction(ref, ds_back, True)).ravel()[0])
    assert np.isclose(pdf_two, pdf_one, rtol=1e-4)
    assert scene_two is not None and scene_one is not None
