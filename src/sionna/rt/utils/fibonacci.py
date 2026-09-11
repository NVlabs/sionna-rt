#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Fibonacci lattice generation utilities."""
from __future__ import annotations

from numbers import Integral
from typing import Tuple

import drjit as dr
import mitsuba as mi

# IEEE754 float64 significand of (1 + sqrt(5)) / 2 (with implicit leading bit)
_PHI_MB = 0x19E3779B97F4A8


def _int_f64_sig_exp(n: int) -> tuple[int, int]:
    """
    Biased exponent and 53-bit significand (with implicit bit) of float64(n).
    """
    if n == 0:
        return 0, 0
    b = n.bit_length()
    return 1023 + b - 1, n << (52 - (b - 1))


def _shifted_div(ma: mi.UInt64, mb: mi.UInt64) -> mi.UInt64:
    r"""
    Returns ``(ma << 53) // mb`` without 128-bit overflow.
    """
    q = mi.UInt64(0)
    r = ma
    for chunk in (11, 11, 11, 11, 9):
        c = mi.UInt64(chunk)
        r = r << c
        q = (q << c) + (r // mb)
        r = r % mb
    return q


def _normalize_quotient(
    q: mi.UInt64,
    exp: mi.UInt64,
) -> tuple[mi.UInt64, mi.UInt64]:
    r"""Normalize a float64 significand quotient to ``[2^52, 2^53)``."""
    q_ge = q >= (mi.UInt64(1) << 53)
    q = dr.select(q_ge, q >> 1, q)
    exp = dr.select(q_ge, exp + 1, exp)
    q_lt = q < (mi.UInt64(1) << 52)
    q = dr.select(q_lt, q << 1, q)
    exp = dr.select(q_lt, exp - 1, exp)
    return q, exp


def _sig_exp_to_f32(
    q: mi.UInt64,
    exp: mi.UInt64,
    active: mi.Bool,
    *,
    fractional: bool,
) -> mi.Float:
    r"""Convert fp64 quotient bits to fp32, optionally taking ``frac(q)``."""
    sig = (q & mi.UInt64(0xFFFFFFFFFFFFF)) | (mi.UInt64(1) << 52)
    e = mi.Int64(exp) - 1023

    if fractional:
        shift = mi.Int64(52) - e
        mask = (mi.UInt64(1) << mi.UInt64(shift)) - mi.UInt64(1)
        sig = sig & mask

    value = mi.Float(sig) * dr.exp2(mi.Float(e - 52))
    return dr.select(active, value, mi.Float(0.0))


def _fibonacci_lattice_f32(num_points: int) -> Tuple[mi.Float, mi.Float]:
    r"""
    Metal implementation using integer fp64 division emulation without fp64.
    """
    eb_d, mb_d = _int_f64_sig_exp(num_points - 1)
    mb_d = mi.UInt64(mb_d)
    eb_d = mi.UInt64(eb_d)

    ns = dr.arange(mi.UInt32, 0, num_points)
    ns64 = mi.UInt64(ns)
    active = ns > 0

    b = dr.select(active, mi.UInt64(64) - dr.lzcnt(ns64), mi.UInt64(0))
    ea = mi.UInt64(1023) + b - 1
    ma = ns64 << (mi.UInt64(53) - b)

    q, exp = _normalize_quotient(_shifted_div(ma, _PHI_MB), ea - 1)
    x = _sig_exp_to_f32(q, exp, active, fractional=True)

    q, exp = _normalize_quotient(
        _shifted_div(ma, mb_d),
        ea - eb_d + mi.UInt64(1023) - 1,
    )
    y = _sig_exp_to_f32(q, exp, active, fractional=False)
    return x, y


def fibonacci_lattice(num_points: int) -> mi.Point2f:
    r"""
    Generates a Fibonacci lattice of size ``num_points`` on the unit square
    :math:`[0, 1] \times [0, 1]`

    :param num_points: Positive integer defining the size of the lattice
    """

    if isinstance(num_points, bool) or not isinstance(num_points, Integral):
        raise TypeError("`num_points` must be an integer")
    if num_points < 1:
        raise ValueError("`num_points` must be greater than or equal to one")
    if num_points == 1:
        return mi.Point2f(0.5, 0.5)

    if dr.backend_v(mi.Float) == dr.JitBackend.Metal:
        # Float64 is not supported on Metal; emulate the fp64 reference in fp32.
        x, y = _fibonacci_lattice_f32(num_points)
    else:
        golden_ratio = (1. + dr.sqrt(mi.ScalarFloat64(5.))) / 2.
        ns = dr.arange(mi.Float64, 0, num_points)

        x = ns / golden_ratio
        x = x - dr.floor(x)
        y = ns / (num_points - 1)

    return mi.Point2f(x, y)
