#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Miscellaneous utilities"""

from contextlib import contextmanager
import drjit as dr
import mitsuba as mi
from .complex import cpx_sqrt

def complex_sqrt(x: mi.Complex2f) -> mi.Complex2f:
    r"""
    Computes the square root of a complex number :math:`x`

    The following formula is implemented to compute the square roots of complex
    numbers:
    https://en.wikipedia.org/wiki/Square_root#Algebraic_formula

    :param x:  Complex number
    """
    return mi.Complex2f(*cpx_sqrt((dr.real(x), dr.imag(x))))

def isclose(
    a: mi.Float,
    b: mi.Float,
    rtol: mi.Float = 1e-5,
    atol: mi.Float = 1e-8
) -> mi.Bool:
    # pylint: disable=line-too-long
    r"""
    Returns an array of boolean in which an element is set to `True` if the
    corresponding entries in ``a`` and ``b`` are equal within a tolerance

    More precisely, this function returns `True` for the :math:`i^{th}` element
    if:

    .. math::
        |\texttt{a}[i] - \texttt{b}[i]| < \texttt{atol} + \texttt{rtol} \cdot |\texttt{b}[i]|

    Elements corresponding to infinite ``b`` or to NaN in ``a`` or ``b`` are
    always `False`.

    :param a: First input array to compare
    :param b: Second input array to compare
    :param rtol: Relative error threshold
    :param atol: Absolute error threshold
    """

    close = dr.abs(a-b) < atol + dr.abs(b)*rtol
    close &= ~dr.isinf(b)
    close &= ~(dr.isnan(a) | dr.isnan(b))
    return close

def log10(x: mi.Float) -> mi.Float:
    r"""
    Evaluates the base-10 logarithm

    :param x: Input value
    """
    return dr.log(x)/dr.log(10.)

def sinc(x: mi.Float) -> mi.Float:
    r"""
    Evaluates the normalized sinc function

    The sinc function is defined as :math:`\sin(\pi x)/(\pi x)`
    for any :math:`x \neq 0` and equals :math:`1` for :math:`x=0`.
    """
    x = dr.pi*x
    return dr.select(x==0, 1, dr.sin(x)*dr.rcp(x))

def watt_to_dbm(x: mi.Float) -> mi.Float:
    r"""
    Converts Watt to dBm

    Implements the following formula:

    .. math::
        P_{dBm} = 30 + 10 \log_{10}(P_W)

    :param x: Power [W]
    """
    return log10(x)*10. + 30.

def dbm_to_watt(x: mi.Float) -> mi.Float:
    r"""
    Converts dBm to Watt

    Implements the following formula:

    .. math::
        P_W = 10^{\frac{P_{dBm}-30}{10}}

    :param x: Power [dBm]
    """
    return dr.power(10., (x - 30.) / 10.)

def spectrum_to_matrix_4f(s: mi.Spectrum) -> mi.Matrix4f:
    r"""
    Builds a :class:`mi.Matrix4f` from a :class:`mi.Spectrum` object

    :param s: Mitsuba Spectrum object
    """
    m = mi.Matrix4f(s.array.x.x.x, s.array.x.y.x, s.array.x.z.x, s.array.x.w.x,
                    s.array.y.x.x, s.array.y.y.x, s.array.y.z.x, s.array.y.w.x,
                    s.array.z.x.x, s.array.z.y.x, s.array.z.z.x, s.array.z.w.x,
                    s.array.w.x.x, s.array.w.y.x, s.array.w.z.x, s.array.w.w.x)
    return m

@contextmanager
def scoped_set_log_level(level: mi.LogLevel):
    r"""
    Context manager for running Mitsuba with a set log level

    :param level: Log level to use within the context
    """
    logger = mi.logger()
    previous = logger.log_level()
    logger.set_log_level(level)
    try:
        yield
    finally:
        logger.set_log_level(previous)

def sigmoid(x: mi.Float) -> mi.Float:
    r"""
    Evaluates the sigmoid of ``x``

    :param x: Input value
    """
    # Clip to avoid extreme exponential values
    x = dr.clip(x, -80, 80)
    y = dr.select(x >= 0.,
                  1.*dr.rcp(1. + dr.exp(-x)),
                  dr.exp(x)*dr.rcp(1. + dr.exp(x)))
    return y

def subcarrier_frequencies(num_subcarriers: int,
                           subcarrier_spacing: float) -> mi.Float:
    # pylint: disable=line-too-long
    r"""
    Compute the baseband frequencies of ``num_subcarriers`` subcarriers spaced by
    ``subcarrier_spacing``, i.e.,

    >>> # If num_subcarriers is even:
    >>> frequencies = [-num_subcarriers/2, ..., 0, ..., num_subcarriers/2-1] * subcarrier_spacing
    >>>
    >>> # If num_subcarriers is odd:
    >>> frequencies = [-(num_subcarriers-1)/2, ..., 0, ..., (num_subcarriers-1)/2] * subcarrier_spacing

    :param num_subcarriers: Number of subcarriers

    :param subcarrier_spacing: Subcarrier spacing [Hz]

    :return: Baseband frequencies of subcarriers
    """
    if num_subcarriers % 2 == 0:
        start=int(-num_subcarriers/2)
        limit=int(num_subcarriers/2)
    else:
        start=int(-(num_subcarriers-1)/2)
        limit=int((num_subcarriers-1)/2+1)

    frequencies = dr.arange(mi.Float, start=start, stop=limit)
    frequencies *= subcarrier_spacing

    return frequencies

def map_angle_to_canonical_range(x):
    r"""
    Maps an angle to the canonical range :math:`[0, 2\pi)`

    :param x: Input angle
    """
    return x - dr.two_pi * dr.floor(x*dr.rcp(dr.two_pi))

def safe_atan2(y: mi.Float, x: mi.Float) -> mi.Float:
    r"""
    Safe implementation of atan2(y, x) that avoids NaN when both inputs
    are zero and gradients are computed

    :param y: Input 1
    :param x: Input 2
    """
    both_zero = (x == 0.0) & (y == 0.0)
    return dr.select(both_zero, 0.0, dr.atan2(y, x))

def cot(x: mi.Float) -> mi.Float:
    r"""
    Computes the cotangent of ``x``

    This is the mathematical cotangent :math:`1/\tan(x)`. Poles where
    :math:`\tan(x)=0` evaluate to IEEE infinities. For the UTD product
    :math:`\cot(\psi)F(x)`, which remains finite at those poles, use
    :func:`~sionna.rt.utils.cot_times_f_utd` instead.

    :param x: Input value
    """
    return dr.rcp(dr.tan(x))
