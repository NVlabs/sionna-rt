#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""EM utilities of Sionna RT"""

import drjit as dr
import mitsuba as mi
from typing import Tuple
from scipy.constants import epsilon_0

from .misc import complex_sqrt
from .complex import cpx_mul


def complex_relative_permittivity(
    eta_r: mi.Float,
    sigma: mi.Float,
    omega: mi.Float
) -> mi.Complex2f:
    r"""
    Computes the complex relative permittivity of a material
    as defined in :eq:`eta`

    :param eta_r: Real component of the relative permittivity
    :param sigma: Conductivity [S/m]
    :param omega: Angular frequency [rad/s]
    """

    eta_i = sigma*dr.rcp(omega*epsilon_0)
    eta = mi.Complex2f(eta_r, -eta_i)
    return eta

def fresnel_reflection_coefficients_simplified(
    cos_theta: mi.Float,
    eta: mi.Complex2f
) -> Tuple[mi.Complex2f, mi.Complex2f]:
    # pylint: disable=line-too-long
    r"""
    Computes the Fresnel transverse electric and magnetic reflection
    coefficients assuming an incident wave propagating in vacuum :eq:`fresnel_vac`

    :param cos_theta: Cosine of the angle of incidence
    :param eta:  Complex-valued relative permittivity of the medium upon which the wave is incident

    :return: Transverse electric :math:`r_{\perp}` and magnetic :math:`r_{\parallel}` Fresnel reflection coefficients
    """

    sin_theta_sqr = 1. - cos_theta*cos_theta # sin^2(theta)
    a = complex_sqrt(eta - sin_theta_sqr)

    # TE coefficient
    r_te = (cos_theta - a)*dr.rcp(cos_theta + a)

    # TM coefficient
    r_tm = (eta*cos_theta - a)*dr.rcp(eta*cos_theta + a)

    return r_te, r_tm

def itu_coefficients_single_layer_slab(
    cos_theta: mi.Float,
    eta: mi.Complex2f,
    d: mi.Float,
    wavelength: mi.Float
    ) -> Tuple[mi.Complex2f, mi.Complex2f, mi.Complex2f, mi.Complex2f]:
    # pylint: disable=line-too-long
    r"""
    Computes the single-layer slab Fresnel transverse electric and
    magnetic reflection and refraction coefficients assuming the incident wave
    propagates in vacuum using recommendation ITU-R P.2040-4 :cite:p:`ITURP20404`

    More precisely, this function implements equations (43) and (44) from
    :cite:p:`ITURP20404`.

    :param cos_theta: Cosine of the angle of incidence
    :param eta: Complex-valued relative permittivity of the medium upon which the wave is incident
    :param d: Thickness of the slab [m]
    :param wavelength:  Wavelength [m]

    :return: Transverse electric reflection coefficient :math:`R_{eTE}`, transverse magnetic reflection coefficient :math:`R_{eTM}`, transverse electric refraction coefficient :math:`T_{eTE}`, and transverse magnetic refraction coefficient :math:`T_{eTM}`
    """

    # sin^2(theta)
    sin_theta_sqr = 1. - dr.square(cos_theta)

    # Compute `q` - Equation (44)
    q = dr.two_pi*d*dr.rcp(wavelength)*complex_sqrt(eta - sin_theta_sqr)

    # Simplified Fresnel coefficients - Equations (37a) and (37b)
    r_te_p, r_tm_p = fresnel_reflection_coefficients_simplified(cos_theta, eta)
    # Squared Fresnel coefficients
    r_te_p_sqr = dr.square(r_te_p)
    r_tm_p_sqr = dr.square(r_tm_p)

    # exp(-jq) and exp(-j2q)
    exp_j_q = dr.exp(mi.Complex2f(0., -1.)*q)
    exp_j_2q = dr.exp(mi.Complex2f(0., -2.)*q)

    # Denominator of Fresnel coefficient
    denom_te = 1. - r_te_p_sqr*exp_j_2q
    inv_denom_te = dr.rcp(denom_te)
    denom_tm = 1. - r_tm_p_sqr*exp_j_2q
    inv_denom_tm = dr.rcp(denom_tm)

    # Reflection coefficients - Equation (43a)
    r_te = r_te_p*(1. - exp_j_2q)*inv_denom_te
    r_tm = r_tm_p*(1. - exp_j_2q)*inv_denom_tm
    # Transmission coefficient - Equation (43b)
    t_te = (1. - r_te_p_sqr)*exp_j_q*inv_denom_te
    t_tm = (1. - r_tm_p_sqr)*exp_j_q*inv_denom_tm

    return r_te, r_tm, t_te, t_tm

def fresnel(x: mi.Float) -> mi.Complex2f:
    # pylint: disable=line-too-long
    r"""
    Computes the complex-valued Fresnel integral

    The complex-valued Fresnel integral is defined as:

    .. math::
        :label: fresnel_integral

        F_c(x) = \int_0^{\sqrt{\frac{2x}{\pi}}} \exp\left(j\frac{\pi s^2}{2}\right)ds = C(x) + jS(x)

    This function computes an approximation of this integral as described in
    Section 2.7 of :cite:p:`ITURP52615`. It has sufficient accuracy for most
    purposes. Note that we let the upper limit of the integral be
    :math:`\sqrt{2x/\pi}` instead of :math:`x`, which is different from the definition
    in :cite:p:`ITURP52615`. Thus, evaluating :math:`F_c(x)` corresponds to
    :math:`F_c(\sqrt{2x/\pi})` in the classical definition.

    :param x: Argument of the Fresnel integral

    :return: Complex-valued Fresnel integral
    """

    # Define Boersma coefficients
    a = [
        +1.595769140,
        -0.000001702,
        -6.808568854,
        -0.000576361,
        +6.920691902,
        -0.016898657,
        -3.050485660,
        -0.075752419,
        +0.850663781,
        -0.025639041,
        -0.150230960,
        +0.034404779
    ]

    b = [
        -0.000000033,
        +4.255387524,
        -0.000092810,
        -7.780020400,
        -0.009520895,
        +5.075161298,
        -0.138341947,
        -1.363729124,
        -0.403349276,
        +0.702222016,
        -0.216195929,
        +0.019547031
    ]

    c = [
        +0.000000000,
        -0.024933975,
        +0.000003936,
        +0.005770956,
        +0.000689892,
        -0.009497136,
        +0.011948809,
        -0.006748873,
        +0.000246420,
        +0.002102967,
        -0.001217930,
        +0.000233939
    ]

    d = [
        +0.199471140,
        +0.000000023,
        -0.009351341,
        +0.000023006,
        +0.004851466,
        +0.001903218,
        -0.017122914,
        +0.029064067,
        -0.027928955,
        +0.016497308,
        -0.005598515,
        +0.000838386
    ]

    # Only consider positive arguments of nu due to symmetry
    # See Eq. (10a,10b)
    x_pos = x>0
    x = dr.abs(x)

    # Eq. (8a, 8b)
    # We treat both cases simultaneously by using a single argument
    cond = x<4
    arg = dr.select(cond, x/4, 4*dr.rcp(x))

    # Unrolling the loop evaluation of a polynomial speeds up the computation
    r_part = 0
    i_part = 0
    arg_pow_n = mi.Float(1)
    r_part += dr.select(cond, a[0], c[0]) * arg_pow_n
    i_part -= dr.select(cond, b[0], d[0]) * arg_pow_n

    arg_pow_n *= arg
    r_part += dr.select(cond, a[1], c[1]) * arg_pow_n
    i_part -= dr.select(cond, b[1], d[1]) * arg_pow_n

    arg_pow_n *= arg
    r_part += dr.select(cond, a[2], c[2]) * arg_pow_n
    i_part -= dr.select(cond, b[2], d[2]) * arg_pow_n

    arg_pow_n *= arg
    r_part += dr.select(cond, a[3], c[3]) * arg_pow_n
    i_part -= dr.select(cond, b[3], d[3]) * arg_pow_n

    arg_pow_n *= arg
    r_part += dr.select(cond, a[4], c[4]) * arg_pow_n
    i_part -= dr.select(cond, b[4], d[4]) * arg_pow_n

    arg_pow_n *= arg
    r_part += dr.select(cond, a[5], c[5]) * arg_pow_n
    i_part -= dr.select(cond, b[5], d[5]) * arg_pow_n

    arg_pow_n *= arg
    r_part += dr.select(cond, a[6], c[6]) * arg_pow_n
    i_part -= dr.select(cond, b[6], d[6]) * arg_pow_n

    arg_pow_n *= arg
    r_part += dr.select(cond, a[7], c[7]) * arg_pow_n
    i_part -= dr.select(cond, b[7], d[7]) * arg_pow_n

    arg_pow_n *= arg
    r_part += dr.select(cond, a[8], c[8]) * arg_pow_n
    i_part -= dr.select(cond, b[8], d[8]) * arg_pow_n

    arg_pow_n *= arg
    r_part += dr.select(cond, a[9], c[9]) * arg_pow_n
    i_part -= dr.select(cond, b[9], d[9]) * arg_pow_n

    arg_pow_n *= arg
    r_part += dr.select(cond, a[10], c[10]) * arg_pow_n
    i_part -= dr.select(cond, b[10], d[10]) * arg_pow_n

    arg_pow_n *= arg
    r_part += dr.select(cond, a[11], c[11]) * arg_pow_n
    i_part -= dr.select(cond, b[11], d[11]) * arg_pow_n

    arg_sqrt = dr.sqrt(arg)
    r_part *= arg_sqrt
    i_part *= arg_sqrt

    sin_arg, cos_arg = dr.sincos(x)

    f_r, f_i = cpx_mul([cos_arg, sin_arg], [r_part, i_part])

    c = dr.select(cond, f_r, f_r+0.5)
    s = dr.select(cond, f_i, f_i+0.5)

    # Change sign if needed
    #  Eq. (10a,10b)
    c = dr.select(x_pos, c, -c)
    s = dr.select(x_pos, s, -s)

    return mi.Complex2f(c, s)

def f_utd(x: mi.Float) -> mi.Complex2f:
    # pylint: disable=line-too-long
    r"""Computes the UTD transition function

    The UTD transition function is defined as:

    .. math::
        F(x) = \sqrt{\frac{\pi x}{2}} e^{jx}\left(1+j-2jF_c^*(x) \right)

    where :math:`F_c^*(x)` is the complex conjugate of the Fresnel integral :eq:`fresnel_integral`.

    :param x: Argument of the UTD transition function

    :return: Real and imaginary parts of the UTD transition function

    Example
    -------
    The following code snippet produces a visualization of the magnitude and phase of
    the UTD transition function which matches that of Fig. 6 in :cite:p:`Kouyoumjian74`.

    .. code-block:: Python

        import numpy as np
        import matplotlib.pyplot as plt
        import drjit as dr
        import mitsuba as mi
        mi.set_variant("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")
        from sionna.rt.utils import f_utd, cpx_convert

        x = np.logspace(-3, 1, 1000)
        y = cpx_convert(f_utd(mi.Float(x)), "numpy")
        fig, ax1 = plt.subplots(figsize=(10, 6.5))

        # Plot magnitude with label
        mag_line, = ax1.semilogx(x, np.abs(y), "k-", label="Magnitude")
        ax1.set_ylabel("Magnitude")

        # Create second y-axis
        ax2 = plt.twinx()

        # Plot phase with label
        phase_line, = ax2.semilogx(x, np.angle(y, deg=True), "r--", label="Phase")
        ax2.set_ylabel("Phase (deg)")

        # Combine lines from both axes for the legend
        lines = [mag_line, phase_line]
        labels = [line.get_label() for line in lines]

        # Add legend with both lines
        ax1.legend(lines, labels, loc='upper center', frameon=True, ncol=2)

        # Set title and x label
        plt.title(r"UTD Transition Function $F(x)$")
        ax1.set_xlabel("x")

        # Adjust limits and ticks
        ax1.set_xlim(x.min(), x.max())
        ax1.set_ylim(np.abs(y).min(), np.abs(y).max())
        ax2.set_ylim(np.angle(y, deg=True).min(), np.angle(y, deg=True).max())
        ax1.set_yticks(np.linspace(0, 1, 6))
        ax2.set_yticks(np.linspace(0, 50, 11))

        plt.show()

    .. figure:: ../figures/f_utd.png

    """
    f = mi.Complex2f(1,1)
    f -= mi.Complex2f(0,2)*dr.conj(fresnel(x))
    f *= dr.sqrt(dr.pi*x/2)
    f *= dr.exp(mi.Complex2f(0,x))

    return f
