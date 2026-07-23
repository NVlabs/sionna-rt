#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os

import pytest
import mitsuba as mi
import numpy as np
import scipy.special as sc
import drjit as dr
from scipy.constants import speed_of_light, epsilon_0
from scipy.spatial.transform import Rotation as scipy_rotation

from sionna.rt.utils import complex_sqrt, fresnel_reflection_coefficients_simplified,\
    complex_relative_permittivity, itu_coefficients_single_layer_slab,\
    rotation_matrix, cpx_abs, cpx_add, cpx_div, cpx_exp, cpx_mul, cpx_sqrt, cpx_sub,\
    cpx_convert, sinc, safe_atan2, transform_mesh, load_mesh, sinc, fresnel, f_utd

#############################################################
# Constants
#############################################################

# Threshold for the relative squared error above which a test fails
MAX_RSE = 1e-5

#############################################################
# Utilities
#############################################################

def ref_complex_relative_permittivity(eta_r, sigma, omega):
    """
    Evalutes the complex relative permittivity from the real component of the
    relative permittivity `eta_r`, the conductivity `sigma` (S/m) and the
    angular frequency `omega` (s^-1)
    """

    eta_i = sigma/(omega*epsilon_0)
    eta = eta_r - 1j*eta_i
    return eta

def ref_fresnel_reflection_coefficients_simplified(cos_theta, eta):
    """
    Computes the Fresnel transverse electric (TE) and transverse magnetic (TM)
    reflection coefficients assuming the medium in which the incident wave
    propagates is air. `cos_theta` is the cosine of the angle of incidence, and
    `eta` the complex relative permittivity.
    """

    # sin^2(theta)
    if dr.is_array_v(eta):
        eta = eta.numpy()
    sin_theta_sqr = 1. - np.square(cos_theta)
    a = np.sqrt(eta - sin_theta_sqr)

    # TE coefficient
    r_te = (cos_theta - a) / (cos_theta + a)

    # TM coefficient
    r_tm = (eta * cos_theta - a) / (eta * cos_theta + a)

    return r_te, r_tm

def ref_itu_coefficient_multi_layer_slab(theta0, eta, d, wavelength, fix_sign=True):
    """
    Implements the multi-layer model from ITU-R P.2040-4 for computing reflection
    and refraction coefficients
    """

    N = len(d)

    sign_fix = -1.0 if fix_sign else 1.0

    ds = np.zeros(N+2)
    ds[1:N+1] = d

    etas = np.ones(N+2, complex)
    etas[1:N+1] = eta

    sin_thetas = np.zeros(N+2, complex)
    sin_thetas[0] = sin_thetas[N+1] = np.sin(theta0)

    cos_thetas = np.zeros(N+2, complex)
    cos_thetas[0] = cos_thetas[N+1] = np.cos(theta0)

    ks = np.zeros(N+2, complex)
    ks[0] = ks[N+1] = 2.*np.pi/wavelength

    gammas = np.zeros(N+2, complex)
    gammas[0] = gammas[N+1] = ks[0]*np.sqrt(1. - np.square(sin_thetas[0]))

    r_tes = np.zeros(N+2, complex)
    r_tms = np.zeros(N+2, complex)

    R_TEs = np.zeros(N+2, complex)
    R_TMs = np.zeros(N+2, complex)

    def k_n(eta_n):
        return 2.*np.pi*np.sqrt(eta_n)/wavelength

    def sin_theta_n(sin_theta_0, eta_n):
        return sin_theta_0/np.sqrt(eta_n)

    def cos_theta_n(sin_theta_n):
        return np.sqrt(1. - np.square(sin_theta_n))

    def gamma_n(k_0, eta_n, sin_theta_0):
        return k_0*np.sqrt(eta_n - np.square(sin_theta_0))

    def r_te_n(eta_n, cos_theta_n, eta_np1, cos_theta_np1):
        sqrt_eta_n = np.sqrt(eta_n)
        sqrt_eta_np1 = np.sqrt(eta_np1)
        return (sqrt_eta_n*cos_theta_n - sqrt_eta_np1*cos_theta_np1)/(sqrt_eta_n*cos_theta_n + sqrt_eta_np1*cos_theta_np1)

    def r_tm_n(eta_n, cos_theta_n, eta_np1, cos_theta_np1):
        sqrt_eta_n = np.sqrt(eta_n)
        sqrt_eta_np1 = np.sqrt(eta_np1)
        return sign_fix*(sqrt_eta_n*cos_theta_np1 - sqrt_eta_np1*cos_theta_n)/(sqrt_eta_n*cos_theta_np1 + sqrt_eta_np1*cos_theta_n)

    def R_n(r_n, R_np1, gamma_np1, d_np1):
        num = r_n + R_np1*np.exp(-2.*1j*gamma_np1*d_np1)
        denum = 1. + r_n*R_np1*np.exp(-2.*1j*gamma_np1*d_np1)
        return num/denum

    T_TE = 1.0
    T_TM = 1.0
    n = N
    while n >= 0:

        ks[n] = k_n(etas[n])

        sin_thetas[n] = sin_theta_n(sin_thetas[0], etas[n])
        cos_thetas[n] = cos_theta_n(sin_thetas[n])

        gammas[n] = gamma_n(ks[0], etas[n], sin_thetas[0])

        r_tes[n] = r_te_n(etas[n], cos_thetas[n], etas[n+1], cos_thetas[n+1])
        r_tms[n] = r_tm_n(etas[n], cos_thetas[n], etas[n+1], cos_thetas[n+1])

        R_TEs[n] = R_n(r_tes[n], R_TEs[n+1], gammas[n+1], ds[n+1])
        R_TMs[n] = R_n(r_tms[n], R_TMs[n+1], gammas[n+1], ds[n+1])

        T_TE *= np.exp(-1j*gammas[n]*ds[n])*(1.+r_tes[n])/(1. + r_tes[n]*R_TEs[n+1]*np.exp(-2j*gammas[n+1]*ds[n+1]))
        T_TM *= np.exp(-1j*gammas[n]*ds[n])*(1.+r_tms[n])/(1. + r_tms[n]*R_TMs[n+1]*np.exp(-2j*gammas[n+1]*ds[n+1]))

        n -= 1

    R_TE = R_TEs[0]
    R_TM = R_TMs[0]

    return R_TE, R_TM, T_TE, T_TM

def itu_concrete(fc):
    """
    Computes conductivity and real component of the relative permittivity
    of the ITU concrete materials for the frequency `fc` (Hz).
    """

    fc_GHz = fc / 1e9

    # From ITU-R P.2040-4, Table 3
    a = 5.24
    b = 0.0
    c = 0.0462
    d = 0.7822

    # From ITU-R P.2040-4, Equations (28), (29)
    sigma = c*np.power(fc_GHz, d)
    eta_r = a*np.power(fc_GHz, b)

    return sigma, eta_r

def itu_metal(fc):
    """
    Computes conductivity and real component of the relative permittivity
    of the ITU metal materials for the frequency `fc` (Hz).
    """

    fc_GHz = fc / 1e9

    # From ITU-R P.2040-4, Table 3
    a = 1.0
    b = 0.0
    c = 1e7
    d = 0.0

    # From ITU-R P.2040-4, Equations (28), (29)
    sigma = c*np.power(fc_GHz, d)
    eta_r = a*np.power(fc_GHz, b)

    return eta_r, sigma

def max_rel_se(u, v):
    """
    Computes the relative max squared error (SE) between `u` and `v`.
    `u` is the reference value.
    """

    rse = np.square(np.abs(u-v)) / np.square(np.abs(u))
    return np.max(rse)

def check_inf_nan(x):
    """
    Raises an assert if `x` contains NaNs or Infs
    """
    is_not_valid = np.logical_or(np.isinf(x), np.isnan(x))
    is_valid = np.logical_not(is_not_valid)
    assert np.all(is_valid)

#############################################################
# Tests - Complex numbers
#############################################################

def test_complex_sqrt():
    """
    Test the `complex_sqrt()` utility by comparing its output against the one
    of numpy
    """

    batch_size = 100

    x = np.random.normal(size=[100,2])

    x_mi = mi.Complex2f(x[:,0], x[:,1])
    x = x[:,0] + 1j*x[:,1]

    sqrt_x_ref = np.sqrt(x)
    sqrt_x = complex_sqrt(x_mi).numpy()

    max_rse = np.square(np.abs(sqrt_x_ref - sqrt_x))/np.square(np.abs(sqrt_x_ref))
    max_rse = np.max(max_rse)
    assert max_rse < MAX_RSE

#############################################################
# Tests - Fresnel coefficients and Materials
#############################################################

def test_fresnel_integral():
    """
    Tests `rt.utils.fresnel` by comparing its output against the one
    of scipy
    """
    nu = np.linspace(-40, 40, 1000000)
    s_ref, c_ref = sc.fresnel(nu)
    f = fresnel(mi.Float(np.sign(nu)*0.5*np.pi*(nu**2)))
    s = f.imag.numpy()
    c = f.real.numpy()
    max_abs_err_s = np.max(np.abs(s_ref-s))
    max_abs_err_c = np.max(np.abs(c_ref-c))
    assert max_abs_err_s < MAX_RSE
    assert max_abs_err_c < MAX_RSE

def test_f_utd():
    """
    Tests `rt.utils.f_utd` by comparing its output against the one
    of scipy
    """
    def f(x):
        """F(x) Eq.(88) in [ITUR_P526]
        """
        s, c = sc.fresnel(np.sqrt(2*x/np.pi))
        f_c = c + 1j*s
        return np.sqrt(np.pi*x/2)*np.exp(1j*x)*(1+1j-2*1j*np.conj(f_c))

    x = np.logspace(-4, 2, 100000)

    y_scipy = f(x)
    y_dr = f_utd(mi.Float(x)).numpy()
    max_abs_err = np.max(np.abs(y_scipy-y_dr))
    assert max_abs_err < MAX_RSE

def test_fresnel_reflection_coefficients_simplified():
    """
    Tests `rt.utils.fresnel_reflection_coefficients_simplified` by comparing its output
    agains a reference implementation `ref_fresnel_reflection_coefficients_simplified`
    """

    def _compute_max_rel_se(eta_r, sigma, fc):

        omega = 2.*np.pi*fc

        # Incident angles
        thetas = np.arange(0., 0.5*np.pi, 0.01) # (rad)

        # Evaluate the reference
        ref_r_tes = []
        ref_r_tms = []
        eta = ref_complex_relative_permittivity(eta_r, sigma, omega)
        cos_thetas = np.cos(thetas)
        ref_r_tes, ref_r_tms = ref_fresnel_reflection_coefficients_simplified(cos_thetas, eta)
        del cos_thetas, eta

        # Evaluate the rt utility
        r_tes = []
        r_tms = []
        eta = complex_relative_permittivity(mi.Float(eta_r), mi.Float(sigma),
                                            mi.Float(omega))
        cos_thetas = dr.cos(mi.Float(thetas))
        r_tes, r_tms = fresnel_reflection_coefficients_simplified(cos_thetas, eta)
        r_tes = r_tes.numpy()
        r_tms = r_tms.numpy()

        max_se_r_te = max_rel_se(ref_r_tes, r_tes)
        max_se_r_tm = max_rel_se(ref_r_tms, r_tms)
        return max_se_r_te, max_se_r_tm

    # Frequency
    fc = 1e9 # (Hz)

    # Concrete
    eta_r, sigma = itu_concrete(fc)
    max_se_r_te, max_se_r_tm = _compute_max_rel_se(eta_r, sigma, fc)
    assert max_se_r_te < MAX_RSE
    assert max_se_r_tm < MAX_RSE
    del eta_r, sigma, max_se_r_te, max_se_r_tm

    # Metal
    eta_r, sigma = itu_metal(fc)
    max_se_r_te, max_se_r_tm = _compute_max_rel_se(eta_r, sigma, fc)
    assert max_se_r_te < MAX_RSE
    assert max_se_r_tm < MAX_RSE

def test_itu_coefficients_single_layer_slab_numerics():
    """
    Tests 'utils.itu_fresnel_coefficients_single_slab()' for NaN and Inf
    """
    # Frequency (Hz)
    fc = 1e9
    # Angular frequency
    omega = 2.*np.pi*fc
    # Wavelength (m)
    wavelength = speed_of_light/fc

    # Evaluated angles of incidence (rad)
    thetas = np.arange(0., 0.5*np.pi, 0.01)
    cos_thetas = dr.cos(mi.Float(thetas))

    # Evaluated slab thickness
    ds = [0.0, 0.5, 1.0, 100.0]

    # Test with concrete
    eta_r, sigma = itu_concrete(fc)
    eta = ref_complex_relative_permittivity(eta_r, sigma, omega)
    eta = mi.Complex2f(mi.Float(eta.real), mi.Float(eta.imag))
    for d in ds:
        r_te, r_tm, t_te, t_tm = itu_coefficients_single_layer_slab(cos_thetas, eta, d, wavelength)
        # Check for NaNs or inf
        check_inf_nan(r_te)
        check_inf_nan(r_tm)
        check_inf_nan(t_te)
        check_inf_nan(t_tm)
    del eta_r, sigma, eta, r_te, r_tm, t_te, t_tm

    # Test with metal
    eta_r, sigma = itu_metal(fc)
    eta = ref_complex_relative_permittivity(eta_r, sigma, omega)
    eta = mi.Complex2f(mi.Float(eta.real), mi.Float(eta.imag))
    for d in ds:
        r_te, r_tm, t_te, t_tm = itu_coefficients_single_layer_slab(cos_thetas, eta, d, wavelength)
        # Check for NaNs or inf
        check_inf_nan(r_te)
        check_inf_nan(r_tm)
        check_inf_nan(t_te)
        check_inf_nan(t_tm)

def test_itu_coefficients_single_layer_slab_large_small_thickness():
    """
    Tests 'utils.itu_fresnel_coefficients_single_slab()' for large values of
    the material thickness and a thickness of 0
    """
    # Frequency (Hz)
    fc = 1e9
    # Angular frequency
    omega = 2.*np.pi*fc
    # Wavelength (m)
    wavelength = speed_of_light/fc

    # Evaluated angles of incidence (rad)
    thetas = np.arange(0., 0.5*np.pi, 0.01)
    cos_thetas = dr.cos(mi.Float(thetas))

    eta_r, sigma = itu_concrete(fc)
    eta = ref_complex_relative_permittivity(eta_r, sigma, omega)
    eta = mi.Complex2f(mi.Float(eta.real), mi.Float(eta.imag))

    ##
    # Test for a thickness of 0
    r_te, r_tm, t_te, t_tm = itu_coefficients_single_layer_slab(cos_thetas, eta, 0.0, wavelength)
    r_te = r_te.numpy()
    r_tm = r_tm.numpy()
    t_te = t_te.numpy()
    t_tm = t_tm.numpy()
    # All the energy should be refracted
    assert np.all(np.isclose(np.abs(r_te), 0.))
    assert np.all(np.isclose(np.abs(r_tm), 0.))
    assert np.all(np.isclose(np.abs(t_te), 1.))
    assert np.all(np.isclose(np.abs(t_tm), 1.))

    ##
    # Test for a thickness of 100m
    r_te, r_tm, t_te, t_tm = itu_coefficients_single_layer_slab(cos_thetas, eta, 100.0, wavelength)
    r_te = r_te.numpy()
    r_tm = r_tm.numpy()
    t_te = t_te.numpy()
    t_tm = t_tm.numpy()
    # The reflection coefficient should match the "textbook" Fresnel ones
    ref_r_te, ref_r_tm = ref_fresnel_reflection_coefficients_simplified(np.cos(thetas), eta)
    max_se_rte = max_rel_se(ref_r_te, r_te)
    assert max_se_rte < MAX_RSE
    max_se_rtm = max_rel_se(ref_r_tm, r_tm)
    assert max_se_rtm < MAX_RSE
    # Close-to-zero energy should be refracted
    assert np.all(np.isclose(np.abs(t_te), 0.))
    assert np.all(np.isclose(np.abs(t_tm), 0.))

def test_itu_coefficients_single_layer_slab_multi_layer():
    """
    Tests that the output of `utils.itu_coefficients_single_layer_slab()` matches
    the ones from the multi-layer model when considering a single layer
    """

    np.random.seed(42)
    batch_size = 1000

    # Frequency (Hz)
    fc = 1e9
    # Angular frequency
    omega = 2.*np.pi*fc
    # Wavelength (m)
    wavelength = speed_of_light/fc

    eta_r, sigma = itu_concrete(fc)
    eta = ref_complex_relative_permittivity(eta_r, sigma, omega)
    eta_mi = mi.Complex2f(mi.Float(eta.real), mi.Float(eta.imag))

    # Sample random angles of incidence (rad)
    thetas = np.random.uniform(low=0.05, high=0.5*np.pi, size=[batch_size])
    cos_thetas_mi = dr.cos(mi.Float(thetas))

    # Sample random thicknesses
    ds = np.random.uniform(low=0.01, high=0.3, size=[batch_size])
    ds_mi = mi.Float(ds)

    r_te, r_tm, t_te, t_tm = itu_coefficients_single_layer_slab(cos_thetas_mi, eta_mi, ds_mi, wavelength)
    r_te = r_te.numpy()
    r_tm = r_tm.numpy()
    t_te = t_te.numpy()
    t_tm = t_tm.numpy()

    # Reference value from the multi-layer model
    ref_r_te = []
    ref_r_tm = []
    ref_t_te = []
    ref_t_tm = []
    for theta, d in zip(thetas, ds):
        r_te_, r_tm_, t_te_, t_tm_ = ref_itu_coefficient_multi_layer_slab(theta, eta, [d], wavelength)
        ref_r_te.append(r_te_)
        ref_r_tm.append(r_tm_)
        ref_t_te.append(t_te_)
        ref_t_tm.append(t_tm_)
    ref_r_te = np.array(ref_r_te)
    ref_r_tm = np.array(ref_r_tm)
    ref_t_te = np.array(ref_t_te)
    ref_t_tm = np.array(ref_t_tm)

    max_se_rte = max_rel_se(ref_r_te, r_te)
    assert max_se_rte < MAX_RSE
    max_se_rtm = max_rel_se(ref_r_tm, r_tm)
    assert max_se_rtm < MAX_RSE
    max_se_tte = max_rel_se(ref_t_te, t_te)
    assert max_se_tte < MAX_RSE
    max_se_ttm = max_rel_se(ref_t_tm, t_tm)
    assert max_se_ttm < MAX_RSE

###########################################################
# Test: Geometry
###########################################################

def test_rotation_matrix():

    batch_size = 100
    np.random.seed(42)

    angles_x = np.random.uniform(low=0.0, high=2.*np.pi, size=[batch_size])
    angles_y = np.random.uniform(low=0.0, high=2.0*np.pi, size=[batch_size])
    angles_z = np.random.uniform(low=0.0, high=2.*np.pi, size=[batch_size])
    angles = np.stack([angles_x, angles_y, angles_z], axis=-1)

    angles_mi = mi.Point3f(angles[:,2], angles[:,1], angles[:,0])
    rotation = rotation_matrix(angles_mi)
    rotation = np.transpose(rotation.numpy(), [2, 0, 1])

    ref_rotations = scipy_rotation.from_euler('xyz', angles)
    ref_rotation_matrices = ref_rotations.as_matrix()

    assert max_rel_se(ref_rotation_matrices, rotation) < MAX_RSE


#############################################################
# Tests - Complex-valued tensors
#############################################################

def test_cpx_abs():
    """
    Test the `cpx_abs()` utility by comparing its output against the one
    of numpy
    """
    dims = [100, 20, 2]
    x = np.random.normal(size=dims)
    x = x[...,0] + 1j*x[...,1]
    abs_x_ref = np.abs(x)
    abs_x = cpx_abs((mi.TensorXf(x.real), mi.TensorXf(x.imag)))
    max_rse = np.square(np.abs(abs_x_ref-abs_x))/np.square(np.abs(abs_x_ref))
    max_rse = np.max(max_rse)
    assert max_rse < MAX_RSE

def test_cpx_add():
    """
    Test the `cpx_add()` utility by comparing its output against the one
    of numpy
    """
    dims = [100, 20, 2]
    a = np.random.normal(size=dims)
    a = a[...,0] + 1j*a[...,1]
    b = np.random.normal(size=dims)
    b = b[...,0] + 1j*b[...,1]
    sum_ref = a+b
    sum_cpx = cpx_add((mi.TensorXf(a.real), mi.TensorXf(a.imag)),
                      (mi.TensorXf(b.real), mi.TensorXf(b.imag)))
    sum_cpx = sum_cpx[0].numpy() + 1j*sum_cpx[1].numpy()
    max_rse = np.square(np.abs(sum_ref-sum_cpx))/np.square(np.abs(sum_ref))
    max_rse = np.max(max_rse)
    assert max_rse < MAX_RSE

def test_cpx_div():
    """
    Test the `cpx_div()` utility by comparing its output against the one
    of numpy
    """
    dims = [100, 20, 2]
    a = np.random.normal(size=dims)
    a = a[...,0] + 1j*a[...,1]
    b = np.random.normal(size=dims)
    b = b[...,0] + 1j*b[...,1]
    div_ref = a/b
    div = cpx_div((mi.TensorXf(a.real), mi.TensorXf(a.imag)),
                  (mi.TensorXf(b.real), mi.TensorXf(b.imag)))
    div = div[0].numpy() + 1j*div[1].numpy()
    max_rse = np.square(np.abs(div_ref-div))/np.square(np.abs(div_ref))
    max_rse = np.max(max_rse)
    assert max_rse < MAX_RSE

def test_cpx_exp():
    """
    Test the `cpx_exp()` utility by comparing its output against the one
    of numpy
    """
    dims = [100, 20, 2]
    x = np.random.normal(size=dims)
    x = x[...,0] + 1j*x[...,1]
    exp_x_ref = np.exp(x)
    exp_x = cpx_exp((mi.TensorXf(x.real), mi.TensorXf(x.imag)))
    exp_x = exp_x[0].numpy() + 1j*exp_x[1].numpy()
    max_rse = np.square(np.abs(exp_x_ref-exp_x))/np.square(np.abs(exp_x_ref))
    max_rse = np.max(max_rse)
    assert max_rse < MAX_RSE

def test_cpx_mul():
    """
    Test the `cpx_mul()` utility by comparing its output against the one
    of numpy
    """
    dims = [100, 20, 2]
    a = np.random.normal(size=dims)
    a = a[:,0] + 1j*a[:,1]
    b = np.random.normal(size=dims)
    b = b[:,0] + 1j*b[:,1]
    mul_ref = a*b
    mul = cpx_mul((mi.TensorXf(a.real), mi.TensorXf(a.imag)),
                  (mi.TensorXf(b.real), mi.TensorXf(b.imag)))
    mul = mul[0].numpy() + 1j*mul[1].numpy()
    max_rse = np.square(np.abs(mul_ref-mul))/np.square(np.abs(mul_ref))
    max_rse = np.max(max_rse)
    assert max_rse < MAX_RSE

def test_cpx_sqrt():
    """
    Test the `cpx_sqrt()` utility by comparing its output against the one
    of numpy
    """
    dims = [100, 20, 2]
    x = np.random.normal(size=dims)
    x = x[...,0] + 1j*x[...,1]
    sqrt_x_ref = np.sqrt(x)
    sqrt_x = cpx_sqrt((mi.TensorXf(x.real), mi.TensorXf(x.imag)))
    sqrt_x = sqrt_x[0].numpy() + 1j*sqrt_x[1].numpy()
    max_rse = np.square(np.abs(sqrt_x_ref-sqrt_x))/np.square(np.abs(sqrt_x_ref))
    max_rse = np.max(max_rse)
    assert max_rse < MAX_RSE

def test_cpx_sub():
    """
    Test the `cpx_sub()` utility by comparing its output against the one
    of numpy
    """
    dims = [100, 20, 2]
    a = np.random.normal(size=dims)
    a = a[...,0] + 1j*a[...,1]
    b = np.random.normal(size=dims)
    b = b[...,0] + 1j*b[...,1]
    sub_ref = a-b
    sub_cpx = cpx_sub((mi.TensorXf(a.real), mi.TensorXf(a.imag)),
                      (mi.TensorXf(b.real), mi.TensorXf(b.imag)))
    sub_cpx = sub_cpx[0].numpy() + 1j*sub_cpx[1].numpy()
    max_rse = np.square(np.abs(sub_ref-sub_cpx))/np.square(np.abs(sub_ref))
    max_rse = np.max(max_rse)
    assert max_rse < MAX_RSE

@pytest.mark.parametrize('out_type', ["numpy", "jax", "tf", "torch", "raise_error"])
def test_cpx_convert(out_type):
    """
    Test the cpx_convert() utility by checking for correct output types
    """
    dims = [100, 20, 2]
    a = np.random.normal(size=dims)
    a = a[...,0] + 1j*a[...,1]
    try:
        a_conv = cpx_convert((mi.TensorXf(a.real), mi.TensorXf(a.imag)),
                             out_type=out_type)
        if out_type == "numpy":
            assert a_conv.dtype == np.complex64
        elif out_type == "tf":
            import tensorflow as tf
            assert a_conv.dtype == tf.complex64
        elif out_type == "torch":
            import torch
            assert a_conv.dtype == torch.complex64
        elif out_type == "jax":
            import jax.numpy as jnp
            assert a_conv.dtype == jnp.complex64
    except ImportError:
        pytest.skip(f"skipped because {out_type} not installed")
    except ValueError as e:
        if out_type!="raise_error":
            raise ValueError from e

#############################################################
# Tests - Miscellaneous
#############################################################

def test_sinc():
    """Test sinc function against numpy"""
    x_low = np.linspace(-10*dr.pi, 0, 1000)
    x_up = np.linspace(0, 10*dr.pi, 1000)
    x = np.concatenate([x_low, x_up], axis=0)
    y = sinc(mi.Float64(x))
    y_np = np.sinc(x)
    assert np.max(np.abs(y-y_np)/np.abs(y_np)) < 1e-6

def test_sinc_gradient():
    """Test gradient at x=0"""
    x = mi.Float([0, 0.1])
    dr.enable_grad(x)
    y = sinc(x)
    dr.backward(y)
    assert x.grad[0] == 0

def test_safe_atan2():
    """Test safe_atan2 function against numpy"""
    x = dr.linspace(mi.Float, 0, 1, 10)
    y = dr.linspace(mi.Float, 0, 1, 10)
    x[-1] = 0
    y[-3] = 0
    for attach_y in (False, True):
        dr.set_grad_enabled(y, attach_y)
        for attach_x in (False, True):
            dr.set_grad_enabled(x, attach_x)
            res_safe = safe_atan2(y, x)
            with dr.suspend_grad():
                res_expected = dr.atan2(y, x)
            assert dr.allclose(res_safe, res_expected)

            if attach_y or attach_x:
                dr.backward(res_safe)
                assert not dr.any(dr.isnan(x.grad))
                assert not dr.any(dr.isnan(y.grad))

                denom = x * x + y * y
                both_zero = (x == 0) & (y == 0)
                grad_y_expected = (x / denom) & (~both_zero)
                grad_x_expected = (-y / denom) & (~both_zero)
                assert dr.allclose(y.grad, grad_y_expected if attach_y else 0)
                assert dr.allclose(x.grad, grad_x_expected if attach_x else 0)

            dr.set_grad(x, 0)
            dr.set_grad(y, 0)

    # Literal case (make sure the result is still attached)
    x = mi.Float(0)
    y = mi.Float(0)
    dr.set_grad_enabled(x, True)
    dr.set_grad_enabled(y, True)
    res_safe = safe_atan2(y, x)
    # Safe result is detached, this is consistent with e.g. `dr.safe_acos()`.
    assert not dr.grad_enabled(res_safe)


#############################################################
# Tests - Shapes
#############################################################

def test_transform_mesh():
    """Test mesh transformation using translation, rotation, and scaling."""

    fname = os.path.join(os.path.dirname(__file__),
                         "../data/subdivided_cube.ply")
    mesh = load_mesh(fname)

    # Read original vertices
    params = mi.traverse(mesh)
    vertices = params["vertex_positions"].numpy()
    vertices = vertices.reshape(-1, 3)

    # Edit the mesh
    t = np.array([0, 0, 0])
    rot = np.array([dr.pi*0.25, dr.pi*0.3, 0])
    scale = np.array([1, 1, 1])
    transform_mesh(mesh,
                   translation=t,
                   rotation=rot,
                   scale=scale)
    # Read the transformed vertices
    params = mi.traverse(mesh)
    vertices_transformed = params["vertex_positions"].numpy()
    vertices_transformed = vertices_transformed.reshape(-1, 3)

    # Apply transformation with numpy/scipy
    t = t[None,:]
    scale = scale[None,:]

    # Center the mesh
    ref_vertices = vertices.copy()
    c = 0.5*(np.max(ref_vertices, axis=0, keepdims=True)
             + np.min(ref_vertices, axis=0, keepdims=True))
    ref_vertices -= c

    # Scale the mesh
    ref_vertices *= scale

    # Rotate the mesh
    rot_matrix = scipy_rotation.from_euler('xyz', [rot[2], rot[1], rot[0]]).as_matrix()
    rot_matrix = rot_matrix[None,...]
    ref_vertices = ref_vertices[...,None]
    ref_vertices = rot_matrix[None,...]@ref_vertices
    ref_vertices = ref_vertices.squeeze()

    # Translate the mesh
    ref_vertices += t + c

    max_rel_ser = np.max(np.abs(vertices_transformed - ref_vertices))
    assert np.isclose(max_rel_ser, 0, atol=1e-6)
