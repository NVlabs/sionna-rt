#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Classes and functions related to antenna patterns"""

from abc import ABC
from typing import Callable, List, Tuple
import inspect
import drjit as dr
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from .utils import theta_phi_from_unit_vec, isclose, to_world_jones_rotator,\
    jones_matrix_rotator_flip_forward, jones_matrix_rotator, theta_hat,\
    implicit_basis_vector
from .registry import Registry

###################################################
# Polarizations
###################################################

# A polarization is defined by either one or two
# slant angles that are applied to the antenna
# pattern.

polarization_registry = Registry()

def register_polarization(name: str, slant_angles: List[float]):
    """Registers a new polarization

    A polarization is defined as a list of one or two slant angles
    that will be applied to a vertically polarized antenna pattern
    function to create the desired polarization directions.

    :param name: Name of the polarization
    :param slant_angles: List of one or two slant angles
    """
    if not isinstance(slant_angles, List):
        raise ValueError("`slant_angles` must be a list")
    elif len(slant_angles) not in [1,2]:
        raise ValueError("`slant_angles` must be a list of length one or two")
    for slant_angle in slant_angles:
        if not isinstance(slant_angle, float):
            raise ValueError("Each slant angle must be a float")
    polarization_registry.register(slant_angles, name)

register_polarization("V", [0.0])
register_polarization("H", [dr.pi/2])
register_polarization("VH", [0.0, dr.pi/2])
register_polarization("cross", [-dr.pi/4, dr.pi/4])

####################################################
# Polarization models
####################################################

# A polarization model use a slant angle to transform
# a vertically polarized antenna pattern into an arbitrarily
# rotated linearly polarized antenna pattern

polarization_model_registry = Registry()

def register_polarization_model(
    name:str,
    model: Callable[[mi.Complex2f, mi.Float, mi.Float, mi.Float],
                    Tuple[mi.Complex2f, mi.Complex2f]]
):
    """Registers a new polarization model

    A polarization model uses a slant angle to transform
    a vertically polarized antenna pattern into an arbitrarily
    rotated linearly polarized antenna pattern

    :param name: Name of the polarization model
    :param model: Polarization model
    """
    if not callable(model):
        raise ValueError("`model` must be a callable")
    if len(inspect.signature(model).parameters) != 4:
        raise ValueError("`model` must take four arguments")
    polarization_model_registry.register(model, name)

@polarization_model_registry.register(name="tr38901_1")
def polarization_model_tr38901_1(
    c_theta_tilde: mi.Complex2f,
    theta: mi.Float,
    phi: mi.Float,
    slant_angle: mi.Float
) -> Tuple[mi.Complex2f, mi.Complex2f]:
    # pylint: disable=line-too-long
    r"""Model-1 for polarized antennas from 3GPP TR 38.901 :cite:p:`TR38901_RT`

    Transforms a vertically polarized antenna pattern
    :math:`\tilde{C}_\theta(\theta, \varphi)`
    into a linearly polarized pattern whose direction
    is specified by a slant angle :math:`\zeta`. For example,
    :math:`\zeta=0` and :math:`\zeta=\pi/2` correspond
    to vertical and horizontal polarization, respectively,
    and :math:`\zeta=\pm \pi/4` to a pair of cross polarized
    antenna elements.

    The transformed antenna pattern is given by (7.3-3) :cite:p:`TR38901_RT`:

    .. math::
        \begin{aligned}
            \begin{bmatrix}
                C_\theta(\theta, \varphi) \\
                C_\varphi(\theta, \varphi)
            \end{bmatrix} &= \begin{bmatrix}
             \cos(\psi) \\
             \sin(\psi)
            \end{bmatrix} \tilde{C}_\theta(\theta, \varphi)\\
            \cos(\psi) &= \frac{\cos(\zeta)\sin(\theta)+\sin(\zeta)\sin(\varphi)\cos(\theta)}{\sqrt{1-\left(\cos(\zeta)\cos(\theta)-\sin(\zeta)\sin(\varphi)\sin(\theta)\right)^2}} \\
            \sin(\psi) &= \frac{\sin(\zeta)\cos(\varphi)}{\sqrt{1-\left(\cos(\zeta)\cos(\theta)-\sin(\zeta)\sin(\varphi)\sin(\theta)\right)^2}}
        \end{aligned}

    :param c_theta_tilde: Vertically polarized zenith pattern
        :math:`\tilde{C}_\theta(\theta, \varphi)`

    :param theta: Zenith angles [rad]

    :param phi: Azimuth angles [rad]

    :param slant_angle: Slant angle of the linear polarization [rad].
        A slant angle of zero means vertical polarization.

    :returns: Zenith (:math:`C_\theta`) and azimuth (:math:`C_\phi`) pattern
    """

    sin_slant, cos_slant = dr.sincos(slant_angle)
    sin_theta, cos_theta = dr.sincos(theta)
    sin_phi, cos_phi = dr.sincos(phi)

    sin_psi = sin_slant*cos_phi
    cos_psi = cos_slant*sin_theta + sin_slant*sin_phi*cos_theta
    norm = dr.sqrt(1. -
                dr.square(cos_slant*cos_theta - sin_slant*sin_phi*sin_theta))
    inv_norm = dr.select(isclose(norm, mi.Float(0.0), atol=1e-5),
                        1.0,
                        dr.rcp(norm))
    sin_psi = sin_psi * inv_norm
    cos_psi = cos_psi * inv_norm
    c_theta = cos_psi * c_theta_tilde
    c_phi = sin_psi * c_theta_tilde
    return c_theta, c_phi

#pylint: disable=unused-argument
@polarization_model_registry.register(name="tr38901_2")
def polarization_model_tr38901_2(
    c_theta_tilde: mi.Complex2f,
    theta: mi.Float,
    phi: mi.Float,
    slant_angle: mi.Float
) -> Tuple[mi.Complex2f, mi.Complex2f]:
    r"""Model-2 for polarized antennas from 3GPP TR 38.901 :cite:p:`TR38901_RT`

    Transforms a vertically polarized antenna pattern
    :math:`\tilde{C}_\theta(\theta, \varphi)`
    into a linearly polarized pattern whose direction
    is specified by a slant angle :math:`\zeta`. For example,
    :math:`\zeta=0` and :math:`\zeta=\pi/2` correspond
    to vertical and horizontal polarization, respectively,
    and :math:`\zeta=\pm \pi/4` to a pair of cross polarized
    antenna elements.

    The transformed antenna pattern is given by (7.3-4/5) :cite:p:`TR38901_RT`:

    .. math::
        \begin{aligned}
            \begin{bmatrix}
                C_\theta(\theta, \varphi) \\
                C_\varphi(\theta, \varphi)
            \end{bmatrix} &= \begin{bmatrix}
             \cos(\zeta) \\
             \sin(\zeta)
            \end{bmatrix} \tilde{C}_\theta(\theta, \varphi)
        \end{aligned}

    :param c_theta_tilde: Vertically polarized zenith pattern
        :math:`\tilde{C}_\theta(\theta, \varphi)`

    :param theta: Zenith angles [rad]

    :param phi: Azimuth angles [-pi, pi) [rad]

    :param slant_angle: Slant angle of the linear polarization [rad].
        A slant angle of zero means vertical polarization.

    :returns: Zenith (:math:`C_\theta`) and azimuth (:math:`C_\phi`) pattern
    """

    sin_slant_angle, cos_slant_angle = dr.sincos(slant_angle)
    c_theta = cos_slant_angle * c_theta_tilde
    c_phi = sin_slant_angle * c_theta_tilde
    return c_theta, c_phi


####################################################
# Vertically polarized antenna pattern functions
####################################################

# These functions are used by the above defined polarization models

# pylint: disable=unused-argument
def v_iso_pattern(
    theta: mi.Float,
    phi: mi.Float
) -> mi.Complex2f:
    r"""
    Vertically polarized isotropic antenna pattern function

    :param theta: Zenith angle [rad]
    :param phi: Azimuth angle [rad]
    """

    # Number of samples
    n = dr.shape(theta)[0]

    # Zenith pattern
    c_theta = dr.ones(mi.Float, n)

    return mi.Complex2f(c_theta, 0)

def v_dipole_pattern(
    theta: mi.Float,
    phi: mi.Float
) -> mi.Complex2f:
    r"""
    Vertically polarized short dipole antenna pattern function
    from (Eq. 4-26a) :cite:p:`Balanis97`

    :param theta: Zenith angle [rad]
    :param phi: Azimuth angle [rad]
    """
    k = dr.sqrt(1.5)
    c_theta = dr.abs(k*dr.sin(theta))

    return mi.Complex2f(c_theta, 0)

def v_hw_dipole_pattern(
    theta: mi.Float,
    phi: mi.Float
) -> mi.Complex2f:
    r"""
    Vertically polarized half-wavelength dipole antenna pattern function
    from (Eq. 4-84) :cite:p:`Balanis97`

    :param theta: Zenith angle [rad]
    :param phi: Azimuth angle [rad]
    """
    k = dr.sqrt(1.643)
    sin_theta = dr.sin(theta)
    inv_sin_theta = dr.select(isclose(sin_theta, mi.Float(0.0), atol=1e-5),
                              1.0,
                              dr.rcp(sin_theta))
    c_theta = k*dr.cos(dr.pi/2.*dr.cos(theta)) * inv_sin_theta
    c_theta = dr.abs(c_theta)

    return mi.Complex2f(c_theta, 0)

def v_tr38901_pattern(
    theta: mi.Float,
    phi: mi.Float
) -> mi.Complex2f:
    r"""
    Vertically polarized antenna pattern function from
    3GPP TR 38.901 (Table 7.3-1) :cite:p:`TR38901_RT`

    :param theta: Zenith angle [rad]
    :param phi: Azimuth angle [rad]
    """
    # Wrap phi to [-PI,PI]
    phi = phi+dr.pi
    phi -= dr.floor(phi/(2.*dr.pi))*2.*dr.pi
    phi -= dr.pi

    # Zenith pattern
    theta_3db = phi_3db = 65./180.*dr.pi
    a_max = sla_v = 30.
    g_e_max = 8.
    a_v = -dr.min([12.*((theta-dr.pi/2.)/theta_3db)**2, sla_v])
    a_h = -dr.min([12.*(phi/phi_3db)**2, a_max])
    a_db = -dr.min([-(a_v + a_h), a_max]) + g_e_max
    a = dr.power(10., a_db/10.)
    c_theta = dr.sqrt(a)

    return mi.Complex2f(c_theta, 0)


####################################################
# Antenna patterns
####################################################

class AntennaPattern(ABC):
    """Abstract class for antenna patterns

    Any instance of this class must implement the
    :attr:`~sionna.rt.AntennaPattern.patterns`
    property which returns a list of one or two antenna patterns
    for single- or dual-polarized antennas, respectively.
    """

    def __getitem__(self, index):
        return self.patterns[index]

    @property
    def patterns(self):
        # pylint: disable=line-too-long
        r"""
        List of antenna patterns for one or two polarization directions.
        The pattern of a specific polarization direction can be also accessed by
        indexing the antenna pattern instance.

        :type: :py:class:`List` [:py:class:`Callable` [[:py:class:`mitsuba.Float`, :py:class:`mitsuba.Float`], :py:class:`Tuple` [:py:class:`mitsuba.Complex2f`, :py:class:`mitsuba.Complex2f`]]]
        """
        return self._patterns

    @patterns.setter
    def patterns(self, v):
        if not isinstance(v, List):
            raise ValueError("`patterns` must be a list")
        if not 1 <= len(v) <= 2:
            raise ValueError("`patterns` must be a list of length 1 or 2")
        for p in v:
            if not callable(p):
                raise ValueError("Each pattern must be a callable")
        self._patterns = v

    def compute_gain(
        self,
        polarization_direction: int = 0,
        num_samples: int = 1000,
        verbose: bool = True
    ) -> Tuple[mi.Float, mi.Float, mi.Float]:
        # pylint: disable=line-too-long
        r"""
        Computes directivity, gain, and radiation efficiency of the antenna
        pattern of one of the polarization directions

        Given a function :math:`f:(\theta,\varphi)\mapsto (C_\theta(\theta, \varphi), C_\varphi(\theta, \varphi))`
        describing an antenna pattern :eq:`C`, this function computes the
        directivity :math:`D`, gain :math:`G`,
        and radiation efficiency :math:`\eta_\text{rad}=G/D`
        (see :eq:`G`).

        :param polarization_direction: Polarization direction (0 | 1)

        :param num_samples: Number of discretization steps
            for numerical integration

        :param verbose: If `True`, the results are pretty printed.

        :return: Directivity :math:`D`, gain :math:`G`, and radiation efficiency
            :math:`\eta_\text{rad}=G/D`

        Example
        -------
        .. code-block:: python

            from sionna.rt import PlanarArray
            array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901", polarization="V")
            d, g, eta = array.antenna_pattern.compute_gain();

        ::

            Directivity [dB]: 9.825768560205825
            Gain [dB]: 7.99998570013805
            Efficiency [%]: 65.67826867103577
        """
        pattern = self[polarization_direction]

        # Create angular meshgrid
        theta = dr.linspace(mi.Float, 0, dr.pi, num_samples, False)
        phi = dr.linspace(mi.Float, -dr.pi, dr.pi, 2*num_samples, False)
        theta_grid, phi_grid = dr.meshgrid(theta, phi, indexing="ij")

        # Compute the gain
        c_theta, c_phi = pattern(theta_grid, phi_grid)
        g = dr.abs(c_theta)**2 + dr.abs(c_phi)**2

        # Find maximum directional gain
        g_max = dr.max(g)

        # Compute radiation efficiency
        dtheta = theta[1]-theta[0]
        dphi = phi[1]-phi[0]
        eta_rad = dr.sum(g*dr.sin(theta_grid)*dtheta*dphi)/(4*dr.pi)

        # Compute directivity
        d = g_max / eta_rad

        if verbose:
            dr.print(f"Directivity [dB]: {10*np.log10(d[0]):.3}")
            dr.print(f"Gain [dB]: {10*np.log10(g_max[0]):.3}")
            dr.print(f"Efficiency [%]: {eta_rad[0]*100:.3}")

        return d, g_max, eta_rad

    def show(
        self,
        polarization_direction: int = 0,
    ) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
        # pylint: disable=line-too-long
        r"""
        Visualizes the antenna gain of the antenna pattern of one
        of the polarization directions

        This function visualizes the directional antenna gain with the help
        of three figures showing the vertical and horizontal cuts as well as a
        three-dimensional visualization.

        :param polarization_direction: Polarization direction (0 | 1)

        :returns: Vertical cut, horizontal cut, and 3D visualization of
            the antenna gain

        Example
        --------
        .. code-block::

            from sionna.rt import PlanarArray
            array = PlanarArray(num_rows=1, num_cols=1, pattern="dipole", polarization="V")
            array.antenna_pattern.show()

        .. figure:: ../figures/pattern_vertical.png
            :align: center
            :scale: 80%
        .. figure:: ../figures/pattern_horizontal.png
            :align: center
            :scale: 80%
        .. figure:: ../figures/pattern_3d.png
            :align: center
            :scale: 80%
        """

        pattern = self[polarization_direction]

        # Vertical cut
        theta = dr.linspace(mi.Float, 0, dr.pi, 1000)
        phi = dr.zeros(mi.Float, 1000)
        c_theta, c_phi = [c.numpy() for c in pattern(theta, phi)]
        g = np.abs(c_theta)**2 + np.abs(c_phi)**2
        g = np.where(g==0, 1e-12, g)
        g_db = 10*np.log10(g)
        g_db_max = np.max(g_db)
        g_db_min = np.min(g_db)
        if g_db_min==g_db_max:
            g_db_min = -30
        else:
            g_db_min = np.maximum(-60., g_db_min)
        fig_v = plt.figure()
        plt.polar(theta.numpy(), g_db)
        fig_v.axes[0].set_rmin(g_db_min)
        fig_v.axes[0].set_rmax(g_db_max+3)
        fig_v.axes[0].set_theta_zero_location("N")
        fig_v.axes[0].set_theta_direction(-1)
        plt.title(r"Vertical cut of the radiation pattern $G(\theta,0)$ ")

        # Horizontal cut
        theta = dr.pi/2*dr.ones(mi.Float, 1000)
        phi = dr.linspace(mi.Float, -dr.pi, dr.pi, 1000)
        c_theta, c_phi = [c.numpy() for c in pattern(theta, phi)]
        g = np.abs(c_theta)**2 + np.abs(c_phi)**2
        g = np.where(g==0, 1e-12, g)
        g_db = 10*np.log10(g)
        g_db_max = np.max(g_db)
        g_db_min = np.min(g_db)
        if (g_db_max - g_db_min)<0.1:
            g_db_min = -30
        else:
            g_db_min = np.maximum(-60., g_db_min)

        fig_h = plt.figure()
        plt.polar(phi.numpy(), g_db)
        fig_h.axes[0].set_rmin(g_db_min)
        fig_h.axes[0].set_rmax(g_db_max+3)
        fig_h.axes[0].set_theta_zero_location("E")
        plt.title(r"Horizontal cut of the radiation pattern $G(\pi/2,\varphi)$")

        # 3D visualization
        n = 100 # sample steps
        theta = dr.linspace(mi.Float, 0, dr.pi, n, False)
        phi = dr.linspace(mi.Float, -dr.pi, dr.pi, n, False)
        theta_grid, phi_grid = dr.meshgrid(theta, phi, indexing='ij')
        c_theta, c_phi = pattern(theta_grid, phi_grid)
        theta_grid = np.reshape(theta_grid, [n, n])
        phi_grid = np.reshape(phi_grid, [n, n])
        c_theta = np.reshape(c_theta, [n, n])
        c_phi = np.reshape(c_phi, [n, n])
        g = np.abs(c_theta)**2 + np.abs(c_phi)**2
        x = g * np.sin(theta_grid) * np.cos(phi_grid)
        y = g * np.sin(theta_grid) * np.sin(phi_grid)
        z = g * np.cos(theta_grid)
        g = np.maximum(g, 1e-5)
        g_db = 10*np.log10(g)

        def norm(x, x_max, x_min):
            """Maps input to [0,1] range"""
            x = 10**(x/10)
            x_max = 10**(x_max/10)
            x_min = 10**(x_min/10)
            if x_min==x_max:
                x = np.ones_like(x)
            else:
                x -= x_min
                x /= np.abs(x_max-x_min)
            return x

        g_db_min = np.min(g_db)
        g_db_max = np.max(g_db)

        fig_3d = plt.figure()
        ax = fig_3d.add_subplot(1,1,1, projection='3d')
        ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0,
                        antialiased=False, alpha=0.7,
                        facecolors=cm.turbo(norm(g_db, g_db_max, g_db_min)))

        sm = cm.ScalarMappable(cmap=plt.cm.turbo)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", location="right",
                            shrink=0.7, pad=0.15)
        xticks = cbar.ax.get_yticks()
        xticklabels = cbar.ax.get_yticklabels()
        xticklabels = g_db_min + xticks*(g_db_max-g_db_min)
        xticklabels = [f"{z:.2f} dB" for z in xticklabels]
        cbar.ax.set_yticks(xticks)
        cbar.ax.set_yticklabels(xticklabels)

        ax.view_init(elev=30., azim=-45)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlabel("z")
        plt.suptitle(
            r"3D visualization of the radiation pattern $G(\theta,\varphi)$")

        return fig_v, fig_h, fig_3d

antenna_pattern_registry = Registry()

def register_antenna_pattern(
    name: str,
    pattern_factory: Callable[..., AntennaPattern]):
    """Registers a new factory method for an antenna pattern

    :param name: Name of the factory method

    :param pattern_factory: A factory method returning an instance of
        :class:`~sionna.rt.AntennaPattern`
    """
    antenna_pattern_registry.register(pattern_factory, name)

class PolarizedAntennaPattern(AntennaPattern):
    """
    Transforms a
    :ref:`vertically polarized antenna pattern function <v_pattern>`
    into an arbitrary single- or dual-polarized antenna pattern
    based on a polarization and polarization model

    :param v_pattern: Vertically polarized antenna pattern function

    :param polarization: Name of registered polarization
        :list-registry:`sionna.rt.antenna_pattern.polarization_registry`

    :param polarization_model: Name of registered polarization model
        :list-registry:`sionna.rt.antenna_pattern.polarization_model_registry`
    """
    def __init__(self,
                 *,
                 v_pattern: Callable[[mi.Float, mi.Float], mi.Complex2f],
                 polarization: str,
                 polarization_model: str = "tr38901_2"):
        super().__init__()
        self.v_pattern = v_pattern
        apply_pol = polarization_model_registry.get(polarization_model)
        patterns = []
        # Apply polarization model to the pattern for all slant angles
        slant_angles = polarization_registry.get(polarization)
        for slant_angle in slant_angles:
            def make_closure(slant_angle):
                def f(theta, phi):
                    return apply_pol(v_pattern(theta, phi),
                                     theta, phi, slant_angle)
                return f
            patterns.append(make_closure(slant_angle))
        self.patterns = patterns

# Register all available antenna patterns
def create_factory(name: str) -> Callable[[str, str], PolarizedAntennaPattern]:
    r"""Create a factory method for the instantiation of polarized antenna
    patterns

    Note that there must be a vertical antenna pattern function with name
    "v_{s}_pattern" which is used.

    :param name: Name under which to register the factory method
    :returns: Callable creating an instance of PolarizedAntennaPattern
    """
    def f(*, polarization, polarization_model="tr38901_2"):
        return PolarizedAntennaPattern(
                                v_pattern=globals()["v_" + name + "_pattern"],
                                polarization=polarization,
                                polarization_model=polarization_model)
    return f

for s in ["iso", "dipole", "hw_dipole", "tr38901"]:
    register_antenna_pattern(s, create_factory(s))


####################################################
# Utilities
####################################################

def antenna_pattern_to_world_implicit(
    pattern: Callable[[mi.Float, mi.Float],
                       Tuple[mi.Complex2f, mi.Complex2f]],
    to_world: mi.Matrix3f,
    k_world: mi.Vector3f,
    direction: str
) -> mi.Vector4f:
    r"""
    Evaluates an antenna pattern for a given direction and
    returns it in the world implicit basis

    For a given direction in the world frame, this function first obtains
    the local zenith and azimuth angles
    :math:`\theta` and :math:`\phi` of the antenna. Then, the antenna pattern
    is evaluated to obtain the complex-valued zenith and azimuth patterns
    :math:`C_\theta` and :math:`C_\phi`, respectively. Both are then
    transformed into the real-valued vectors

    .. math::

        \mathbf{f}_\text{real} = \begin{bmatrix}
                                    \Re\{C_\theta(\theta,\phi)\} \\
                                    \Re\{C_\phi(\theta,\phi)\}
                                 \end{bmatrix}

        \mathbf{f}_\text{imag} = \begin{bmatrix}
                                    \Im\{C_\theta(\theta,\phi)\} \\
                                    \Im\{C_\phi(\theta,\phi)\}
                                 \end{bmatrix}.

    The final output is obtained by applying a to-world rotation
    matrix :math:`\mathbf{W}` to both vectors before they are stacked:

    .. math::

        \mathbf{v}_\text{out} = \begin{bmatrix}
                                    \mathbf{W} \mathbf{f}_\text{real}\\
                                    \mathbf{W} \mathbf{f}_\text{imag}
                                \end{bmatrix}.

    The parameter `direction` indicates the direction of propagation of the
    transverse wave with respect to the antenna, i.e., away from the antenna
    (`direction = "out"`) or towards the antenna (`direction = "in"`). If the
    wave propagates towards the antenna, then the evaluated antenna pattern
    is rotated to be represented in the world frame.

    :param pattern: Antenna pattern

    :param to_world: To-world rotation matrix

    :param k_world: Direction in which to evaluate the antenna pattern in
        the world frame

    :param direction: Direction of propagation with respect
        to the antenna  ("in" | "out")

    :return: Antenna pattern in the world implicit basis as a
        real-valued vector
    """

    to_local = to_world.T

    # Direction of propagation in the local frame
    k_local = to_local@k_world

    # Evaluate the antenna pattern

    # Zenith and azimuth angle in the local frame
    theta_local, phi_local = theta_phi_from_unit_vec(k_local)

    # Evaluate the antenna pattern in the local spherical frame
    c_theta, c_phi = pattern(theta_local, phi_local)
    f_real, f_imag = complex2real_antenna_pattern(c_theta, c_phi)

    # The antenna pattern is evaluated in the local spherical basis, whereas
    # Jones vectors are represented in the implicit basis.
    rot_to_local_implicit = jones_matrix_rotator(
        k_local,
        theta_hat(theta_local, phi_local),
        implicit_basis_vector(k_local))

    # Rotation matrix to the world implicit basis from the local implicit basis
    rot_to_world_implicit = to_world_jones_rotator(to_world, k_local)

    # If the wave propagates towards the antenna, then `k_world` points in the
    # opposite direction, and the antenna pattern is rotated to match the
    # frame in which the wave is represented.
    if direction == "in":
        flip_rotator = jones_matrix_rotator_flip_forward(k_world)
        rotator = flip_rotator@rot_to_world_implicit
    elif direction == "out":
        rotator = rot_to_world_implicit
    else:
        raise ValueError('`direction` must be either "in" or "out"')
    rotator = rotator@rot_to_local_implicit

    # Apply the rotation
    f_real = rotator@f_real
    f_imag = rotator@f_imag

    # Returns the antenna pattern vector as a real vector of dimension 4
    f = mi.Vector4f(f_real.x, f_real.y, f_imag.x, f_imag.y)
    return f

def complex2real_antenna_pattern(
    c_theta: mi.Complex2f,
    c_phi: mi.Complex2f,
) -> Tuple[mi.Vector2f, mi.Vector2f]:
    """
    Converts a complex-valued antenna pattern to
    a real-valued representation

    :param c_theta: Zenith antenna pattern

    :param c_phi: Azimuth antenna pattern

    :returns: Tuple of the real and imaginary
        parts of the zenith and azimuth antenna patterns
    """

    c_real = mi.Vector2f(c_theta.real, c_phi.real)
    c_imag = mi.Vector2f(c_theta.imag, c_phi.imag)

    return c_real, c_imag
