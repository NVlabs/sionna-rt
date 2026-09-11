#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Scattering patterns"""

from abc import abstractmethod
from numbers import Integral
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib import cm
import drjit as dr
import mitsuba as mi
import scipy
import numpy as np
from ..registry import Registry
from ..utils import r_hat, theta_phi_from_unit_vec

class ScatteringPattern:
    """
    Abstract class implementing a scattering pattern
    """
    @abstractmethod
    def __call__(self,
                 ki_local: mi.Vector3f,
                 ko_local: mi.Vector3f
    )-> mi.Float:
        r"""
        :param ki_local: Direction of propagation of the incident wave
            in local frame

        :param ko_local: Direction of propagation of the scattered wave
            in local frame

        :return: Scattering pattern value
        """
        pass

    def show(self,
             k_i_local: mi.Vector3f = (0.7071, 0., -0.7071),
             show_directions: bool = False
    )-> tuple[plt.Figure, plt.Figure]:
        # pylint: disable=line-too-long
        r"""
        Visualizes the scattering pattern

        It is assumed that the surface normal points toward the
        positive z-axis.

        :param k_i_local: Incoming direction

        :param show_directions: Show incoming and specular reflection
            directions

        :return: 3D visualization of the scattering pattern

        :return: Visualization of the incident plane cut through
            the scattering pattern
        """
        k_i_local = mi.Vector3f(k_i_local)

        ###
        ### 3D visualization
        ###
        theta = dr.linspace(mi.Float, 0, dr.pi/2, 50, False)
        phi = dr.linspace(mi.Float, -dr.pi, dr.pi, 100, False)
        theta_grid, phi_grid = dr.meshgrid(theta, phi, indexing='ij')
        k_o_local = r_hat(theta_grid, phi_grid)
        pattern = self(k_i_local, k_o_local)
        sin_phi_grid, cos_phi_grid = dr.sincos(phi_grid)
        sin_theta_grid, cos_theta_grid = dr.sincos(theta_grid)
        x = pattern * sin_theta_grid * cos_phi_grid
        y = pattern * sin_theta_grid * sin_phi_grid
        z = pattern * cos_theta_grid

        # Reshape to tensor for visualization
        pattern, x, y, z = [dr.reshape(mi.TensorXf, s, [50, 100]).numpy()
                            for s in [pattern, x, y, z]]
        p_min = np.min(pattern)
        p_max = np.max(pattern)

        def norm(x):
            """Maps input to [0,1] range"""
            x_min = np.min(x)
            x_max = np.max(x)
            if x_min==x_max:
                x = np.ones_like(x)
            else:
                x -= x_min
                x /= np.abs(x_max-x_min)
            return x

        fig_3d = plt.figure()
        ax = fig_3d.add_subplot(1,1,1, projection='3d', computed_zorder=False)

        ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0,
                            antialiased=False, alpha=0.7,
                            facecolors=cm.turbo(norm(pattern)))

        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

        if show_directions:
            r = p_max*1.1
            uvw = -k_i_local.numpy()[:,0]
            plt.quiver(*r*uvw, *-uvw, length=r, linestyle='dashed',
                       color="black", arrow_length_ratio=0.07, alpha=0.5)
            ax.text(r*uvw[0], r*uvw[1], r*uvw[2],
                    r"$\hat{\mathbf{k}}_\mathrm{i}$")

            theta_i, phi_i = theta_phi_from_unit_vec(-k_i_local)
            theta_r, phi_r = theta_i, phi_i + dr.pi
            k_r = r_hat(theta_r, phi_r).numpy()[:,0]
            plt.quiver(*[0,0,0], *k_r, length=r, linestyle='dashed',
                       color="black", arrow_length_ratio=0.07, alpha=0.5)
            ax.text(r*k_r[0], r*k_r[1], r*k_r[2],
                    r"$\hat{\mathbf{k}}_\mathrm{r}$")

        sm = cm.ScalarMappable(cmap=plt.cm.turbo)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical",
                            location="right", shrink=0.7, pad=0.15)

        xticks = cbar.ax.get_yticks()
        xticklabels = cbar.ax.get_yticklabels()
        xticklabels = p_min + xticks*(p_max-p_min)
        xticklabels = [f"{z:.2f}" for z in xticklabels]

        cbar.ax.set_yticks(xticks)
        cbar.ax.set_yticklabels(xticklabels)
        ax.view_init(elev=30., azim=-60)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlabel("z")
        ax.set_aspect('auto')
        plt.suptitle(r"3D visualization of the scattering pattern $f_\mathrm{s}(\hat{\mathbf{k}}_\mathrm{i}, \hat{\mathbf{k}}_\mathrm{s})$")

        ###
        ### Incident plane cut through the scattering pattern
        ###
        theta_i, phi_i = theta_phi_from_unit_vec(-k_i_local)
        theta_r, phi_r = theta_i, phi_i + dr.pi

        # Pattern around reflected direction
        theta_s = dr.linspace(mi.Float, 0, dr.pi/2, 100, False)
        phi_s = phi_r
        k_o_local = r_hat(theta_s, phi_s)
        pattern = self(k_i_local, k_o_local)

        # Pattern around incident direction
        k_s = r_hat(theta_s, phi_s+dr.pi)
        pattern2 = self(k_i_local, k_s)

        fig_cut = plt.figure()
        plt.polar(theta_s.numpy(), pattern.numpy(), color='C0')
        plt.polar(2*dr.pi-theta_s.numpy() , pattern2.numpy(), color='C0')

        ax = fig_cut.axes[0]
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)

        if show_directions:
            theta_i = theta_i.numpy()[0]
            xticks = list(ax.get_xticks())
            if not theta_i in xticks:
                xticks += [theta_i]
            if not -theta_i in xticks:
                xticks += [-theta_i]
            ax.set_xticks(xticks)
            ax.text(-theta_i-10*dr.pi/180, ax.get_yticks()[-1]*2/3,
                    r"$\hat{\mathbf{k}}_\mathrm{i}$",
                    horizontalalignment='center')
            ax.text(theta_i+10*dr.pi/180, ax.get_yticks()[-1]*2/3,
                    r"$\hat{\mathbf{k}}_\mathrm{r}$",
                     horizontalalignment='center')
            plt.quiver([0], [0], [np.sin(theta_i)], [np.cos(theta_i)],
                       scale=1., linestyle='dashed', color="black",  alpha=0.5)
            plt.quiver([0], [0], [-np.sin(theta_i)], [np.cos(theta_i)],
                       scale=1., linestyle='dashed', color="black", alpha=0.5)

        plt.title(r"Incident plane cut through the scattering pattern $f_\mathrm{s}(\hat{\mathbf{k}}_\mathrm{i}, \hat{\mathbf{k}}_\mathrm{s})$ ($\phi_\mathrm{s}=\phi_\mathrm{i}+\pi$)")
        plt.tight_layout()

        return fig_3d, fig_cut

# Registry for scattering patterns
scattering_pattern_registry = Registry()

def register_scattering_pattern(
    name: str,
    scattering_pattern_factory: Callable[..., ScatteringPattern]):
    """Registers a new factory method for a scattering pattern

    :param name: Name of the factory method

    :param scattering_pattern_factory: A factory method returning an instance
        of :class:`~sionna.rt.ScatteringPattern`
    """
    scattering_pattern_registry.register(scattering_pattern_factory, name)

@scattering_pattern_registry.register(name="lambertian")
class LambertianPattern(ScatteringPattern):
    r"""
    Lambertian scattering model from :cite:p:`Degli-Esposti07` as given in
    :eq:`lambertian_model`

    Example
    -------
    >>> LambertianPattern().show()

    .. figure:: ../figures/lambertian_pattern_3d.png
        :align: center

    .. figure:: ../figures/lambertian_pattern_cut.png
        :align: center
    """
    def __call__(self,
                 ki_local: mi.Vector3f,
                 ko_local: mi.Vector3f
    )-> mi.Float:
        r"""
        :param ki_local: Direction of propagation of the incident wave
            in local frame

        :param ko_local: Direction of propagation of the scattered wave
            in local frame

        :return: Scattering pattern value
        """
        cos_theta_o = dr.abs(ko_local.z)
        w = cos_theta_o/dr.pi
        return w

@scattering_pattern_registry.register(name="backscattering")
class BackscatteringPattern(ScatteringPattern):
    r"""
    Backscattering model from :cite:p:`Degli-Esposti07` as given in
    :eq:`backscattering_model`

    :param alpha_r: Positive integer parameter related to the width of the
        scattering lobe in the direction of the specular reflection

    :param alpha_i: Positive integer parameter related to the width of the
        scattering lobe in the incoming direction

    :param lambda_: Parameter determining the percentage of the diffusely
        reflected energy in the lobe around the specular reflection.
        Must be in the range :math:`[0,1]`

    Example
    -------
    >>> from sionna.rt import BackscatteringPattern
    >>> BackscatteringPattern(alpha_r=20, alpha_i=30, lambda_=0.7).show()

    .. figure:: ../figures/backscattering_pattern_3d.png
        :align: center

    .. figure:: ../figures/backscattering_pattern_cut.png
        :align: center
    """
    def __init__(self, alpha_r: int=1, alpha_i: int=1, lambda_: mi.Float=1.0):
        self.alpha_r = alpha_r
        self.alpha_i = alpha_i
        self.lambda_ = lambda_
        super().__init__()

    @property
    def alpha_r(self):
        r"""Get/set :math:`\alpha_\text{R}`

        :type: :py:class:`int`
        """
        return self._alpha_r

    @alpha_r.setter
    def alpha_r(self, v):
        if isinstance(v, bool) or not isinstance(v, Integral):
            raise TypeError("`alpha_r` must be an integer")
        if v < 1:
            raise ValueError("`alpha_r` must be greater than or equal to one")
        self._alpha_r = int(v)

    @property
    def alpha_i(self):
        r"""Get/set :math:`\alpha_\text{I}`

        :type: :py:class:`int`
        """
        return self._alpha_i

    @alpha_i.setter
    def alpha_i(self, v):
        if isinstance(v, bool) or not isinstance(v, Integral):
            raise TypeError("`alpha_i` must be an integer")
        if v < 1:
            raise ValueError("`alpha_i` must be greater than or equal to one")
        self._alpha_i = int(v)

    @property
    def lambda_(self):
        r"""Get/set :math:`\Lambda`

        :type: :py:class:`mi.float`
        """
        return self._lambda_

    @lambda_.setter
    def lambda_(self, v):
        v = mi.Float(v)
        if dr.any(~dr.isfinite(v) | (v < 0.) | (v > 1.)):
            raise ValueError("`lambda_` must be in the range [0,1]")
        self._lambda_ = v

    def __call__(self,
                 ki_local: mi.Vector3f,
                 ko_local: mi.Vector3f
    )-> mi.Float:
        r"""
        :param ki_local: Direction of propagation of the incident wave
            in local frame

        :param ko_local: Direction of propagation of the scattered wave
            in local frame

        :return: Scattering pattern value
        """

        # Direction of specular reflection
        ks_local = mi.reflect(-ki_local)

        cos_psi_r = dr.dot(ks_local, ko_local)
        cos_psi_i = -dr.dot(ki_local, ko_local)

        v_r = dr.power(0.5*(1.+cos_psi_r), self.alpha_r)
        v_i = dr.power(0.5*(1.+cos_psi_i), self.alpha_i)

        w = self.lambda_*v_r + (1.-self.lambda_)*v_i

        # Computes the normalization factor, denoted by F_{alpha_r,alpha_i}
        # in [Degli-Esposi07].

        cos_theta_i = -ki_local.z
        sin_theta_i = dr.sqrt(1. - dr.square(cos_theta_i))

        # F_alpha_i and F_alpha_r
        f_alpha_i = dr.zeros(mi.Float)
        f_alpha_r = dr.zeros(mi.Float)

        # K_n
        # n ranges from 0 to n_max, and will be computed
        # sequentially thereafter
        k_n = dr.zeros(mi.Float)

        # As parallelization is done over samples, i.e.,
        # each thread computes a sample, the compute of
        # the normalization factor is done sequentially for
        # each sample.

        # Compute I_j for odd values of j
        alpha_max = np.maximum(self.alpha_i, self.alpha_r)
        for j in range(alpha_max+1):

            # Even j
            if (j % 2) == 0:
                # For even j, I_j is independant of the incidence
                # direction and therefore the same for all samples
                # ()
                i_j = 2.*dr.pi/(j+1)

            # Odd j
            else:
                # ()
                n = (j-1)//2

                # Compute k_n
                # [num_samples]
                v = dr.power(sin_theta_i, 2*n)
                v *= scipy.special.binom(2*n, n)
                v /= np.power(2., 2.*n)
                k_n = k_n + v

                # Compute I_j
                # [num_samples]
                i_j = cos_theta_i*k_n*2.*dr.pi/float(j+1)

            # Update f_alpha_i
            mask_i = 1.0 if j <= self.alpha_i else 0.0
            f_alpha_i += i_j*scipy.special.binom(self.alpha_i, j)*mask_i
            # Update f_alpha_r
            mask_r = 1.0 if j <= self.alpha_r else 0.0
            f_alpha_r += i_j*scipy.special.binom(self.alpha_r, j)*mask_r

        f_alpha_i /= np.power(2., self.alpha_i)
        f_alpha_r /= np.power(2., self.alpha_r)

        # Compute normalization factor
        f = self.lambda_*f_alpha_r + (1.-self.lambda_)*f_alpha_i

        w_normalized = w*dr.rcp(f)
        return w_normalized

@scattering_pattern_registry.register(name="directive")
class DirectivePattern(BackscatteringPattern):
    r"""
    Directive scattering model from :cite:p:`Degli-Esposti07` as given in
    :eq:`directive_model`

    :param alpha_r: Positive integer parameter related to the width of the
        scattering lobe in the direction of the specular reflection

    Example
    -------
    >>> from sionna.rt import DirectivePattern
    >>> DirectivePattern(alpha_r=10).show()

    .. figure:: ../figures/directive_pattern_3d.png
        :align: center

    .. figure:: ../figures/directive_pattern_cut.png
        :align: center
    """
    def __init__(self, alpha_r: int=1):
        super().__init__(alpha_r=alpha_r, alpha_i=1, lambda_=1)
