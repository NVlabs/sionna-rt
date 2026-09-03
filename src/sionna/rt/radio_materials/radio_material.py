#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class implementing a radio material"""

import drjit as dr
import mitsuba as mi
from typing import Tuple, Callable

from sionna.rt.utils import itu_coefficients_single_layer_slab,\
    complex_relative_permittivity, jones_matrix_to_world_implicit,\
        f_utd, jones_matrix_rotator, implicit_basis_vector,\
        wedge_interior_angle, cot, sample_keller_cone
from sionna.rt.constants import InteractionType, DEFAULT_THICKNESS,\
    DEFAULT_FREQUENCY, NO_JONES_MATRIX
from .radio_material_base import RadioMaterialBase
from .scattering_pattern import scattering_pattern_registry, \
                                ScatteringPattern
from ..registry import Registry

from scipy.constants import speed_of_light


# Registry for custom radio materials
radio_material_registry = Registry()


def register_radio_material(rm: RadioMaterialBase) -> None:
    # pylint: disable=line-too-long
    r"""
    Registers a custom radio material instance to be used when loading scene files.

    :param rm: An instance of :class:`~sionna.rt.RadioMaterialBase` (or a subclass)
    """
    radio_material_registry.register(rm, rm.name)


class RadioMaterial(RadioMaterialBase):
    # pylint: disable=line-too-long
    r"""
    Class implementing the radio material model described in the `Primer on Electromagnetics <../em_primer.html>`_

    This class implements :class:`RadioMaterialBase`.

    A radio material is defined by its relative permittivity
    :math:`\varepsilon_r` and conductivity :math:`\sigma` (see :eq:`eta`), its
    thickness :math:`d` (see :eq:`q_fresnel_slab`),
    as well as optional parameters related to diffuse reflection (see `Scattering`_), such as the
    scattering coefficient :math:`S`, cross-polarization discrimination
    coefficient :math:`K_x`, and scattering pattern :math:`f_\text{s}(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s})`.

    We assume non-ionized and non-magnetic materials, and therefore the
    permeability :math:`\mu` of the material is assumed to be equal
    to the permeability of vacuum i.e., :math:`\mu_r=1.0`.

    Reflection and refraction coefficients are computed assuming that the
    intersected surface models a slab with the specified ``thickness``
    (see :eq:`fresnel_slab`). The computed coefficients therefore account for
    the multiple internal reflections inside the slab, meaning that this
    class should be used when objects like walls are modeled as single flat
    surfaces. The figure below illustrates this model, where :math:`E_i` is the
    incident electric field, :math:`E_r` is the reflected field and :math:`E_t` is
    the transmitted field. The Jones matrices, :math:`\mathbf{R}(d)` and
    :math:`\mathbf{T}(d)`, represent the effects of reflection and transmission,
    respectively, and depend on the slab thickness, :math:`d`.

    .. figure:: ../figures/transmission_model.png
        :width: 80%
        :align: center

    For frequency-dependent materials, it is possible to
    specify a callback function ``frequency_update_callback`` that computes
    the material properties :math:`(\varepsilon_r, \sigma)` from the
    frequency. If a callback function is specified, the material properties
    cannot be set and the values specified at instantiation are ignored.

    In addition to the following inputs, additional keyword arguments can be
    provided that will be passed to the scattering pattern as keyword
    arguments.

    :param name: Unique name of the material. Ignored if ``props`` is provided.
    :param thickness: Thickness of the material [m]. Ignored if ``props`` is provided.
    :param relative_permittivity: Relative permittivity of the material. Must be larger or equal to 1. Ignored if ``frequency_update_callback`` or ``props`` is provided.
    :param conductivity: Conductivity of the material [S/m].  Must be non-negative. Ignored if ``frequency_update_callback`` or ``props`` is provided.
    :param scattering_coefficient: Scattering coefficient :math:`S\in[0,1]` as defined in :eq:`scattering_coefficient`. Ignored if ``props`` is provided.
    :param xpd_coefficient:  Cross-polarization discrimination coefficient :math:`K_x\in[0,1]` as defined in :eq:`xpd`. Only relevant if ``scattering_coefficient`` is not equal to zero. Ignored if ``props`` is provided.
    :param scattering_pattern: Name of a registered scattering pattern for
        diffuse reflections
        :list-registry:`sionna.rt.radio_materials.scattering_pattern_registry`.
        Only relevant if ``scattering_coefficient`` is not equal to zero. Ignored if ``props`` is provided.
    :param frequency_update_callback: Callable used to update the material parameters when the frequency is set. This callable must take as input the frequency [Hz] and must return the material properties as a tuple: ``(relative_permittivity, conductivity)``. If set to :py:class:`None`, then material properties are constant and equal to the value set at instantiation or using the corresponding setters.
    :param color: RGB (red, green, blue) color for the radio material as displayed in the previewer and renderer. Each RGB component must have a value within the range :math:`[0,1]`. If set to :py:class:`None`, then a random color is used.
    :param props: Mitsuba container storing the material properties, and used
        when loading a scene to initialize the radio material.

    Keyword Arguments
    -----------------
    ``**kwargs``: :py:class:`Any`
        Depending on the chosen scattering antenna pattern, other keyword arguments
        must be provided.
        See the :ref:`Developer Guide <dev_custom_scattering_patterns>` for
        more details.
    """

    # pylint: disable=line-too-long
    def __init__(
        self,
        name: str | None = None,
        thickness: float | mi.Float = DEFAULT_THICKNESS,
        relative_permittivity: float | mi.Float = 1.0,
        conductivity: float | mi.Float = 0.0,
        scattering_coefficient: float | mi.Float = 0.0,
        xpd_coefficient: float | mi.Float = 0.0,
        scattering_pattern: str = "lambertian",
        frequency_update_callback: Callable[[mi.Float], Tuple[mi.Float, mi.Float]] | None = None,
        color: Tuple[float, float, float] | None = None,
        props: mi.Properties | None = None,
        **kwargs):

        if props is None:
            props = self._build_mi_props_from_params(name,
                                                     thickness,
                                                     relative_permittivity,
                                                     conductivity,
                                                     scattering_coefficient,
                                                     xpd_coefficient,
                                                     color,
                                                     **kwargs)

        # Real part of the relative permittivity
        eta_r = 1.0
        if 'relative_permittivity' in props:
            eta_r = props['relative_permittivity']
            del props['relative_permittivity']
        self.relative_permittivity = eta_r

        # Conductivity [S/m]
        sigma = 0.0
        if 'conductivity' in props:
            sigma = props['conductivity']
            del props['conductivity']
        self.conductivity = sigma

        # Material thickness [m]
        if 'thickness' in props:
            d = props['thickness']
            del props['thickness']
        else:
            d = DEFAULT_THICKNESS
        self.thickness = d

        # Scattering coefficient
        s = 0.0
        if 'scattering_coefficient' in props:
            s = props['scattering_coefficient']
            del props['scattering_coefficient']
        self.scattering_coefficient = s

        # XPD coefficient
        kx = 0.0
        if "xpd_coefficient" in props:
            kx = props["xpd_coefficient"]
            del props['xpd_coefficient']
        self.xpd_coefficient = kx

        super().__init__(props)

        # Gather the other properties as keyword arguments for the
        # scattering pattern
        scattering_pattern_attributes = {}
        for prop_name in props.keys():
            scattering_pattern_attributes[prop_name] = props[prop_name]

        # Set the scattering pattern if provided
        if scattering_pattern is None:
            scattering_pattern = "lambertian"
        factory = scattering_pattern_registry.get(scattering_pattern)
        self.scattering_pattern = factory(**scattering_pattern_attributes)

        # Set the frequency update callback
        self.frequency_update_callback = frequency_update_callback

    @RadioMaterialBase.scene.setter
    def scene(self, scene):
        # We need to overwrite this setter to make sure that the  material
        # parameters are correctly updated if a frequency callback is defined
        RadioMaterialBase.scene.fset(self, scene)
        self.frequency_update()

    @property
    def relative_permittivity(self):
        r"""
        Get/set the relative permittivity :math:`\varepsilon_r` :eq:`eta`

        :type: :py:class:`mi.Float`
        """
        return self._eta_r

    @relative_permittivity.setter
    def relative_permittivity(self, eta_r):
        if eta_r < 1.0:
            raise ValueError("Real part of the relative permittivity must be"
                             " greater or equal to 1")
        self._eta_r = mi.Float(eta_r)

    @property
    def conductivity(self):
        r"""Get/set the conductivity :math:`\sigma` [S/m] :eq:`eta`

        :type: :py:class:`mi.Float`
        """
        return self._sigma

    @conductivity.setter
    def conductivity(self, sigma):
        if sigma < 0.0:
            raise ValueError("The conductivity must be greater or equal to 0")
        self._sigma = mi.Float(sigma)

    @property
    def thickness(self):
        r"""Get/set the material thickness [m]

        :type: :py:class:`mi.Float`
        """
        return self._d

    @thickness.setter
    def thickness(self, d):
        if d < 0.0:
            raise ValueError("The material thickness must be positive")
        self._d = mi.Float(d)

    @property
    def scattering_coefficient(self):
        r"""Get/set the scattering coefficient :math:`S\in[0,1]`
        :eq:`scattering_coefficient`

        :type: :py:class:`mi.Float`
        """
        return self._s

    @scattering_coefficient.setter
    def scattering_coefficient(self, s):
        if s < 0.0 or s > 1.0:
            raise ValueError("Scattering coefficient must be in range (0,1)")
        self._s = mi.Float(s)

    @property
    def xpd_coefficient(self):
        r"""Get/set the cross-polarization discrimination coefficient
        :math:`K_x\in[0,1]` :eq:`xpd`

        :type: :py:class:`mi.Float`
        """
        return self._kx

    @xpd_coefficient.setter
    def xpd_coefficient(self, kx):
        if kx < 0.0 or kx > 1.0:
            raise ValueError("XPD coefficient must be in the range (0,1)")
        self._kx = mi.Float(kx)
        self._build_xpd_jones_mat()

    @property
    def scattering_pattern(self):
        # pylint: disable=line-too-long
        r"""Get/set the scattering pattern

        :type: :class:`~sionna.rt.ScatteringPattern`
        """
        return self._scattering_pattern

    @scattering_pattern.setter
    def scattering_pattern(self, sp):
        if not isinstance(sp, ScatteringPattern):
            raise ValueError("Not an instance of ScatteringPattern")
        self._scattering_pattern = sp

    def clone(self,
              name: str | None = None,
              **overrides) -> "RadioMaterial":
        r"""
        Returns a new :class:`RadioMaterial`, identical to this one except
        for any specified ``overrides``

        This method performs a shallow clone: non-overridden properties
        share their underlying values and arrays/tensors with the origin
        material. In differentiable ray tracing, this allows backpropagating
        gradients from interactions on cloned materials to the shared
        parameters of the origin material. Note that each clone is an
        independent object: updating a property on one instance via its
        setter (e.g., applying a gradient descent step) re-binds that
        attribute on that instance and does not implicitly re-bind attributes
        on other clones.

        See :meth:`RadioMaterialBase.clone` for details.

        :param name: Optional new name for the cloned material.
        :param overrides: Keyword arguments specifying properties to override.

        :return: New :class:`RadioMaterial` with the specified overrides.
        """
        kwargs = {
            "name": name or self.name,
            "thickness": self.thickness,
            "relative_permittivity": self.relative_permittivity,
            "conductivity": self.conductivity,
            "scattering_coefficient": self.scattering_coefficient,
            "xpd_coefficient": self.xpd_coefficient,
            "frequency_update_callback": self.frequency_update_callback,
            "color": self.color,
        }
        sp_override = overrides.pop("scattering_pattern", None)
        kwargs.update(overrides)
        new = RadioMaterial(**kwargs)
        if sp_override is not None:
            if isinstance(sp_override, str):
                factory = scattering_pattern_registry.get(sp_override)
                sp_override = factory()
            new.scattering_pattern = sp_override
        else:
            new.scattering_pattern = self.scattering_pattern
        return new

    @property
    def frequency_update_callback(self):
        # pylint: disable=line-too-long
        r"""
        Get/set the frequency update callback

        :type: :py:class:`Callable` [[:py:class:`mi.Float`], [:py:class:`mi.Float`, :py:class:`mi.Float`]]
        """
        return self._frequency_update_callback

    @frequency_update_callback.setter
    def frequency_update_callback(self, value):
        self._frequency_update_callback = value
        self.frequency_update()

    def frequency_update(self):
        r"""Updates the material parameters according to the set frequency"""

        if self._frequency_update_callback is None:
            return
        if self.scene is None:
            return

        parameters = self._frequency_update_callback(self.scene.frequency)
        relative_permittivity, conductivity = parameters
        self.relative_permittivity = relative_permittivity
        self.conductivity = conductivity

    def sample(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        sample1: mi.Float,
        sample2: mi.Point2f,
        active: bool | mi.Bool = True
    ) -> Tuple[mi.BSDFSample3f, mi.Spectrum]:
        # pylint: disable=line-too-long
        r"""
        Samples the radio material

        This function samples a radio material interaction and returns
        both the interaction details and its corresponding Jones matrix.
        It determines the type of interaction (specular reflection, diffuse
        reflection, refraction, or diffraction) and the direction of
        propagation for the scattered ray. The returned radio material sample
        contains the sampled interaction type, scattered ray direction, and
        probability that the selected interaction was sampled.
        The returned Jones matrix describes the electromagnetic transformation of
        the wave.

        The following assumptions are made on the inputs:

        - ``si.wi`` is the direction of propagation of the incident wave in
          the local frame
        - ``si.sh_frame`` is the surface frame where ``sh_frame.n`` is the normal to
          the intersected surface in the world coordinate system
        - ``si.dn_du`` stores the edge vector in the local frame.
            This is only used for diffraction.
        - ``si.dn_dv`` stores the normal to the n-face in the local frame.
            This is only used for diffraction.
        - ``si.dp_du`` stores the distance from the diffraction point to the source,
          the distance from the diffraction point to the receiver, and the flags indicating
          the enabled interaction types. The first two components are only used when
          diffraction is sampled.
        - ``si.duv_dx`` and ``si.duv_dy`` stores the incident field
        - ``si.t`` stores the solid angle of the incident ray tube.

        Note that all quantities related to diffraction are integrated to the paths only
        if diffraction is sampled.

        The outputs are set as follows:

        - ``bs.wo`` is the direction of propagation of the sampled scattered ray in
        the world frame
        - ``bs.sampled_component`` is the sampled interaction type
        - ``bs.pdf`` is the probability of the sampled interaction type
        - ``jones_mat`` is the Jones matrix describing the transformation incurred
        by the incident wave in the implicit world frame

        :param ctx: A context data structure used to specify which interaction
            types are enabled
        :param si: Surface interaction data structure describing the underlying
            surface position
        :param sample1: A uniformly distributed sample on :math:`[0,1]` used to
            sample the type of interaction
        :param sample2: A uniformly distributed sample on :math:`[0,1]^2` used
            to sample the direction of the reflected wave in the case of diffuse
            reflection, or the directon of the diffracted ray on the Keller cone
        :param active: Mask to specify active rays

        :return: Radio material sample and Jones matrix as a :math:`4 \times 4` real-valued matrix
        """
        # Incident direction of propagation in the local frame
        ki_local = si.wi
        # Cosine of the angle of arrival
        # In the local interaction frame, the normal is z+
        cos_theta_i = -ki_local.z

        # If the Jones matrix is not needed, we can avoid computing it.
        # This can significantly speed up the computation.
        compute_jones_matrix =  ctx.component & NO_JONES_MATRIX == 0

        # Interaction types can be globally disabled, i.e., their sampling is not possible.
        # When diffraction is globally disabled, we can avoid running the related code to
        # speed up the computation.
        diffraction_enabled = (ctx.component & InteractionType.DIFFRACTION) > 0

        # If an interaction type is enabled, then it can also be disabled locally, i.e.,
        # at the scale of a single path and single intersection point.
        loc_en_inter = dr.reinterpret_array(mi.UInt, si.dp_du.z)
        diffraction = active & (loc_en_inter == InteractionType.DIFFRACTION)

        # If scene is not set, uses the default frequency for evaluating the
        # material.
        # This is required to enable Dr.Jit to trace this code when the material
        # is instantiated but not assigned to a scene yet.
        if self.scene is None:
            angular_frequency = dr.two_pi*DEFAULT_FREQUENCY
            wavelength = speed_of_light/DEFAULT_FREQUENCY
            wavenumber = dr.two_pi/wavelength
        else:
            angular_frequency = self.scene.angular_frequency
            wavelength = self.scene.wavelength
            wavenumber = self.scene.wavenumber

        # To-world transform as a 3x3 matrix
        to_world = mi.Matrix3f(si.sh_frame.s, si.sh_frame.t, si.sh_frame.n).T

        # Relative permittivity of the material
        eta = complex_relative_permittivity(self._eta_r, self._sigma,
                                            angular_frequency)

        # ITU coefficients
        # r_te, r_tm: TE/TM reflection coefficients
        # t_te, t_tm: TE/TM transmission coefficients
        r_te, r_tm, t_te, t_tm =\
            itu_coefficients_single_layer_slab(cos_theta_i, eta, self._d,
                                               wavelength)

        # Sample an event type
        sampled_event, probs, specular, diffuse, _ = \
            self._sample_event_type(sample1, r_te, r_tm, t_te, t_tm,
                                    loc_en_inter)
        reflection = specular | diffuse
        sampled_event[diffraction] = InteractionType.DIFFRACTION

        # Direction of propagation of specularly reflected and transmitted
        # wave
        ko_spec_trans_local =\
            self._specular_reflection_transmission_direction(ki_local,
                                                             reflection)

        # Direction of propagation of scattered wave for the diffuse reflection
        ko_diffuse_local = self._diffuse_reflection_direction(sample2)

        # Direction of propagation of the diffracted ray
        if diffraction_enabled:
            ko_diffr_local = self._diffraction_direction(si, sample2.x,
                                                        ki_local)
        else:
            ko_diffr_local = mi.Vector3f(0.0)

        # If the Jones matrix is not needed, we can avoid computing it.
        if compute_jones_matrix:
            # Computes the Jones matrix and direction of propagation of the
            # scattered wave for specular reflection and transmission.
            # Because the specular reflection matrix is needed to compute the
            # diffusely scattered field, it is also computed when a diffuse
            # reflection was sampled.
            spec_trans_mat = self._specular_reflection_transmission_matrix(to_world,
                ki_local, ko_spec_trans_local, reflection, r_te, r_tm, t_te, t_tm)

            # Computes Jones matrix for diffuse reflection
            diff_mat = self._diffuse_reflection_matrix(si, ki_local, ko_diffuse_local,
                                                    spec_trans_mat)

            # Computes Jones matrix for diffraction
            if diffraction_enabled:
                diffr_mat = self._diffraction_matrix(to_world, ki_local, ko_diffr_local,
                                                    si, eta, wavenumber)
            else:
                diffr_mat = mi.Matrix4f(0.0)

            # Jones matrix selected according to the sampled event type
            jones_mat = dr.select(diffuse, diff_mat, spec_trans_mat)
            jones_mat = dr.select(diffraction, diffr_mat, jones_mat)

            # Apply multiplication by scattering coefficient and the importance
            # sampling weighting.
            # Scaling by `1/sqrt(probs)` cancels the weighting by `probs` that
            # arises from sampling the interaction types according to these
            # probabilities.
            s = dr.select(reflection, dr.sqrt(1. - dr.square(self._s)), 1.)
            s = dr.select(diffuse, self._s, s)
            # Block differentiation through the importance sampling weighting
            jones_mat *= s*dr.detach(dr.rsqrt(probs))

            # Cast the Jones matrix to a mi.Spectrum to meet the requirements of
            # the BSDF interface
            jones_mat = mi.Spectrum(jones_mat)
        else:
            jones_mat = mi.Spectrum(0.0)

        # Direction of propagation of the scattered wave selected according to
        # the sampled event type
        ko_local = dr.select(diffuse, ko_diffuse_local, ko_spec_trans_local)
        ko_local = dr.select(diffraction, ko_diffr_local, ko_local)

        # Instantiate and set the BSDFSample object
        bs = mi.BSDFSample3f()
        bs.sampled_component = sampled_event
        # Direction of the scattered wave in the world frame
        bs.wo = si.to_world(ko_local)
        # Probability of the event to be sampled
        bs.pdf = probs

        # Not used but must be set
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.DeltaReflection)
        bs.eta = 1.0

        return bs, jones_mat

    def eval(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool | mi.Bool = True
    ) -> mi.Spectrum:
        # pylint: disable=line-too-long
        r"""
        Evaluates the radio material

        This function evaluates the Jones matrix of the radio material for the scattered
        direction ``wo`` and for the interaction type stored in ``si.dp_du.z``.

        - ``si.wi`` is the direction of propagation of the incident wave in
            the local frame
        - ``si.sh_frame`` is the surface frame where ``sh_frame.n`` is the normal to
            the intersected surface in the world coordinate system
        - ``si.dn_du`` stores the edge vector in the local frame.
            This is only used for diffraction.
        - ``si.dn_dv`` stores the normal to the n-face in the local frame.
            This is only used for diffraction.
        - ``si.dp_du`` stores the distance from the diffraction point to the source,
            the distance from the diffraction point to the receiver, and the interaction
            type. The first two components are only used when diffraction is sampled.
        - ``si.duv_dx`` and ``si.duv_dy`` stores the incident field
        - ``si.t`` stores the solid angle of the incident ray tube.

        :param ctx: A context data structure used to specify which interaction types are enabled
        :param si: Surface interaction data structure describing the underlying surface position
        :param wo: Direction of propagation of the scattered wave in the world frame
        :param active: Mask to specify active rays

        :return: Jones matrix as a :math:`4 \times 4` real-valued matrix
        """
        # Incident direction of propagation in the local frame
        ki_local = si.wi
        # Cosine of the angle of arrival
        # In the local interaction frame, the normal is z+
        cos_theta_i = -ki_local.z

        # Interaction types can be globally disabled, i.e., their sampling is not possible.
        # When diffraction is globally disabled, we can avoid running the related code to
        # speed up the computation.
        diffraction_enabled = (ctx.component & InteractionType.DIFFRACTION) > 0

        # If scene is not set, uses the default frequency for evaluating the
        # material.
        # This is required to enable Dr.Jit to trace this code when the material
        # is instantiated but not assigned to a scene yet.
        if self.scene is None:
            angular_frequency = dr.two_pi * DEFAULT_FREQUENCY
            wavelength = speed_of_light / DEFAULT_FREQUENCY
            wavenumber = dr.two_pi / wavelength
        else:
            angular_frequency = self.scene.angular_frequency
            wavelength = self.scene.wavelength
            wavenumber = self.scene.wavenumber

        # Direction of propagation of the scattered wave in the world frame
        ko_world = wo
        ko_local = si.to_local(ko_world)

        # To-world transform as a 3x3 matrix
        to_world = mi.Matrix3f(si.sh_frame.s, si.sh_frame.t, si.sh_frame.n).T

        # Relative permittivity of the material
        eta = complex_relative_permittivity(self._eta_r, self._sigma,
                                            angular_frequency)

        # ITU coefficients
        # r_te, r_tm: TE/TM reflection coefficients
        # t_te, t_tm: TE/TM transmission coefficients
        r_te, r_tm, t_te, t_tm =\
            itu_coefficients_single_layer_slab(cos_theta_i, eta, self._d,
                                               wavelength)

        sampled_event =  dr.reinterpret_array(mi.UInt, si.dp_du.z)
        reflection = (sampled_event == InteractionType.SPECULAR) |\
                     (sampled_event == InteractionType.DIFFUSE)
        diffuse = sampled_event == InteractionType.DIFFUSE
        diffraction = sampled_event == InteractionType.DIFFRACTION

        # Direction of propagation of specularly reflected and transmitted
        # wave
        ko_spec_trans_local =\
            self._specular_reflection_transmission_direction(ki_local,
                                                            reflection)

        # Computes the Jones matrix and direction of propagation of the
        # scattered wave for specular reflection and transmission.
        # Because the specular reflection matrix is needed to compute the
        # diffusely scattered field, it is also computed when a diffuse
        # reflection was sampled.
        spec_trans_mat = self._specular_reflection_transmission_matrix(to_world,
            ki_local, ko_spec_trans_local, reflection, r_te, r_tm, t_te, t_tm)

        # Computes Jones matrix for diffuse reflection
        diff_mat = self._diffuse_reflection_matrix(si, ki_local, ko_local,
                                                  spec_trans_mat)

        # Computes Jones matrix for diffraction
        if diffraction_enabled:
            diffr_mat = self._diffraction_matrix(to_world, ki_local, ko_local,
                                                si, eta, wavenumber)
        else:
            diffr_mat = mi.Matrix4f(0.0)

        # Jones matrix selected according to the sampled event type
        jones_mat = dr.select(diffuse, diff_mat, spec_trans_mat)
        jones_mat = dr.select(diffraction, diffr_mat, jones_mat)

        # Apply multiplication by scattering coefficient
        s = dr.select(reflection, dr.sqrt(1. - dr.square(self._s)), 1.)
        s = dr.select(diffuse, self._s, s)
        jones_mat *= s

        # Cast the Jones matrix to a mi.Spectrum to meet the requirements of
        # the BSDF interface
        jones_mat = mi.Spectrum(jones_mat)

        return jones_mat

    def pdf(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool | mi.Bool = True
    ) -> mi.Float:
        # pylint: disable=line-too-long
        r"""
        Evaluates the probability of the sampled interaction type and direction of scattered ray

        This function evaluates the probability density of the radio material for the scattered
        direction ``wo`` and for the interaction type stored in ``si.dp_du.z``.

        - ``si.wi`` is the direction of propagation of the incident wave in
            the local frame
        - ``si.sh_frame`` is the surface frame where ``sh_frame.n`` is the normal to
            the intersected surface in the world coordinate system
        - ``si.dn_du`` stores the edge vector in the local frame.
            This is only used for diffraction.
        - ``si.dn_dv`` stores the normal to the n-face in the local frame.
            This is only used for diffraction.
        - ``si.dp_du`` stores the distance from the diffraction point to the source,
            the distance from the diffraction point to the receiver, and the interaction
            type. The first two components are only used when diffraction is sampled.
        - ``si.duv_dx`` and ``si.duv_dy`` stores the incident field
        - ``si.t`` stores the solid angle of the incident ray tube.

        :param ctx: A context data structure used to specify which interaction types are enabled
        :param si: Surface interaction data structure describing the underlying surface position
        :param wo: Direction of propagation of the scattered wave in the world frame
        :param active: Mask to specify active rays

        :return: Probability density value
        """
        # Incident direction of propagation in the local frame
        ki_local = si.wi
        # Cosine of the angle of arrival
        # In the local interaction frame, the normal is z+
        cos_theta_i = -ki_local.z

        # If scene is not set, uses the default frequency for evaluating the
        # material.
        # This is required to enable Dr.Jit to trace this code when the material
        # is instantiated but not assigned to a scene yet.
        if self.scene is None:
            angular_frequency = dr.two_pi * DEFAULT_FREQUENCY
            wavelength = speed_of_light / DEFAULT_FREQUENCY
        else:
            angular_frequency = self.scene.angular_frequency
            wavelength = self.scene.wavelength

        # Read the flag indicating the enabled interaction types
        loc_en_inter = dr.reinterpret_array(mi.UInt, si.dp_du.z)

        # Relative permittivity of the material
        eta = complex_relative_permittivity(self._eta_r, self._sigma,
                                            angular_frequency)

        # ITU coefficients
        # r_te, r_tm: TE/TM reflection coefficients
        # t_te, t_tm: TE/TM transmission coefficients
        r_te, r_tm, t_te, t_tm = \
            itu_coefficients_single_layer_slab(cos_theta_i, eta, self._d,
                                               wavelength)

        sampled_event =  dr.reinterpret_array(mi.UInt, si.dp_du.z)
        specular = sampled_event == InteractionType.SPECULAR
        transmission = sampled_event == InteractionType.REFRACTION
        diffraction = sampled_event == InteractionType.DIFFRACTION

        prs, prd, pt, pd = self._event_probabilities(r_te, r_tm,
                                                     t_te, t_tm,
                                                     loc_en_inter)

        probs = dr.select(specular, prs, prd)
        probs[transmission] = pt
        probs[diffraction] = pd
        return probs

    def traverse(self, callback: mi.TraversalCallback):
        # pylint: disable=line-too-long
        r"""
        Traverse the attributes and objects of this instance

        Implementing this function is required for Mitsuba to traverse a scene graph,
        including materials, and determine the differentiable parameters.

        :param callback: Object used to traverse the scene graph
        """
        callback.put('eta_r', self._eta_r, mi.ParamFlags.Differentiable)
        callback.put('sigma', self._sigma, mi.ParamFlags.Differentiable)
        callback.put('d', self._d, mi.ParamFlags.Differentiable)
        callback.put('s', self._s, mi.ParamFlags.Differentiable)
        callback.put('xpd_coefficient', self._kx, mi.ParamFlags.Differentiable)

    def to_string(self) -> str:
        r"""
        Returns a string describing the object
        """
        s = f"RadioMaterial eta_r={self._eta_r[0]:.3f}\n"\
            f"              sigma={self._sigma[0]:.3f}\n"\
            f"              thickness={self._d[0]:.3f}\n"\
            f"              scattering_coefficient={self._s[0]:.3f}\n"\
            f"              xpd_coefficient={self._kx[0]:.3f}"
        return s

    ##############################################
    # Internal methods
    ##############################################

    def _event_probabilities(
        self,
        r_te: mi.Complex2f,
        r_tm: mi.Complex2f,
        t_te: mi.Complex2f,
        t_tm: mi.Complex2f,
        enabled_interactions: mi.UInt
    ) -> Tuple[mi.Float, mi.Float, mi.Float, mi.Bool]:
        # pylint: disable=line-too-long
        r"""
        Compute probabilities with which to sample interaction types

        The probabilities of sampling an interaction type is set proportionally
        to the corresponding gain.

        To compute the probability of sampling a transmission or a reflection,
        it is assumed that the energy is evenly split between the two
        polarization directions.

        :param r_te: Transverse electric reflection coefficient
        :param r_tm: Transverse magnetic reflection coefficient
        :param t_te: Transverse electric refraction coefficient
        :param t_tm: Transverse magnetic refraction coefficient
        :param enabled_interactions: Bitmask indicating which interaction types are enabled

        :return: Probability with which to sample a specular reflection
        :return: Probability with which to sample a diffuse reflection
        :return: Probability with which to sample a transmission
        :return: Flag indicating if the sampled interaction type
        """

        # The probability of reflection and transmission are sampled
        # proportionally to the reflection and transmission coefficients
        r = dr.square(dr.abs(r_te)) + dr.square(dr.abs(r_tm))
        t = dr.square(dr.abs(t_te)) + dr.square(dr.abs(t_tm))
        # Scattering coefficient
        s = dr.square(self._s)

        specular_reflection_enabled = enabled_interactions & InteractionType.SPECULAR > 0
        diffuse_reflection_enabled = enabled_interactions & InteractionType.DIFFUSE > 0
        refraction_enabled = enabled_interactions & InteractionType.REFRACTION > 0

        # Probability of sampling a specular reflection
        prs = dr.select(specular_reflection_enabled, r * (1. - s), mi.Float(0.))
        # Probability of sampling a diffuse reflection
        prd = dr.select(diffuse_reflection_enabled, r * s, mi.Float(0.))
        # Probability of a transmission
        pt = dr.select(refraction_enabled, t, mi.Float(0.))

        # Normalize to ensure the probabilities sum to 1
        sum_probs = prs + prd + pt
        none_int = sum_probs <= 0.
        norm_factor = dr.select(none_int, 1.0, dr.rcp(sum_probs))
        prs *= norm_factor
        prd *= norm_factor
        pt *= norm_factor

        return prs, prd, pt, none_int

    def _sample_event_type(
        self,
        sample1: mi.Float,
        r_te: mi.Complex2f,
        r_tm: mi.Complex2f,
        t_te: mi.Complex2f,
        t_tm: mi.Complex2f,
        enabled_interactions: mi.UInt
    ) -> Tuple[mi.UInt, mi.Float, mi.Bool, mi.Bool, mi.Bool]:
        # pylint: disable=line-too-long
        """
        Samples event types

        :param sample1: Uniformly distributed float in :math:`[0,1]`
        :param r_te: Transverse electric reflection coefficient
        :param r_tm: Transverse magnetic reflection coefficient
        :param t_te: Transverse electric refraction coefficient
        :param t_tm: Transverse magnetic refraction coefficient
        :param enabled_interactions: Bitmask indicating which interaction types are enabled
        :return: Tuple containing:
            - Sampled type of interaction
            - Probability of sampling the selected event
            - Flag set to `True` if a specular reflection was sampled
            - Flag set to `True` if a diffuse reflection was sampled
            - Flag set to `True` if a transmission was sampled
        """

        prs, prd, pt, none_int = self._event_probabilities(r_te, r_tm,
                                                           t_te, t_tm,
                                                           enabled_interactions)

        specular = sample1 < prs
        diffuse = (prs <= sample1) & (sample1 < prd + prs)
        transmission = sample1 >= prd + prs

        #
        sampled_event = dr.select(specular, InteractionType.SPECULAR,
                                  InteractionType.DIFFUSE)
        sampled_event[transmission] = InteractionType.REFRACTION
        sampled_event[none_int] = InteractionType.NONE

        # Probability of sampling the selected event
        probs = dr.select(specular, prs, prd)
        probs[transmission] = pt

        output = (sampled_event, probs, specular,
                  diffuse, transmission)
        return output

    def _specular_reflection_transmission_direction(
        self,
        ki_local: mi.Vector3f,
        reflection: mi.Bool
    ) -> mi.Vector3f:
        # pylint: disable=line-too-long
        r"""
        Computes the direction of propagation of specularly reflected and
        transmitted wave according to the sampled event, indicated by the
        ``reflection`` mask

        :param ki_local: Direction of propagation of the incident field in the local frame
        :param reflection: Mask indicating if a reflection (specular or diffuse) was sampled

        :return: Direction of propagation of the scattered wave in the local frame
        """

        # Direction of the scattered field
        ko_local_spec_refl = mi.reflect(-ki_local)
        ko_local_trans = ki_local
        ko_local = dr.select(reflection, ko_local_spec_refl, ko_local_trans)

        return ko_local

    def _specular_reflection_transmission_matrix(
        self,
        to_world: mi.Matrix3f,
        ki_local: mi.Vector3f,
        ko_local: mi.Vector3f,
        reflection: mi.Bool,
        r_te: mi.Complex2f,
        r_tm: mi.Complex2f,
        t_te: mi.Complex2f,
        t_tm: mi.Complex2f
    ) -> mi.Matrix4f:
        # pylint: disable=line-too-long
        r"""
        Computes the Jones matrix of the scattered wave for specular reflection
        and transmission according to the sampled event indicated by the
        ``reflection`` mask

        :param to_world: To-world transform as a :math:`3 \times 3` matrix
        :param ki_local: Direction of propagation of the incident field in the local frame
        :param ko_local: Direction of propagation of the scattered wave in the local frame
        :param reflection: Mask indicating if a reflection (specular or diffuse) was sampled
        :param r_te: Transverse electric reflection coefficient
        :param r_tm: Transverse magnetic reflection coefficient
        :param t_te: Transverse electric refraction coefficient
        :param t_tm: Transverse magnetic refraction coefficient

        :return: Jones matrix as a :math:`4 \times 4` real-valued matrix in the world frame
        """

        # Selects the reflection/refraction coefficients, direction of
        # scattering, and S basis vector according to the sampled event.
        # Coefficients
        c1 = dr.select(reflection, r_te, t_te)
        c2 = dr.select(reflection, r_tm, t_tm)

        # Jones matrix in the world implicit base
        jones_mat = jones_matrix_to_world_implicit(c1, c2, to_world, ki_local,
                                                   ko_local)

        return jones_mat

    def _diffuse_reflection_direction(
        self,
        sample2: mi.Point2f
    ) -> mi.Vector3f:
        # pylint: disable=line-too-long
        r"""
        Computes a unit vector on the hemisphere from a 2D uniform sample on the
        unit square `sample2`

        :param sample2: A uniformly distributed sample on :math:`[0,1]^2`

        :return: Unit vector on the hemisphere
        """

        ko_local = mi.warp.square_to_uniform_hemisphere(sample2)
        # Due to numerical error, it could be that kd_local.z is slightly
        # negative
        ko_local.z = dr.abs(ko_local.z)
        return ko_local

    def _diffuse_reflection_matrix(
        self,
        si: mi.SurfaceInteraction3f,
        ki_local: mi.Vector3f,
        ko_local: mi.Vector3f,
        specular_reflection_mat: mi.Matrix4f
    ) -> mi.Matrix4f:
        # pylint: disable=line-too-long
        r"""
        Computes the Jones matrix for diffuse reflection

        :param si: Surface interaction data structure describing the underlying surface position
        :param ki_local: Direction of propagation of the incident wave in the local frame
        :param ko_local: Direction of propagation of the scattered wave in the local frame
        :param specular_reflection_mat: Jones matrix for specular reflection as :math:`4 \times 4` real-valued matrix in the implicit world frame

        :return: Jones matrix as a :math:`4 \times 4` real-valued matrix in the world frame
        """

        # Incident field Jones vector
        ei = mi.Vector4f(si.duv_dx.x, # Real component of S
                         si.duv_dx.y, # Real component of P
                         si.duv_dy.x, # Imag component of S
                         si.duv_dy.y) # Imag component of P
        # Solid angle
        solid_angle = si.t

        # Amplitude of the reflected field
        er_spec = specular_reflection_mat @ ei
        er_spec_norm = dr.norm(er_spec)
        # Amplitude of the incident field
        ei_norm = dr.norm(ei)
        # Gamma coefficient
        gamma = dr.select(ei_norm > 0.,
                          er_spec_norm * dr.rcp(ei_norm),
                          mi.Float(0.))

        # Scattering pattern
        fs = self._scattering_pattern(ki_local, ko_local)

        # Jones matrix for diffuse reflection
        # As the implicit basis are the spherical unit vectors
        # (theta_hat, phi_hat), we do not need to apply change-of-basis matrices
        # from the implicit basis to the spherical basis. Note that this would
        # be needed otherwise, as the model used for diffuse scattering operates
        # in the basis defined by the spherical unit vectors.
        jones_mat = dr.sqrt(fs * solid_angle) * gamma * self._xpd_jones_mat

        return jones_mat

    def _diffraction_matrix(
        self,
        to_world: mi.Matrix3f,
        ki_local: mi.Vector3f,
        ko_local: mi.Vector3f,
        si: mi.SurfaceInteraction3f,
        eta: mi.Complex2f,
        wavenumber: mi.Float,
    ) -> mi.Matrix4f:
        # pylint: disable=line-too-long
        r"""
        Computes the Jones matrix of diffracted rays

        This method computes the Jones matrix for electromagnetic wave diffraction
        at a wedge edge using the uniform theory of diffraction (UTD). The matrix
        describes the transformation of the incident wave into the diffracted wave.

        :param to_world: To-world transform as a :math:`3 \times 3` matrix
        :param ki_local: Direction of propagation of the incident ray in the local frame
        :param ko_local: Direction of propagation of the diffracted ray in the local frame
        :param si: Surface interaction data structure describing the underlying
            surface position
        :param eta: Complex-valued relative permittivity of the wedge material
        :param wavenumber: Wavenumber of the incident wave [m^-1]

        :return: Jones matrix as a :math:`4 \times 4` real-valued matrix in the world frame
        """

        e_hat_local = si.dn_du
        nn_local = si.dn_dv
        n0_local = mi.Vector3f(0, 0, 1)

        # Read the wedge opening angle
        wedge_angle = wedge_interior_angle(n0_local, nn_local)
        # Read the angle of incidence on the wedge
        beta0 = dr.safe_acos(dr.abs(dr.dot(ki_local, e_hat_local)))

        # Wedge exterior angle
        exterior_angle = dr.two_pi - wedge_angle

        # Distance from the diffraction point to the source and receiver
        s = si.dp_du.x
        s_prime = si.dp_du.y

        # Compute vector tangential to 0-face
        to_hat_local = dr.normalize(dr.cross(n0_local, e_hat_local))

        # Compute projected incoming and outgoing directions
        ki_local_proj = ki_local - dr.dot(ki_local, e_hat_local) * e_hat_local
        ki_local_proj = dr.normalize(ki_local_proj)

        ko_local_proj = ko_local - dr.dot(ko_local, e_hat_local) * e_hat_local
        ko_local_proj = dr.normalize(ko_local_proj)

        # Compute phi_prime and phi
        phi_prime = dr.pi-dr.safe_acos(-dr.dot(ki_local_proj, to_hat_local))
        phi_prime *= -dr.sign(-dr.dot(ki_local_proj, n0_local))
        phi_prime += dr.pi

        phi = dr.pi-dr.safe_acos(dr.dot(ko_local_proj, to_hat_local))
        phi *= -dr.sign(dr.dot(ko_local_proj, n0_local))
        phi += dr.pi

        # Compute L
        l = s * s_prime * dr.rcp(s + s_prime) * dr.sin(beta0)**2

        # Compute diffraction coefficients
        n = exterior_angle * dr.rcp(dr.pi)
        dif_phi = phi - phi_prime
        sum_phi = phi + phi_prime

        def a_p_m(beta):
            # Compute n_p and n_m
            n_p = dr.round( (beta + dr.pi) * dr.rcp(2 * exterior_angle) )
            n_m = dr.round( (beta - dr.pi) * dr.rcp(2 * exterior_angle) )

            # Compute a_p and a_m
            a_p = 2 * dr.cos(exterior_angle * n_p - beta / 2) ** 2
            a_m = 2 * dr.cos(exterior_angle * n_m - beta / 2) ** 2

            return a_p, a_m

        a1, a2 = a_p_m(dif_phi)
        a3, a4 = a_p_m(sum_phi)

        factor = -dr.exp(mi.Complex2f(0, -dr.pi/4))
        factor *= dr.rcp(2 * n * dr.safe_sqrt(dr.two_pi * wavenumber)
                         * dr.sin(beta0))

        d1 = cot( (dr.pi + dif_phi) * dr.rcp(2 * n) )
        d2 = cot( (dr.pi - dif_phi) * dr.rcp(2 * n) )
        d3 = cot( (dr.pi + sum_phi) * dr.rcp(2 * n) )
        d4 = cot( (dr.pi - sum_phi) * dr.rcp(2 * n) )

        # Complex-valued diffraction coefficients
        d1 *= factor * f_utd(wavenumber * l * a1)
        d2 *= factor * f_utd(wavenumber * l * a2)
        d3 *= factor * f_utd(wavenumber * l * a3)
        d4 *= factor * f_utd(wavenumber * l * a4)

        # Compute various vectors for basis changes
        phi_hat_prime = dr.normalize(dr.cross(ki_local, e_hat_local))
        phi_hat = -dr.normalize(dr.cross(ko_local, e_hat_local))

        e_i_s_0_hat = dr.normalize(dr.cross(ki_local, n0_local))
        e_r_s_0_hat = e_i_s_0_hat

        e_i_s_n_hat = dr.normalize(dr.cross(ki_local, nn_local))
        e_r_s_n_hat = e_i_s_n_hat

        # Compute basis change matrices
        w_0_in = jones_matrix_rotator(ki_local, phi_hat_prime, e_i_s_0_hat)
        w_0_out = jones_matrix_rotator(ko_local, e_r_s_0_hat, phi_hat)
        w_n_in = jones_matrix_rotator(ki_local, phi_hat_prime, e_i_s_n_hat)
        w_n_out = jones_matrix_rotator(ko_local, e_r_s_n_hat, phi_hat)

        # Compute fresnel coefficients for both faces
        wavelength = dr.two_pi / wavenumber
        r_te_0, r_tm_0, _, _ =\
            itu_coefficients_single_layer_slab(dr.abs(dr.sin(phi_prime)),
                                               eta,
                                               self._d,
                                               wavelength)
        r_te_n, r_tm_n, _, _ =\
            itu_coefficients_single_layer_slab(dr.abs(dr.sin(exterior_angle-phi)),
                                               eta,
                                               self._d,
                                               wavelength)
        #
        # Construct R_0 and R_n matrices
        #

        # Multiply reflection coefficients by diffraction coefficients
        d12 = -(d1 + d2)
        r_te_0 *= d4
        r_tm_0 *= d4
        r_te_n *= d3
        r_tm_n *= d3

        # Compute the final diffraction matrix
        r_0_real = mi.Matrix2f(r_te_0.real, 0,
                               0,           r_tm_0.real)
        r_0_imag = mi.Matrix2f(r_te_0.imag, 0,
                               0,           r_tm_0.imag)
        r_n_real = mi.Matrix2f(r_te_n.real, 0,
                               0,           r_tm_n.real)
        r_n_imag = mi.Matrix2f(r_te_n.imag, 0,
                               0,           r_tm_n.imag)

        r_0_real = w_0_out @ r_0_real @ w_0_in
        r_0_imag = w_0_out @ r_0_imag @ w_0_in
        r_n_real = w_n_out @ r_n_real @ w_n_in
        r_n_imag = w_n_out @ r_n_imag @ w_n_in

        d_12_real = mi.Matrix2f(d12.real,        0,
                                       0, d12.real)
        d_12_imag = mi.Matrix2f(d12.imag,        0,
                                       0, d12.imag)

        real = d_12_real + r_0_real + r_n_real
        imag = d_12_imag + r_0_imag + r_n_imag

        # Embed into to_world transformation
        ki_world = to_world@ki_local
        ko_world = to_world@ko_local
        theta_i_world = implicit_basis_vector(ki_world)
        theta_o_world = implicit_basis_vector(ko_world)
        theta_i_local = to_world.T @ theta_i_world
        theta_o_local = to_world.T @ theta_o_world
        w_in = jones_matrix_rotator(ki_local, theta_i_local, phi_hat_prime)
        w_out = jones_matrix_rotator(ko_local, phi_hat, theta_o_local)

        real = w_out @ real @ w_in
        imag = w_out @ imag @ w_in

        # Jones matrix is returned as a 4x4 real-valued matrix
        m4f = mi.Matrix4f(real[0,0], real[0,1], -imag[0,0], -imag[0,1],
                          real[1,0], real[1,1], -imag[1,0], -imag[1,1],
                          imag[0,0], imag[0,1],  real[0,0],  real[0,1],
                          imag[1,0], imag[1,1],  real[1,0],  real[1,1])

        return m4f

    def _build_xpd_jones_mat(self):
        # pylint: disable=line-too-long
        r"""
        Builds the Jones matrix from the XPD coefficient that models the
        rotation of the polarization direction

        The stored Jones matrix is represented by a :math:`4 \times 4` real-valued matrix as follows:

        .. math::

        J =
            \begin{bmatrix}
                \begin{array}{c|c}
                    \sqrt(1-K_x)    & -\sqrt(Kx)    & 0                 & 0             \\ \hline
                    \sqrt(Kx)       & \sqrt(1-K_x)  & 0                 & 0             \\ \hline
                    0               & 0             & \sqrt(1-K_x)      & -\sqrt(Kx)    \\ \hline
                    0               & 0             & \sqrt(K_x)        & \sqrt(1-Kx)   \\
                \end{array}
            \end{bmatrix}

        where :math:`K_x` is the XPD coefficient.

        The built Jones matrix is stored in the `self._xpd_jones_mat` attribute.
        """

        a = dr.sqrt(1. - self._kx)
        b = dr.sqrt(self._kx)

        m = mi.Matrix4f(a,  -b, 0., 0.,
                        b,   a, 0., 0.,
                        0., 0., a,  -b,
                        0., 0., b,  a)
        self._xpd_jones_mat = m

    def _diffraction_direction(
        self,
        si: mi.SurfaceInteraction3f,
        sample: mi.Float,
        ki_local: mi.Vector3f,
    ) -> mi.Vector3f:
        r"""
        Samples a direction on the Keller cone for diffraction

        This function samples a diffracted ray direction on the Keller cone.
        The Keller cone is such that the angle between the diffracted rays and
        the edge direction is equal to the angle between the incident ray and
        the edge direction.

        :param si: Surface interaction
        :param sample: Random sample in [0, 1) used for sampling the Keller cone
        :param ki_local: Incident ray direction in the local coordinate system

        :return: Normalized direction vector of the diffracted ray in the
            local coordinate system
        """

        e_hat_local = si.dn_du
        nn_local = si.dn_dv

        # In the local coordinate system, the normal is z+
        n0_local = mi.Vector3f(0., 0., 1.)

        # Sample the Keller cone
        k_diffr = sample_keller_cone(e_hat_local, n0_local, nn_local,
                                     sample, ki_local, True)

        return k_diffr

    def _build_mi_props_from_params(
        self,
        name: str,
        thickness: float | mi.Float,
        relative_permittivity: float | mi.Float,
        conductivity: float | mi.Float,
        scattering_coefficient: float | mi.Float,
        xpd_coefficient: float | mi.Float,
        color: Tuple[float, float, float] | None,
        **kwargs
    ) -> mi.Properties:
        # pylint: disable=line-too-long
        r"""
        Builds a :class:`mitsuba.Properties` object from a set of material
        properties

        Additional keyword arguments can be provided to be passed to the
        scattering pattern.

        :param name: Unique name of the material
        :param thickness: Thickness of the material [m]
        :param relative_permittivity: Relative permittivity of the material
        :param conductivity: Conductivity of the material [S/m]
        :param scattering_coefficient: Scattering coefficient
        :param xpd_coefficient: Cross-polarization discrimination coefficient
        :param color: Optional RGB (red, green, blue) color for the radio material as displayed in the previewer and renderer
        """
        props = mi.Properties("radio-material")
        # Name of the radio material
        props.set_id(name)

        # BSDF parameters
        props["relative_permittivity"] = relative_permittivity
        props["conductivity"] = conductivity
        props["scattering_coefficient"] = scattering_coefficient
        props["thickness"] = thickness
        props["xpd_coefficient"] = xpd_coefficient
        if color is not None:
            props["color"] = mi.ScalarColor3f(color)

        # Additional keywords arguments will be passed to the
        # scattering pattern
        for k, v in kwargs.items():
            props[k] = v

        return props

# Register this custom BSDF plugin
mi.register_bsdf("radio-material", lambda props: RadioMaterial(props=props))
