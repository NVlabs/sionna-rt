#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""ITU radio materials"""

import mitsuba as mi
from typing import Tuple, Callable, Mapping

from .itu import itu_material, ITU_MATERIALS_PROPERTIES
from .radio_material import RadioMaterial
from .scattering_pattern import scattering_pattern_registry


class ITURadioMaterial(RadioMaterial):
    # pylint: disable=line-too-long
    r"""
    Class implementing the materials defined in the ITU-R P.2040-4 recommendation :cite:p:`ITURP20404`

    This class inherits from :class:`~sionna.rt.RadioMaterial`.

    The models from the ITU-R P.2040-4 recommendation are based on curve fitting
    to measurement results and assume non-ionized and non-magnetic materials (:math:`\mu_r = 1`).
    Frequency dependence is modeled by

    .. math::

        \begin{aligned}
            \varepsilon_r &= a f_{\text{GHz}}^b\\
            \sigma &= c f_{\text{GHz}}^d
        \end{aligned}

    where :math:`f_{\text{GHz}}` is the frequency in GHz, and the constants
    :math:`a`, :math:`b`, :math:`c`, and :math:`d` characterize the material.

    Note that the relative permittivity :math:`\varepsilon_r` and
    conductivity :math:`\sigma` of all materials are updated automatically when
    the frequency is set through the scene's property :class:`~.rt.Scene.frequency`.

    In addition to the following inputs, additional keyword arguments can be
    provided that will be passed to the scattering pattern as keyword
    arguments.

    :param name: Unique name of the material. Ignored if ``props`` is provided.
    :param itu_type: Type the ITU material. The available materials are listed in :ref:`the corresponding table <provided-materials>`. Ignored if ``props`` is provided.
    :param thickness: Thickness of the material [m]. Ignored if ``props`` is provided.
    :param scattering_coefficient: Scattering coefficient :math:`S\in[0,1]` as defined in :eq:`scattering_coefficient`. Ignored if ``props`` is provided.
    :param xpd_coefficient:  Cross-polarization discrimination coefficient :math:`K_x\in[0,1]` as defined in :eq:`xpd`. Only relevant if ``scattering_coefficient`` is not equal to zero. Ignored if ``props`` is provided.
    :param scattering_pattern: Scattering pattern to use for diffuse reflection. Only relevant if ``scattering_coefficient`` is not equal to zero. Ignored if ``props`` is provided. Defaults to :func:`~sionna.rt.lambertian_pattern`.
    :param color: RGB (red, green, blue) color for the radio material as displayed in the previewer and renderer. Each RGB component must have a value within the range :math:`[0,1]`. If set to :py:class:`None`, then a random color is used.
    :param props: Mitsuba container storing the material properties, and used when loading a scene to initialize the radio material.
    """

    # ITU material colors
    ITU_MATERIAL_COLORS = {
        "marble": (0.701, 0.644, 0.485),
        "concrete": (0.539, 0.539, 0.539),
        "wood": (0.266, 0.109, 0.060),
        "metal": (0.220, 0.220, 0.254),
        "brick": (0.402, 0.112, 0.087),
        "glass": (0.168, 0.139, 0.509),
        "floorboard": (0.539, 0.386, 0.025),
        "ceiling_board": (0.376, 0.539, 0.117),
        "chipboard": (0.509, 0.159, 0.323),
        "plasterboard": (0.051, 0.539, 0.133),
        "plywood": (0.136, 0.076, 0.539),
        "clear_acrylic": (0.8, 0.9, 0.95),
        "vinyl_tile": (0.75, 0.75, 0.72),
        "carpet_tile": (0.32, 0.35, 0.42),
        "asphalt_concrete": (0.18, 0.18, 0.18),
        "very_dry_ground": (0.539, 0.319, 0.223),
        "medium_dry_ground": (0.539, 0.181, 0.076),
        "wet_ground": (0.539, 0.027, 0.147)
    }

    # pylint: disable=line-too-long
    def __init__(
        self,
        name: str | None = None,
        itu_type: str | None = None,
        thickness: float | mi.Float | None = None,
        scattering_coefficient: float | mi.Float = 0.0,
        xpd_coefficient: float | mi.Float = 0.0,
        scattering_pattern: Callable[[mi.Vector3f, mi.Vector3f, ...], mi.Float] | None = None,
        color: Tuple[float, float, float] | None = None,
        props: mi.Properties | None = None,
        **kwargs):

        has_props = props is not None
        if has_props:
            direct_args_none = (
                (name is None) and (itu_type is None) and (thickness is None)
                and (scattering_coefficient == 0.0) and (xpd_coefficient == 0.0)
            )
            if not direct_args_none:
                raise ValueError(
                    "When providing a `props` dictionary, not argument other"
                    " than `scattering_pattern` and `color` should be provided."
                )
            if 'type' not in props:
                raise ValueError(
                    "Missing property \"type\" (string) to select the ITU"
                    " material type."
                )
            itu_type = props['type']
            del props['type']

        if itu_type not in ITU_MATERIALS_PROPERTIES:
            raise ValueError(f"Invalid ITU material type \"{itu_type}\"")
        self._itu_type = itu_type

        # Order of priority to set the visual color of this ITU material:
        # 1. `color` keyword argument
        # 2. `color`, `reflectance` or `base_color` property specified in the
        #    props (scene dictionary or XML file).
        # 3. Default color from `ITU_MATERIAL_COLORS`.
        # 4. Set color to :py:class:`None`, which results in a random color being used.
        if color is None:
            if has_props:
                for pname in ("color", "reflectance", "base_color"):
                    if pname in props:
                        color = tuple(props[pname])
                        del props[pname]
                        break
            if color is None:
                color = ITURadioMaterial.ITU_MATERIAL_COLORS.get(itu_type, None)  # Color is allowed to be left unspecified (e.g., for custom user-defined ITU materials)

        if color is not None and has_props:
            props["color"] = mi.ScalarColor3f(color)

        # Frequency update callback
        def cb(f: float):
            return itu_material(itu_type, f)

        if has_props:
            super().__init__(scattering_pattern=scattering_pattern,
                             frequency_update_callback=cb,
                             props=props,
                             **kwargs)
        else:
            super().__init__(name=name,
                             thickness=thickness,
                             scattering_coefficient=scattering_coefficient,
                             xpd_coefficient=xpd_coefficient,
                             scattering_pattern=scattering_pattern,
                             frequency_update_callback=cb,
                             color=color,
                             **kwargs)

    @property
    def itu_type(self):
        r"""
        Get the ITU type

        :type: :py:class:`str`
        """
        return self._itu_type

    def clone(self,
              name: str | None = None,
              **overrides) -> "ITURadioMaterial":
        r"""
        Returns a new :class:`ITURadioMaterial`, identical to this one except
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

        See :meth:`~sionna.rt.RadioMaterialBase.clone` for details.

        :param name: Optional new name for the cloned material.
        :param overrides: Keyword arguments specifying properties to override.

        :return: New :class:`ITURadioMaterial` with the specified overrides.
        """
        kwargs = {
            "name": name or self.name,
            "itu_type": self.itu_type,
            "thickness": self.thickness,
            "scattering_coefficient": self.scattering_coefficient,
            "xpd_coefficient": self.xpd_coefficient,
            "color": self.color,
        }
        sp_override = overrides.pop("scattering_pattern", None)
        kwargs.update(overrides)
        new = ITURadioMaterial(**kwargs)
        if sp_override is not None:
            if isinstance(sp_override, str):
                factory = scattering_pattern_registry.get(sp_override)
                sp_override = factory()
            new.scattering_pattern = sp_override
        else:
            new.scattering_pattern = self.scattering_pattern
        return new

    def to_string(self) -> str:
        r"""
        Returns a string describing the object
        """
        s = f"ITURadioMaterial type={self._itu_type}\n"\
            f"                 eta_r={self._eta_r[0]:.3f}\n"\
            f"                 sigma={self._sigma[0]:.3f}\n"\
            f"                 thickness={self._d[0]:.3f}\n"\
            f"                 scattering_coefficient={self._s[0]:.3f}\n"\
            f"                 xpd_coefficient={self._kx[0]:.3f}"
        return s


mi.register_bsdf("itu-radio-material",
                 lambda props: ITURadioMaterial(props=props))


def register_itu_radio_material(
    name: str,
    parameters: Mapping[tuple[float, float], tuple[float, float, float, float]],
    color: tuple[float, float, float] | None = None
) -> None:
    # pylint: disable=line-too-long
    r"""
    Registers a custom ITU radio material or updates an existing ITU material definition.

    :param name: Name of the ITU radio material to register.
    :param parameters: A mapping of frequency ranges in GHz ``(f_min, f_max)`` to tuples of ITU parameters ``(a, b, c, d)`` as defined in recommendation ITU-R P.2040.
    :param color: Optional RGB (red, green, blue) color tuple for rendering/previewing, where each component is in :math:`[0, 1]`. If set to :py:class:`None`, then a random color is used.
    """
    ITU_MATERIALS_PROPERTIES[name] = dict(parameters)
    if color is not None:
        ITURadioMaterial.ITU_MATERIAL_COLORS[name] = color
