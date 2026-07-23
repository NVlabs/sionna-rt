#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Material properties from Section 2.1.4 of recommendation ITU-R P2040"""

import drjit as dr
import mitsuba as mi
from typing import Tuple


# Data structure storing the properties from Table 3 of ITU-R P2040.
# A material can be mapped to multiple frequency ranges, each with a
# different set of parameters.
#
# Structure :
#   material_name: { (min_freq [GHz], max_freq [GHz]): (a, b, c, d) }
ITU_MATERIALS_PROPERTIES = {
    "vacuum"            :   { (0.001, 100.) :   (1.0, 0.0, 0.0, 0.0) },
    
    "concrete"          :   { (1., 100.)    :   (5.24, 0.0, 0.0462, 0.7822),
                              (110., 330.)  :   (5.17, 0.0, 0.0145, 1.0900) },

    "brick"             :   { (1., 40.)     :   (3.91, 0.0, 0.0238, 0.16),
                              (110., 330.)  :   (4.15, 0.0, 0.0006, 1.5712) },

    "plasterboard"      :   { (1., 100.)    :   (2.73, 0.0, 0.0085, 0.9395),
                              (110., 330.)  :   (2.56, 0.0, 0.0001, 1.7799),
                              (100., 400.)  :   (2.65, 0.0, 0.0002, 1.598) },

    "wood"              :   { (0.001, 100.) :   (1.99, 0.0, 0.0047, 1.0718),
                              (110., 330.)  :   (1.82, 0.0, 0.0040, 1.0761),
                              (100., 400.)  :   (2.1183, 0.0, 0.0055, 1.1113) },

    "glass"             :   { (0.1, 100.)   :   (6.31, 0.0, 0.0036, 1.3394),
                              (220., 450.)  :   (5.79, 0.0, 0.0004, 1.658),
                              (100., 400.)  :   (6.5767, 0.0, 0.0012, 1.4697) },

    "clear_acrylic"     :   { (110., 330.)  :   (2.58, 0.0, 0.0001, 1.6524) },

    "ceiling_board"     :   { (1.0, 100.)   :   (1.48, 0.0, 0.0011, 1.0750),
                              (220., 450.)  :   (1.52, 0.0, 0.0029, 1.029),
                              (100., 400.)  :   (1.2567, 0.0, 0.00013, 1.454) },

    "chipboard"         :   { (1.0, 100.)   :   (2.58, 0.0, 0.0217, 0.7800),
                              (100., 200.)  :   (2.16, 0.0, 0.0023, 1.359) },

    "plywood"           :   { (1.0, 40.)    :   (2.71, 0.0, 0.33, 0.0),
                              (110., 330.)  :   (1.94, 0.0, 0.0067, 0.9982),
                              (100., 400.)  :   (2.17, 0.0, 0.0063, 1.045) },

    "marble"            :   { (1.0, 60.)    :   (7.074, 0.0, 0.0055, 0.9262),
                              (110., 330.)  :   (7.94, 0.0, 0.0001, 1.7330),
                              (100., 400.)  :   (8.62, 0.0, 0.0027, 1.15) },

    "floorboard"        :   { (50., 100.)   :   (3.66, 0.0, 0.0044, 1.3515),
                              (220., 300.)  :   (5.27, 0.0, 2.22e-17, 7.3413),
                              (300., 400.)  :   (5.27, 0.0, 0.0003, 2.0298),
                              (400., 450.)  :   (5.27, 0.0, 49.8726, 0.0),
                              (100., 400.)  :   (3.1575, 0.0, 0.001675, 1.32775) },

    "vinyl_tile"        :   { (1.0, 40.)    :   (3.62, 0.0, 0.0051, 0.8422) },

    "carpet_tile"       :   { (1.0, 40.)    :   (2.08, 0.0, 0.0009, 0.8200) },

    "asphalt_concrete"  :   { (1.0, 40.)    :   (4.83, 0.0, 0.0108, 1.3969) },

    "metal"             :   { (1.0, 100.)   :   (1.0, 0.0, 1e7, 0.0) },

    "very_dry_ground"   :   { (1.0, 10.)    :   (3.0, 0.0, 0.00015, 2.52) },

    "medium_dry_ground" :   { (1.0, 10.)    :   (15., -0.1, 0.035, 1.63) },

    "wet_ground"        :   { (1.0, 10.)    :   (30., -0.4, 0.15, 1.30) }
}


def itu_material(name: str, f: mi.Float) -> Tuple[mi.Float, mi.Float]:
    r"""
    Evaluates the real component of the relative permittivity and the
    conductivity [S/m] of the ITU material `name` for the frequency `f` [Hz]

    Implements model from Section 2.1.4 of recommendation ITU-R P2040.

    :param name: Name of the ITU material to evaluate.
        Must be a key of `ITU_MATERIALS_PROPERTIES`.
    :param f: Frequency [Hz]

    :return: Real component of the relative permittivity and conductivity [S/m]
    """

    if name not in ITU_MATERIALS_PROPERTIES:
        raise ValueError(f"Unknown ITU material '{name}'")
    props = ITU_MATERIALS_PROPERTIES[name]

    f_ghz = f/1e9

    # Extract the properties to use according to the frequency
    # If the frequency is in none of the valid ranges, an exception is raised
    valid_freq = False
    for f_ranges, params in props.items():
        if f_ranges[0] <= f_ghz <= f_ranges[1]:
            a, b, c, d = params
            valid_freq = True
            break
    if not valid_freq:
        raise ValueError(f"Properties of ITU material '{name}' are not defined"
                         " for this frequency")

    # Evaluate the material properties
    eta_r = a * dr.power(f_ghz, b)
    sigma = c * dr.power(f_ghz, d)

    return eta_r, sigma
