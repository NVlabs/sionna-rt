#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Sionna Ray Tracing (RT) Package"""

# pylint: disable=wrong-import-position

__version__ = "1.1.0"

import importlib

import mitsuba as mi
if mi.variant() is None:
    try:
        mi.set_variant("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")
    except ImportError:
        mi.set_variant("llvm_ad_mono_polarized")

from .utils import *
from .antenna_pattern import AntennaPattern, \
                             PolarizedAntennaPattern, \
                             register_antenna_pattern, \
                             register_polarization, \
                             register_polarization_model
from .antenna_array import AntennaArray, PlanarArray
from .camera import Camera
from .radio_devices import Transmitter, Receiver, RadioDevice
from .scene import Scene, load_scene
from .radio_materials import *
from .constants import InteractionType,\
                       DEFAULT_FREQUENCY,\
                       DEFAULT_BANDWIDTH,\
                       DEFAULT_TEMPERATURE,\
                       INVALID_SHAPE,\
                       INVALID_PRIMITIVE
from .path_solvers import PathSolver, Paths
from .radio_map_solvers import RadioMapSolver, PlanarRadioMap, MeshRadioMap, DemRadioMap, RadioMap
from .preview import Previewer
from .scene_object import SceneObject
from .sliced_integrator import SlicedPathIntegrator, SlicedDepthIntegrator
from .twosided_area import TwosidedAreaEmitter

# Register the defined materials once a Mitsuba variant is set
def sionna_rt_variant_cb(old: str, new: str):
    # --- Radio materials
    if (new is not None) and "mono_polarized" in new:
        # TODO: reload all radio materials for the new variant.
        #       Probably requires not using `import *` above.
        pass

    # --- Sliced path tracer integrator
    # pylint: disable=import-outside-toplevel
    from . import sliced_integrator
    importlib.reload(sliced_integrator)
    global SlicedPathIntegrator, SlicedDepthIntegrator
    SlicedPathIntegrator = sliced_integrator.SlicedPathIntegrator
    SlicedDepthIntegrator = sliced_integrator.SlicedDepthIntegrator

    # --- Twosided area emitter
    # pylint: disable=import-outside-toplevel
    from . import twosided_area
    importlib.reload(twosided_area)
    global TwosidedAreaEmitter
    TwosidedAreaEmitter = twosided_area.TwosidedAreaEmitter

mi.detail.add_variant_callback(sionna_rt_variant_cb)
