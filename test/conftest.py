#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Pytest configuration for the entire project."""

import sys

import mitsuba as mi

sys.path.append("../src")

def pytest_addoption(parser):
    parser.addoption("--cpu", action="store_true", default=False, help="Run tests on CPU. Overrides --gpu setting.")


def pytest_configure(config):
    if config.getoption("cpu"):
        print("\n========================================\n")
        print("           Running tests on CPU")
        print("\n========================================\n")
        mi.set_variant("llvm_ad_mono_polarized")
    else:
        print("\n========================================\n")
        print("    Running tests on GPU if available")
        print("\n========================================\n")
        mi.set_variant("cuda_ad_mono_polarized", "metal_ad_mono_polarized", "llvm_ad_mono_polarized")
