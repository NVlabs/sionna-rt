#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "Sionna RT"
copyright = "2021-2026 NVIDIA CORPORATION"

# Read version number from sionna.__init__
from importlib.machinery import SourceFileLoader
release = SourceFileLoader("version",
                           "../../src/sionna/rt/__init__.py").load_module().__version__


# -- General configuration ---------------------------------------------------

#import sphinx_rtd_theme
extensions = ["sphinx_rtd_theme",
              "sphinx.ext.napoleon",
              "sphinx_autodoc_typehints",
              "sphinx.ext.viewcode",
              "sphinx.ext.mathjax",
              "sphinx_copybutton",
              "nbsphinx",
              "_ext.list_registry",
              "_ext.itu_materials_table",
             ]
autodoc_typehints = "description"
typehints_fully_qualified = True
simplify_optional_unions = True
nbsphinx_execute = 'never'

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": False,
    "navigation_depth": 5,
    }
html_show_sourcelink = False
pygments_style = "default"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ['css/sionna.css']

napoleon_custom_sections = [("Input shape", "params_style"),
                            ("Output shape", "params_style"),
                            ("Attributes", "params_style"),
                            ("Input", "params_style"),
                            ("Output", "params_style"),
                            ("Keyword Arguments", "params_style"),
                            ]
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_keyword = True
numfig = True
