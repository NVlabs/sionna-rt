#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import ast
import os
import sys

from sphinx.ext.intersphinx import missing_reference as intersphinx_missing_reference

sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "Sionna RT"
copyright = "2021-2026 NVIDIA CORPORATION"

# Read version number from sionna/rt/__init__.py without executing the module
_rt_init_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../src/sionna/rt/__init__.py")
)
with open(_rt_init_path, encoding="utf-8") as _f:
    _tree = ast.parse(_f.read(), filename=_rt_init_path)
release = None
for _node in _tree.body:
    if isinstance(_node, ast.Assign):
        for _target in _node.targets:
            if isinstance(_target, ast.Name) and _target.id == "__version__":
                release = ast.literal_eval(_node.value)
                break
    if release is not None:
        break
if release is None:
    raise ValueError(f"Could not find __version__ in {_rt_init_path}")
version = release

# -- General configuration ---------------------------------------------------

#import sphinx_rtd_theme
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "pydata_sphinx_theme",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "_ext.list_registry",
    "_ext.itu_materials_table",
]
nbsphinx_execute = 'never'

# Internal helper types referenced from public signatures but not documented
# on the RT API pages.
nitpick_ignore = [
    ("py:class", "sionna.rt.path_solvers.paths_buffer.PathsBuffer"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "mitsuba": ("https://mitsuba.readthedocs.io/en/latest/", None),
    "drjit": ("https://drjit.readthedocs.io/en/latest/", None),
}

# Docstrings refer to Mitsuba types through the conventional ``mi`` alias, which
# is not the name under which they are published. Retry such references against
# the Mitsuba inventory using the fully qualified name.
_MITSUBA_ALIAS = "mi."


def _resolve_mitsuba_alias(app, env, node, contnode):
    target = node.get("reftarget", "")
    if not target.startswith(_MITSUBA_ALIAS):
        return None
    node["reftarget"] = "mitsuba." + target[len(_MITSUBA_ALIAS):]
    return intersphinx_missing_reference(app, env, node, contnode)


def setup(app):
    app.connect("missing-reference", _resolve_mitsuba_alias)


# -- sphinx_autodoc_typehints ----------------------------------------------------

autodoc_typehints = "description"
typehints_fully_qualified = True
simplify_optional_unions = True


# -- sphinxcontrib-bibtex ----------------------------------------------------

bibtex_bibfiles = ["rt.bib"]
# bibtex_default_style = "plain_keylabel"
bibtex_reference_style = "label"
bibtex_tooltips = True


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

base_path = os.environ.get("BASE_PATH", "/sionna-rt")
html_baseurl = f"{base_path}/"  # Base URL for version switcher absolute paths


html_theme_options = {
    "logo": {
        "text": "Sionna",
    },
    "header_links_before_dropdown": 5,
    "switcher": {
        "json_url": f"{base_path}/versions.json",
        "version_match": version,
    },
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
    "check_switcher": False,
    "show_version_warning_banner": True,
    "navbar_center": ["navbar-nav"],
    "navbar_end": [
        "version-switcher",
        "navbar-icon-links",
        "theme-switcher",
    ],
    "search_bar_text": "Search...",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVlabs/sionna",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/sionna/",
            "icon": "fa-custom fa-pypi",
        },
    ],
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "navigation_with_keys": False,
    "show_nav_level": 1,
    "navigation_depth": 5,
    "show_toc_level": 2,
    "navbar_align": "left",
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
