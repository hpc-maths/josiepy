# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "josiepy"
copyright = "2020, Ruben Di Battista"
author = "Ruben Di Battista"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.imgmath",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",
    "recommonmark",
    "sphinx_markdown_tables",
    "sphinx_rtd_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# -- Extensions Configuration -------------------------------------------------

# :: sphinx_autodoc_typehints
set_type_checking_flag = False

# :: sphinx.ext.imgmath
imgmath_image_format = "svg"
imgmath_latex_preamble = r"""
\usepackage{physics}
"""

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# # Substitutions
# rst_epilog = r"""
#  .. |pde| replace :: :math:`\hat{n}`
#
#  .. |pde_normal| replace :: :math:`\hat{\vb{n}}`
#
#  .. |pde_state| replace :: \vb{q}
#
#  .. |pde_convective_flux| replace :: \vb{F}\qty(|state|)
#
#  .. |pde_non_conservative_multiplier| replace :: \vb{B}\qty(|state|)
#
#  .. |pde_gradient| replace :: \gradient{|state|}
#
#  .. |pde_source| replace :: \vb{s}\qty(|state|)
#
#  .. |pde_term_list| replace ::
#     * :math:`|pde_convective_flux|`
#     * :math`|pde_non_conservative_multiplier|`
#     * :math:`|pde_source|`
#
#  .. |pde_full| replace :: \pdv{|pde_state|}{t} + \div{|pde_convective_flux| +
#     |pde_non_conservative_multiplier| \cdot |pde_gradient| = |pde_source|
#
#  .. |scheme_convective_flux| replace :: \int_{V_i}
#     \divergence{|pde_convective_flux|} \dd{V}
#
#  .. |scheme_convective_flux_S| replace :: \oint{\partial V_i}
#     |pde_convective_flux| \cdot |pde_normal| \dd{S}
#
#  .. |scheme_gradient| replace :: \oint{\partial V_i} |pde_state||pde_normal|
#     \dd{S}
#
#  .. |scheme_source| replace :: \int_{V_i} |pde_source| \dd{V}
#
#  .. |scheme_terms| replace ::
#     * :math:`|scheme_convective_flux| = |scheme_convective_flux_S|`
#     * :math:`|scheme_gradient|`
#     * :math:`|scheme_source|`
#
# """
