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
import sys

print(sys.path)


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
imgmath_font_size = 14
imgmath_use_preview = True
imgmath_latex_preamble = r"""
\usepackage{physics}
\usepackage{lmodern}
\usepackage[T1]{fontenc}

\newcommand{\pdeNormal}{\ensuremath{\hat{\vb{n}}}}

\newcommand{\pdeState}{\ensuremath{\vb{q}}}

\newcommand{\pdeConvective}{\ensuremath{\vb{F}\qty(\pdeState)}}

\newcommand{\pdeNonConservativeMultiplier}{\ensuremath{\vb{B}\qty{\pdeState}}}

\newcommand{\pdeGradient}{\ensuremath{\gradient{\pdeState}}}

\newcommand{\pdeSource}{\ensuremath{\vb{s}\qty(\pdeState)}}

\newcommand{\pdeTermList}{\ensuremath{%
         \pdeConvective, \pdeNonConservativeMultiplier, \pdeSource
}}

\newcommand{\pdeFull}{\ensuremath{%
    \pdv{\pdeState}{t} + \divergence{\pdeConvective} +
    \pdeNonConservativeMultiplier \cdot \pdeGradient = \pdeSource
}}

\newcommand{\numConvective}{%
    \sum_{f \in \text{faces}} \qty|\vb{F} \cdot \hat{\vb{n}}|_f S_f
}

\newcommand{\numConvectiveFull}{%
    \int_{V_i} \div{\vb{F}\qty(\vb{q})} \dd{V} =
        \oint_{\partial V_i} \vb{F}\qty(\vb{q}) \cdot \hat{\vb{n}} \dd{S}
        \approx \numConvective
}


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
