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

# :: napoleon
napoleon_use_param = True

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
\newcommand{\ipdeNormal}{\ensuremath{\hat{n_r}}}

\newcommand{\pdeState}{\ensuremath{\vb{q}}}
\newcommand{\ipdeState}{\ensuremath{q_p}}

\newcommand{\pdeConvective}{\ensuremath{\vb{F}\qty(\pdeState)}}
\newcommand{\ipdeConvective}{\ensuremath{F_{pr}\qty(\ipdeState)}}

\newcommand{\pdeNonConservativeMultiplier}{\ensuremath{\vb{B}\qty(\pdeState)}}
\newcommand{\ipdeNonConservativeMultiplier}{%
    \ensuremath{B_{pqr}\qty(\ipdeState)}
}

\newcommand{\pdeGradient}{\ensuremath{\gradient{\pdeState}}}
\newcommand{\ipdeGradient}{\ensuremath{\pdv{\ipdeState}{x_r}}}

\newcommand{\pdeSource}{\ensuremath{\vb{s}\qty(\pdeState)}}
\newcommand{\ipdeSource}{\ensuremath{s_p\qty(\ipdeState)}}

\newcommand{\pdeTermList}{\ensuremath{%
         \pdeConvective, \pdeNonConservativeMultiplier, \pdeSource
}}

\newcommand{\pdeFull}{\ensuremath{%
    \pdv{\pdeState}{t} + \divergence{\pdeConvective} +
    \pdeNonConservativeMultiplier \cdot \pdeGradient = \pdeSource \\
    \pdv{\ipdeState}{t} + \pdv{\ipdeConvective}{x_r} +
    \ipdeNonConservativeMultiplier \ipdeGradient = \ipdeSource
}}

\newcommand{\numConvective}{%
    \qty|\vb{F} \cdot \hat{\vb{n}}|_f S_f
}

\newcommand{\numConvectiveFaces}{%
    \sum_{f \in \text{faces}} \numConvective
}

\newcommand{\numConvectiveFull}{%
    \int_{V_i} \div{\pdeConvective} \dd{V} =
        \oint_{\partial V_i} \pdeConvective \cdot \pdeNormal \dd{S}
        \approx \numConvectiveFaces
}

\newcommand{\numSource}{%
    \expval{\pdeSource}_{V_i} V_i
}

\newcommand{\numSourceFull}{%
    \int_{V_i} \pdeSource \dd{V} \approx \numSource
}

\newcommand{\numNonConservative}{%
    \qty|\pdeState \hat{\vb{n}}|_f S_f
}

\newcommand{\numNonConservativeFaces}{%
    \sum_{f \in \text{faces}} \numNonConservative
}

\newcommand{\numPreMultipliedNonConservativeFaces}{%
    \qty|\pdeNonConservativeMultiplier|_i \cdot \numNonConservativeFaces
}

\newcommand{\numNonConservativeFull}{%
    \int_{V_i} \pdeNonConservativeMultiplier \cdot \pdeGradient \dd{V} \approx
    \oint_{\partial V_i} \pdeNonConservativeMultiplier \cdot \pdeState
    \dd{S} \approx \numPreMultipliedNonConservativeFaces
}

\newcommand{\numSpaceTerms}{%
    \numConvectiveFaces, \numPreMultipliedNonConservativeFaces, \numSource
}

\newcommand{\numTime}{%
    \int_t \vb{f}\qty(\pdeState^*,\pdeGradient^*) \dd{t}
}

\newcommand{\numTimeFull}{%
    \pdeState^{k+1} = \pdeState^k + \numTime
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
