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
import os
import re
import sys

# The package lives in the "src" layout, i.e. "src" is a container rather than a package
# and "biobox" sits inside it. Putting "src" on sys.path lets the documentation build
# without the package having been installed first; if it has been installed
# ("pip install -e ."), this line is harmless.
REPO_ROOT = os.path.abspath('../..')
SRC_DIR = os.path.join(REPO_ROOT, 'src')

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# -- Project information -----------------------------------------------------

project = 'biobox'
copyright = '2014-2026, M. T. Degiacomi'
author = 'M. T. Degiacomi'

# The full version, including alpha/beta/rc tags, read from the package so that the
# two cannot drift apart
with open(os.path.join(SRC_DIR, 'biobox', '__init__.py')) as handle:
    release = re.search(r"__version__\s*=\s*'([^']+)'", handle.read()).group(1)
version = release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

# Render ".. todo::" notes in the docstrings rather than silently dropping them, so that
# they stay visible to whoever picks the work up.
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinxdoc'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'bioboxdoc'

# -- Options for Texinfo output -------------------------------------------
add_module_names = False
autoclass_content = "both"

# biobox's compiled Cython extensions. Read the Docs installs the requirements in
# docs/requirements.txt but does not build the package, so these modules do not exist
# there. Mocking them lets autodoc import every module of the package: none of them is
# called at import time or used as a default argument, so no mock object reaches a
# rendered signature. The packages that are imported at module load and cannot be
# mocked are listed in docs/requirements.txt instead.
autodoc_mock_imports = [
    "biobox.lib.fastmath",
    "biobox.lib.graph",
    "biobox.lib.e_density",
]
