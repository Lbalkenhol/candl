# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(-1, os.path.abspath(f'../candl'))

#import os
#import sys
#sys.path.insert(0, os.path.abspath('./../..'))
#sys.path.insert(0, os.path.abspath('./..'))
#sys.path.insert(0, os.path.abspath('./../candl'))
#sys.path.insert(0, os.path.abspath('./candl'))
#sys.path.insert(0, os.path.abspath("."))
#sys.path.insert(0, '/home/docs/checkouts/readthedocs.org/user_builds/candl')

#print("WHAT EVEN IS THE PATH")
#print(sys.path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "candl"
copyright = "2023, Lennart Balkenhol"
author = "Lennart Balkenhol"
release = "2023"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx_collapse",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "numpydoc",
]

# Things we need for the docs to render, but don't actually need to import
autodoc_mock_imports = ["camb", "cobaya"]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "logos/candl_wordmark&symbol_stacked_col_RGB.png"
