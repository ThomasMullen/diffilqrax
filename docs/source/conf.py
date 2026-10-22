# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../.."))

project = 'diffilqrax'
copyright = '2024-2026, Thomas Soares Mullen & Marine Schimel'
author = 'Thomas Soares Mullen & Marine Schimel'
release = (Path(__file__).resolve().parents[2] / "diffilqrax" / "VERSION").read_text().strip()
version = release


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# -- Load extensions ------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    # "sphinx.ext.imgmath",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "sphinx_gallery.load_style",
    # "IPython.sphinxext.ipython_console_highlighting",  # lowercase didn't work
    # "numpydoc",
    # "myst_nb",
    "sphinxcontrib.bibtex",
    "sphinx_design",
    # "sphinx_design_elements",
    # "sphinx.ext.autosectionlabel",
]

bibtex_bibfiles = ["refs.bib"]          # required, path relative to conf.py
bibtex_default_style = "unsrt"          # or "plain", "alpha"
bibtex_reference_style = "author_year"  # or "label" for [1]-style numbers

templates_path = ['_templates']
exclude_patterns = ["build", "**.ipynb_checkpoints"]

# notebooks are stored with their outputs; the docs build never executes them
nbsphinx_execute = "never"

autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True  # NamedTuple fields are already documented by autodoc

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

html_theme_options = {
   "logo": {
        "text": "Diffilqrax documentation",
        "image_dark": "./_static/images/diffilqrax_logo_dm2_short2.png",
        "image_light": "./_static/images/diffilqrax_logo_dm2_short.png",
   },
   "icon_links": [
       {
           "name": "GitHub",
           "url": "https://github.com/ThomasMullen/diffilqrax",
           "icon": "fa-brands fa-github",
       }
   ],
   # Navigation bar links
    # "navbar_links": [
    #     {"name": "Getting started", "url": "userguide/getting_started"},
    #     {"name": "Tutorials", "url": "examples/index"},
    #     {"name": "Design principle", "url": "userguide/design_principles"},
    #     {"name": "API", "url": "api/modules"},
    # ],
}

