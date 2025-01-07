# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'diffilqrax'
copyright = '2024, Thomas Soares Mullen & Marine Schimel'
author = 'Thomas Soares Mullen & Marine Schimel'

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
    # "sphinxcontrib.bibtex",
    "sphinx_design",
    # "sphinx_design_elements",
    # "sphinx.ext.autosectionlabel",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

html_theme_options = {
   "logo": {
        "text": "Diffilqrax documentation",
        "image_dark": "./_static/images/diffilqrax_logo_dm2_short2.png",
        "image_light": "./_static/images/diffilqrax_logo_dm2_short.png",
   }
}
