"""Sphinx configuration for the skysurvey documentation."""
import datetime
import os
import sys

# Document the local source tree (on readthedocs the package is also pip-installed).
sys.path.insert(0, os.path.abspath('../src'))

import skysurvey # noqa: E402

# -- Project information -----------------------------------------------------

project = "skysurvey"
author = "Mickael Rigault"
copyright = f"2022-{datetime.date.today().year}, Mickael Rigault"
release = skysurvey.__version__
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'matplotlib.sphinxext.plot_directive',
    'myst_nb',
    'sphinx_copybutton',
    ]

# Notebooks are stored executed: they are never run at build time.
nb_execution_mode = "off"

# MyST (markdown cells of notebooks and .md pages)
myst_enable_extensions = [
    "amsmath",     # allows \begin{equation} ... \end{equation}
    "dollarmath",  # allows $ ... $ and $$ ... $$ math
    "colon_fence", # allows ::: fenced directives (e.g. :::{note})
    "deflist",
]
myst_heading_anchors = 3

# -- API documentation -------------------------------------------------------

autosummary_generate = True        # one page per object listed in docs/api/*.rst
autosummary_imported_members = True
autoclass_content = "class"        # only use the class docstring, ignore __init__
autodoc_member_order = "bysource"
autodoc_typehints = "none"

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True           # "Attributes" sections rendered as field lists
napoleon_use_rtype = False

# Prefix section labels with the document path to avoid conflicts between notebooks sharing the same section
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 1

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'sncosmo': ('https://sncosmo.readthedocs.io/en/stable/', None),
    'shapely': ('https://shapely.readthedocs.io/en/stable/', None),
    'geopandas': ('https://geopandas.org/en/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
                    '**.ipynb_checkpoints',
                    '**/*-Copy*.ipynb', 'Untitled*.ipynb',
                    'gallery/*.key']

source_suffix = {'.rst': 'restructuredtext',
                 '.ipynb': 'myst-nb',
                 '.md': 'myst-nb'}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = f"skysurvey {version}"
html_logo = '_static/skysurvey_logo.png'
html_favicon = '_static/favicon.png'
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme_options = {
    'show_toc_level': 2,
    'home_page_in_toc': True,
    'navigation_with_keys': False,
    'repository_url': 'https://github.com/MickaelRigault/skysurvey',
    'repository_branch': 'main',
    'path_to_docs': 'docs',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_edit_page_button': True,
    'use_download_button': True,
    'launch_buttons': {'colab_url': 'https://colab.research.google.com'},
}
