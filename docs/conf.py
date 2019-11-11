import time
import warnings
from empymod import __version__
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

# ==== 1. Extensions  ====

# Load extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    # 'sphinx.ext.intersphinx',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
]

# Numpydoc settings
numpydoc_show_class_members = False
# numfig = True
# numfig_format = {'figure': 'Figure %s:'}

# Todo settings
todo_include_todos = True

# Sphinx gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'examples',
    'subsection_order': ExplicitOrder([
        '../examples/frequency_domain',
        '../examples/time_domain',
        '../examples/comparisons',
        '../examples/add_ons',
        '../examples/reproducing',
        '../examples/educational',
        '../examples/published',
        ]),
    'capture_repr': ('_repr_html_', '__repr__'),
    # Patter to search for example files
    "filename_pattern": r"\.py",
    # Sort gallery example by file name instead of number of lines (default)
    "within_subsection_order": FileNameSortKey,
}

# https://github.com/sphinx-gallery/sphinx-gallery/pull/521/files
# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')

# Intersphinx configuration
# intersphinx_mapping = {
#     "numpy": ("https://docs.scipy.org/doc/numpy/", None),
#     "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
# }

# ==== 2. General Settings ====
description = 'A multigrid solver for 3D electromagnetic diffusion.'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'friendly'

# The templates path.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'empymod'
copyright = u'2016-{}, The empymod Developers.'.format(time.strftime("%Y"))
author = 'The empymod Developers'

# |version| and |today| tags (|release|-tag is not used).
version = __version__
release = __version__
today_fmt = '%d %B %Y'

# List of patterns to ignore, relative to source directory.
exclude_patterns = ['_build', 'PermissionToRelicenseFilters.txt',
                    'LaTeX', '../tests']

# ==== 3. HTML settings ====
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'both',
}
html_static_path = ['_static']
html_logo = '_static/logo-empymod-plain.png'
html_favicon = '_static/favicon.ico'
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
    ]
}

html_context = {
    'menu_links_name': 'Links',
    'menu_links': [
        ('<i class="fa fa-link fa-fw"></i> Website',
         'https://empymod.github.io'),
        ('<i class="fa fa-github fa-fw"></i> Source Code',
         'https://github.com/empymod/empymod'),
    ],
}

htmlhelp_basename = 'empymoddoc'


# -- CSS fixes --
def setup(app):
    app.add_stylesheet("style.css")


# ==== 4. Other Document Type Settings ====
# Options for LaTeX output
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
}
latex_documents = [
    (master_doc, 'empymod.tex', 'empymod Documentation',
     'The empymod Developers', 'manual'),
]

# Options for manual page output
man_pages = [
    (master_doc, 'empymod', 'empymod Documentation',
     [author], 1)
]

# Options for Texinfo output
texinfo_documents = [
    (master_doc, 'empymod', 'empymod Documentation',
     author, 'empymod', description,
     'Electromagnetic geophysical modelling'),
]


# ==== 5. linkcheck ====

# software.seg.org results in "403 Client Error: Forbidden for url" on Travis.
linkcheck_ignore = [
    'https://software.seg.org',
]
