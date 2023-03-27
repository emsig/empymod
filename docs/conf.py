import time
import warnings
from empymod import __version__
from sphinx_gallery.sorting import FileNameSortKey

# ==== 1. Extensions  ====

# Load extensions
extensions = [
    # 'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx_design',
    'sphinx.ext.intersphinx',
    # 'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx_gallery.gen_gallery',
    'sphinx_automodapi.automodapi',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
]
autosummary_generate = True
add_module_names = True
add_function_parentheses = False

# Numpydoc settings
numpydoc_show_class_members = False
# numfig = True
# numfig_format = {'figure': 'Figure %s:'}
# Make numpydoc to generate plots for example sections
numpydoc_use_plots = True

# Todo settings
todo_include_todos = True

# Sphinx gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': [
        '../examples/frequency_domain',
        '../examples/time_domain',
        '../examples/educational',
        '../examples/reproducing',
        '../examples/published',
        ],
    'gallery_dirs': [
        'gallery/fdomain',
        'gallery/tdomain',
        'gallery/educational',
        'gallery/reproducing',
        'gallery/published',
        ],
    'capture_repr': ('_repr_html_', '__repr__'),
    # Patter to search for example files
    "filename_pattern": r"\.py",
    # Sort gallery example by file name instead of number of lines (default)
    "within_subsection_order": FileNameSortKey,
    # Remove the settings (e.g., sphinx_gallery_thumbnail_number)
    'remove_config_comments': True,
    # Show memory
    'show_memory': True,
    # Custom first notebook cell
    'first_notebook_cell': '%matplotlib notebook',
}

# https://github.com/sphinx-gallery/sphinx-gallery/pull/521/files
# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')

# Intersphinx configuration
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# ==== 2. General Settings ====
description = '3D EM modeller for 1D VTI media.'

# The templates path.
# templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'empymod'
author = 'The emsig community'
copyright = f'2016-{time.strftime("%Y")}, {author}'

# |version| and |today| tags (|release|-tag is not used).
version = __version__
release = __version__
today_fmt = '%d %B %Y'

# List of patterns to ignore, relative to source directory.
exclude_patterns = ['_build', 'PermissionToRelicenseFilters.txt',
                    'LaTeX', '../tests']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'friendly'

# ==== 3. HTML settings ====
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_logo = '_static/empymod-logo.svg'
html_favicon = '_static/favicon.ico'

html_theme_options = {
    "github_url": "https://github.com/emsig/empymod",
    "external_links": [
        {"name": "emsig", "url": "https://emsig.xyz"},
    ],
    # "use_edit_page_button": True,
}

html_context = {
    "github_user": "emsig",
    "github_repo": "empymod",
    "github_version": "main",
    "doc_path": "docs",
}

html_use_modindex = True
html_file_suffix = '.html'
htmlhelp_basename = 'empymod'
html_css_files = [
    "style.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/" +
    "css/font-awesome.min.css"
]

# ==== 4. linkcheck ====

linkcheck_ignore = [
    'https://doi.org/10.1046/j.1365-2478.1997.500292.x',
    'https://doi.org/10.1046/j.1365-8711.2000.03071.x',
    'https://doi.org/10.1111/j.1365-2478.2006.00555.x',
    'https://doi.org/10.1111/j.1365-2478.2006.00585.x',
    'https://doi.org/10.1111/j.1365-246X.1988.tb01131.x',
    'https://doi.org/10.1137/1.9780898719598.ch12',
    'https://doi.org/10.1190/1.9781560802631.ch4',
    'https://doi.org/10.1002/sapm19553411',
    'https://doi.org/10.1029/JB087iB07p05327',
    'https://doi.org/10.1093/gji/ggt251',
    'https://doi.org/10.1190/geo2013-0411.1',
    'https://doi.org/10.1190/geo2016-0626.1',
    'https://doi.org/10.1190/geo2018-0069.1',
    'https://doi.org/10.1190/tle36040352.1',

]
