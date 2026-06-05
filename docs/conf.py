# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import importlib.machinery
import sys
import types
from datetime import datetime

try:
    import tomllib
except ImportError:
    import tomli as tomllib
from pathlib import Path


class _MockMeta(type):
    """Metaclass for mock classes — supports the operations doc-time code
    performs on classes: ``X | Y`` unions, subscripting, attribute access,
    and decorator-style calls."""

    def __or__(cls, other):
        return cls

    def __ror__(cls, other):
        return cls

    def __getitem__(cls, item):
        return cls

    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _mock_attr(f"{cls.__name__}.{name}")

    def __call__(cls, *args, **kwargs):
        # Decorator usage (@mock) → return the decorated object unchanged.
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return cls


_attr_cache = {}


def _mock_attr(qualname):
    """Return a cached mock *class* standing in for an attribute (a class,
    function, or constant) of a mocked module.

    Returning a real class is what keeps ``issubclass``/``isinstance`` checks
    working — e.g. scipy's ``issubclass(x, torch.Tensor)`` torch-detection at
    import time, and use of ``torch.nn.Module`` as a base class.
    """
    if qualname not in _attr_cache:
        _attr_cache[qualname] = _MockMeta(qualname, (), {})
    return _attr_cache[qualname]


class _MockModule(types.ModuleType):
    """Mock module that satisfies Python's import system for doc builds.

    Registered submodules (see ``_MOCKED_MODULES``) resolve to their module
    objects so submodule imports (e.g. ``torch.utils.data`` from inside
    anndata) work; any other attribute resolves to a mock *class* via
    ``_mock_attr``.
    """

    def __init__(self, name):
        super().__init__(name)
        self.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        self.__path__ = []
        self.__package__ = name

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        child_name = f"{self.__name__}.{name}"
        if child_name in sys.modules:
            value = sys.modules[child_name]
        else:
            value = _mock_attr(child_name)
        self.__dict__[name] = value
        return value

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return _mock_attr(self.__name__)


# Pre-populate sys.modules so both autosummary and autodoc see mocked deps
# before any import can trigger them.
_MOCKED_MODULES = [
    "torch", "torch.nn", "torch.nn.functional",
    "torch.distributions", "torch.optim", "torch.optim.lr_scheduler",
    "torch.utils", "torch.utils.data",  # anndata 0.11+ imports these unconditionally
    "torch_geometric", "torch_geometric.data", "torch_geometric.loader",
    "torch_geometric.nn", "torch_geometric.utils",
    "torch_sparse",
    "torch_scatter",
    "scvi", "scvi.data", "scvi.data.fields", "scvi.dataloaders",
    "scvi.distributions", "scvi.model", "scvi.model._utils",
    "scvi.model.base", "scvi.module", "scvi.module._classifier",
    "scvi.module.base", "scvi.nn", "scvi.train", "scvi.utils",
]
for _mod_name in _MOCKED_MODULES:
    sys.modules[_mod_name] = _MockModule(_mod_name)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))
sys.path.insert(0, str(HERE.parent / "src"))


# -- Project information -----------------------------------------------------

with open(HERE.parent / "pyproject.toml", "rb") as f:
    _pyproject = tomllib.load(f)
project_name = _pyproject["project"]["name"]
version = _pyproject["project"]["version"]
author = ", ".join(a.get("name", "") for a in _pyproject["project"].get("authors", []))
copyright = f"{datetime.now():%Y}, {author}."
repository_url = f"https://github.com/PMBio/{project_name}"

# The full version, including alpha/beta/rc tags
release = version

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = False
needs_sphinx = "4.0"

# The tutorial renders a page-local bibliography ({bibliography} filtered by
# docname) while references.md renders the global ``:cited:`` list, so each key
# is intentionally emitted by two bibliography directives — silence the
# resulting duplicate-citation warnings.
suppress_warnings = ["bibtex.duplicate_citation"]

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "PMBio",  # Username
    "github_repo": project_name,  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]


autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "torch_sparse",
    "torch_scatter",
    "scvi",
]

autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
myst_heading_anchors = 3  # create anchors for h1-h3
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = project_name
html_logo = "logo.svg"

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
}

pygments_style = "default"

nitpick_ignore = [
    # If building the documentation fails because of a missing link that is outside your control,
    # you can add an exception to this list.
    #     ("py:class", "igraph.Graph"),
]


def setup(app):
    """App setup hook."""
    app.add_config_value(
        "recommonmark_config",
        {
            "auto_toc_tree_section": "Contents",
            "enable_auto_toc_tree": True,
            "enable_math": True,
            "enable_inline_math": False,
            "enable_eval_rst": True,
        },
        True,
    )
