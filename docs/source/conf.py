from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "Reservoir Benchmark"
author = "Yafei Wang"
copyright = f"{date.today().year}, {author}"
release = "0.1.0"

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
]

autosectionlabel_prefix_document = True
templates_path = ["_templates"]
exclude_patterns = []

language = "en"

try:
    import sphinx_rtd_theme  # noqa: F401

    html_theme = "sphinx_rtd_theme"
    html_theme_options = {
        "navigation_depth": 4,
        "collapse_navigation": False,
    }
except Exception:
    html_theme = "alabaster"
    html_theme_options = {}

html_title = "Reservoir Benchmark Documentation"
html_show_sourcelink = False
