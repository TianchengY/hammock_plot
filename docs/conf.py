from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "gallery"))

project = "hammock_plot"
author = "Tiancheng Yang"

extensions = [
    "sphinx_gallery.gen_gallery",
]

html_theme = "alabaster"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

sphinx_gallery_conf = {
    "examples_dirs": str(ROOT / "examples" / "gallery"),
    "gallery_dirs": "gallery",
    "filename_pattern": r".*\.py",
    "ignore_pattern": r"_gallery_utils\.py",
    "download_all_examples": False,
    "remove_config_comments": True,
}
