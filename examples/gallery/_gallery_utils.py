from pathlib import Path
import sys


def repo_root():
    starts = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    for start in starts:
        for path in (start, *start.parents):
            if (path / "data").is_dir() and (path / "hammock_plot").is_dir():
                return path
    raise RuntimeError("Could not locate the hammock_plot repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def data_path(filename):
    return ROOT / "data" / filename


def gallery_image_path(filename):
    return ROOT / "image" / "gallery" / filename
