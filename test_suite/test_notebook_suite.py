import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


PIXEL_TOLERANCE = 5    # per-channel delta for a pixel to count as different
EDGE_TOLERANCE = 8     # local colour swing (in a 5x5 window) that marks an edge


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "test_suite" / "test_suite.ipynb"
EXPECTED_DIR = ROOT / "test_suite" / "expected_plots"
RESULT_DIR = ROOT / "test_suite" / "test_plots"
DIFF_DIR = ROOT / "test_suite" / "diff_plots"


def assert_error_check(plot_func_error, expected_msg=None):
    try:
        plot_func_error()
    except Exception as exc:
        error_message = str(exc)
        if expected_msg is not None and expected_msg != error_message:
            raise AssertionError(
                f"Expected error {expected_msg!r}, got {error_message!r}"
            ) from exc
        return

    raise AssertionError("Expected an exception, but no exception was raised")


def _edge_band(arr):
    """Pixels on or within ~2px of an anti-aliased edge (text, shape outlines)."""
    gray = arr.max(axis=2)
    swing = ndimage.maximum_filter(gray, 5) - ndimage.minimum_filter(gray, 5)
    return swing > EDGE_TOLERANCE


def compare_images(expected_path, actual_path, diff_path):
    expected = Image.open(expected_path).convert("RGB")
    actual = Image.open(actual_path).convert("RGB")

    if expected.size != actual.size:
        raise AssertionError(
            f"Image size changed for {actual_path.name}: "
            f"expected {expected.size}, got {actual.size}"
        )

    expected_arr = np.asarray(expected, dtype=np.int16)
    actual_arr = np.asarray(actual, dtype=np.int16)
    delta = np.abs(expected_arr - actual_arr)
    differing = np.any(delta > PIXEL_TOLERANCE, axis=2)

    # Text and shape outlines are anti-aliased differently on each OS, so those
    # edge pixels never match across platforms; solid fills are
    # pixel-identical. Ignore diffs on edges and flag only a sizeable diff in a
    # flat region. The threshold is ~1% of the image's mean side (12px floor).
    real = differing & ~_edge_band(expected_arr)
    labels, n = ndimage.label(real)
    largest = int(np.bincount(labels.ravel())[1:].max()) if n else 0
    min_blob = max(12, round((expected_arr.shape[0] * expected_arr.shape[1]) ** 0.5 / 100))

    if largest >= min_blob:
        diff = np.clip(delta * 8, 0, 255).astype(np.uint8)
        Image.fromarray(diff).save(diff_path)
        raise AssertionError(
            f"{actual_path.name} differs from expected in a flat (non-edge) "
            f"region: largest diff blob = {largest}px (allowed < {min_blob}px). "
            f"Diff saved to {diff_path}"
        )


def assert_expected_vs_actual(title, filename):
    expected_file = EXPECTED_DIR / filename
    actual_file = RESULT_DIR / filename
    diff_file = DIFF_DIR / filename

    if not expected_file.exists():
        raise AssertionError(f"Missing expected image: {expected_file}")
    if not actual_file.exists():
        raise AssertionError(f"Missing actual image for {title}: {actual_file}")

    compare_images(expected_file, actual_file, diff_file)


def iter_code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            yield index, "".join(cell.get("source", []))


def test_existing_notebook_suite(monkeypatch):
    os.environ.setdefault("MPLBACKEND", "Agg")
    RESULT_DIR.mkdir(exist_ok=True)
    DIFF_DIR.mkdir(exist_ok=True)

    monkeypatch.chdir(ROOT / "test_suite")

    namespace = {
        "__name__": "__notebook_test_suite__",
    }

    for index, source in iter_code_cells():
        exec(compile(source, f"{NOTEBOOK}:cell-{index}", "exec"), namespace)

        if source.lstrip().startswith("def run_error_check"):
            namespace["run_error_check"] = assert_error_check
        elif source.lstrip().startswith("def show_expected_vs_actual"):
            namespace["show_expected_vs_actual"] = assert_expected_vs_actual
