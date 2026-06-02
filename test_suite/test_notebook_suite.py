import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


PIXEL_TOLERANCE = 5
MAX_DIFF_RATIO = 0.01


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
    bad_pixels = np.any(delta > PIXEL_TOLERANCE, axis=2)
    diff_ratio = bad_pixels.mean()

    if diff_ratio > MAX_DIFF_RATIO:
        diff = np.clip(delta * 8, 0, 255).astype(np.uint8)
        Image.fromarray(diff).save(diff_path)
        raise AssertionError(
            f"{actual_path.name} differs from expected image by "
            f"{diff_ratio:.2%}, above allowed {MAX_DIFF_RATIO:.2%}. "
            f"Diff image saved to {diff_path}"
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
