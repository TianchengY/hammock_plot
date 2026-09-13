import re
import colorsys
import warnings
import matplotlib.colors as mcolors
import pandas as pd
from typing import List, Dict, Any
import numpy as np

class Defaults:
    # General
    MISSING_PLACEHOLDER: str = "missing"

    # Colours
    COLORS: list = [ "#fdc086",  "#386cb0", "#7fc97f", "#f0027f"]
    DEFAULT_COLOR: str = "#beaed4"

    # COLORS: list = ["#e06a85", "#aa9000", "#00aa5a", "#00a6c9"]
    # DEFAULT_COLOR: str = "#b675df"

    # Layout
    UNI_VFILL: float = 0.08 # default unibar vertical fill
    CONNECTOR_FRACTION: float = 1 # default proportion fraction of connectors : vfill
    UNI_HFILL: float = 0.3 # default horizontal fill
    MIN_BAR_HEIGHT_UNIBAR: float = 0.15 # minimum drawn height of a unibar (and its colour segments)
    MIN_BAR_HEIGHT_CONNECTORS: float = 0.12 # minimum drawn thickness of a connector; slightly less than the unibar floor
    BAR_UNIT: float = 1.0 # default bar unit (how many pixels/obs.) is recalculated on init.
    XMARGIN: float = 0.02 # margin on x axis
    YMARGIN: float = 0.04 # margin on y axis
    SCALE: float = 10 #  width/height/location multiplier to get pixel values
    GAP_BTWN_UNI_MULTI: float = 2 # required gap between unibars and the multivariate connectors
    MIN_MULTI_WIDTH: float = 3 # in pixels
    SPACE_ABOVE_MISSING: float = 2 # space above missing values (separating missing from non-missing)
    NUM_LEVELS = 7 # default number of levels for a numeric variable
    ALPHA = 0.7 # alpha value of the colours
    WHITE_DIVIDER_HEIGHT = 0.3 # height of white divider when uni_vfill =1
    SPIKE_THICKNESS = 0.3 # width of spikes in spiky beanplot
    # Display types drawn as a point (not a bar): their connectors attach at the
    # value centre (parallel-coordinate fan) instead of stacking to fill a bar.
    CENTER_ATTACH_DISPLAYS = {"box", "violin"}


def clean_expression(expr: str) -> str:
    """
    Cleans up a logical expression string by inserting necessary spaces
    around logical operators, comparison operators, and parentheses.
    """
    expr = re.sub(r'(?i)(\w)(and|or|not)(\w)', r'\1 \2 \3', expr)
    expr = re.sub(r'(?i)(\W)(and|or|not)(\w)', r'\1 \2 \3', expr)
    expr = re.sub(r'(?i)(\w)(and|or|not)(\W)', r'\1 \2 \3', expr)
    expr = re.sub(r'(?i)\b(and|or|not)\b', r' \1 ', expr)
    expr = re.sub(r'([<>!=]=?|==)', r' \1 ', expr)
    expr = re.sub(r'([a-zA-Z0-9_])(\()', r'\1 \2', expr)
    expr = re.sub(r'(\))([a-zA-Z0-9_])', r'\1 \2', expr)
    expr = re.sub(r'\s+', ' ', expr)
    return expr.strip()


def is_in_range(x: float, expr: str) -> bool:
    """
    Evaluates whether the given x satisfies the cleaned expression.
    Only 'x' is available inside the eval environment.
    """
    expr = clean_expression(expr)
    try:
        return eval(expr, {"__builtins__": {}}, {"x": x})
    except Exception as e:
        raise ValueError(f"Invalid expression: '{expr}'") from e

def validate_expression(expr: str) -> bool:
    """
    Validates whether an expression string can be parsed and evaluated safely.
    Returns True if it's either:
      - A valid numeric range expression for `is_in_range`
      - A valid regex pattern
    """
    # First, check if it can be evaluated as a numeric range
    try:
        _ = is_in_range(0, expr)
        return True
    except Exception:
        pass

    # Next, check if it's a valid regex
    try:
        re.compile(expr)
        return True
    except re.error:
        return False

def safe_numeric(val):
            """Try to convert to float, return float if possible, else return original value."""
            try:
                return float(val)
            except (ValueError, TypeError):
                return val

def clamp_unit(value, name):
    """Clamp a 0-1 layout fraction, warning (with the parameter's name) when the
    given value falls outside the range."""
    if value < 0:
        warnings.warn(f"{name} < 0. Value has been clamped to 0.")
        return 0
    elif value > 1:
        warnings.warn(f"{name} > 1. Value has been clamped to 1.")
        return 1
    return value

def resolve_ordering(orders):
    """
        Merge several category orderings (from variables in same_scale) into one
        ordering that doesn't disagree with any of them. Returns None if they
        conflict (e.g. one says A before B and another says B before A).
    """
    # all_categories keeps every category in the order we first see it.
    all_categories = []
    for seq in orders:
        for cat in seq:
            if cat not in all_categories:
                all_categories.append(cat)

    # must_come_after[x] = the categories that have to be placed after x
    must_come_after = {cat: [] for cat in all_categories}
    for seq in orders:
        for i in range(1, len(seq)):
            before = seq[i - 1]
            after = seq[i]
            if after not in must_come_after[before]:
                must_come_after[before].append(after)

    result = []
    while len(result) < len(all_categories):
        # find the next category we are allowed to place: one that isn't placed
        # yet and has nothing left that must come before it.
        next_cat = None
        for cat in all_categories:
            if cat in result:
                continue
            waiting_on = [other for other in all_categories
                          if other not in result and cat in must_come_after[other]]
            if len(waiting_on) == 0:
                next_cat = cat
                break

        if next_cat is None:
            # nothing can be placed but some categories are left -> there's a loop
            return None

        result.append(next_cat)

    return result

def edge_color_from_face(facecolor, delta=0.3):
    """
    Compute an edge color based on a face color by adjusting brightness.

    Parameters:
        facecolor: str or tuple
            Hex string (e.g. '#FFAA00') or RGB tuple (r,g,b) in [0,1].
        delta: float
            How much to increase/decrease brightness.
            If face is light, brightness is reduced by delta.
            If face is dark, brightness is increased by delta.

    Returns:
        edgecolor: RGB tuple (r,g,b)
    """
    # Convert hex to RGB if necessary
    if isinstance(facecolor, str):
        rgb = mcolors.to_rgb(facecolor)
    else:
        rgb = facecolor

    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(*rgb)

    # Adjust brightness based on current value
    if v > 0.6:  # light color
        v = max(0, v - delta)
    else:        # dark color
        v = min(1, v + delta)

    # Convert back to RGB
    edge_rgb = colorsys.hsv_to_rgb(h, s, v)
    return edge_rgb

# -----------------------------
# Color indexing helpers
# -----------------------------
def _compute_color_index(val: Any, hi_missing, hi_value) -> int:
    """
        Calculates the appropriate colour for a specific row, given the hi_missing and hi_value parameters
    """
    missing_buffer = 1 if hi_missing else 0

    if pd.isna(val):
        return 0

    if isinstance(hi_value, list):
        try:
            return hi_value.index(val) + 1 + missing_buffer
        except ValueError:
            # Try numeric matching
            if not isinstance(val, str):
                for i, value in enumerate(hi_value):
                    try:
                        if float(value) == val:
                            return i + 1 + missing_buffer
                    except (ValueError, TypeError):
                        continue
            return 0

    if isinstance(hi_value, str):
        # regex
        try:
            regex = re.compile(hi_value)
            if isinstance(val, str) and regex.search(val):
                return 1 + missing_buffer
        except re.error:
            pass

        # numeric expression
        try:
            numeric_val = float(val) if isinstance(val, str) else val
            if isinstance(numeric_val, (int, float)) and is_in_range(numeric_val, hi_value):
                return 1 + missing_buffer
        except (ValueError, TypeError):
            return 0

    return 0

def assign_color_index(df: pd.DataFrame, var_list: List[str], hi_missing, missing_placeholder, hi_var, hi_value) -> pd.DataFrame:
    """
        Assigns each row in the dataframe with a colour index; calculates which rows are highlighted which colour.
    """

    df["color_index"] = 0  # default

    # Highlight missing values first
    if hi_missing and missing_placeholder is not None:
        for v in var_list:
            if v != hi_var:
                continue
            df.loc[df[v] == missing_placeholder, "color_index"] = 1

    # Then apply hi_value highlighting, but only where color_index is still 0
    if hi_var and hi_value is not None:
        for v in var_list:
            if v != hi_var:
                continue
            mask = df["color_index"] == 0
            col = df.loc[mask, v]
            if isinstance(hi_value, list):
                # Fast path: a list of highlight values is a straight lookup.
                # Build the same mapping _compute_color_index applies — first
                # occurrence in hi_value wins (list.index semantics), exact match
                # first, then a numeric-coercion fallback for non-string values.
                buffer = 1 if hi_missing else 0
                exact_map = {}
                numeric_map = {}
                for i, hv in enumerate(hi_value):
                    if hv not in exact_map:
                        exact_map[hv] = i + 1 + buffer
                    try:
                        fk = float(hv)
                    except (ValueError, TypeError):
                        continue
                    if fk not in numeric_map:
                        numeric_map[fk] = i + 1 + buffer

                def _numeric_index(val):
                    # numeric fallback only applies to non-string, non-NaN values
                    if isinstance(val, str) or pd.isna(val):
                        return np.nan
                    try:
                        return numeric_map.get(float(val), np.nan)
                    except (ValueError, TypeError):
                        return np.nan

                result = col.map(exact_map).fillna(col.map(_numeric_index))
                df.loc[mask, "color_index"] = result.fillna(0).astype(int)
            else:
                # regex / numeric-range expressions stay on the row-wise path
                df.loc[mask, "color_index"] = col.apply(lambda val: _compute_color_index(val, hi_missing, hi_value))
    return df

# -----------------------------
# Numeric axis label spacing
# -----------------------------
# The "nice" ladder: 1, 2, 2.5, 5 x 10^n. (10 x 10^n is the same number as
# 1 x 10^(n+1), so it is already covered.)
_NICE_MANTISSAS = (1.0, 2.0, 2.5, 5.0)


def _nice_steps_near(raw_step: float, integer_only: bool = False) -> List[float]:
    """
    The 1 / 2 / 2.5 / 5 x 10^n ladder spanning the decades around raw_step,
    ascending. With integer_only, non-integral rungs (0.5, 2.5, ...) are dropped
    so an integer variable never gets a fractional step.
    """
    exponent = int(np.floor(np.log10(raw_step)))
    steps = [m * (10.0 ** e)
             for e in range(exponent - 1, exponent + 4)
             for m in _NICE_MANTISSAS]
    if integer_only:
        steps = [float(round(s)) for s in steps
                 if s >= 1 and abs(s - round(s)) < 1e-9 * max(1.0, s)]
    return sorted(steps)


def nice_step(raw_step: float, integer_only: bool = False) -> float:
    """
    Round a raw label spacing to the nearest step on the 1 / 2 / 2.5 / 5 x 10^n
    ladder, i.e. one of ... 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50 ...
    "Nearest" is measured in log space, so the choice doesn't depend on the units.
    """
    if not np.isfinite(raw_step) or raw_step <= 0:
        return raw_step
    steps = _nice_steps_near(raw_step, integer_only)
    return min(steps, key=lambda s: abs(np.log10(s / raw_step)))


# An interior label closer than this fraction of a step to the min or the max is
# dropped: the end itself is labelled there, and two labels that close together
# would collide.
_END_GAP_FRACTION = 1 / 3


def _levels_on_step(min_val: float, max_val: float, step: float) -> List[float]:
    """
    Multiples of step strictly inside (min_val, max_val), minus any that crowd
    the min or max label (see _END_GAP_FRACTION).
    """
    gap = step * _END_GAP_FRACTION
    decimals = max(0, -int(np.floor(np.log10(step))) + 1)
    levels = []
    for k in range(int(np.ceil(min_val / step)), int(np.floor(max_val / step)) + 1):
        v = k * step
        if (v - min_val) < gap or (max_val - v) < gap:
            continue
        levels.append(round(v, decimals))
    return levels


def nice_level_values(min_val: float, max_val: float, num_levels: int,
                      integer_only: bool = False) -> np.ndarray:
    """
    Label values for a numeric axis. The minimum and maximum are always kept as
    they are; the levels in between land on "nice" numbers: the raw spacing
    (max - min) / (num_levels - 1) is rounded to the nearest step on the
    1 / 2 / 2.5 / 5 x 10^n ladder and the interior labels sit on multiples of
    that step. num_levels is therefore a target, not a hard limit -- rounding the
    step down can fit in a few more labels than asked for.
    """
    min_val, max_val = float(min_val), float(max_val)
    if not (np.isfinite(min_val) and np.isfinite(max_val)) or max_val <= min_val:
        return np.array([min_val], dtype=float)
    if num_levels is None or num_levels < 3:
        return np.array([min_val, max_val], dtype=float)

    step = nice_step((max_val - min_val) / (num_levels - 1), integer_only)
    interior = _levels_on_step(min_val, max_val, step)
    return np.array([min_val] + interior + [max_val], dtype=float)


def get_formatted_label(datatype, value):
    """
        Returns the formatted label version (in particular, returns something nicer for numeric vals)
    """
    if value is None or pd.isna(value):
        return value
    # if the label is a string
    if datatype == np.str_:
        return value
    # otherwise, it should be a numerical value
    value = float(value)
    if abs(value) >= 1000000 or 0 < abs(value) < 0.01: # threshold for displaying scientific notation
        return f"{value:.2e}"
    if datatype == np.integer:
        return str(int(value))
    if datatype == np.floating:
        return f"{value:.2f}" # round to 2 decimal places