import re
import colorsys
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
    MIN_BAR_HEIGHT: float = 0.15 # minimum bar height
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
            df.loc[mask, "color_index"] = df.loc[mask, v].apply(lambda val: _compute_color_index(val, hi_missing, hi_value))
    return df

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