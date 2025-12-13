import re
from collections import defaultdict, deque
import colorsys
import matplotlib.colors as mcolors
import pandas as pd
from typing import List, Dict, Any
import numpy as np

class Defaults:
    # General
    MISSING_PLACEHOLDER: str = "missing"

    # Colours
    COLORS: list = ["#e31a1c", "#fb9a99", "#33a02c", "#b2df8a", "#ff7f00", "#fdbf6f", "#6a3d9a", "#cab2d6", "#b15928", "#1f78b4"]
    DEFAULT_COLOR: str = "#a6cee3"

    # Layout
    UNI_VFILL: float = 0.08
    CONNECTOR_FRACTION: float = 1
    UNI_HFILL: float = 0.3
    MIN_BAR_HEIGHT: float = 0.15
    BAR_UNIT: float = 1.0
    XMARGIN: float = 0.02
    YMARGIN: float = 0.04
    SCALE: float = 10
    GAP_BTWN_UNI_MULTI: float = 2
    MIN_MULTI_WIDTH: float = 3 # in pixels
    SPACE_ABOVE_MISSING: float = 2
    NUM_LEVELS = 7
    ALPHA = 0.7
    WHITE_DIVIDER_HEIGHT = 0.3
    SPIKE_THICKNESS = 0.3

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
    graph = defaultdict(set)
    indegree = defaultdict(int)
    nodes = set()

    # Build graph
    for seq in orders:
        for i in range(len(seq)):
            nodes.add(seq[i])
            if i > 0:
                u, v = seq[i-1], seq[i]
                if v not in graph[u]:
                    graph[u].add(v)
                    indegree[v] += 1

    # Initialize queue with nodes of indegree 0
    q = deque([n for n in nodes if indegree[n] == 0])
    order = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)

    # If we used all nodes, success
    if len(order) == len(nodes):
        return order
    else:
        return None

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
    # if hi_missing is true, increase each index by 1
    missing_buffer = 1 if hi_missing else 0

    if isinstance(hi_value, list):
        try:
            idx = hi_value.index(val) + 1 + missing_buffer
            return idx
        except ValueError:
            if not type(val) is str:
                for i, value in enumerate(hi_value):
                    try:
                        match = float(value) == val
                        if match:
                            return i + 1 + missing_buffer
                    except ValueError:
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
        except ValueError:
            return 0
    return 0

def assign_color_index(df: pd.DataFrame, var_list: List[str], hi_missing, missing_placeholder, hi_var, hi_value) -> pd.DataFrame:
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
    if value is None:
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