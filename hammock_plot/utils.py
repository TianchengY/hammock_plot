import re
from collections import defaultdict, deque

class Defaults:
    # General
    MISSING_PLACEHOLDER: str = "missing"

    # Colours
    COLORS: list = ["#e31a1c", "#33a02c", "#fb9a99", "#33a02c", "#b2df8a", "#ff7f00", "#fdbf6f", "#6a3d9a", "#cab2d6", "#b15928", "#1f78b4"]
    DEFAULT_COLOR: str = "#a6cee3"

    # Layout
    UNI_FRACTION: float = 0.08
    SPACE: float = 0.3
    MIN_BAR_HEIGHT: float = 0.1
    BAR_UNIT: float = 1.0
    XMARGIN: float = 0.02
    YMARGIN: float = 0.04
    SCALE: float = 10
    GAP_BTWN_UNI_MULTI: float = 2
    MIN_MULTI_WIDTH: float = 3 # in pixels
    SPACE_ABOVE_MISSING: float = 2


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
    Returns True if valid, False if not.
    """
    try:
        _ = is_in_range(0, expr)
        return True
    except Exception:
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