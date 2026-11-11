"""White dividers between adjacent categorical bars.

With uni_vfill close to 1 the leftover gap between categorical bars vanishes,
so a divider is drawn wherever the gap
is narrower than Defaults.WHITE_DIVIDER_HEIGHT (reported with a 15-level
interviewer variable at uni_vfill=0.99).
"""
import os

import matplotlib.colors as mcolors
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from hammock_plot import Hammock

N_LEVELS = 15


@pytest.fixture
def many_levels():
    # unequal counts, several small levels, like the interviewer column
    counts = [70, 56, 55, 54, 53, 47, 45, 26, 19, 15, 12, 12, 10, 10, 10]
    names = [f"name{i:02d}" for i in range(N_LEVELS)]
    rows = [n for n, c in zip(names, counts) for _ in range(c)]
    return pd.DataFrame({"a": rows, "b": rows[::-1]})


def n_white_patches(ax):
    white = mcolors.to_rgba("white")
    return sum(1 for p in ax.patches if tuple(p.get_facecolor()) == white)


@pytest.mark.parametrize("vfill", [0.99, 1])
def test_dividers_drawn_when_bars_nearly_touch(many_levels, vfill):
    ax = Hammock(data_df=many_levels).plot(var=["a", "b"], uni_vfill=vfill)
    assert n_white_patches(ax) == 2 * (N_LEVELS - 1)


def test_no_dividers_when_gaps_are_visible(many_levels):
    ax = Hammock(data_df=many_levels).plot(var=["a", "b"], uni_vfill=0.5)
    assert n_white_patches(ax) == 0
