"""Tests for display_names"""
import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from hammock_plot import Hammock


ROOT = Path(__file__).resolve().parents[1]
PENGUINS = ROOT / "data" / "data_penguins.csv"

PENGUIN_VARS = ["species", "island", "bill_length_mm", "body_mass_g"]


@pytest.fixture
def penguins():
    return pd.read_csv(PENGUINS)


def axis_names(ax):
    return [t.get_text() for t in ax.get_xticklabels()]


def test_no_display_names_uses_column_names(penguins):
    ax = Hammock(data_df=penguins).plot(var=PENGUIN_VARS)
    assert axis_names(ax) == PENGUIN_VARS


def test_display_names_replace_only_the_given_variables(penguins):
    ax = Hammock(data_df=penguins).plot(
        var=PENGUIN_VARS,
        display_names={"bill_length_mm": "bill\nlength", "body_mass_g": "mass (g)"})
    assert axis_names(ax) == ["species", "island", "bill\nlength", "mass (g)"]


def test_label_options_still_keyed_by_column_name(penguins):
    ax = Hammock(data_df=penguins).plot(
        var=PENGUIN_VARS,
        display_names={"island": "Island"},
        label_options={"island": {"fontsize": 17}})
    assert ax.get_xticklabels()[1].get_fontsize() == 17


def test_display_names_leave_summary_on_column_names(penguins):
    hammock = Hammock(data_df=penguins)
    hammock.plot(var=PENGUIN_VARS, display_names={"island": "Island"},
                 display_figure=False)
    assert hammock.summary.var == PENGUIN_VARS


def test_display_names_for_unplotted_variable_raises(penguins):
    with pytest.raises(ValueError, match="display_names"):
        Hammock(data_df=penguins).plot(var=PENGUIN_VARS,
                                       display_names={"sex": "Sex"})


def test_display_names_must_be_strings(penguins):
    with pytest.raises(ValueError, match="display_names"):
        Hammock(data_df=penguins).plot(var=PENGUIN_VARS,
                                       display_names={"island": 3})
