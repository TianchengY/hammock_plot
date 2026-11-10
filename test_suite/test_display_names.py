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


def test_name_options_keyed_by_column_name(penguins):
    ax = Hammock(data_df=penguins).plot(
        var=PENGUIN_VARS,
        display_names={"island": "Island"},
        name_options={"island": {"fontsize": 17}})
    assert ax.get_xticklabels()[1].get_fontsize() == 17


ISLANDS = {"Biscoe", "Dream", "Torgersen"}


def value_label_sizes(ax, texts):
    return {t.get_fontsize() for t in ax.texts if t.get_text() in texts}


def test_label_options_and_name_options_are_independent(penguins):
    ax = Hammock(data_df=penguins).plot(
        var=PENGUIN_VARS,
        label_options={"island": {"fontsize": 8}},
        name_options={"island": {"fontsize": 20}})
    assert ax.get_xticklabels()[1].get_fontsize() == 20
    assert value_label_sizes(ax, ISLANDS) == {8}


def test_label_options_no_longer_style_variable_names(penguins):
    default = Hammock(data_df=penguins).plot(var=PENGUIN_VARS)
    ax = Hammock(data_df=penguins).plot(
        var=PENGUIN_VARS, label_options={"island": {"fontsize": 25}})
    assert (ax.get_xticklabels()[1].get_fontsize()
            == default.get_xticklabels()[1].get_fontsize())
    assert value_label_sizes(ax, ISLANDS) == {25}


def test_name_options_for_unplotted_variable_raises(penguins):
    with pytest.raises(ValueError, match="name_options"):
        Hammock(data_df=penguins).plot(var=PENGUIN_VARS,
                                       name_options={"sex": {"fontsize": 12}})


def test_name_options_must_be_dicts(penguins):
    with pytest.raises(ValueError, match="name_options"):
        Hammock(data_df=penguins).plot(var=PENGUIN_VARS,
                                       name_options={"island": 12})


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
