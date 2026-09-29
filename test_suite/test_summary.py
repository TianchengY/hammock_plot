"""Tests for the readback produced by print(hammock) and hammock.summary.

These assert against the summary's fields. 
Two tests for the print layout.
"""
import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from hammock_plot import Hammock


ROOT = Path(__file__).resolve().parents[1]
PENGUINS = ROOT / "data" / "data_penguins.csv"
ASTHMA = ROOT / "data" / "data_asthma.csv"

PENGUIN_VARS = ["species", "island", "bill_length_mm", "bill_depth_mm",
                "flipper_length_mm", "body_mass_g"]


@pytest.fixture
def penguins():
    # rowid is a row counter, not data; dropping it keeps the frame at 8 columns
    return pd.read_csv(PENGUINS).drop(columns=["rowid"])


@pytest.fixture
def asthma():
    return pd.read_csv(ASTHMA)


def plot_penguins(df, **kwargs):
    kwargs.setdefault("display_figure", False)
    hammock = Hammock(data_df=df)
    hammock.plot(var=PENGUIN_VARS, **kwargs)
    return hammock


# ---------------------------------------------------------------
# The two states
# ---------------------------------------------------------------
def test_summary_exists_before_plot(penguins):
    hammock = Hammock(data_df=penguins)
    assert hammock.summary.plotted is False
    assert hammock.summary.n_observations == 344
    assert hammock.summary.n_variables == 8
    assert hammock.summary.columns[0] == "species"


def test_columns_listed_only_before_plot(penguins):
    hammock = Hammock(data_df=penguins)
    assert "Columns:" in str(hammock)
    hammock.plot(var=PENGUIN_VARS, display_figure=False)
    assert "Columns:" not in str(hammock)


def test_neither_state_raises_on_str(penguins):
    # __str__ gets called by f-strings, logging and debugger panes without
    # being asked, so raising from it would break all three
    hammock = Hammock(data_df=penguins)
    assert f"{hammock}"
    hammock.plot(var=PENGUIN_VARS, display_figure=False)
    assert f"{hammock}"


def test_repr_stays_one_line(penguins):
    hammock = Hammock(data_df=penguins)
    assert "\n" not in repr(hammock)
    hammock.plot(var=PENGUIN_VARS, display_figure=False)
    assert "\n" not in repr(hammock)


def test_str_matches_summary(penguins):
    hammock = plot_penguins(penguins)
    assert str(hammock) == str(hammock.summary)


# ---------------------------------------------------------------
# Resolved per-variable values
# ---------------------------------------------------------------
def test_display_type_defaults_by_dtype(penguins):
    summary = plot_penguins(penguins).summary
    assert summary.display_types["species"] == "stacked_bar"
    assert summary.display_types["bill_length_mm"] == "rug"


def test_display_type_override_is_reported(penguins):
    summary = plot_penguins(penguins, display_type={"bill_length_mm": "box"}).summary
    assert summary.display_types["bill_length_mm"] == "box"
    assert summary.display_types["bill_depth_mm"] == "rug"


def test_data_types(penguins):
    summary = plot_penguins(penguins).summary
    assert summary.data_types["island"] == "categorical"
    assert summary.data_types["body_mass_g"] == "numerical"


def test_variables_are_in_axis_order(penguins):
    summary = plot_penguins(penguins).summary
    assert [v.name for v in summary.variables] == PENGUIN_VARS


def test_numerical_levels_default_and_override(penguins):
    summary = plot_penguins(penguins, numerical_var_levels={"body_mass_g": 8}).summary
    assert summary.numerical_levels["bill_length_mm"] == 7
    assert summary.numerical_levels["body_mass_g"] == 8
    # categorical axes have no level count at all
    assert "species" not in summary.numerical_levels


def test_numerical_levels_by_value_under_seven_distinct(asthma):
    # a numerical axis with fewer than 7 distinct values is labelled per value
    # and never binned, so a level count would name a binning that didn't happen
    subset = asthma[asthma["hospitalizations"] < 5]
    hammock = Hammock(data_df=subset)
    hammock.plot(var=["hospitalizations", "group"], display_figure=False)
    assert hammock.summary.numerical_levels["hospitalizations"] == "by value"


def test_numerical_levels_by_value_when_set_to_none(asthma):
    hammock = Hammock(data_df=asthma)
    hammock.plot(var=["hospitalizations", "group"],
                 numerical_var_levels={"hospitalizations": None},
                 display_figure=False)
    assert hammock.summary.numerical_levels["hospitalizations"] == "by value"


# ---------------------------------------------------------------
# value_order
# ---------------------------------------------------------------
def test_value_order_lists_only_what_the_user_set(penguins):
    summary = plot_penguins(
        penguins, value_order={"species": ["Adelie", "Chinstrap", "Gentoo"]}).summary
    assert list(summary.value_order) == ["species"]
    assert summary.value_order["species"] == ["Adelie", "Chinstrap", "Gentoo"]


def test_value_order_shows_the_missing_placeholder(penguins):
    # the placeholder is drawn on the axis, so it belongs in the drawn order
    summary = plot_penguins(
        penguins, missing=True,
        value_order={"species": ["Adelie", "Chinstrap", "Gentoo"]}).summary
    assert summary.value_order["species"][0] == "missing"


def test_same_scale_propagation_is_not_reported_as_user_order(asthma):
    # same_scale builds an order for every sibling; none of that was asked for
    hammock = Hammock(data_df=asthma)
    hammock.plot(var=["group", "gender"], same_scale=["group", "gender"],
                 display_figure=False)
    assert hammock.summary.value_order == {}


def test_default_ordering_is_not_reported(penguins):
    summary = plot_penguins(penguins).summary
    assert summary.value_order == {}


def test_value_order_on_a_numeric_variable_is_flagged(asthma):
    # ordering a numeric column retypes it as categorical, which reads as a bug
    # unless the readback says why
    hammock = Hammock(data_df=asthma)
    hammock.plot(var=["hospitalizations", "group"],
                 value_order={"hospitalizations": [0, 1, 2, 3, 4, 5, 20]},
                 display_figure=False)
    assert hammock.summary.data_types["hospitalizations"] == "categorical"
    assert "(set by value_order)" in str(hammock)


# ---------------------------------------------------------------
# Header fields
# ---------------------------------------------------------------
def test_observations_plotted_appears_only_when_rows_are_dropped(penguins):
    dropped = plot_penguins(penguins, missing=False).summary
    assert dropped.n_plotted == 342
    assert "Observations plotted: 342 of 344" in str(dropped)

    kept = plot_penguins(penguins, missing=True).summary
    assert kept.n_plotted == 344
    assert "Observations plotted" not in str(kept)


def test_highlight_lines_suppress_without_a_highlight(penguins):
    text = str(plot_penguins(penguins))
    assert "Highlight variable: none" in text
    assert "Highlight values" not in text
    assert "Highlight box layout" not in text


def test_highlight_values_from_a_list(penguins):
    summary = plot_penguins(penguins, hi_var="island",
                            hi_value=["Torgersen", "Dream"]).summary
    assert summary.hi_values == ["Torgersen", "Dream"]


def test_highlight_values_from_an_expression(asthma):
    hammock = Hammock(data_df=asthma)
    hammock.plot(var=["hospitalizations", "group"], hi_var="hospitalizations",
                 hi_value="x>3", display_figure=False)
    assert hammock.summary.hi_values == ["x>3"]


def test_hi_missing_appends_the_placeholder(penguins):
    summary = plot_penguins(penguins, missing=True, hi_var="island",
                            hi_value="Torgersen", hi_missing=True,
                            missing_placeholder="(not recorded)").summary
    assert summary.hi_values == ["Torgersen", "(not recorded)"]
    assert "Missing placeholder: (not recorded)" in str(summary)


def test_missing_placeholder_hidden_when_nothing_is_placeheld(penguins):
    assert "Missing placeholder" not in str(plot_penguins(penguins, missing=False))


def test_violin_bandwidth_only_with_a_violin(penguins):
    without = plot_penguins(penguins)
    assert "Violin bandwidth" not in str(without)

    with_violin = plot_penguins(penguins, display_type={"flipper_length_mm": "violin"})
    assert "Violin bandwidth: scott" in str(with_violin)


def test_save_path_only_when_given(penguins, tmp_path):
    assert "Saved to" not in str(plot_penguins(penguins))

    target = tmp_path / "hammock.png"
    saved = plot_penguins(penguins, save_path=str(target))
    assert f"Saved to: {target}" in str(saved)


def test_toggles_render_lowercase(penguins):
    text = str(plot_penguins(penguins, unibar=False, label=True))
    assert "Unibars shown: no" in text
    assert "Labels shown: yes" in text


def test_weights_and_same_scale(asthma):
    weighted = asthma.copy()
    weighted["wt"] = 1.5
    hammock = Hammock(data_df=weighted)
    hammock.plot(var=["hospitalizations", "comorbidities", "group"], weights="wt",
                 same_scale=["hospitalizations", "comorbidities"], display_figure=False)
    text = str(hammock)
    assert "Weight variable: wt" in text
    assert "Same scale: hospitalizations, comorbidities" in text


# ---------------------------------------------------------------
# Everything else
# ---------------------------------------------------------------
def test_caller_lists_are_copied(asthma):
    var = ["group", "gender"]
    same_scale = ["group", "gender"]
    hammock = Hammock(data_df=asthma)
    hammock.plot(var=var, same_scale=same_scale, display_figure=False)

    var.append("comorbidities")
    same_scale.append("comorbidities")
    assert hammock.summary.var == ["group", "gender"]
    assert hammock.summary.same_scale == ["group", "gender"]


@pytest.mark.parametrize("same_scale", [
    {"comorbidities", "hospitalizations"},
    ("hospitalizations", "comorbidities"),
])
def test_same_scale_accepts_set_and_tuple(asthma, same_scale):
    # a set used to crash with "'set' object is not subscriptable"; it takes
    # the order of var since a set has none
    hammock = Hammock(data_df=asthma)
    hammock.plot(var=["hospitalizations", "comorbidities", "group"],
                 same_scale=same_scale, display_figure=False)
    assert hammock.summary.same_scale == ["hospitalizations", "comorbidities"]


@pytest.mark.parametrize("missing", [True, False])
def test_nullable_integer_column(asthma, missing):
    # an Int64 column with missing values used to crash on fillna
    nullable = asthma.copy()
    nullable.loc[:4, "comorbidities"] = None
    nullable["comorbidities"] = nullable["comorbidities"].astype("Int64")
    plain = asthma.copy()
    plain.loc[:4, "comorbidities"] = None

    var = ["comorbidities", "group"]
    got = Hammock(data_df=nullable)
    got.plot(var=var, missing=missing, display_figure=False)
    want = Hammock(data_df=plain)
    want.plot(var=var, missing=missing, display_figure=False)
    assert str(got) == str(want)
    # the caller's frame is left alone
    assert nullable["comorbidities"].dtype == "Int64"


def test_summary_is_independent_of_figure_size(penguins):
    small = plot_penguins(penguins, height=5, width=8)
    large = plot_penguins(penguins, height=20, width=30)
    assert str(small) == str(large)


def test_long_variable_lists_wrap(penguins):
    wide = penguins.rename(
        columns={"bill_length_mm": "a_really_quite_long_variable_name_for_bill_length"})
    hammock = Hammock(data_df=wide)
    hammock.plot(var=["a_really_quite_long_variable_name_for_bill_length",
                      "species", "island", "body_mass_g"], display_figure=False)

    lines = [line for line in str(hammock).splitlines() if line.strip()]
    assert all(len(line) <= 72 for line in lines)
    # continuation aligns under the first name rather than under the label
    variables_line = next(i for i, l in enumerate(lines) if l.startswith("  Variables:"))
    assert lines[variables_line + 1].startswith(" " * len("  Variables: "))


# ---------------------------------------------------------------
# Golden text - the layout itself
# ---------------------------------------------------------------
EXPECTED_NOT_PLOTTED = """Hammock object

Data:
  Observations: 692
  Num Variables: 4
  Columns: hospitalizations, group, gender, comorbidities

Not yet plotted. Call plot(var=[...]) to draw."""


EXPECTED_PLOTTED = """Hammock object

Data:
  Observations: 692
  Num Variables: 4

Plot specification:
  Variables: hospitalizations, group, gender
  Highlight variable: group
  Highlight values: child, missing
  Highlight box layout: side-by-side
  Missing values shown: yes
  Missing placeholder: missing
  Same scale: none
  Weight variable: none
  Unibars shown: yes
  Labels shown: yes

Variable specifications:
  hospitalizations
    Data type: numerical
    Display type: rug
    Numerical levels: 7

  group
    Data type: categorical
    Display type: stacked_bar
    Value order: missing, child, adolescent, adult

  gender
    Data type: categorical
    Display type: stacked_bar"""


def test_golden_not_plotted(asthma):
    assert str(Hammock(data_df=asthma)) == EXPECTED_NOT_PLOTTED


def test_golden_plotted(asthma):
    hammock = Hammock(data_df=asthma)
    hammock.plot(var=["hospitalizations", "group", "gender"],
                 hi_var="group", hi_value="child", hi_missing=True, missing=True,
                 value_order={"group": ["child", "adolescent", "adult"]},
                 display_figure=False)
    assert str(hammock) == EXPECTED_PLOTTED
