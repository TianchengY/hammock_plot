"""Gallery example: missing values and top coding in diabetes satisfaction scales."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hammock_plot

DATA = ROOT / "data"
OUT = ROOT / "image" / "gallery"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "data_diabetes.csv")

hammock = hammock_plot.Hammock(data_df=df)
hammock.plot(
    var=["sataces", "satcomm", "satrate"],
    missing=True,
    numerical_var_levels={"sataces": None, "satcomm": None, "satrate": None},
    min_bar_height=0.2,
    default_color="#4e79a7",
    width=9,
    height=6,
    display_figure=False,
    save_path=OUT / "diabetes_missing_satisfaction.png",
)
