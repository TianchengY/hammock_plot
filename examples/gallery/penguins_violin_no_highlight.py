"""Gallery example: unhighlighted violin plots for numerical penguin measures."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hammock_plot

DATA = ROOT / "data"
OUT = ROOT / "image" / "gallery"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "data_penguins.csv")

hammock = hammock_plot.Hammock(data_df=df)
hammock.plot(
    var=[
        "species",
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ],
    missing=True,
    display_type={
        "bill_length_mm": "violin",
        "bill_depth_mm": "violin",
        "flipper_length_mm": "violin",
        "body_mass_g": "violin",
    },
    default_color="#4e79a7",
    width=15,
    height=8,
    display_figure=False,
    save_path=OUT / "penguins_violin_no_highlight.png",
)
