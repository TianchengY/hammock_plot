"""Gallery example: rugplots with narrow unibars and prominent connectors."""

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
    hi_var="island",
    hi_value=["Torgersen", "Biscoe"],
    colors=["#f28e2b", "#76b7b2"],
    default_color="#4e79a7",
    missing=True,
    uni_hfill=0.3,
    display_type={
        "bill_length_mm": "rugplot",
        "bill_depth_mm": "rugplot",
        "flipper_length_mm": "rugplot",
        "body_mass_g": "rugplot",
    },
    width=15,
    height=8,
    display_figure=False,
    save_path=OUT / "penguins_rugplots_compact_hfill.png",
)
