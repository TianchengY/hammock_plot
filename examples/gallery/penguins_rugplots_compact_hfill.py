"""
Penguins Compact Rugplots
=========================

Use compact horizontal fill settings for numerical rugplots.
"""

from _gallery_utils import data_path, gallery_image_path

import hammock_plot
import pandas as pd


df = pd.read_csv(data_path("data_penguins.csv"))

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
    save_path=gallery_image_path("penguins_rugplots_compact_hfill.png"),
)
