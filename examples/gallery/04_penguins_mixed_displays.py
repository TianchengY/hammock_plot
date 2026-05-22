"""
Penguins Mixed Displays
=======================

Combine bar charts and box plots in one hammock plot.
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
    uni_vfill=0.7,
    uni_hfill=0.99,
    connector_fraction=0.1,
    display_type={
        "species": "bar chart",
        "island": "bar chart",
        "bill_length_mm": "box",
        "bill_depth_mm": "box",
        "flipper_length_mm": "box",
        "body_mass_g": "box",
    },
    width=15,
    height=8,
    save_path=gallery_image_path("penguins_mixed_displays.png"),
)
