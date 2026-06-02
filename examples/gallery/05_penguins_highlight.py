"""
Highlighting
============

Highlighting. Also missing values.
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
    hi_value=["Torgersen"],
    missing=True,
    uni_vfill=.4,
    connector_fraction=.2,
    display_type={
        "bill_length_mm":"box",
        "bill_depth_mm": "rug",
        "flipper_length_mm": "violin",
        "body_mass_g":"box"
    },
    width=15,
    height=8,
    save_path=gallery_image_path("penguins_highlight.png"),
)
