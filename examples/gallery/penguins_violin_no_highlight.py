"""
Large categorical bars / violin plots
===========================================

Display numerical variables using violin plots. Increase bar size of categorical variables.
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
    missing=True,
    uni_vfill=.9,
    connector_fraction=.2,
    display_type={
        "bill_length_mm": "violin",
        "bill_depth_mm": "violin",
        "flipper_length_mm": "violin",
        "body_mass_g": "violin",
    },
    width=15,
    height=8,
    save_path=gallery_image_path("penguins_violin_no_highlight.png"),
)
