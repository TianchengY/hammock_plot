"""
Marginal displays Only
=======================

Remove connectors to focus on the marginal displays. Also large bars.
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
    connector_fraction=0,
    uni_hfill=.5,
    uni_vfill=.99,
    display_type={
        "species": "stacked bar",
        "island": "stacked bar",
        "bill_length_mm": "box",
        "bill_depth_mm": "box",
        "flipper_length_mm": "box",
        "body_mass_g": "box",
    },
    save_path=gallery_image_path("penguins_unibars_only.png"),
)
