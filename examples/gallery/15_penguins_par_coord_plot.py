"""
Parallel coordinate plot
========================

In the hammock plot, we hide unibars and labels and show connectors only.
"""

from _gallery_utils import data_path, gallery_image_path

import hammock_plot
import pandas as pd

from pathlib import Path
print("cwd:", Path.cwd())


df = pd.read_csv(data_path("data_penguins.csv"))

hammock = hammock_plot.Hammock(data_df=df)
hammock.plot(
    var=[
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ],
    hi_var="island",
    hi_value=["Torgersen"],
    unibar=False,
    label=False,
    width=10,
    height=9,
    # optionally increase line thickness
    min_bar_height_connectors=.12,
    save_path=gallery_image_path("penguins_par_coord_plot.png"),
)


