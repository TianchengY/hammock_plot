"""
Slope Graph
========================

In a slope graph, all variables are on the same scale. In the hammock plot we hide unibars and show connectors only.
"""

from _gallery_utils import data_path, gallery_image_path

import hammock_plot
import pandas as pd


df = pd.read_csv(data_path("data_penguins.csv"))


hammock = hammock_plot.Hammock(data_df=df)
hammock.plot(
    var=[
        "bill_length_mm",
        "bill_depth_mm",
    ],
    same_scale=["bill_length_mm","bill_depth_mm"],
    unibar=False,
    label=True,
    width=7,
    height=10,
    uni_hfill=0.1, # allocates less space to labels
    save_path=gallery_image_path("penguins_slope_graph.png"),
)
