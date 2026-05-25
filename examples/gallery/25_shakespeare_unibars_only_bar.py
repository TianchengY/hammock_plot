"""
Marginal Display only -  horizontal barcharts
==================================

display type "bar chart" for categorical vars; removing connectors; same scale.
"""

from _gallery_utils import data_path, gallery_image_path

import hammock_plot
import pandas as pd


df = pd.read_csv(data_path("data_shakespeare_v5.csv"))
speaker_order = [ "Royalty", "Nobility", "Gentry", "Citizens", "Yeomanry","Beggars"]
sex_order = [ "M","F"]

hammock = hammock_plot.Hammock(data_df=df)
hammock.plot(
    var=["type",  "speaker1", "speaker2"],
    hi_box="stacked",
    value_order={"speaker1": speaker_order, "speaker2": speaker_order},
    same_scale=["speaker1", "speaker2"],
    missing=False,
    uni_vfill=.95,
    uni_hfill=.7,
    connector_fraction=0,
    width=5,
    height=10,
    display_type={"type": "bar chart", "speaker1":"bar chart", "speaker2": "bar chart"},
    save_path=gallery_image_path("shakespeare_box.png"),
)

import matplotlib.pyplot as plt
plt.show()
