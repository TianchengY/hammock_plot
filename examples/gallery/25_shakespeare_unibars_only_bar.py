"""
Marginal display only -  horizontal barcharts
=============================================

Display type "bar" for categorical vars; removing connectors; force same scale for some vars.
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
    display_type={"type": "bar", "speaker1":"bar", "speaker2": "bar"},
    save_path=gallery_image_path("shakespeare_bar.png"),
)

import matplotlib.pyplot as plt
plt.show()
