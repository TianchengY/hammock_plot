"""
Diabetes Satisfaction Scales
============================

Show missing values across related satisfaction scale variables.
"""

from _gallery_utils import data_path, gallery_image_path

import hammock_plot
import pandas as pd


df = pd.read_csv(data_path("data_diabetes.csv"))

hammock = hammock_plot.Hammock(data_df=df)
hammock.plot(
    var=["sataces", "satcomm", "satrate"],
    missing=True,
    numerical_var_levels={"sataces": None, "satcomm": None, "satrate": None},
    min_bar_height=0.2,
    uni_vfill=.3,
    save_path=gallery_image_path("diabetes_missing_satisfaction.png"),
)
