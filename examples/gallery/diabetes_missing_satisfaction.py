import hammock_plot
import pandas as pd


df = pd.read_csv("../../data/data_diabetes.csv")

hammock = hammock_plot.Hammock(data_df=df)
hammock.plot(
    var=["sataces", "satcomm", "satrate"],
    missing=True,
    numerical_var_levels={"sataces": None, "satcomm": None, "satrate": None},
    min_bar_height=0.2,
    uni_vfill=.3,
    save_path="../../image/gallery/diabetes_missing_satisfaction.png",
)
