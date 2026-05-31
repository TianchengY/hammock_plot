import hammock_plot
import pandas as pd


df = pd.read_csv("../../data/data_penguins.csv")

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
    uni_hfill=0.3,
    display_type={
        "bill_length_mm": "rug",
        "bill_depth_mm": "rug",
        "flipper_length_mm": "rug",
        "body_mass_g": "rug",
    },
    width=15,
    height=8,
    save_path="../../image/gallery/penguins_rugplots_compact_hfill.png",
)
