"""Gallery example: ordered asthma categories with a highlighted subgroup."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hammock_plot

DATA = ROOT / "data"
OUT = ROOT / "image" / "gallery"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "data_asthma.csv")

hammock = hammock_plot.Hammock(data_df=df)
hammock.plot(
    var=["hospitalizations", "group", "gender", "comorbidities"],
    value_order={"group": ["child", "adolescent", "adult"]},
    numerical_var_levels={"hospitalizations": None, "comorbidities": None},
    hi_var="comorbidities",
    hi_value=[0],
    colors=["#f28e2b"],
    default_color="#4e79a7",
    width=12,
    height=7,
    display_figure=False,
    save_path=OUT / "asthma_highlight.png",
)
