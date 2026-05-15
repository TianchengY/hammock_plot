"""Gallery example: Shakespeare speakers ordered by social class."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hammock_plot

DATA = ROOT / "data"
OUT = ROOT / "image" / "gallery"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "data_shakespeare.csv")
speaker_order = ["Beggars", "Royalty", "Nobility", "Gentry", "Citizens", "Yeomanry"]

hammock = hammock_plot.Hammock(data_df=df)
hammock.plot(
    var=["type", "speaker1", "speaker2", "sex1", "sex2"],
    hi_var="speaker1",
    hi_value=["Beggars", "Citizens", "Gentry"],
    colors=["#fdc086", "#386cb0", "#7fc97f"],
    default_color="#666666",
    value_order={"speaker1": speaker_order, "speaker2": speaker_order},
    missing=True,
    width=14,
    height=7,
    display_figure=False,
    save_path=OUT / "shakespeare_social_order.png",
)
