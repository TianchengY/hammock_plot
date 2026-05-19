import hammock_plot
import pandas as pd


df = pd.read_csv("../../data/data_shakespeare_v5.csv")
speaker_order = [ "Royalty", "Nobility", "Gentry", "Citizens", "Yeomanry","Beggars"]
sex_order = [ "M","F"]

hammock = hammock_plot.Hammock(data_df=df)
hammock.plot(
    var=["type", "characters", "speaker1", "speaker2", "sex1", "sex2"],
    hi_var="type",
    hi_value=["comedy", "history"],
    hi_box="stacked",
    value_order={"speaker1": speaker_order, "speaker2": speaker_order, "sex1": sex_order, "sex2": sex_order},
    same_scale=["speaker1", "speaker2"],
    missing=True,
    uni_vfill=.4,
    display_type={"characters": "box"},
    save_path="../../image/gallery/shakespeare_box.png",
)
