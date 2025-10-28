import hammock_plot
import pandas as pd
import os

DATA_DIR="data/"

df_asthma = pd.read_csv(os.path.join(DATA_DIR, "data_asthma.csv"))
df_diabetes = pd.read_csv(os.path.join(DATA_DIR, "data_diabetes.csv"))
df_asthma_2 = pd.read_csv(os.path.join(DATA_DIR, "data_asthma_2.csv"))
df_shakespeare = pd.read_csv(os.path.join(DATA_DIR, "data_shakespeare.csv"))
df_penguins = pd.read_csv(os.path.join(DATA_DIR, "data_penguins.csv"))

def minimal_example():
    df = df_asthma
    var = ["hospitalizations","group","gender","comorbidities"]
    hammock = hammock_plot.Hammock(data_df = df)
    ax = hammock.plot(var=var, save_path="image/asthma_minimal.png")

def numeric_var_levels():
    df = df_asthma
    var = ["hospitalizations","group","gender","comorbidities"]
    hammock = hammock_plot.Hammock(data_df = df)
    ax = hammock.plot(var=var, numerical_var_levels={"comorbidities": None, "hospitalizations": None}, save_path="image/asthma_levels.png")

def value_order():
    df = df_asthma
    var = ["hospitalizations","group","gender","comorbidities"]
    group_order = ["child", "adolescent", "adult"]
    value_order = {"group": group_order}
    hammock = hammock_plot.Hammock(data_df = df)
    ax = hammock.plot(var=var, value_order=value_order, numerical_var_levels={"comorbidities": None, "hospitalizations": None}, save_path="image/asthma_value_order.png")

def highlighting():
    df = df_asthma
    var = ["hospitalizations","group","gender","comorbidities"]
    hammock = hammock_plot.Hammock(data_df = df)
    ax = hammock.plot(var=var ,hi_var="comorbidities", hi_value=[0], colors=["red"], numerical_var_levels={"comorbidities": None, "hospitalizations": None}, save_path="image/asthma_highlighting.png")

def missing_true():
    df = df_diabetes
    var = ["sataces","satcomm","satrate"]
    hammock = hammock_plot.Hammock(data_df = df)
    ax = hammock.plot(var=var, missing=True, numerical_var_levels={"sataces": None, "satcomm": None, "satrate": None}, min_bar_height=0.2, save_path="image/diabetes.png") 

def speaker_order():
    df = df_shakespeare
    var_lst = ["type","speaker1","speaker2","sex1"]
    color_lst = ["#fb9a99","#6a3d9a","#ff7f00"]
    hi_value = ["Beggars","Citizens","Gentry"]

    speaker_order=["Beggars", "Royalty", "Nobility", "Gentry", "Citizens", "Yeomanry"]

    hammock = hammock_plot.Hammock(data_df = df)
    ax = hammock.plot(var=var_lst,hi_var = "speaker1", hi_value=hi_value,colors=color_lst,missing=True,
                    value_order ={"speaker1":speaker_order,"speaker2":speaker_order},
                    save_path="image/shakespeare_order.png")

def same_scale():
    df = df_shakespeare
    var_lst = ["type","speaker1","speaker2","sex1"]
    color_lst = ["#fb9a99","#6a3d9a","#ff7f00"]
    hi_value = ["Beggars","Citizens","Gentry"]

    speaker_order=["Beggars", "Royalty", "Nobility", "Gentry", "Citizens", "Yeomanry"]

    hammock = hammock_plot.Hammock(data_df = df)
    ax = hammock.plot(var=var_lst,hi_var = "speaker1", hi_value=hi_value,colors=color_lst,missing=True,
                    value_order ={"speaker1":speaker_order}, same_scale=["speaker1", "speaker2"],
                    save_path="image/shakespeare_scale.png")

def numerical_display_type():
    df = df_penguins
    hammock = hammock_plot.Hammock(df)

    ax = hammock.plot(
        var= ["species", "island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
        display_figure=False,
        hi_var="island",
        hi_value=["Torgersen", "Biscoe"],
        missing=True,
        save_path="image/penguin_display_types.png",
        numerical_display_type={"bill_length_mm":"box", "bill_depth_mm": "box", "flipper_length_mm": "box", "body_mass_g":"box"},
    )

def numerical_display_violin():
    df = df_penguins
    hammock = hammock_plot.Hammock(df)
    ax = hammock.plot(
        var= ["species", "island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
        display_figure=False,
        hi_var="island",
        hi_value=["Torgersen"],
        missing=True,
        save_path="image/penguin_display_violin.png",
        numerical_display_type={"bill_length_mm":"box", "bill_depth_mm": "rugplot", "flipper_length_mm": "violin", "body_mass_g":"box"},
    )
