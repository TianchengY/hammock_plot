import hammock_plot
import pandas as pd


df = pd.read_csv("../../data/data_asthma_2.csv")
binary_labels = {
    "Diagnosis": {0: "No asthma", 1: "Asthma"},
    "Gender": {0: "Female", 1: "Male"},
    "Smoking": {0: "No smoking", 1: "Smoking"},
    "FamilyHistoryAsthma": {0: "No family history", 1: "Family history"},
}
for column, labels in binary_labels.items():
    df[column] = df[column].map(labels)

hammock = hammock_plot.Hammock(data_df=df)
hammock.plot(
    var=[
        "Diagnosis",
        "Gender",
        "Smoking",
        "FamilyHistoryAsthma",
        "Age",
        "LungFunctionFEV1",
    ],
    hi_var="Diagnosis",
    hi_value=["Asthma"],
    display_type={
        "Age": "box",
        "LungFunctionFEV1": "violin",
        "Gender": "bar chart",
        "Smoking": "bar chart",
        "Diagnosis": "bar chart",
        "FamilyHistoryAsthma": "bar chart",
    },
    # number of levels displayed for corresponding numerical variable
    numerical_var_levels={"Age": 6, "LungFunctionFEV1": 6},
    uni_hfill=0.9,
    uni_vfill=0.99,
    save_path="../../image/gallery/asthma_clinical_profile.png",
)
