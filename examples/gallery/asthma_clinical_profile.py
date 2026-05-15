"""Gallery example: mixed clinical indicators from the larger asthma dataset."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hammock_plot

DATA = ROOT / "data"
OUT = ROOT / "image" / "gallery"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "data_asthma_2.csv")
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
    colors=["#e15759"],
    default_color="#59a14f",
    display_type={
        "Age": "box",
        "LungFunctionFEV1": "violin",
        "Gender": "bar chart",
        "Smoking": "bar chart",
        "FamilyHistoryAsthma": "bar chart",
    },
    numerical_var_levels={"Age": 6, "LungFunctionFEV1": 6},
    missing=True,
    uni_vfill=0.55,
    connector_fraction=0.3,
    width=14,
    height=8,
    display_figure=False,
    save_path=OUT / "asthma_clinical_profile.png",
)
