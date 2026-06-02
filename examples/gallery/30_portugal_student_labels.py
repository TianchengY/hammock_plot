"""
Changing names, labels and size
===============================

Changing labels of categorical variables, variable names size of names/labels (Portugal Student Data)
"""

from _gallery_utils import data_path, gallery_image_path
import hammock_plot
import pandas as pd

df = pd.read_csv(data_path("portugal_student.csv"))

education_labels = {
    0: "none",
    1: "1-5th grade",
    2: "5-9th grade",
    3: "secondary",
    4: "higher educ",
}
studytime_labels = {
    1: "<2 hrs",
    2: "2-5 hrs",
    3: "5-10 hrs",
    4: ">10 hrs",
}
low_high_labels = {
    1: "very low",
    2: "low",
    3: "medium",
    4: "high",
    5: "very high",
}

df["medu"] = df["medu"].map(education_labels)
df["fedu"] = df["fedu"].map(education_labels)
df["studytime"] = df["studytime"].map(studytime_labels)
df["goout"] = df["goout"].map(low_high_labels)
df["dalc"] = df["dalc"].map(low_high_labels)
df["walc"] = df["walc"].map(low_high_labels)


plot_vars = [
    "medu",
    "fedu",
    "studytime",
    "failures",
    "goout",
    "walc",
    "g1",
    "g2",
    "g3",
]

display_names = {
    "medu": "mother\neducation",
    "fedu": "father\neduction",
    "goout": "go out",
    "walc": "weekly\nalcohol",
}

label_options = {var: {"fontsize": 14, "color": "#006D77"} for var in plot_vars}

hammock = hammock_plot.Hammock(data_df=df)
ax = hammock.plot(
    var=plot_vars,
    value_order={
        "medu": ["none", "1-5th grade", "5-9th grade", "secondary",
                 "higher educ"],
        "fedu": ["none", "1-5th grade", "5-9th grade", "secondary",
                 "higher educ"],
        "studytime": ["<2 hrs", "2-5 hrs", "5-10 hrs", ">10 hrs"],
        "failures": [0, 1, 2, 3],
        "walc": ["very low", "low", "medium", "high", "very high"],
        "goout": ["very low", "low", "medium", "high", "very high"],
    },
    display_type={"g1": "box", "g2": "box", "g3": "box"},
    same_scale=["g1", "g2", "g3"],
    missing=True,
    height=8,
    width=14,
    uni_vfill=0.99,
    connector_fraction=0.2,
    label=True,
    label_options=label_options,
    default_color="#D95F45",
)

ax.set_xticklabels([display_names.get(var, var) for var in plot_vars])
for tick_label in ax.get_xticklabels():
    tick_label.set_color("black")

ax.get_figure().savefig(gallery_image_path("portugal_student_labels.png"))
