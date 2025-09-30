# Hammock plot


## Description

    The hammock plot draws a graph to visualize categorical or mixed categorical / continuous data.
    Variables are lined up parallel to the vertical axis. Categories within a variable are spread out along a
    vertical line. Categories of adjacent variables are connected by boxes. (The boxes are parallelograms; we
    use boxes for brevity). The "width" of a box is proportional to the number of observations that correspond
    to that box (i.e. have the same values/categories for the two variables). The "width" of a box refers to the
    distance between the longer set of parallel lines rather than the vertical distance.

    If the boxes degenerate to a single line, and no labels or missing values are used the hammock plot
    corresponds to a parallel coordinate plot. Boxes degenerate into a single line if barwidth is so small that
    the boxes for categorical variables appear to be a single line. For continuous variables boxes will usually
    appear to be a single line because each category typically only contains one observation.

    The order of variables in varlist determines the order of variables in the graph.  All variables in varlist
    must be numerical. String variables should be converted to numerical variables first, e.g. using encode or
    destring.




## Getting started

You can install hammock from `pip`:

```shell
pip install hammock_plot
```


### Example: Asthma data

We import the diabetes dataset:

```python
import hammock_plot
import pandas as pd
df = pd.read_csv('./data/data_asthma.csv')
```

Minimal example of a hammock plot: 
```python
var = ["hospitalizations","group","gender","comorbidities"]
hammock = hammock_plot.Hammock(data_df = df)
ax = hammock.plot(var=var)
```
<img src="image/asthma_minimal.png" alt="Minimal example for a Hammock plot" width="600"/>

The labels for the numerical variables aren't as desired; we would like the labels directly drawn on the data. We specify that we want no levels for our numerical variables.

```python
numeric_levels = {"comorbidities": None, "hospitalizations": None}
ax = hammock.plot(var=var, numerical_var_levels=numeric_levels)
```

<img src="image/asthma_levels.png" alt="Hammock plot" width="600"/>

The ordering of the child-adolescent-adult variable is not in the desired order; adult should not be in the middle. We now specify a specific order, child-adolescent-adult. 

```python
group_order = ["child", "adolescent", "adult"]
value_order = {"group": group_order}
hammock = hammock_plot.Hammock(data_df = df)
ax = hammock.plot(var=var, value_order=value_order, numerical_var_levels=numeric_levels)
```

<!--- to restrict image size, I am using a an html command, rather than the standard ![](image.png) --->
<!---    ![Hammock plot ](image/asthma1.png)   --->
<img src="image/asthma_value_order.png" alt="Hammock plot" width="600"/>

We highlight observations with comorbidities=0  in red:

```python
ax = hammock.plot(var=var ,hi_var="comorbidities", hi_value=[0], colors=["red"], numerical_var_levels=numeric_levels)
```

<!---   ![Hammock plot with highlighting](image/asthma_highlighting.png)    --->
<img src="image/asthma_highlighting.png" alt="Hammock plot with highlighting" width="600"/>


### Example Satisfaction scales for the diabetes data

We import the diabetes dataset:

```python
import hammock_plot
import pandas as pd
df = pd.read_csv('./data/data_diabetes.csv')
```

The three variables represent different ordinal scales for satisfaction. We are checking for missing values: 
```python
var = ["sataces","satcomm","satrate"]
hammock = hammock_plot.Hammock(data_df = df)
ax = hammock.plot(var=var, missing=True, min_bar_height=0.2,numerical_var_levels={"sataces": None, "satcomm": None, "satrate": None}) 
```

<img src="image/diabetes.png" alt="Hammock plot for the Diabetes Data" width="600"/>

The missing value category is shown at the bottom for each variable. We find missing values for all 3 variables, but fewest for the last one. We also see a phenomenon called "top coding", where 
satisfied respondents simply choose the highest value.

### Example value_order for the Shakespeare data

We import the Shakespeare dataset:

```python
import hammock_plot
import pandas as pd
df = pd.read_csv('./data/data_shakespeare.csv')
```

We use `speaker_dict` to map the values of the variables `speaker1` and `speaker2` according to the social class hierarchy.
```python
var_lst = ["type","speaker1","speaker2","sex1"]
color_lst = ["red","yellow","green"]
hi_value = ["Beggars","Citizens","Gentry"]

speaker_order=["Beggars", "Royalty", "Nobility", "Gentry", "Citizens", "Yeomanry"]

hammock = hammock_plot.Hammock(data_df = data_df)
ax = hammock.plot(var=var_lst,hi_var = "speaker1", hi_value=hi_value,color=color_lst, bar_width=0.6,missing=True,
                value_order ={"speaker1":speaker_order,"speaker2":speaker_order} )
```

<img src="image/shakespeare.png" alt="Hammock plot for the Diabetes Data" width="600"/>



## API Reference

```
  hammock()
```

| Category | Parameter | Type     | Description                |
| --- | :-------- | :------- | :-------------------------  |
| General |     `var` | `List[str]` | List of variables to display. |
| |             `value_order` | `Dict[str, List[int]]`  |  If specified, the order of the values in the plot follows the order of values in the list supplied in the dictionary. Only applicable to categorical variables |
| |            `numerical_var_levels` | `Dict[str, int \| None]` | Specifies the number of subdivisions in the y-axis for numerical variables. Example: {"NumericalVarname": 9, "NumericalVarname2": None}. Default is 7. |
| |            `numerical_display_type` | `Dict[str, str]` | Specifies the type of plot (rugplot, box plot, violin plot) for numerical variable display. Example: {"NumericalVarname": "rugplot", "NumericalVarname2": "violin", "NumericalVarname3": "box"}. Default is "rugplot". |
| |             `missing` | `bool` | Whether or not to add a category for missing values at the bottom of the plot.  If False, observations that have a missing value for any variable in the data frame (even those not used in the hammock plot) are removed.  Default is False. |
| |             `label` | `bool` | Whether or not to display labels between the plotting segments |
| |             `unibar`| `bool` | Whether or not to display unibars between the plotting segments |
| Highlighting | `hi_var` | `str` |  Variable to be highlighted. Default is none. |
| | `hi_value` | `List[str or int] or str or int` | Value(s) of `hi_var` to be highlighted. You can highlighted one or multiple values. You can also pass an expression (e.g. "x>1 and (x>5 or x<4)") in string when you want to specify a range for a numeric hi_var.|
| | `hi_box` | `str` | Controls how highlighted values are displayed within category labels. Options are "vertical" for vertically stacked color segments or "horizontal" for horizontally split color segments. Default is "vertical".|
| | `hi_missing` | `bool` | Whether or not missing values for `hi_var` should be highlighted. |
| | `color` | `List[str]` | List of colors corresponding to the list of values to be highlighted. Each color can be specified as a plain color name (e.g., `"red"`, `"yellow"`) or in the format `"color=alpha"` (e.g., `"red=0.5"`) to control transparency/intensity, where `alpha` is a decimal between 0 and 1. The default highlight color list is `["red", "green", "yellow", "lightblue", "orange", "gray", "brown", "olive", "pink", "cyan", "magenta"]`. |
| | `default_color` | `str` |  Default color of plotting elements for boxes that are not highlighted. Default is "blue" |
| Manipulating Spacing and Layout |   `uni_fraction` | `float`  | Fraction of vertical space that should be populated by data. Adjusts the height of the data points. Defaults is 0.08. | 
| |              `space` |  `float`  |Fraction of horizontal space allocated to labels/univ. bars rather than to connecting boxes. Default is 0.3 | 
| |              `label_options` |  `Dict[str, Dict[str, Any]]`  | Manipulates the size and look of the labels. Args following the options in the website: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html Example:{"ExampleVarname":{"fontsize":12,"fontstyle":"italic","fontweight":"black","color":"b"}}  Default is None. | 
| |              `height` |  `float`  | Height of the plot in inches. Default is 10. | 
| |              `width` |  `float`  |  Width of the plot in inches. Default is 15. Caution: Width too narrow may distort the plot. | 
| |               `alpha` | `float` | Alpha value for the colours in the plot. Float from 0-1. Default is 0.7. |
| |              `min_bar_height` | `float` | Minimal bar height. Bars representing only a tiny fraction of the data may be so narrow, that they are invivisible in a plot. The default value tries to ensure this does not happen.  Default is 0.1.
| Other options |              `shape` |  `str`  | Shape of the boxes. "rectangle" (default) or "parallelogram". | 
| |              `same_scale` |  `List[str]`  | List of variables that have the same scale. Default is None. | 
| |              `display_figure` |  `bool`  | Whether or not to display the figure. This can be useful if you just want to save the plots. Default is 'True'. | 
| |              `save_path` |  `str`  |   If it is not None, the figure will be saved to the given path with given name and format. Default is None. | 


## Historical context

In 1898, Sankey diagrams were developed to visualize flows of energy and materials. 

In 1985, Inselberg popularized parallel coordinates to visualize continuous variables only. The central contribution is the use of parallel axes.

In 2003, Schonlau proposed the hammock plot. This was the first plot to visualize categorical data (or mixed categorical continuous data) on parallel axes. 

In 2010, Rosvall proposed alluvial plots to visualize network variables over time. Rather than using bars to connect axes, alluvial plots use rounded curves. Alluvial plots are now also used to visualize categorical data.

There are several additional variations that also visualize categorical data including Parallel Set plots (Bendix et al, 2005), Right Angle plots (Hofmann and Vendettuoli, 2013),
and generalized parallel coordinate plots (GPCPs) (popularized by VanderPlas et al., 2023). 

### References 
Bendix, F., Kosara, R., & Hauser, H. (2005). Parallel sets: visual analysis of categorical data. In IEEE Symposium on Information Visualization, 2005. INFOVIS 2005. 133-140. 

Hofmann, H., & Vendettuoli, M. (2013). Common angle plots as perception-true visualizations of categorical associations. IEEE transactions on visualization and computer graphics, 19(12), 2297-2305.

Inselberg, A., & Dimsdale, B. (2009). Parallel coordinates. Human-Machine Interactive Systems, 199-233.

Rosvall, Martin, & Bergstrom, C.T. (2010) "Mapping change in large networks." PloS one 5.1: e8694.

Sankey, H. (1898). Introductory note on the thermal efficiency of steam-engines. report of
the committee appointed on the 31st march, 1896, to consider and report to the council
upon the subject of the definition of a standard or standards of thermal efficiency for
steam-engines: With an introductory note. In Minutes of proceedings of the institution
of civil engineers, Volume 134, pp. 278–283.

Schonlau M. 
*[Visualizing Categorical Data Arising in the Health Sciences Using Hammock Plots.](http://www.schonlau.net/publication/03jsm_hammockplot.pdf)* 
In Proceedings of the Section on Statistical Graphics, American Statistical Association; 2003

VanderPlas, S., Ge, Y., Unwin, A., & Hofmann, H. (2023). 
Penguins Go Parallel: a grammar of graphics framework for generalized parallel coordinate plots. 
Journal of Computational and Graphical Statistics, 1-16. (online first)

### Other implementations of the hammock plot 
There is also a Stata implementation `hammock` (available from the Stata archive SSC) and an R implementation as part of the package `ggparallel`.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)



## Authors

- Tiancheng Yang t77yang@uwaterloo.ca


