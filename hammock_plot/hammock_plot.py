# hammock_plot.py
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from figure import Figure
from utils import Defaults
import numpy as np
from utils import safe_numeric, validate_expression, resolve_ordering
import warnings

class Hammock:
    def __init__(self, data_df: pd.DataFrame):
        if data_df is None or data_df.empty:
            raise ValueError("data_df must be provided and non-empty.")
        self.data_df = data_df.copy()

    def plot(self,
             # General
             var: List[str] = None,
             value_order: Dict[str, List[str]] = None,
             numerical_var_levels: Dict[str, int] = None,
             missing: bool = False,
             missing_placeholder: str = Defaults.MISSING_PLACEHOLDER,
             label: bool = True,
             unibar: bool = True,

             # Highlighting
             hi_var: str = None,
             hi_value=None,
             hi_box: str = "vertical",
             hi_missing: bool = False,
             colors: list = Defaults.COLORS,
             default_color: str = Defaults.DEFAULT_COLOR,

             # Layout
             uni_fraction: float = Defaults.UNI_FRACTION, 
             space: float = Defaults.SPACE, 
             label_options: dict = None,
             height: float = 10,
             width: float = 15,
             min_bar_height: float = Defaults.MIN_BAR_HEIGHT,

             # Other
             shape: str = "rectangle",
             same_scale: list = None,
             display_figure: bool = True,
             save_path: str = None):
        
        data_df_columns = self.data_df.columns.tolist()
        
        # no variable names passed
        if var is None or len(var) == 0:
            raise ValueError("There must be some variable names passed to the argument 'var'.")
        
        # invalid variable name passed
        if not set(var) <= set(data_df_columns):
            error_values = (set(var) ^ set(data_df_columns)) & set(var)
            raise ValueError(
                f'the variables: {error_values} in var_lst is not in data or value names user given does not match the data '
            )
        
        # make dictionary with variable types
        var_types = {}
            
        for varname in var:
            temp = self.data_df[varname].dropna()
            datatype = temp.dtype
            if np.issubdtype(datatype, np.integer):
                var_types[varname] = np.integer
            elif np.issubdtype(datatype, np.number):
                if (temp == temp.astype(int)).all():
                    var_types[varname] = np.integer
                else: var_types[varname] = np.floating
            else:
                var_types[varname] = np.str_
    
        if numerical_var_levels:
            # invalid variable name passed to numerical_var_levels dictionary
            if not set(numerical_var_levels.keys()).issubset(set(var)):
                error_values =  set(numerical_var_levels.keys()) - set(var)
                raise ValueError(
                    f'The value: {error_values} in numerical_var_levels is not in data.'
                )
            
            # validating that all variables in numerical_var_levels is numeric data type
            for k, v in numerical_var_levels.items():
                if var_types[k] == np.str_: # categorical data type
                    raise ValueError(
                        f'{k} is a categorical data type and thus cannot belong to numerical_var_levels'
                    )
                if not (v is None or isinstance(v, int)):
                    raise ValueError(
                        f'You can only specify integer values or None for the number of levels for variables labeled by intervals'
                    )
                if v is not None and v < 0:
                    raise ValueError(
                        f'Levels must be nonnegative: error with pair ({k}, {v})'
                    )
        
        # 'color' argument is not a list of strings
        if colors and type(colors) != type([]):
            raise ValueError(
                f'Argument "colors" must be a list of str.'
            )
        for color in colors:
            if type(color) != str:
                raise ValueError(
                    f'Argument "colors" must be a list of str.'
                )
        
        # invalid variable name in same_scale list
        if same_scale and not set(same_scale) <= set(var):
            # gets a list of the items that are in same_scale but not in data_df_columns
            error_values = set(same_scale) - (set(same_scale) & set(var))
            raise ValueError(
                f'the variables: {error_values} in same_scale is not in var_lst or value names user given does not match the data '
            )
        
        same_scale_type = None

        if same_scale:
            # should all be categorical or numerical
            cur_type = var_types[same_scale[0]]
            if cur_type == np.str_: # categorical data type
                same_scale_type = "categorical"
                for cur_var in same_scale:
                    if np.issubdtype(var_types[cur_var], np.integer) or np.issubdtype(var_types[cur_var], np.floating):
                        raise ValueError(
                            "Variables in same_scale must either all be numerical or all be categorical."
                        )
            else: # numerical data type
                same_scale_type = "numerical"
                for cur_var in same_scale:
                    if var_types[cur_var] == np.str_:
                        raise ValueError(
                            "Variables in same_scale must either all be numerical or all be categorical."
                        )
                
        if same_scale and value_order:
            orders = []
            for cur_var in same_scale:
                if cur_var in value_order:
                    orders.append(value_order[cur_var])
            resolved_order = resolve_ordering(orders)
            if resolved_order == None:
                raise ValueError(
                    "value_order has conflict with same_scale."
                )
            else:
                for cur_var in same_scale:
                    # set the new value_order to be the full, resolved order
                    value_order[cur_var] = resolved_order
        elif same_scale and same_scale_type == "categorical":
            combined_uni_vals = []
            seen = set()
            for cur_var in same_scale:
                uni_vals = list(self.data_df[cur_var].dropna().unique())
                for val in uni_vals:
                    if val not in seen:
                        seen.add(val)
                        combined_uni_vals.append(val)
            # now set the combined order for all variables
            if not value_order: value_order = {}
            for cur_var in same_scale:
                value_order[cur_var] = sorted(combined_uni_vals)
                    
        
        # highlight variable is not in data
        if hi_var and not hi_var in data_df_columns:
            raise ValueError(
                f'highlight variable is not in data. '
            )

        if hi_missing and not missing:
            raise ValueError(
                f'missing must be True if hi_missing is True.'
            )
        
        # hi_value or hi_missing not specified when hi_var was specified
        if hi_var and not (hi_value or hi_missing):
            raise ValueError(
                f'hi_value or hi_missing must be speicified as hi_var is given.'
            )
        
        # check if hi_value is valid
        if hi_var != None and hi_value != None:
            if type(hi_value) == list:
                hi_var_unique_set = set(self.data_df[hi_var].unique())
                hi_value_set = set(hi_value)
                hi_var_unique_set = {safe_numeric(v) for v in hi_var_unique_set}
                hi_value_set = {safe_numeric(v) for v in hi_value_set}
                # check if hi_value is not a subset of all values of hi_var in the dataset
                if not hi_value_set <= hi_var_unique_set:
                    error_values = (set(hi_value) ^ hi_var_unique_set) & set(hi_value)
                    raise ValueError(
                        f'The value(s): {error_values} in hi_value is not in data.'
                    )
            else: # check the edge case where hi_value is an expression of a range for the numeric hi_var
                if not validate_expression(hi_value):
                    raise ValueError(
                        f'Invalid expression: {hi_value}.'
                    )
                else:
                    if var_types[hi_var] == np.str_:
                        raise ValueError(
                            "Range based highlighting for categorical variables is not allowed."
                        )

        num_hi_colors = len(hi_value) if isinstance(hi_value, list) else 0
        num_hi_colors = 1 if isinstance(hi_value, str) else num_hi_colors
        num_hi_colors += 1 if hi_missing else 0

        colors = colors[0:num_hi_colors] if len(colors) >= num_hi_colors else colors
        # default colour in colors
        if hi_var != None and default_color in colors:
            raise ValueError(
                f'The current highlight colors {colors} conflict with the default color {default_color}. Please choose another default color or other highlight colors'
            )
        
        # automatically extend colors
        if hi_var != None:
            if num_hi_colors > len(colors):
                for _ in range(num_hi_colors - len(colors)):
                    for c in Defaults.COLORS:
                        if c not in colors:
                            colors.append(c)
                            break
                warnings.warn(
                    f"Warning: The length of color is less than the total number of (hi values and missing), color was automatically extended to {colors}")

        # shape check
        if shape != "parallelogram" and shape != "rectangle":
            raise ValueError(
                f"Invalid shape {shape} provided. shape must be either 'parallelogram' or 'rectangle'."
            )
            
        fig = Figure.from_dataframe(
            # general
            self.data_df,
            var_list=var,
            value_order=value_order,
            numerical_var_levels=numerical_var_levels,
            missing=missing,
            missing_placeholder=(missing_placeholder if missing else None),
            label=label,
            unibar=unibar,

            # highlighting
            hi_var=hi_var,
            hi_value=hi_value,
            hi_box=hi_box,
            hi_missing=hi_missing,
            default_color=default_color,
            colors=colors,

            # Layout
            width=width,
            height=height,
            uni_fraction=uni_fraction,
            min_bar_height=min_bar_height,
            space=space,
            
            # Other
            shape_type=shape,
            same_scale=same_scale,
            same_scale_type=same_scale_type,
        )

        ax = fig.draw_unibars(label_opts=label_options)
        ax = fig.draw_connections(ax=ax)

        if save_path:
            ax.get_figure().savefig(save_path)
        if display_figure:
            return ax
        else:
            plt.close()
            return None
