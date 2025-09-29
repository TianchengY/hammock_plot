# figure.py
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from hammock_plot.shapes import Rectangle, Parallelogram, FigureBase
from hammock_plot.unibar import Unibar
from hammock_plot.value import Value
import pandas as pd
from hammock_plot.utils import clean_expression, is_in_range, validate_expression
from hammock_plot.utils import Defaults
import re

class Figure:
    def __init__(self,
                width: float,
                height: float,
                uni_fraction: float,
                min_bar_height: float,
                space: float,

                unibar: bool,
                label: bool,
            
                colors: List[str],
                default_color: str,
                hi_var: str,
                hi_value: List[str],
                missing_placeholder: str,
                hi_missing: bool,

                value_order: Dict[str, List[str]],

                shape_type: str,
                same_scale: List[str],
                ):
        
        self.height = height
        self.width = width
        self.uni_fraction = uni_fraction
        self.space = space
        self.min_bar_height = min_bar_height

        self.scale = Defaults.SCALE
        self.xmargin = Defaults.XMARGIN
        self.ymargin = Defaults.YMARGIN
        self.bar_unit = Defaults.BAR_UNIT

        self.unibar = unibar
        self.label = label
        self.unibars: List[Unibar] = []
        self.fig_painter: FigureBase = Rectangle() if shape_type == "rectangle" else Parallelogram()

        self.value_order = value_order
        
        self.colors = [default_color] + colors
        self.default_color = default_color
        self.missing_placeholder = missing_placeholder
        self.hi_missing = hi_missing

        self.hi_var = hi_var
        self.hi_value = hi_value

        self.same_scale = same_scale

        self.gap_btwn_uni_multi = Defaults.GAP_BTWN_UNI_MULTI if unibar or label else 0

    # -----------------------------
    # Highlight helpers (instance methods)
    # -----------------------------
    def check_hi_var(self, varname: str) -> bool:
        """Return True if this variable should be highlighted."""
        return varname in self.hi_var

    def check_hi_value(self, value: Any) -> bool:
        """
        Return True if this value should be highlighted.
        Supports:
          - list of literal values
          - regex pattern (if hi_value is a string and compiles as regex)
          - logical expression (if hi_value is a string that looks like an expression)
        """
        if self.hi_value is None:
            return False

        # Case 1: Literal list
        if isinstance(self.hi_value, list):
            return value in self.hi_value

        # Case 2: Regex string
        if isinstance(self.hi_value, str):
            try:
                regex = re.compile(self.hi_value)
                if isinstance(value, str) and regex.search(value):
                    return True
            except re.error:
                pass  # not valid regex, try as expression instead

            # Case 3: Expression string
            try:
                if isinstance(value, (int, float)):
                    return is_in_range(value, self.hi_value)
            except ValueError:
                return False

        return False
    
    # -----------------------------
    # Color indexing helpers
    # -----------------------------
    def assign_color_index(self, df: pd.DataFrame, var_list: List[str]) -> pd.DataFrame:
        df["color_index"] = 0  # default

        # Highlight missing values first
        if self.hi_missing and self.missing_placeholder is not None:
            for v in var_list:
                if v != self.hi_var:
                    continue
                df.loc[df[v] == self.missing_placeholder, "color_index"] = 1

        # Then apply hi_value highlighting, but only where color_index is still 0
        if self.hi_var and self.hi_value is not None:
            for v in var_list:
                if v != self.hi_var:
                    continue
                mask = df["color_index"] == 0
                df.loc[mask, "color_index"] = df.loc[mask, v].apply(self._compute_color_index)

        #print(df[[self.hi_var, "color_index"]])
        return df


    def _compute_color_index(self, val: Any) -> int:
        # if hi_missing is true, increase each index by 1
        missing_buffer = 1 if self.hi_missing else 0

        if isinstance(self.hi_value, list):
            try:
                idx = self.hi_value.index(val) + 1 + missing_buffer
                return idx
            except ValueError:
                return 0

        if isinstance(self.hi_value, str):
            # regex
            try:
                regex = re.compile(self.hi_value)
                if isinstance(val, str) and regex.search(val):
                    return 1 + missing_buffer
            except re.error:
                pass
            # expression
            try:
                if isinstance(val, (int, float)) and is_in_range(val, self.hi_value):
                    return 1 + missing_buffer
            except ValueError:
                return 0

        return 0
    
    def add_unibar(self, unibar: Unibar):
        self.unibars.append(unibar)

    def layout_unibars(self):
        n = len(self.unibars)
        if n == 0:
            return

        # Use margins as fractions of width/height
        edge_x = self.xmargin * self.width * self.scale
        edge_y = self.ymargin * self.height * self.scale

        # Plotting extents
        x_start = edge_x
        x_end = self.width * self.scale - edge_x
        y_start = edge_y
        y_end = self.height * self.scale - edge_y

        x_total = x_end - x_start  # total drawable width

        # --- slot-based unibar math ---
        raw_width = x_total / n                     # slot width for each unibar
        unibar_width = raw_width * self.space

        self.unibar_width = unibar_width if self.unibar or self.label else 0

        # Compute leftover spacing inside each slot for connections
        multi_width = raw_width - unibar_width - 2 * self.gap_btwn_uni_multi

        if multi_width < Defaults.MIN_MULTI_WIDTH:
            multi_width = 0
        
        self.multi_width = multi_width

        # Centers of slots (always fill x_start..x_end)
        xs = [x_start + raw_width/2 + i * raw_width for i in range(n)]

        # Assign positions
        for uni, x in zip(self.unibars, xs):
            uni.set_measurements(pos_x=x)
            uni.missing_placeholder = self.missing_placeholder

        # Compute vertical layout with consistent margins
        for uni in self.unibars:
            uni.set_measurements(width = unibar_width)
            uni.compute_vertical_positions(
                y_start=y_start,
                y_end=y_end
            )

        # Store useful attributes
        self.y_start = y_start
        self.y_end = y_end
        self.edge_y = edge_y
        self.scale_x = self.scale * self.width
        self.scale_y = self.scale * self.height


    def draw_unibars(self, alpha, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(self.width, self.height))
        ax.set_xlim(0, self.scale * self.width)
        ax.set_ylim(0, self.scale * self.height)
        ax.set_yticks([])
        ax.set_xticks([c.pos_x for c in self.unibars])
        ax.set_xticklabels([c.name for c in self.unibars])
        ax.tick_params(axis='x', which='major', pad=10)
        rect_painter = self.fig_painter
        for uni in self.unibars:
            uni.draw(ax,
                    rectangle_painter=rect_painter,
                    bar_unit=self.bar_unit,
                    y_start=self.y_start,
                    y_end=self.y_end,
                    alpha=alpha)
        return ax

    def draw_connections(self, alpha, ax=None):
        # nothing to draw if no multi width (no room for connections)
        if self.multi_width == 0:
            return ax

        rect_painter = self.fig_painter

        # For each adjacent pair of unibars, build connection polygons
        for i in range(len(self.unibars) - 1):
            left_uni = self.unibars[i]
            right_uni = self.unibars[i + 1]

            left_name = left_uni.name
            right_name = right_uni.name

            # Group by left, right and color_index to get counts per colour for each pair
            grouped = (
                self.data_df
                .groupby([left_name, right_name, "color_index"], observed=True)
                .size()
                .to_dict()
            )

            # Aggregate into pair -> weight-vector
            pairs: Dict[Tuple[Any, Any], List[float]] = {}
            for (lv, rv, color_idx), cnt in grouped.items():
                key = (lv, rv)
                if key not in pairs:
                    pairs[key] = [0.0] * len(self.colors)
                # ensure color_idx is an int and within range
                try:
                    ci = int(color_idx)
                except Exception:
                    ci = 0
                if 0 <= ci < len(self.colors):
                    pairs[key][ci] += float(cnt)
                else:
                    pairs[key][0] += float(cnt)

            # Build geometry arrays for this unibar pair
            left_center_pts = []
            right_center_pts = []
            heights = []
            weights = []

            for (lv, rv), wts in pairs.items():
                total_cnt = sum(wts)
                if total_cnt <= 0:
                    continue

                # find corresponding Value objects to get vertical centres
                lv_obj = left_uni.get_value_by_id(str(lv))
                rv_obj = right_uni.get_value_by_id(str(rv))
                if lv_obj is None or rv_obj is None:
                    # If get_value_by_id fails, skip this pair
                    continue

                # x positions: just inside each unibar toward the middle gap
                lx = left_uni.pos_x + self.unibar_width / 2 + self.gap_btwn_uni_multi
                rx = right_uni.pos_x - self.unibar_width / 2 - self.gap_btwn_uni_multi

                ly = lv_obj.vert_centre
                ry = rv_obj.vert_centre

                left_center_pts.append((lx, ly))
                right_center_pts.append((rx, ry))
                heights.append(total_cnt * self.bar_unit)
                weights.append(wts)

            # draw the batch for this adjacent pair
            if left_center_pts:
                rect_painter.plot(
                    ax=ax,
                    alpha=alpha,
                    left_center_pts=left_center_pts,
                    right_center_pts=right_center_pts,
                    heights=heights,
                    colors=self.colors,
                    weights=weights,
                    orientation="horizontal",
                )

        return ax

    @classmethod
    def from_dataframe(cls,
                        # general
                        df: "pd.DataFrame",
                        var_list: List[str],
                        value_order: Dict[str, List[str]],
                        numerical_var_levels:  Dict[str, int],
                        numerical_display_type,#: Dict[str, str],
                        missing: bool,
                        missing_placeholder: str,
                        label: bool,
                        unibar: bool,

                        # highlighting
                        hi_var: str,
                        hi_value,
                        hi_missing: bool,
                        hi_box: str,
                        default_color: str,
                        colors: List[str],

                        # Layout
                        width: float,
                        height: float,
                        uni_fraction: float,
                        min_bar_height: float,
                        space: float,

                        # Other
                        label_options: dict,
                        shape_type,
                        same_scale,
                        same_scale_type,
                        var_types,
                    ):

        fig = cls(width = width,
                  height = height,
                  uni_fraction = uni_fraction,
                  min_bar_height = min_bar_height,
                  space=space,

                  unibar=unibar,
                  label=label,
                  
                  colors = colors,
                  default_color = default_color,
                  hi_var = hi_var,
                  hi_value = hi_value,
                  hi_missing = hi_missing,
                  missing_placeholder = missing_placeholder,
                  value_order = value_order,

                  shape_type = shape_type,
                  same_scale = same_scale)

        data_df = df.copy()
        
        # # Precompute unibar data types
        # var_types = {}

        # for varname in var_list:
        #     temp = data_df[varname].dropna()
        #     datatype = temp.dtype
        #     if np.issubdtype(datatype, np.integer):
        #         var_types[varname] = np.integer
        #     elif np.issubdtype(datatype, np.number):
        #         # Check if all numbers are effectively integers
        #         if (temp == temp.astype(int)).all():
        #             var_types[varname] = np.integer
        #         else:
        #             var_types[varname] = np.floating
        #     else:
        #         var_types[varname] = np.str_
        
        if missing_placeholder is not None:
            data_df = data_df.fillna(missing_placeholder)
        else:
            data_df = data_df.dropna()

        data_df = fig.assign_color_index(data_df, var_list)
        
        fig.data_df = data_df
        
        colors = [default_color] + colors if colors else [default_color]

        # Build unibars
        for i, v in enumerate(var_list):
            uni_series = data_df[v]
            dtype = var_types[v]

            # ordering: use value_order if provided, else default sort
            if value_order and v in value_order:
                order = value_order[v]
            else:
                uniq = uni_series.dropna().unique().tolist()
                order = uniq

            display_type = "rugplot" # default
            if numerical_display_type and v in numerical_display_type:
                display_type = numerical_display_type[v]
            # display_type = numerical_display_type[i]
            label_type = "default"

            num_levels = Defaults.NUM_LEVELS # default num levels

            if display_type == "violin" or display_type == "box":
                label_type = "levels"

            if numerical_var_levels and v in numerical_var_levels.keys():
                if numerical_var_levels[v]:
                    label_type="levels"
                    num_levels = numerical_var_levels[v]
                elif display_type == "rugplot": # v: None - labels are by value only if display is rugplot
                    label_type = "values"

            label_opts = label_options[v] if label_options and v in label_options else None

            uni = Unibar(
                df=data_df,
                name=v,
                val_type=dtype,
                unibar=unibar,
                label=label,
                missing=(missing_placeholder is not None),
                missing_placeholder=missing_placeholder,
                val_order=order,
                min_bar_height=fig.min_bar_height,
                colors=colors,
                hi_box=hi_box,
                display_type = display_type,
                label_type = label_type,
                num_levels = num_levels,
                label_options=label_opts
            )

            fig.add_unibar(uni)

        
        # adjust some variables for drawing
        available_height = fig.height * fig.scale * fig.uni_fraction
        max_total_occurrences = max(sum(v.occurrences for v in uni.values) for uni in fig.unibars)
        fig.bar_unit = available_height / max_total_occurrences

        max_missing_occ = max(
            sum(v.occurrences for v in uni.values if str(v.id) == fig.missing_placeholder)
            for uni in fig.unibars
        )
        missing_padding = max_missing_occ * fig.bar_unit

        for unibar in fig.unibars:
            unibar.set_measurements(bar_unit=fig.bar_unit,
                                    missing_padding=max(min_bar_height, missing_padding) + Defaults.SPACE_ABOVE_MISSING)
            
        if same_scale_type and same_scale_type == "numerical":
            # Determine ranges for unibars that should use same_scale
            range = None
            if same_scale:
                # Collect all numeric values across the same_scale group
                combined_vals = []
                for uni_name in same_scale:
                    uni_series = data_df[uni_name]
                    numeric_vals = pd.to_numeric(uni_series, errors="coerce").dropna()
                    combined_vals.extend(numeric_vals.tolist())

                if combined_vals:
                    global_min, global_max = min(combined_vals), max(combined_vals)
                    # Assign the same global range to all unibars in same_scale
                    for uni_name in same_scale:
                        range = (global_min, global_max)
                
                max_min_occ = 0
                max_max_occ = 0
                for uni in fig.unibars:
                    if uni.name in same_scale:
                        for val in uni.values:
                            if val.numeric == global_min:
                                max_min_occ = max(val.occurrences, max_min_occ)
                            if val.numeric == global_max:
                                max_max_occ = max(val.occurrences, max_max_occ)
                min_max_pos = (max_min_occ * fig.bar_unit / 2, max_max_occ * fig.bar_unit / 2)

                for uni in fig.unibars:
                    if uni.name in same_scale:
                        uni.range = range
                        uni.min_max_pos = min_max_pos

        elif same_scale_type and same_scale_type == "categorical":
            # determine the positions of the first and last categories to make them line up
            if same_scale:
                max_btm_occ = 0
                max_top_occ = 0
                for uni in fig.unibars:
                    if uni.name in same_scale:
                        for val in uni.values:
                            if val.id == value_order[uni.name][0]:
                                max_btm_occ = max(max_btm_occ, val.occurrences)
                            if val.id == value_order[uni.name][-1]:
                                max_top_occ = max(max_top_occ, val.occurrences)
                min_max_pos = (max_btm_occ * fig.bar_unit / 2, max_top_occ * fig.bar_unit / 2)

                for uni in fig.unibars:
                    if uni.name in same_scale:
                        uni.min_max_pos = min_max_pos
        
        fig.layout_unibars()

        return fig