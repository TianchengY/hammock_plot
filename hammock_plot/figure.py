# figure.py
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
from hammock_plot.shapes import Rectangle, Parallelogram, FigureBase
from hammock_plot.unibar import Unibar
import pandas as pd
from hammock_plot.utils import Defaults
import numpy as np
from collections import defaultdict

class Figure:
    """
        Initializes a Figure.
        Parameters:
        df:
        - is computed to have a "color_index" column
        - either has missing variables labeled as missing_placeholder or missing values are dropped
        var_list: the variable list
        value_order:
        - dictionary that contains the order that variables are listed
        - is populated with each variable
        colors:
        -  is a List: [default color] + highlight colors
        same_scale_type:
        - determines if same_scale is for numerical or categorical data
        var_types:
        - Dict of the types of each variable. Either: np.str_, np.floating, or np.integer

        numerical_var_levels, numerical_display_type, missing, missing_placeholder, label, unibar, hi_box, width, height, uni_fraction, connector_fraction, min_bar_height, space, label_options, shape_type, same_scale, violin_bw_method: refer to README file
    """
    def __init__(self,
                # general
                df: pd.DataFrame,
                var_list: List[str],
                value_order: Dict[str, List[str]],
                numerical_var_levels:  Dict[str, int],
                numerical_display_type,#: Dict[str, str],
                missing: bool,
                missing_placeholder: str,
                label: bool,
                unibar: bool,

                # highlighting
                hi_box: str,
                colors: List[str],

                # Layout
                width: float,
                height: float,
                uni_fraction: float,
                connector_fraction: float,
                min_bar_height: float,
                space: float,

                # Other
                label_options: dict,
                shape_type,
                same_scale,
                same_scale_type,
                var_types,
                violin_bw_method
                ):

        self.var_list = var_list
        self.value_order = value_order
        self.missing = missing
        self.missing_placeholder = missing_placeholder
        self.label = label
        self.unibar = unibar

        self.hi_box = hi_box
        self.colors = colors # is a List: [default color] + highlight colors
        
        self.data_df = df

        self.width = width # width of the entire plot
        self.height = height # height of the entire plot
        self.uni_fraction = uni_fraction
        self.connector_fraction = connector_fraction
        self.min_bar_height = min_bar_height
        self.space = space
        
        self.label_options = label_options
        self.fig_painter: FigureBase = Rectangle() if shape_type == "rectangle" else Parallelogram()
        self.shape = shape_type

        self.scale = Defaults.SCALE
        self.xmargin = Defaults.XMARGIN
        self.ymargin = Defaults.YMARGIN
        self.bar_unit = Defaults.BAR_UNIT

        # slight gap that should be in between unibars and multivariate connectors
        self.gap_btwn_uni_multi = Defaults.GAP_BTWN_UNI_MULTI if unibar or label else 0
        
        # build and layout unibars
        self.unibars: List[Unibar] = []

        # initialize the unibars
        self.build_unibars(var_types=var_types,
                           numerical_display_type=numerical_display_type,
                           numerical_var_levels=numerical_var_levels,
                           violin_bw_method=violin_bw_method)

        # layout the unibars
        self.layout_unibars(same_scale, same_scale_type)
    
    """
        adds a unibar to the list of unibars maintained in a Figure
    """
    def add_unibar(self, unibar: Unibar):
        self.unibars.append(unibar)
    
    """
        initializes unibars with data
    """
    def build_unibars(self, var_types, numerical_display_type, numerical_var_levels, violin_bw_method):
        # Build unibars
        for i, v in enumerate(self.var_list):
            dtype = var_types[v]
            order = self.value_order[v]
            label_opts = self.label_options[v] if self.label_options and v in self.label_options else None

            # -------------- DETERMINE DISPLAY AND LABEL TYPES ---------------------
            display_type = "rugplot" # default
            if numerical_display_type and v in numerical_display_type:
                display_type = numerical_display_type[v]
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
                
            # long boolean expression represents the conditions for drawing small white lines to divide rugplot rectangles
            draw_white_dividers = display_type == "rugplot" and dtype == np.str_ and self.uni_fraction == 1 and not self.missing

            uni = Unibar(
                df=self.data_df,
                name=v,
                val_type=dtype,
                unibar=self.unibar,
                label=self.label,
                missing=self.missing,
                missing_placeholder=self.missing_placeholder,
                val_order=order,
                min_bar_height=self.min_bar_height,
                colors=self.colors,
                hi_box=self.hi_box,
                display_type = display_type,
                label_type = label_type,
                num_levels = num_levels,
                label_options=label_opts,
                violin_bw_method=violin_bw_method,
                draw_white_dividers=draw_white_dividers
            )

            self.add_unibar(uni)

    """
        Layout the unibars on the axes
    """
    def layout_unibars(self, same_scale, same_scale_type):
        n = len(self.unibars)
        if n == 0:
            return
        
        # ------------------- ADJUST VARIABLES FOR DRAWING ---------------------------
        available_height = (self.height - 2 * self.ymargin * self.height) * self.scale * self.uni_fraction
        total_occurrences = len(self.data_df)
        
        # avoid divide by 0
        if total_occurrences > 0:
            self.bar_unit = available_height / total_occurrences

        # find the maximum missing occurrences to determine how large the missing padding should be
        max_missing_occ = max(
            sum(v.occurrences for v in uni.values if str(v.id) == self.missing_placeholder)
            for uni in self.unibars
        )
        max_missing_height = (max_missing_occ / total_occurrences) * available_height

        # set bar_unit in unibars, set missing_padding in unibars
        for unibar in self.unibars:
            unibar.set_measurements(bar_unit=self.bar_unit,
                                    missing_padding=max(self.min_bar_height, max_missing_height) + Defaults.SPACE_ABOVE_MISSING)
        
        # determine same_scale positioning
        if same_scale_type and same_scale_type == "numerical":
            # Determine ranges for unibars that should use same_scale
            global_range = None
            if same_scale:
                # Collect all numeric values across the same_scale group
                combined_vals = []
                for uni_name in same_scale:
                    uni_series = self.data_df[uni_name]
                    numeric_vals = pd.to_numeric(uni_series, errors="coerce").dropna()
                    combined_vals.extend(numeric_vals.tolist())

                if combined_vals:
                    global_min, global_max = min(combined_vals), max(combined_vals)
                    # Assign the same global range to all unibars in same_scale
                    for uni_name in same_scale:
                        global_range = (global_min, global_max)
                
                # set variables so that same_scale variables align with each other
                max_min_occ = 0 # maximum occurrences observed across all values that are at the minimum value on same_scale
                max_max_occ = 0 # maximum occurrences observed across all values that are at the maximum value on same_scale
                for uni in self.unibars:
                    if uni.name in same_scale:
                        for val in uni.values:
                            if val.numeric == global_min:
                                max_min_occ = max(val.occurrences, max_min_occ)
                            if val.numeric == global_max:
                                max_max_occ = max(val.occurrences, max_max_occ)
                # determine the centres of the items that should be at min, max
                min_max_pos = (max_min_occ * self.bar_unit / 2, max_max_occ * self.bar_unit / 2)
                # set the positions of the minimum and maximum values for each unibar in same_scale
                for uni in self.unibars:
                    if uni.name in same_scale:
                        uni.range = global_range
                        uni.min_max_pos = min_max_pos

        elif same_scale_type and same_scale_type == "categorical":
            # determine the positions of the first and last categories to make them line up
            if same_scale:
                max_btm_occ = 0
                max_top_occ = 0
                for uni in self.unibars:
                    if uni.name in same_scale:
                        for val in uni.values:
                            if val.id == self.value_order[uni.name][0]:
                                max_btm_occ = max(max_btm_occ, val.occurrences)
                            if val.id == self.value_order[uni.name][-1]:
                                max_top_occ = max(max_top_occ, val.occurrences)
                min_max_pos = (max_btm_occ * self.bar_unit / 2, max_top_occ * self.bar_unit / 2)

                for uni in self.unibars:
                    if uni.name in same_scale:
                        uni.min_max_pos = min_max_pos

        # ----------------------- set specific drawing parameters for the unibars --------------------------
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

    """
        Draws the unibars at the appropriate spots on the axes
    """
    def draw_unibars(self, alpha, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(self.width, self.height))

        # set axes limits
        ax.set_xlim(0, self.scale * self.width)
        ax.set_ylim(0, self.scale * self.height)
        ax.set_yticks([])

        # xticks + labels
        ax.set_xticks([c.pos_x for c in self.unibars])
        ax.tick_params(axis='x', length=0) 
        ax.set_xticklabels([c.name for c in self.unibars])

        # apply per-label formatting
        for label, c in zip(ax.get_xticklabels(), self.unibars):
            if self.label_options:
                opts = self.label_options.get(c.name, {})
                if opts:
                    label.set(**opts)

        # spacing for x ticks
        ax.tick_params(axis='x', which='major', pad=10)

        # draw the unibars
        rect_painter = self.fig_painter
        for uni in self.unibars:
            uni.draw(
                ax,
                rectangle_painter=rect_painter,
                y_start=self.y_start,
                y_end=self.y_end,
                alpha=alpha,
            )

        return ax

    def draw_connections(self, alpha, ax=None):
        if self.multi_width == 0:
            return ax

        shape_painter = self.fig_painter

        # Get geometry data depending on the shape type
        if self.shape == "rectangle":
            conn_data = self.get_connect_params()
        elif self.shape == "parallelogram":
            conn_data = self.get_connect_params()
        else:
            raise ValueError(f"Unknown connection shape: {self.shape}")

        # Draw each connection group
        for left_center_pts, right_center_pts, heights, weights in conn_data:
            shape_painter.plot(
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


    def _group_pairs(self, left_uni, right_uni):
        """Return a dictionary of connection pairs with color weights."""
        grouped = (
            self.data_df
            .groupby([left_uni.name, right_uni.name, "color_index"], observed=True)
            .size()
            .to_dict()
        )

        pairs = defaultdict(lambda: [0.0] * len(self.colors))
        for (lv, rv, color_idx), cnt in grouped.items():
            ci = int(color_idx) if str(color_idx).isdigit() else 0
            ci = ci if 0 <= ci < len(self.colors) else 0
            pairs[(lv, rv)][ci] += float(cnt)

        return pairs


    def _compute_stacked_centers(self, left_uni, right_uni, pairs):
        """
        Compute vertical centers for each connection.
        For parallelogram, spacing is expanded according to slope.
        """
        outgoing = defaultdict(list)
        incoming = defaultdict(list)

        right_index = {v.id: idx for idx, v in enumerate(right_uni.values)}
        left_index = {v.id: idx for idx, v in enumerate(left_uni.values)}

        # Prepare connections (keep zero heights)
        for (lv, rv), wts in pairs.items():
            height = sum(wts) * self.bar_unit * self.connector_fraction
            outgoing[lv].append((rv, wts, height))
            incoming[rv].append((lv, wts, height))

        def stack_connections(conns, obj, index_map):
            """Compute stacked centers for one side (left or right)."""
            conns.sort(key=lambda c: index_map.get(str(c[0]), 0))
            total_h = obj.occurrences * self.bar_unit * self.connector_fraction
            bottom_y = obj.vert_centre - total_h / 2.0
            current_y = bottom_y
            new_conns = []

            for other_id, wts, h in conns:
                if self.shape == "parallelogram" and h > 0:
                    # compute vertical spacing for parallelogram (centers only)
                    # width along x will be used for slope, vertical_h used only for spacing
                    center_y = current_y + h / 2.0
                    current_y += h / np.cos(0)  # placeholder, actual slope used later
                else:
                    center_y = current_y + h / 2.0
                    current_y += h

                new_conns.append((other_id, wts, h, center_y))
            return new_conns

        # Stack left
        for lv, conns in outgoing.items():
            lv_obj = left_uni.get_value_by_id(str(lv))
            if lv_obj is None:
                continue
            outgoing[lv] = stack_connections(conns, lv_obj, right_index)

        # Stack right
        for rv, conns in incoming.items():
            rv_obj = right_uni.get_value_by_id(str(rv))
            if rv_obj is None:
                continue
            incoming[rv] = stack_connections(conns, rv_obj, left_index)

        return outgoing, incoming


    def get_connect_params(self):
        """
        Generic method for both rectangle and parallelogram connections.
        Returns a list of tuples: (left_pts, right_pts, heights, weights)
        """
        conn_data = []

        for i in range(len(self.unibars) - 1):
            left_uni = self.unibars[i]
            right_uni = self.unibars[i + 1]

            pairs = self._group_pairs(left_uni, right_uni)
            outgoing, incoming = self._compute_stacked_centers(left_uni, right_uni, pairs)

            left_center_pts, right_center_pts, heights, weights = [], [], [], []

            for (lv, rv), wts in pairs.items():
                total_cnt = sum(wts)

                lv_obj = left_uni.get_value_by_id(str(lv))
                rv_obj = right_uni.get_value_by_id(str(rv))
                if lv_obj is None or rv_obj is None:
                    continue

                # stacked centers
                ly = next((cy for r, _, _, cy in outgoing[lv] if r == rv), lv_obj.vert_centre)
                ry = next((cy for l, _, _, cy in incoming[rv] if l == lv), rv_obj.vert_centre)

                lx = left_uni.pos_x + self.unibar_width / 2 + self.gap_btwn_uni_multi
                rx = right_uni.pos_x - self.unibar_width / 2 - self.gap_btwn_uni_multi

                if self.shape == "parallelogram" and total_cnt > 0:
                    # adjust centers for slant
                    alpha_local = np.arctan(abs(ly - ry) / abs(lx - rx)) if lx != rx else np.pi / 2
                    vertical_h = (total_cnt * self.bar_unit * self.connector_fraction) / np.cos(alpha_local)
                    ly += (vertical_h - total_cnt * self.bar_unit * self.connector_fraction)/2
                    ry -= (vertical_h - total_cnt * self.bar_unit * self.connector_fraction)/2

                left_center_pts.append((lx, ly))
                right_center_pts.append((rx, ry))
                heights.append(total_cnt * self.bar_unit * self.connector_fraction)  # true height
                weights.append(wts)

            conn_data.append((left_center_pts, right_center_pts, heights, weights))

        return conn_data