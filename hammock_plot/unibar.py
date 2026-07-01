# unibar.py
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from hammock_plot.value import Value
from hammock_plot.utils import edge_color_from_face
from .utils import Defaults, get_formatted_label
from scipy.stats import gaussian_kde

class Unibar:
    def __init__(self,
                 df,
                 weights: str,
                 name: str,
                 val_type,
                 unibar: bool,
                 label: bool,
                 missing: bool,
                 missing_placeholder: str,
                 val_order: List[str],
                 min_bar_height,
                 colors,
                 hi_box,
                 num_levels: int,
                 display_type: str,
                 label_type: str,
                 label_options: dict,
                 violin_bw_method,
                 draw_white_dividers):
        self.df = df
        self.weights = weights # None if none...
        self.name = name
        self.display_type = display_type
        self.label_type = label_type
        self.val_type = val_type          # "np.str_" or "np.floating" or "np.integer"
        self.num_levels = num_levels
        self.unibar = unibar
        self.label = label
        self.missing = missing
        self.y_top = 0.0
        self.y_bottom = 0.0 # bottom of non missing values
        self.draw_y_start = 0.0 # adjusted drawable bottom (incl. missing_padding + bottom_adjustment)
        self.draw_y_end = 0.0 # adjusted drawable top (incl. top_adjustment)
        self.missing_placeholder = missing_placeholder
        self.val_order = val_order
        self.min_bar_height = min_bar_height
        self.label_options = label_options
        self.violin_bw_method = violin_bw_method

        self._build_values() # build values list

        self.hi_box = hi_box
        self.colors = colors

        # for same_scale variables
        self.range = None # if numerical, will be a min val and a max val
        self.min_max_pos = None # records the centre positions of the top and bottom values
        self.draw_white_dividers = draw_white_dividers

    def _build_values(
        self,
    ) -> List[Value]:
        """
        Create Value objects for this unibar from self.df.
        Each Value has total occurrences and breakdown by colour_index.
        """
        values: List[Value] = []

        dtype = self.val_type

        # global set of color indices
        all_colors = list(range(int(self.df["color_index"].max()) + 1))

        # Determine order
        order = self.val_order

        # Count occurrences per (value, colour) in one grouped pass rather than
        # scanning the whole frame once per value. Rows = each value, columns =
        # each colour index; weighted sums the weight column, else just counts.
        grouped = self.df.groupby([self.name, "color_index"], observed=True)
        if self.weights is None:
            occ_table = grouped.size().unstack("color_index", fill_value=0)
        else:
            occ_table = grouped[self.weights].sum().unstack("color_index", fill_value=0)
        occ_table = occ_table.reindex(columns=all_colors, fill_value=0)

        for val in order:
            # look up this value's per-colour occurrences. A value that's in the
            # order but absent from the data (e.g. an empty same_scale slot) isn't
            # in the table, so it gets all zeros.
            if val in occ_table.index:
                occ_by_colour = occ_table.loc[val].tolist()
                cnt = sum(occ_by_colour)
            else:
                occ_by_colour = [0] * len(all_colors)
                cnt = 0

            # puts the constructed Value in a list associated with the Unibar.
            values.append(Value(
                id=str(val),
                occurrences=cnt,
                occ_by_colour=occ_by_colour,
                dtype=dtype if str(val) != self.missing_placeholder else np.str_
            ))

        # Set display type if there is no specified display type
        if np.issubdtype(dtype, np.number) and self.label_type == "default":
            if len(values) >= 7:
                self.label_type = "levels"
            else:
                self.label_type = "values"
        elif self.label_type == "default":
            self.label_type = "values"

        self.values = values

        # sort values before separating missing and non-missing values
        self._sort_values()

        # id -> Value lookup so get_value_by_id doesn't rescan the list each call
        self._values_by_id = {v.id: v for v in self.values}

        # Separate missing and non-missing values
        self.missing_vals = [v for v in self.values
                            if self.missing_placeholder is not None and str(v.id) == str(self.missing_placeholder)]
        self.non_missing_vals = [v for v in self.values if v not in self.missing_vals]

    
    def set_measurements(self, pos_x=None, width=None, bar_unit=None, missing_padding=None,
                        hbar_height=None):
        if pos_x is not None:
            self.pos_x = pos_x
        if width is not None:
            self.width = width
        if bar_unit is not None:
            self.bar_unit = bar_unit
        if missing_padding is not None:
            self.missing_padding = missing_padding
        if hbar_height is not None:
            self.hbar_height = hbar_height

    def compute_vertical_positions(self, y_start: float, y_end: float):    
        bottom = y_start
        top = y_end

        # --- Handle missing bar at bottom ---
        if self.missing:
            mv = self.missing_vals[0] if self.missing_vals else None
            mv_height = max(self.min_bar_height, mv.occurrences * self.bar_unit) if mv else 0
            missing_center = bottom + mv_height / 2
            if mv: mv.set_y(centre=missing_center)
            # Update bottom for non-missing values: start above missing bar + padding
            bottom += self.missing_padding

        # Box / violin draw each value as a thin point, not a bar, so no half-bar
        # padding is reserved at the extremes: their value centres (where connectors
        # attach) span the full [draw_y_start, draw_y_end], matching where the level
        # labels and the box/violin body are drawn. Bar-like displays (rug /
        # stacked_bar / bar / lumpy beanplot) still reserve half a bar. The same_scale
        # path encodes the same exclusion via bar_like_for_same_scale in figure.py.
        point_like = self.display_type in Defaults.CENTER_ATTACH_DISPLAYS

        # --- Adjust top for last non-missing bar ---
        if self.min_max_pos:
            top_adjustment =  max(self.min_bar_height / 2, self.min_max_pos[1]) if self.min_max_pos[1] != 0 else 0
        elif point_like:
            top_adjustment = 0
        else:
            if self.display_type != "bar":
                top_height = self.non_missing_vals[-1].occurrences * self.bar_unit
            else:
                top_height = self.hbar_height
            top_adjustment = max(self.min_bar_height, top_height) / 2 if self.non_missing_vals and self.non_missing_vals[-1].occurrences != 0 else 0
        top -= top_adjustment

        if self.min_max_pos:
            bottom_adjustment = max(self.min_bar_height / 2, self.min_max_pos[0]) if self.min_max_pos[0] != 0 else 0
        elif point_like:
            bottom_adjustment = 0
        else:
            if self.display_type != "bar":
                bottom_height = self.non_missing_vals[0].occurrences * self.bar_unit
            else:
                bottom_height = self.hbar_height
            bottom_adjustment =  max(self.min_bar_height, bottom_height) / 2 if self.non_missing_vals and self.non_missing_vals[0].occurrences != 0 else 0
        bottom += bottom_adjustment

        # For non rugplot bars, the bar-height-based adjustments don't apply. When same_scale is on,
        # use the adjusted [bottom, top] so the shape aligns with rugplots in
        # the group. Otherwise let the shape span the panel (minus missing area).
        if self.min_max_pos:
            self.draw_y_start = bottom
            self.draw_y_end = top
        else:
            self.draw_y_start = (y_start + self.missing_padding) if self.missing else y_start
            self.draw_y_end = y_end

        # --- Numeric values ---
        if self.val_type in [np.integer, np.floating] and self.non_missing_vals:
            numeric_vals = []
            for v in self.non_missing_vals:
                if v.numeric is not None:
                    numeric_vals.append((v.numeric, v))
                else:
                    try:
                        numeric_vals.append((float(v.id), v))
                    except Exception:
                        continue

            if numeric_vals:
                numeric_vals.sort(key=lambda x: x[0])
                nums = [x[0] for x in numeric_vals]
                vals = [x[1] for x in numeric_vals]

                # Determine range
                minv, maxv = self.range if self.range else (min(nums), max(nums))

                # Map to vertical coordinates
                if maxv == minv:
                    # All identical: equally spaced
                    gap = (top - bottom) / max(1, len(vals) - 1)
                    positions = [bottom + i * gap for i in range(len(vals))]
                else:
                    positions = [bottom + (n - minv) / (maxv - minv) * (top - bottom) for n in nums]

                # assign positions
                for v, p in zip(vals, positions):
                    v.set_y(centre=p)

        # --- String/Categorical values (with same_scale) ---
        elif self.val_type == np.str_ and self.non_missing_vals and (self.min_max_pos or self.display_type == "bar"):
            n = len(self.non_missing_vals)

            if n == 1:
                # Single shared slot: nothing to space; centre it in the band.
                self.non_missing_vals[0].set_y(centre=(bottom + top) / 2)
            else:
                # spacing between centers
                step = (top - bottom) / (n - 1)

                for i, val in enumerate(self.non_missing_vals):
                    # place at center of each interval
                    pos = bottom + i * step
                    val.set_y(pos)
        
        # --- String/Categorical values (without same_scale) ---
        elif self.val_type == np.str_ and self.non_missing_vals:
            n = len(self.non_missing_vals)

            if n == 1:
                # Single bar: just put it in the middle
                self.non_missing_vals[0].set_y(centre=(bottom + top) / 2)
            else:
                # --- Step 1: compute natural positions without compression ---
                total_coloured_y = sum((max(val.occurrences * self.bar_unit, self.min_bar_height) if val.occurrences != 0 else 0)
                                    for val in self.non_missing_vals)
                coloured_y_with_adjustments = total_coloured_y - bottom_adjustment - top_adjustment

                # spacing between bars
                gap = (top - bottom - coloured_y_with_adjustments) / (n - 1)

                positions = []
                cur_y = bottom
                self.non_missing_vals[0].set_y(centre=bottom)
                positions.append(bottom)

                for i in range(1, n):
                    prev_half = max(self.non_missing_vals[i-1].occurrences * self.bar_unit,
                                    self.min_bar_height) / 2 if self.non_missing_vals[i-1].occurrences != 0 else 0
                    cur_half = max(self.non_missing_vals[i].occurrences * self.bar_unit,
                                self.min_bar_height) / 2 if self.non_missing_vals[i].occurrences != 0 else 0
                    cur_y += prev_half + cur_half + gap
                    positions.append(cur_y)

                # --- assign positions ---
                for v, p in zip(self.non_missing_vals, positions):
                    v.set_y(centre=p)

        # --- Set final unibar bounds ---
        self.y_bottom = (y_start + self.missing_padding) if self.missing else y_start # bottom of the non-missing values in the unibar
        self.y_top = y_end # true top of the unibar
    
    """
        sorts values based on val_order
    """
    def _sort_values(self):
        if self.val_order is not None:
            # Map each name to its position in val_order
            order_map = {name: i for i, name in enumerate(self.val_order)}
            self.values.sort(key=lambda v: order_map.get(v.id, len(order_map)))
            

    def draw(self, ax, alpha, rectangle_painter=None, color="lightskyblue"):
        """
        Template Method for drawing a unibar:
        1. Draw the background according to display_type
        2. Draw the labels according to label_type
        Assumes that compute_vertical_positions has already been called
        """
        self.alpha = alpha

        # Step 1: Draw background based on display_type
        if self.unibar:
            self._draw_background(ax, rectangle_painter)

        # Step 2: Draw labels
        if self.label:
            self._draw_labels(ax)

        return ax

    # ---------- Template Method ----------
    def _draw_background(self, ax, rectangle_painter):
        """
        Template Method for drawing the backgrounds in a unibar
        3 types of backgrounds:
        1. rugplots (draws rectangles behind values)
        2. violin plots (draws a violin plot)
        3. boxplots (draws a boxplot)
        4. lumpy beanplots (draws a bean plot with a lumpy rugplot inside)
        5. spiky beanplots (draws a bean plot with a spikeplot inside)
        """
        if self.missing:
            # draw missing values
            self._draw_rectangles(ax, self.missing_vals, rectangle_painter)

        if self.display_type == "rug" or self.display_type == "stacked_bar":
            self._draw_rectangles(ax, self.non_missing_vals, rectangle_painter)
        elif self.display_type == "violin":
            self._draw_violin(ax, self.draw_y_start, self.draw_y_end)
        elif self.display_type == "box":
            self._draw_boxplot(ax, self.draw_y_start, self.draw_y_end)
        elif self.display_type == "lumpy beanplot":
            self._draw_lumpy_beanplot(ax, rectangle_painter)
        elif self.display_type == "spiky beanplot":
            self._draw_spiky_beanplot(ax, self.draw_y_start, self.draw_y_end, rectangle_painter)
        elif self.display_type == "bar":
            self._draw_hbar(ax, self.non_missing_vals, rectangle_painter)
        else:
            raise ValueError(f"Unknown display_type: {self.display_type}")

    def _draw_rectangles(self, ax, values, rectangle_painter, width=None):
        """
        Draw rectangles
        """
        left_pts, right_pts, heights, weights = [], [], [], []

        if not width:
            width = self.width
        
        half_label_space = width / 2

        for val in values:
            # Compute vertical bar height
            bar_height = val.occurrences * self.bar_unit
            bar_height = max(bar_height, self.min_bar_height) if bar_height != 0 else 0 # enforce minimum bar height unless there are no such occurrences

            heights.append(bar_height)
            # Horizontal coordinates
            
            left_pts.append((self.pos_x - half_label_space, val.vert_centre))
            right_pts.append((self.pos_x + half_label_space, val.vert_centre))
            weights.append(val.occ_by_colour)

        rectangle_painter.plot(ax, self.alpha, left_pts, right_pts, heights, self.colors, weights, orientation=self.hi_box,zorder=1,
                               check_overlap=True, unibar_name=self.name, min_seg_height=self.min_bar_height)

        if self.draw_white_dividers and len(values) > 1:
            # each rectangle's edge is half its own bar height from its centre
            half_heights = [h / 2 for h in heights]
            self._draw_white_dividers(ax, values, rectangle_painter, half_heights, width)

    def _draw_white_dividers(self, ax, values, rectangle_painter, half_heights, width):
        """
        Draw thin white lines dividing adjacent bars (used when uni_vfill == 1).
        half_heights[i] is the half-height of values[i], so each divider lands
        midway between the top edge of one bar and the bottom edge of the next.
        """
        divider_height = Defaults.WHITE_DIVIDER_HEIGHT

        divider_left_pts = []
        divider_right_pts = []
        divider_heights = []
        divider_weights = []

        half_label_space = width / 2

        for i in range(len(values) - 1):
            top_of_i = values[i].vert_centre + half_heights[i]
            bottom_of_next = values[i + 1].vert_centre - half_heights[i + 1]
            divider_y = (top_of_i + bottom_of_next) / 2

            divider_left_pts.append((self.pos_x - half_label_space, divider_y))
            divider_right_pts.append((self.pos_x + half_label_space, divider_y))
            divider_heights.append(divider_height)

            # white divider bar (use 2D structure)
            divider_weights.append([1])

        rectangle_painter.plot(
            ax,
            alpha=1,
            left_center_pts=divider_left_pts,
            right_center_pts=divider_right_pts,
            heights=divider_heights,
            colors=["white"],
            weights=divider_weights,
            orientation=self.hi_box,
            zorder=2,  # slightly above bars
            check_overlap=False
        )

    def _prepare_scaled_data(self, y_start, y_end):
        """
        Collect the y-positions for the box/violin plots, split by colour, along
        with the frequency sitting at each position.

        Each value's number is mapped onto the [y_start, y_end] span, and that
        position is recorded once per colour it appears in. The matching entry in
        weights_per_color carries how many observations sit there (occurrence
        count, or weight-sum if a weights column is set), so the KDE and quantile
        code can zip the two together entry for entry.

        Args:
            y_start, y_end: bottom and top of the drawable vertical span.

        Returns (data_per_color, weights_per_color, facecolors, edgecolors): the
        y-positions and their frequencies per colour, the fill colours, and their
        matching edge colours. Empty lists if there are no non-missing values.
        """
        if not self.non_missing_vals:
            return [], [], [], []

        n_colors = len(self.colors)
        data_per_color = [[] for _ in range(n_colors)]
        weights_per_color = [[] for _ in range(n_colors)]

        all_numeric_vals = [v.numeric for v in self.non_missing_vals]
        min_val, max_val = self.range if self.range else (min(all_numeric_vals), max(all_numeric_vals))

        def scale_y(val):
            if max_val == min_val:
                return (y_start + y_end) / 2
            return y_start + (val - min_val) / (max_val - min_val) * (y_end - y_start)

        for v in self.non_missing_vals:
            occs = v.occ_by_colour
            if len(occs) < n_colors:
                occs = occs + [0] * (n_colors - len(occs))
            scaled = scale_y(v.numeric)
            for i, occ in enumerate(occs):
                if occ > 0:
                    data_per_color[i].append(scaled)
                    # if no weight column, occ is an integer count — use it directly as the weight
                    weights_per_color[i].append(float(occ))

        return data_per_color, weights_per_color, self.colors, [edge_color_from_face(c) for c in self.colors]

    def _weighted_quantile(self, data, weights, quantiles):
        """
        Weighted quantiles matching matplotlib boxplot's default convention
        (numpy.percentile / Hyndman-Fan type 7): target position p*(W-1)+1
        on the expanded sample, with linear interpolation between adjacent
        order statistics. For unit weights this equals numpy.quantile(...).
        For integer weights it equals expanding-then-quantiling.
        data and weights must be the same length.
        """
        data = np.array(data, dtype=float)
        weights = np.array(weights, dtype=float)

        if len(data) != len(weights):
            raise ValueError(
                f"_weighted_quantile: data length {len(data)} != weights length {len(weights)}."
            )

        sorter = np.argsort(data)
        data = data[sorter]
        weights = weights[sorter]
        cum = np.cumsum(weights)
        W = cum[-1]

        quantiles = np.asarray(quantiles, dtype=float)
        # we decided to use the p*(n-1)+1 method in line with other software packages. See closed issue on github.
        target = quantiles * (W - 1) + 1
        lower = np.floor(target)
        frac = target - lower

        n = len(data)
        lo_idx = np.clip(np.searchsorted(cum, lower, side='left'), 0, n - 1)
        hi_idx = np.clip(np.searchsorted(cum, lower + 1, side='left'), 0, n - 1)
        return (1 - frac) * data[lo_idx] + frac * data[hi_idx]

    def _draw_violin(self, ax, y_start, y_end, draw_boxplot=True):
        """
        Draw a violin plot with optional split halves and overlaid boxplots.
        Uses weighted KDE (scipy); each unique value's frequency comes from
        its occurrence count or weight-sum.
        """

        data_per_color, weights_per_color, facecolors, edgecolors = self._prepare_scaled_data(y_start, y_end)

        # ---- helpers ----

        def make_kde_path(data, weights=None):
            data = np.array(data, dtype=float)
            if weights is not None:
                w = np.array(weights, dtype=float)
                # repeat each value proportionally so KDE sees correct sample size
                counts = np.round(w / w.min()).astype(int)  # relative integer counts
                data = np.repeat(data, counts)
            kde = gaussian_kde(data, bw_method=self.violin_bw_method)
            y_grid = np.linspace(data.min(), data.max(), 1000)
            density = kde(y_grid)
            density = density / density.max() * self.width / 2
            return y_grid, density

        def fill_violin(y_grid, density, color, side='full'):
            """Draw violin as a filled polygon directly on ax."""
            d = density / density.max() * self.width / 2
            if side == 'full':
                xl, xr = self.pos_x - d, self.pos_x + d
            elif side == 'left':
                xl, xr = self.pos_x - d, np.full_like(d, self.pos_x)
            else:  # right
                xl, xr = np.full_like(d, self.pos_x), self.pos_x + d
            ax.fill_betweenx(y_grid, xl, xr, color=color, alpha=self.alpha)

        def draw_inner_box(data, weights, pos_x, width, edgecolor):
            """Draw box/whisker using weighted quantiles."""
            q1, median, q3 = self._weighted_quantile(data, weights, [0.25, 0.5, 0.75])
            iqr = q3 - q1
            arr = np.array(data, dtype=float)
            lo = arr[arr >= q1 - 1.5 * iqr].min()
            hi = arr[arr <= q3 + 1.5 * iqr].max()
            hw = width / 2
            ax.broken_barh([(pos_x - hw, width)], (q1, q3 - q1),
                           facecolors='none', edgecolors=edgecolor, linewidth=1.2)
            ax.plot([pos_x - hw, pos_x + hw], [median, median], color=edgecolor, linewidth=1.5)
            ax.plot([pos_x, pos_x], [lo, q1], color=edgecolor, linewidth=1)
            ax.plot([pos_x, pos_x], [q3, hi], color=edgecolor, linewidth=1)

        # ---- single violin ----
        if len(data_per_color) == 1:
            data = data_per_color[0]
            if not data:
                return
            weights = weights_per_color[0]

            y_grid, density = make_kde_path(data, weights)
            fill_violin(y_grid, density, facecolors[0], side='full')

            if draw_boxplot:
                draw_inner_box(data, weights, self.pos_x, self.width * 0.1, edgecolors[0])

        # ---- split violin ----
        else:
            right_data    = data_per_color[0]
            left_data     = data_per_color[1]
            right_weights = weights_per_color[0]
            left_weights  = weights_per_color[1]

            if right_data:
                y_grid, density = make_kde_path(right_data, right_weights)
                fill_violin(y_grid, density, facecolors[0], side='right')
            if left_data:
                y_grid, density = make_kde_path(left_data, left_weights)
                fill_violin(y_grid, density, facecolors[1], side='left')

            offset = self.width * 0.05
            if draw_boxplot:
                if right_data:
                    draw_inner_box(right_data, right_weights,
                                   self.pos_x + offset, self.width * 0.05, edgecolors[0])
                if left_data:
                    draw_inner_box(left_data, left_weights,
                                   self.pos_x - offset, self.width * 0.05, edgecolors[1])

    def _draw_boxplot(self, ax, y_start, y_end, gap_ratio=0.02):
        """
        Draw a boxplot using weighted quantiles. Each unique value's
        frequency comes from its occurrence count or weight-sum.
        """
        def rotate_left(lst):
            if len(lst) > 1:
                return lst[1:] + lst[:1]
            return lst

        data_per_color, weights_per_color, facecolors, edgecolors = self._prepare_scaled_data(y_start, y_end)

        n = len(data_per_color)
        if n == 0:
            return

        data_per_color  = rotate_left(data_per_color)
        facecolors      = rotate_left(facecolors)
        edgecolors      = rotate_left(edgecolors)
        if weights_per_color:
            weights_per_color = rotate_left(weights_per_color)

        # Ensure color lists match
        if len(facecolors) < n:
            facecolors = (facecolors * n)[:n]
        if len(edgecolors) < n:
            edgecolors = (edgecolors * n)[:n]

        # Proportional widths — weight sum if weighted, count if not
        if weights_per_color:
            totals = [sum(w) for w in weights_per_color]
        else:
            totals = [len(d) for d in data_per_color]

        grand_total = sum(totals)
        if grand_total == 0:
            return

        gap       = self.width * gap_ratio
        box_widths = [(self.width - gap * (n - 1)) * (t / grand_total) for t in totals]
        total_width = sum(box_widths) + gap * (n - 1)
        start_x   = self.pos_x - total_width / 2

        offsets = []
        current_x = start_x
        for w in box_widths:
            offsets.append(current_x + w / 2)
            current_x += w + gap

        for i, data in enumerate(data_per_color):
            if not data:
                continue

            weights = weights_per_color[i]
            pos_x   = offsets[i]
            bw      = box_widths[i]

            q1, median, q3 = self._weighted_quantile(data, weights, [0.25, 0.5, 0.75])
            iqr   = q3 - q1
            arr   = np.array(data, dtype=float)
            lo    = arr[arr >= q1 - 1.5 * iqr].min()
            hi    = arr[arr <= q3 + 1.5 * iqr].max()
            hw    = bw / 2

            # Box fill
            ax.broken_barh(
                [(pos_x - hw, bw)], (q1, q3 - q1),
                facecolors=facecolors[i], edgecolors=edgecolors[i],
                linewidth=1.2, alpha=self.alpha,
            )
            # Median line
            ax.plot([pos_x - hw, pos_x + hw], [median, median],
                    color=edgecolors[i], linewidth=1.5)
            # Whiskers
            ax.plot([pos_x, pos_x], [lo, q1], color=edgecolors[i], linewidth=1)
            ax.plot([pos_x, pos_x], [q3, hi], color=edgecolors[i], linewidth=1)
            # Caps
            ax.plot([pos_x - hw * 0.5, pos_x + hw * 0.5], [lo, lo],
                    color=edgecolors[i], linewidth=1)
            ax.plot([pos_x - hw * 0.5, pos_x + hw * 0.5], [hi, hi],
                    color=edgecolors[i], linewidth=1)
            # Fliers
            flier_mask = (arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)
            fliers = arr[flier_mask]
            if len(fliers):
                ax.scatter(
                    [pos_x] * len(fliers), fliers,
                    marker='o', color=facecolors[i], edgecolors='none',
                    s=10, alpha=self.alpha, zorder=3,
                )
    
    def _draw_lumpy_beanplot(self, ax, rectangle_painter):
        self._draw_violin(ax,
                          y_start = self.non_missing_vals[0].vert_centre,
                          y_end = self.non_missing_vals[-1].vert_centre,
                          draw_boxplot=False)
        self._draw_rectangles(ax,
                             self.non_missing_vals,
                             rectangle_painter)
    
    def _weighted_centre(self, weight_fn):
        """
        Weighted mean of vert_centre across the non-missing values, weighting each
        value by weight_fn(val). Returns None when the weights sum to zero.
        """
        total = sum(weight_fn(val) for val in self.non_missing_vals)
        if total == 0:
            return None
        return sum(val.vert_centre * weight_fn(val) for val in self.non_missing_vals) / total

    def _draw_spiky_beanplot(self, ax, y_start, y_end, rectangle_painter):
        # draw violin
        self._draw_violin(ax, y_start, y_end, draw_boxplot=False)

        # draw spike plot
        n_colors = len(self.colors)

        # only one colour
        if n_colors == 1:
            heights = [Defaults.SPIKE_THICKNESS] * len(self.non_missing_vals)
            left_pts, right_pts, weights = [], [], []   

            max_occ = max(val.occurrences for val in self.non_missing_vals)
            width_per_occ = self.width / max_occ
            color = [edge_color_from_face(self.colors[0])] # darker/lighter colour based on face color

            for val in self.non_missing_vals:
                half_label_space = width_per_occ * val.occurrences / 2
                left_pts.append((self.pos_x - half_label_space, val.vert_centre))
                right_pts.append((self.pos_x + half_label_space, val.vert_centre))
                weights.append([val.occurrences])
            
            rectangle_painter.plot(ax, alpha=self.alpha,
                                left_center_pts=left_pts,
                                right_center_pts=right_pts,
                                heights=heights,
                                colors=color,
                                weights=weights,
                                zorder=1)
            
            # draw the mean line
            mean_y = self._weighted_centre(lambda val: val.occurrences)

            rectangle_painter.plot(ax, alpha=1,
                                   left_center_pts=[(self.pos_x - self.width / 2, mean_y)], 
                                   right_center_pts=[(self.pos_x + self.width / 2, mean_y)],
                                   heights=[Defaults.SPIKE_THICKNESS],
                                   colors=["#000000"], # black
                                   weights=[[1]],
                                   zorder=3)
        else: # divide two halves
            # colors
            l_color = [edge_color_from_face(self.colors[1])]
            r_color = [edge_color_from_face(self.colors[0])]

            l_left_pts, l_right_pts, l_weights = [], [], [] # highlighted colour
            r_left_pts, r_right_pts, r_weights = [], [], [] # non highlighted colour

            l_max_occ, r_max_occ = 0, 0

            for val in self.non_missing_vals:
                l_max_occ = max(val.occ_by_colour[1], l_max_occ)
                r_max_occ = max(val.occ_by_colour[0], r_max_occ)
            
            max_occ = max(l_max_occ, r_max_occ)
            width_per_occ = self.width / max_occ
            
            # spike lengths (left and right)
            for val in self.non_missing_vals:
                l_label_space = width_per_occ * val.occ_by_colour[1] / 2
                if l_label_space > 0:
                    l_left_pts.append((self.pos_x - l_label_space, val.vert_centre))
                    l_right_pts.append((self.pos_x, val.vert_centre))
                    l_weights.append([val.occ_by_colour[1]])

                r_label_space = width_per_occ * val.occ_by_colour[0] / 2
                if r_label_space > 0:
                    r_left_pts.append((self.pos_x, val.vert_centre))
                    r_right_pts.append((self.pos_x + r_label_space, val.vert_centre))
                    r_weights.append([val.occ_by_colour[0]])
            
            # heights
            l_heights = [Defaults.SPIKE_THICKNESS] * len(l_left_pts)
            r_heights = [Defaults.SPIKE_THICKNESS] * len(r_left_pts)

            # draw left
            rectangle_painter.plot(ax, alpha=self.alpha,
                                left_center_pts=l_left_pts,
                                right_center_pts=l_right_pts,
                                heights=l_heights,
                                colors=l_color,
                                weights=l_weights,
                                zorder=1)

            # draw right
            rectangle_painter.plot(ax, alpha=self.alpha,
                                left_center_pts=r_left_pts,
                                right_center_pts=r_right_pts,
                                heights=r_heights,
                                colors=r_color,
                                weights=r_weights,
                                zorder=1)
            
            # draw the mean lines
            # LEFT (highlighted)
            l_mean_y = self._weighted_centre(lambda val: val.occ_by_colour[1])
            if l_mean_y is not None:
                rectangle_painter.plot(
                    ax,
                    alpha=1,
                    left_center_pts=[(self.pos_x - self.width / 2, l_mean_y)],
                    right_center_pts=[(self.pos_x, l_mean_y)],
                    heights=[Defaults.SPIKE_THICKNESS],
                    colors=["#000000"],  # black
                    weights=[[1]],
                    zorder=3,
                )

            # RIGHT (non-highlighted)
            r_mean_y = self._weighted_centre(lambda val: val.occ_by_colour[0])
            if r_mean_y is not None:
                rectangle_painter.plot(
                    ax,
                    alpha=1,
                    left_center_pts=[(self.pos_x, r_mean_y)],
                    right_center_pts=[(self.pos_x + self.width / 2, r_mean_y)],
                    heights=[Defaults.SPIKE_THICKNESS],
                    colors=["#000000"],  # black
                    weights=[[1]],
                    zorder=3
                )

    
    # draws missing values as well
    def _draw_hbar(self, ax, values, rectangle_painter):
        """
            Used to draw the unibar background for the horizontal bar chart display option (categorical data only)
        """
        n = len(values)
        
        left_pts, right_pts, weights = [], [], []
        # determine value coordinates for drawing
        left_xpos = self.pos_x - self.width / 2
        heights = [self.hbar_height] * n # constant height

        for v in values:
            total_area = self.bar_unit * v.occurrences * self.width
            hbar_width = max(total_area / self.hbar_height, self.min_bar_height) if total_area != 0 else 0
            left_pts.append((left_xpos, v.vert_centre))
            right_pts.append((left_xpos + hbar_width, v.vert_centre))
            weights.append(v.occ_by_colour)

        # draw values        
        rectangle_painter.plot(ax, self.alpha,
                               left_pts,
                               right_pts,
                               heights,
                               self.colors,
                               weights,
                               orientation=self.hi_box,
                               zorder=1)
        
        if self.draw_white_dividers and len(values) > 1:
            # bar charts draw every value at a constant hbar_height
            half_heights = [self.hbar_height / 2] * len(values)
            self._draw_white_dividers(ax, values, rectangle_painter, half_heights, self.width)
    # ---------- Label Drawing ----------
    def _draw_labels(self, ax):
        """
        Draws labels depending on the display type.
        2 types of labels:
        1. Values (draws labels directly on values)
        2. Levels (draws labels at even increments, depending on a predetermined number of levels (default 7))
        """
        x = self.pos_x
        # Draw missing labels first at proper bottom offset
        if self.missing:
            for mv in self.missing_vals:
                # don't draw the labels if there are no missing values
                if mv.occurrences > 0:
                    # Place missing labels just above the bottom with missing_padding
                    ax.text(x, mv.vert_centre, self.missing_placeholder, ha='center', va='center', **(self.label_options or {}))

        if self.label_type == "values":
            self._draw_value_labels(ax) #draws labels directly according to the values
        elif self.label_type == "levels":
            self._draw_level_labels(ax)
        else:
            raise ValueError(f"invalid label_type {self.label_type}")

    # --------- Label drawing directly onto values --------
    def _draw_value_labels(self, ax):
        """
        Draws labels directly on values
        """
        for val in self.non_missing_vals:
            if val.occurrences > 0: # do not draw if no values exist
                ax.text(self.pos_x, val.vert_centre, self._get_formatted_label(val.dtype, val.id), ha='center', va='center', **(self.label_options or {}))

    # -------- Label drawing - levels (starting from y_start and ending at y_end) ------
    def _draw_level_labels(self, ax):
        """
        2 ways to draw levels:
        1. Display type == rug
            - this means that the bottommost and topmost values are NOT centred at the bottom and the top of the drawable space.
            - Labels must be offset slightly to accomodate for the adjustment made
        2. Display type == box or violin
            - this means that the labels should span the entire vertical range of the drawable area.
        """
        # Draw numeric levels if display_type="levels"
        min_val, max_val = self.range if self.range else (min(v.numeric for v in self.non_missing_vals),
                                                        max(v.numeric for v in self.non_missing_vals))
        num_levels = self.num_levels

        # Handle integer vs float
        if self.val_type == np.floating:
            level_vals = np.linspace(min_val, max_val, num_levels)
        elif self.val_type == np.integer:
            possible_vals = np.arange(int(np.floor(min_val)), int(np.ceil(max_val)) + 1)
            if len(possible_vals) <= num_levels:
                level_vals = possible_vals
            else:
                indices = np.linspace(0, len(possible_vals) - 1, num_levels, dtype=int)
                level_vals = possible_vals[indices]
        
        if self.display_type == "rug" or self.display_type == "lumpy beanplot":
            # Compute coordinate range based on first and last non-missing bar centers
            first_center = self.non_missing_vals[0].vert_centre
            last_center = self.non_missing_vals[-1].vert_centre

            bottom_coord = (self.y_bottom + self.min_max_pos[0]) if self.min_max_pos else first_center
            top_coord = (self.y_top - self.min_max_pos[1]) if self.min_max_pos else last_center
        else: # "box" or "violin" or "spiky beanplot"
            bottom_coord = self.draw_y_start
            top_coord = self.draw_y_end

        level_coords = [bottom_coord + (top_coord - bottom_coord) * (v - min_val) / (max_val - min_val)
                        for v in level_vals]

        for tick_val, tick_y in zip(level_vals, level_coords):
            ax.text(self.pos_x, tick_y, self._get_formatted_label(self.val_type, tick_val), ha='center', va='center', **(self.label_options or {}))
    
    def _get_formatted_label(self, datatype, value):
        """
        Determines how a label should be formatted.
        Categorical Data:
        - Strings are labeled as-is
        Numerical Data:
        - Scientific notation is used if the absolute value of a value is > 1000000 or < 0.01
        - Floats are rounded to 2 decimal places
        - Integers are written with no decimal places
        """
        if value == self.missing_placeholder:
            return value
        return get_formatted_label(datatype, value)

    def get_value_by_id(self, id: str):
        """
        Returns a Value, given its id
        Assumes that all ids are unique (true)
        """
        return self._values_by_id.get(id)

    def __repr__(self):
        return f"unibar(name={self.name!r}, x={self.pos_x:.2f}, nvals={len(self.values)})"
