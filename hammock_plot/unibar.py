# unibar.py
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from hammock_plot.value import Value
from hammock_plot.utils import edge_color_from_face
from .utils import Defaults, get_formatted_label

class Unibar:
    def __init__(self,
                 df,
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
        uni_series = self.df[self.name]
        counts = uni_series.value_counts()
        values: List[Value] = []

        dtype = self.val_type

        # global set of color indices
        all_colors = list(range(self.df["color_index"].max() + 1))

        # Determine order
        order = self.val_order

        for val in order:
            cnt = int(counts.get(val, 0))
            if cnt > 0:
                subset = self.df[self.df[self.name] == val]
                # reindex to cover all colors (missing → 0)
                occ_by_colour = (
                    subset["color_index"]
                    .value_counts()
                    .reindex(all_colors, fill_value=0)
                    .tolist()
                )
            else:
                occ_by_colour = [0] * len(all_colors)

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

        # Separate missing and non-missing values
        self.missing_vals = [v for v in self.values
                            if self.missing_placeholder is not None and str(v.id) == str(self.missing_placeholder)]
        self.non_missing_vals = [v for v in self.values if v not in self.missing_vals]

    
    def set_measurements(self, pos_x=None, width=None, bar_unit=None, missing_padding=None,
                        scale_ypos: Tuple[float, float] = None):
        if pos_x is not None:
            self.pos_x = pos_x
        if width is not None:
            self.width = width
        if bar_unit is not None:
            self.bar_unit = bar_unit
        if missing_padding is not None:
            self.missing_padding = missing_padding
        if scale_ypos is not None:
            self.scale_ypos = scale_ypos

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
        
        if self.display_type == "hbar":
            self.hbar_height = (top - bottom) / len(self.non_missing_vals) * Defaults.HBAR_HEIGHT_FRAC

        # --- Adjust top for last non-missing bar ---
        if self.min_max_pos:
            top_adjustment =  max(self.min_bar_height / 2, self.min_max_pos[1]) if self.min_max_pos[1] != 0 else 0
        else:
            if self.display_type != "hbar":
                top_height = self.non_missing_vals[-1].occurrences * self.bar_unit
            else:
                top_height = self.hbar_height
            top_adjustment = max(self.min_bar_height, top_height) / 2 if self.non_missing_vals and self.non_missing_vals[-1].occurrences != 0 else 0
        top -= top_adjustment
        
        if self.min_max_pos:
            bottom_adjustment = max(self.min_bar_height / 2, self.min_max_pos[0]) if self.min_max_pos[0] != 0 else 0
        else:
            if self.display_type != "hbar":
                bottom_height = self.non_missing_vals[0].occurrences * self.bar_unit
            else:
                bottom_height = self.hbar_height
            bottom_adjustment =  max(self.min_bar_height, bottom_height) / 2 if self.non_missing_vals and self.non_missing_vals[0].occurrences != 0 else 0
        bottom += bottom_adjustment

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
        elif self.val_type == np.str_ and self.non_missing_vals and (self.min_max_pos or self.display_type == "hbar"):
            n = len(self.non_missing_vals)

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
            

    def draw(self, ax, alpha, rectangle_painter=None,
             color="lightskyblue", y_start: int = None, y_end: int = None):
        """
        Template Method for drawing a unibar:
        1. Draw the background according to display_type
        2. Draw the labels according to label_type
        Assumes that compute_vertical_positions has already been called
        """
        self.alpha = alpha

        # Step 1: Draw background based on display_type
        if self.unibar:
            self._draw_background(ax, rectangle_painter, y_start, y_end)

        # Step 2: Draw labels
        if self.label:
            self._draw_labels(ax, y_start, y_end)

        return ax

    # ---------- Template Method ----------
    def _draw_background(self, ax, rectangle_painter, y_start, y_end):
        """
        Template Method for drawing the backgrounds in a unibar
        3 types of backgrounds:
        1. rugplots (draws rectangles behind values)
        2. violin plots (draws a violin plot)
        3. boxplots (draws a boxplot)
        """
        if self.missing:
            y_start += self.missing_padding
            # draw missing values
            self._draw_rectangles(ax, self.missing_vals, rectangle_painter)

        if self.display_type == "rugplot":
            self._draw_rectangles(ax, self.non_missing_vals, rectangle_painter)
        elif self.display_type == "violin":
            self._draw_violin(ax, y_start, y_end)
        elif self.display_type == "box":
            self._draw_boxplot(ax, y_start, y_end)
        elif self.display_type == "hbar":
            self._draw_hbar(ax, rectangle_painter)
        else:
            raise ValueError(f"Unknown display_type: {self.display_type}")

    def _draw_rectangles(self, ax, values, rectangle_painter):
        """
        Draw rectangles
        """
        left_pts, right_pts, heights, weights = [], [], [], []

        for val in values:
            # Compute vertical bar height
            bar_height = val.occurrences * self.bar_unit
            bar_height = max(bar_height, self.min_bar_height) if bar_height != 0 else 0 # enforce minimum bar height unless there are no such occurrences

            heights.append(bar_height)

            # Horizontal coordinates
            half_label_space = self.width / 2
            left_pts.append((self.pos_x - half_label_space, val.vert_centre))
            right_pts.append((self.pos_x + half_label_space, val.vert_centre))
            weights.append(val.occ_by_colour)

        rectangle_painter.plot(ax, self.alpha, left_pts, right_pts, heights, self.colors, weights, orientation=self.hi_box,zorder=1, 
                               check_overlap=True, unibar_name=self.name)
        
        if self.draw_white_dividers and len(values) > 1:
            divider_height = Defaults.WHITE_DIVIDER_HEIGHT

            divider_left_pts = []
            divider_right_pts = []
            divider_heights = []
            divider_weights = []

            for i in range(len(values) - 1):
                top_of_i = values[i].vert_centre + heights[i] / 2
                bottom_of_next = values[i + 1].vert_centre - heights[i + 1] / 2
                divider_y = (top_of_i + bottom_of_next) / 2
                half_label_space = self.width / 2

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
                check_overlap=False,
                unibar_name=self.name + "_divider"
            )

    def _prepare_scaled_data(self, y_start, y_end):
        """
        Prepare scaled data and colors for plotting.
        Each element in data_scaled corresponds to one highlight (color),
        with the last element holding the non-highlighted data.
        Expands each Value according to occ_by_colour, then scales numeric
        values globally into [y_start, y_end].
        
        Returns:
            data_scaled: list of lists, each list is the scaled values for a highlight
            facecolors: list of colors for each dataset
            edgecolors: uses function in utils.py to make a darker/lighter edge colour relative to the face colour
        """
        if not self.non_missing_vals:
            return [], [], []

        n_colors = len(self.colors)
        data_expanded = [[] for _ in range(n_colors)]

        # Collect all numeric values to compute global min/max
        all_numeric_vals = []
        for v in self.non_missing_vals:
            val_numeric = v.numeric
            all_numeric_vals.append(val_numeric)

        
        min_val, max_val = self.range if self.range else (min(all_numeric_vals), max(all_numeric_vals))

        # Scaling function
        def scale_y(val):
            if max_val == min_val:
                return (y_start + y_end) / 2
            return y_start + (val - min_val) / (max_val - min_val) * (y_end - y_start)

        # Expand each Value according to occ_by_colour
        for v in self.non_missing_vals:
            val_numeric = v.numeric
            occs = v.occ_by_colour

            # Pad occs if fewer than colors
            if len(occs) < n_colors:
                occs = occs + [0]*(n_colors - len(occs))

            for i, occ in enumerate(occs):
                data_expanded[i].extend([val_numeric] * occ)

        # Scale each dataset
        data_scaled = [[scale_y(val) for val in dataset] for dataset in data_expanded]

        return data_scaled, self.colors, [edge_color_from_face(color) for color in self.colors]

    def _draw_violin(self, ax, y_start, y_end):
        """
        Draw a violin plot with optional split halves and overlaid boxplots.
        """
        data_scaled, facecolors, edgecolors = self._prepare_scaled_data(y_start, y_end)

        if len(data_scaled) == 1:
            # Single violin (no split)
            parts = ax.violinplot(
                dataset=[data_scaled[0]],
                positions=[self.pos_x],
                widths=self.width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method=self.violin_bw_method,
            )
            for pc in parts['bodies']:
                pc.set_facecolor(facecolors[0])
                pc.set_edgecolor('none')
                pc.set_alpha(self.alpha)

            # Boxplot inside violin
            box = ax.boxplot(
                data_scaled[0],
                positions=[self.pos_x],
                widths=self.width * 0.1,
                patch_artist=True,
                showcaps=False,
                boxprops=dict(facecolor='none', edgecolor=edgecolors[0], linewidth=1.2),
                whiskerprops=dict(color=edgecolors[0], linewidth=1),
                medianprops=dict(color=edgecolors[0], linewidth=1.5),
                showfliers=False,
                manage_ticks=False,
            )

        else:
            # Split violin
            left_scaled = data_scaled[1]
            right_scaled = data_scaled[0]

            # Left half
            parts_left = ax.violinplot(
                dataset=[left_scaled],
                positions=[self.pos_x],
                widths=self.width,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )
            for pc in parts_left['bodies']:
                verts = pc.get_paths()[0].vertices
                verts[:, 0] = np.clip(verts[:, 0], -np.inf, self.pos_x)
                pc.set_facecolor(facecolors[1])
                pc.set_edgecolor('none')
                pc.set_alpha(self.alpha)

            # Right half
            parts_right = ax.violinplot(
                dataset=[right_scaled],
                positions=[self.pos_x],
                widths=self.width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for pc in parts_right['bodies']:
                verts = pc.get_paths()[0].vertices
                verts[:, 0] = np.clip(verts[:, 0], self.pos_x, np.inf)
                pc.set_facecolor(facecolors[0])
                pc.set_edgecolor('none')
                pc.set_alpha(self.alpha)

            # Offset for half-boxplots (move slightly away from center)
            offset = self.width * 0.05

            # Left half box (shift left)
            left_box = ax.boxplot(
                left_scaled,
                positions=[self.pos_x - offset],
                widths=self.width * 0.05,
                patch_artist=True,
                showcaps=False,
                boxprops=dict(facecolor='none', edgecolor=edgecolors[1], linewidth=1.2),
                whiskerprops=dict(color=edgecolors[1], linewidth=1),
                medianprops=dict(color=edgecolors[1], linewidth=1.5),
                manage_ticks=False,
                showfliers=False,
            )
            for patch in left_box['boxes']:
                verts = patch.get_path().vertices
                verts[:, 0] = np.clip(verts[:, 0], -np.inf, self.pos_x)

            # Right half box (shift right)
            right_box = ax.boxplot(
                right_scaled,
                positions=[self.pos_x + offset],
                widths=self.width * 0.05,
                patch_artist=True,
                showcaps=False,
                boxprops=dict(facecolor='none', edgecolor=edgecolors[0], linewidth=1.2),
                whiskerprops=dict(color=edgecolors[0], linewidth=1),
                medianprops=dict(color=edgecolors[0], linewidth=1.5),
                manage_ticks=False,
                showfliers=False,
            )
            for patch in right_box['boxes']:
                verts = patch.get_path().vertices
                verts[:, 0] = np.clip(verts[:, 0], self.pos_x, np.inf)

    def _draw_boxplot(self, ax, y_start, y_end, gap_ratio=0.02):
        """
        Draw a boxplot
        """
        def rotate_left(lst):
            if len(lst) > 1:
                return lst[1:] + lst[:1]
            return lst

        data_scaled, facecolors, edgecolors = self._prepare_scaled_data(y_start, y_end)
        n = len(data_scaled)
        if n == 0:
            return

        # Reverse all lists so visual order = highlighted 2 -> highlighted 1 -> not highlighted
        data_scaled = rotate_left(data_scaled)
        facecolors = rotate_left(facecolors)
        edgecolors = rotate_left(edgecolors)

        # Ensure colors match number of boxes
        if len(facecolors) < n:
            facecolors = (facecolors * n)[:n]
        if len(edgecolors) < n:
            edgecolors = (edgecolors * n)[:n]

        # Calculate proportional widths
        num_points = [len(values) for values in data_scaled]
        total_points = sum(num_points)
        if total_points == 0:
            return
        gap = self.width * gap_ratio
        num_boxes = len(data_scaled)
        box_widths = [(self.width - gap *  (num_boxes - 1))* (cnt / total_points) for cnt in num_points]

        # Compute total width including gaps
        total_width = sum(box_widths) + gap * (n - 1)
        start_x = self.pos_x - total_width / 2  # leftmost edge of first box

        # Compute offsets sequentially
        offsets = []
        current_x = start_x
        for w in box_widths:
            center = current_x + w / 2
            offsets.append(center)
            current_x += w + gap

        # Draw boxes
        for i, values in enumerate(data_scaled):
            if not values:
                continue

            bp = ax.boxplot(
                x=[values],
                positions=[offsets[i]],
                widths=box_widths[i],
                vert=True,
                patch_artist=True,
                manage_ticks=False,
                showfliers=True,
                flierprops=dict(marker='o', markerfacecolor=facecolors[i], markeredgecolor='none')
            )

            # Set edge colors
            for element in ['whiskers', 'caps', 'medians']:
                for artist in bp[element]:
                    artist.set_color(edgecolors[i])

            # Set face colors
            for patch in bp['boxes']:
                patch.set_facecolor(facecolors[i])
                patch.set_edgecolor(edgecolors[i])
                patch.set_alpha(self.alpha)
    
    # draws missing values as well
    def _draw_hbar(self, ax, rectangle_painter):
        n = len(self.values)
        
        max_val_occ = max([v.occurrences for v in self.values])

        bar_unit = self.width / max_val_occ # this is the horizontal bar unit (how much "width" a value occurrence should contribute)

        left_pts, right_pts, weights = [], [], []
        # determine value coordinates for drawing
        left_xpos = self.pos_x - self.width / 2
        heights = [self.hbar_height] * n # constant height

        for v in self.values:
            hbar_width = bar_unit * v.occurrences
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

    # ---------- Label Drawing ----------
    def _draw_labels(self, ax, y_start, y_end):
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
                if mv.occurrences == 0: continue
                # Place missing labels just above the bottom with missing_padding
                ax.text(x, mv.vert_centre, self.missing_placeholder, ha='center', va='center', **(self.label_options or {}))
        
        if self.label_type == "values":
            self._draw_value_labels(ax) #draws labels directly according to the values
        elif self.label_type == "levels":
            self._draw_level_labels(ax, y_start, y_end)
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
    def _draw_level_labels(self, ax, y_start, y_end):
        """
        2 ways to draw levels:
        1. Display type == rugplot
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
        
        if self.display_type == "rugplot":
            # Compute coordinate range based on first and last non-missing bar centers
            first_center = self.non_missing_vals[0].vert_centre
            last_center = self.non_missing_vals[-1].vert_centre
            
            bottom_coord =  (self.y_bottom + self.min_max_pos[0]) if self.min_max_pos else first_center
            top_coord = (self.y_top - self.min_max_pos[1]) if self.min_max_pos else last_center
        else: # "box" or "violin" display types
            bottom_coord = y_start
            top_coord = y_end
    
        if self.missing and self.min_max_pos and bottom_coord == self.y_bottom + self.min_max_pos[0]:
            bottom_coord += self.missing_padding
        elif self.missing and self.display_type in ["violin", "box"]:
            bottom_coord += self.missing_padding
        
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
        for v in self.values:
            if v.id == id:
                return v
        return None

    def __repr__(self):
        return f"unibar(name={self.name!r}, x={self.pos_x:.2f}, nvals={len(self.values)})"
