# unibar.py
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from value import Value
from utils import Defaults

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
                 display_type: str,):
        self.df = df
        self.name = name
        self.display_type = display_type
        self.val_type = val_type          # "np.str_" or "np.floating" or "np.integer"
        self.num_levels = num_levels
        self.unibar = unibar
        self.label = label
        self.missing = missing
        self.y_top = 0.0
        self.y_bottom = 0.0
        self.highlight_colors: Optional[List[str]] = None
        self.missing_placeholder = missing_placeholder
        self.val_order = val_order
        self.min_bar_height = min_bar_height

        self._build_values()
        self.sort_values()

        self.hi_box = hi_box
        self.colors = colors

        # for same_scale variables
        self.range = None # if numerical, will be a min val and a max val
        self.min_max_pos = None # records the centre positions of the top and bottom values

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
        all_colors = sorted(self.df["color_index"].unique())

        # Determine order
        if self.val_order:
            order = self.val_order
        else:
            order = uni_series.dropna().unique().tolist()

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
        if np.issubdtype(dtype, np.number) and self.display_type == "default":
            if len(values) >= 7:
                self.display_type = "levels"
            else:
                self.display_type = "rugplot"
        elif self.display_type == "default":
            self.display_type = "rugplot"

        self.values = values
        return values
    
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
        if len(self.values) == 0:
            return

        bottom = y_start
        top = y_end

        # Separate missing and non-missing values
        self.missing_vals = [v for v in self.values
                            if self.missing_placeholder is not None and str(v.id) == str(self.missing_placeholder)]
        self.non_missing_vals = [v for v in self.values if v not in self.missing_vals]

        # --- Handle missing bar at bottom (only one missing value assumed) ---
        if self.missing:
            mv = self.missing_vals[0] if self.missing_vals else None
            mv_height = max(self.min_bar_height, mv.occurrences * self.bar_unit) if mv else 0
            missing_center = bottom + mv_height / 2
            if mv: mv.set_y(centre=missing_center)
            # Update bottom for non-missing values: start above missing bar + padding
            bottom += self.missing_padding

        # --- Adjust top for last non-missing bar ---
        if self.min_max_pos:
            top_adjustment =  max(self.min_bar_height / 2, self.min_max_pos[1]) if self.min_max_pos[1] != 0 else 0
        else:
            top_adjustment = max(self.min_bar_height, self.non_missing_vals[-1].occurrences * self.bar_unit) / 2 if self.non_missing_vals[-1].occurrences != 0 else 0
        top -= top_adjustment
        
        if self.min_max_pos:
            bottom_adjustment = max(self.min_bar_height / 2, self.min_max_pos[0]) if self.min_max_pos[0] != 0 else 0
        else:
            bottom_adjustment =  max(self.min_bar_height, self.non_missing_vals[0].occurrences * self.bar_unit) / 2 if self.non_missing_vals[0].occurrences != 0 else 0
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

        # --- String/Categorical values (without same_scale) ---
        elif self.val_type == np.str_ and self.non_missing_vals and self.min_max_pos:
            n = len(self.non_missing_vals)

            # spacing between centers
            step = (top - bottom) / n

            for i, val in enumerate(self.non_missing_vals):
                # place at center of each interval
                pos = bottom + (i + 0.5) * step
                val.set_y(pos)
        # --- String/Categorical values (with same_scale) ---
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
                space = (top - bottom - coloured_y_with_adjustments) / (n - 1)

                positions = []
                cur_y = bottom
                self.non_missing_vals[0].set_y(centre=bottom)
                positions.append(bottom)

                for i in range(1, n):
                    prev_half = max(self.non_missing_vals[i-1].occurrences * self.bar_unit,
                                    self.min_bar_height) / 2 if self.non_missing_vals[i-1].occurrences != 0 else 0
                    cur_half = max(self.non_missing_vals[i].occurrences * self.bar_unit,
                                self.min_bar_height) / 2 if self.non_missing_vals[i].occurrences != 0 else 0
                    cur_y += prev_half + cur_half + space
                    positions.append(cur_y)

                # --- assign positions ---
                for v, p in zip(self.non_missing_vals, positions):
                    v.set_y(centre=p)

        # --- Set final unibar bounds ---
        self.y_bottom = y_start  # true bottom of the unibar
        self.y_top = top
    
    def sort_values(self):
        # --- Categorical/string sorting ---
        if self.val_type == np.str_:
            if self.val_order is not None:
                # Map each name to its position in val_order
                order_map = {name: i for i, name in enumerate(self.val_order)}
                self.values.sort(key=lambda v: order_map.get(v.id, len(order_map)))
            # If no val_order, leave values as-is
            return

        # --- Numeric sorting ---
        if np.issubdtype(self.val_type, np.number):
            # Sort by .numeric from smallest to largest
            self.values.sort(key=lambda v: (v.numeric is None, v.numeric), reverse=False)
            


    def draw(self, ax, label_opts: Dict = None, rectangle_painter=None,
             color="lightskyblue", bar_unit: float = 1.5,
             default_highlight_colors: Optional[List[str]] = None):
        """
        Template Method for drawing a unibar:
        1. Draw the background according to display_type
        2. Draw the labels
        """
        # Step 1: Draw background based on display_type
        self._draw_background(ax, rectangle_painter, color, bar_unit)

        # Step 2: Draw labels
        if self.label:
            self._draw_labels(ax, label_opts)

        return ax

    # ---------- Template Method ----------
    def _draw_background(self, ax, rectangle_painter, color, bar_unit):
        if self.display_type in ["rugplot", "levels"]:
            # self._draw_rectangles(ax, rectangle_painter, bar_unit)
            if self.val_type != np.str_:
                #self._draw_violin(ax)
                self._draw_boxplot(ax)
                missing_values = []
                for val in self.values:
                    if val.id == self.missing_placeholder:
                        missing_values.append(val)
                self._draw_rectangles(ax, missing_values, rectangle_painter, bar_unit)
            else:
                self._draw_rectangles(ax, self.values, rectangle_painter, bar_unit)
        elif self.display_type == "violin":
            self._draw_violin(ax)
        else:
            raise ValueError(f"Unknown display_type: {self.display_type}")

    def _draw_rectangles(self, ax, values, rectangle_painter, bar_unit):
        """
        Draw rectangles
        """
        if not self.unibar or rectangle_painter is None:
            return

        left_pts, right_pts, heights, weights = [], [], [], []

        for val in values:
            # Compute vertical bar height
            bar_height = val.occurrences * bar_unit
            bar_height = max(bar_height, self.min_bar_height) if bar_height != 0 else 0 # enforce minimum bar height unless there are no such occurrences

            heights.append(bar_height)

            # Horizontal coordinates
            half_label_space = self.width / 2
            left_pts.append((self.pos_x - half_label_space, val.vert_centre))
            right_pts.append((self.pos_x + half_label_space, val.vert_centre))
            weights.append(val.occ_by_colour)

        rectangle_painter.plot(ax, left_pts, right_pts, heights, self.colors, weights, orientation=self.hi_box,zorder=1)

    def _prepare_scaled_data(self):
        """
        Shared helper: prepare scaled left/right data and colors for plotting.
        Returns: (left_scaled, right_scaled, colors)
        """
        if not self.non_missing_vals:
            return [], [], []

        # Determine colors
        if len(self.colors) == 1:
            colors = [self.colors[0], self.colors[0]]
        else:
            colors = self.colors[:2]

        # Determine bottom and top for scaling
        bottom_coord = self.y_bottom
        if self.missing and self.missing_vals:
            bottom_coord = self.missing_vals[0].vert_centre + getattr(self, "missing_padding", 0)
        top_coord = self.y_top

        left_data, right_data, numeric_vals = [], [], []

        for v in self.non_missing_vals:
            val_numeric = v.numeric if v.numeric is not None else float(v.id)
            numeric_vals.append(val_numeric)

            if len(v.occ_by_colour) > 1:
                left_data.extend([val_numeric] * v.occ_by_colour[1])
                right_data.extend([val_numeric] * v.occ_by_colour[0])
            else:
                right_data.extend([val_numeric] * v.occurrences)
                left_data.extend([val_numeric] * v.occurrences)

        min_val, max_val = min(numeric_vals), max(numeric_vals)

        # Scaling function
        def scale_y(val):
            if max_val == min_val:
                return (bottom_coord + top_coord) / 2
            return bottom_coord + (val - min_val) / (max_val - min_val) * (top_coord - bottom_coord)

        left_scaled = [scale_y(d) for d in left_data]
        right_scaled = [scale_y(d) for d in right_data]

        return left_scaled, right_scaled, colors

    def _draw_violin(self, ax):
        left_scaled, right_scaled, colors = self._prepare_scaled_data()
        if not left_scaled and not right_scaled:
            return

        # Left half
        if left_scaled:
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
                pc.set_facecolor(colors[1])
                pc.set_edgecolor(colors[1])
                pc.set_alpha(0.7)

        # Right half
        if right_scaled:
            parts_right = ax.violinplot(
                dataset=[right_scaled],
                positions=[self.pos_x],
                widths=self.width,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )
            for pc in parts_right['bodies']:
                verts = pc.get_paths()[0].vertices
                verts[:, 0] = np.clip(verts[:, 0], self.pos_x, np.inf)
                pc.set_facecolor(colors[0])
                pc.set_edgecolor(colors[0])
                pc.set_alpha(0.7)

    def _draw_boxplot(self, ax):
        left_scaled, right_scaled, colors = self._prepare_scaled_data()
        if not left_scaled and not right_scaled:
            return

        # Half-width for split look
        half_width = self.width / 2

        # Left half (highlighted)
        if left_scaled:
            bp_left = ax.boxplot(
                x=[left_scaled],
                positions=[self.pos_x - half_width/2],
                widths=half_width,
                vert=True,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False  # 🔥 removes weird circles
            )
            for element in ['boxes', 'whiskers', 'caps', 'medians']:
                for artist in bp_left[element]:
                    artist.set_color(colors[1])
            for patch in bp_left['boxes']:
                patch.set_facecolor(colors[1])
                patch.set_alpha(0.7)

        # Right half (non-highlighted)
        if right_scaled:
            bp_right = ax.boxplot(
                x=[right_scaled],
                positions=[self.pos_x + half_width/2],
                widths=half_width,
                vert=True,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False
            )
            for element in ['boxes', 'whiskers', 'caps', 'medians']:
                for artist in bp_right[element]:
                    artist.set_color(colors[0])
            for patch in bp_right['boxes']:
                patch.set_facecolor(colors[0])
                patch.set_alpha(0.7)

    # ---------- Label Drawing ----------
    def _draw_labels(self, ax, label_opts: Dict = None):
        """
        Draw labels depending on variable type, missing data, and numeric levels.
        """
        x = self.pos_x

        # Draw missing labels first at proper bottom offset
        if self.missing_vals:
            for mv in self.missing_vals:
                # Place missing labels just above the bottom with missing_padding
                ax.text(x, mv.vert_centre, "missing", ha='center', va='center', **(label_opts or {}))

        if self.display_type == "rugplot":
            # Draw non-missing data labels
            for val in self.non_missing_vals:
                ax.text(x, val.vert_centre, self._get_formatted_label(val.dtype, val.id), ha='center', va='center', **(label_opts or {}))

        # Draw numeric levels if display_type="levels"
        if self.display_type == "levels" and np.issubdtype(self.val_type, np.number) and self.non_missing_vals:
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

            # Compute coordinate range based on first and last non-missing bar centers
            first_center = self.non_missing_vals[0].vert_centre
            last_center = self.non_missing_vals[-1].vert_centre
            bottom_coord =  (self.y_bottom + self.min_max_pos[0]) if self.min_max_pos else first_center
            if self.missing and self.min_max_pos and bottom_coord == self.y_bottom + self.min_max_pos[0]:
                bottom_coord += self.missing_padding
            top_coord = (self.y_top - self.min_max_pos[1]) if self.min_max_pos else last_center
            
            level_coords = [bottom_coord + (top_coord - bottom_coord) * (v - min_val) / (max_val - min_val) 
                            for v in level_vals]

            for tick_val, tick_y in zip(level_vals, level_coords):
                ax.text(x, tick_y, self._get_formatted_label(self.val_type, tick_val), ha='center', va='center', **(label_opts or {}))
    
    def _get_formatted_label(self, datatype, value):
        # if the label is a string
        if datatype == np.str_ or value == self.missing_placeholder:
            return value
        # otherwise, it should be a numerical value
        value = float(value)
        if abs(value) >= 1000000 or (0 < abs(value) < 0.01): # threshold for displaying scientific notation
            return f"{value:.2e}"
        if datatype == np.integer:
            return str(int(value))
        if datatype == np.floating:
            return f"{value:.2f}" # round to 2 decimal places

    def get_value_by_id(self, id: str):
        for v in self.values:
            if v.id == id:
                return v
        return None

    def __repr__(self):
        return f"unibar(name={self.name!r}, x={self.pos_x:.2f}, nvals={len(self.values)})"
