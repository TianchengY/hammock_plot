# shapes.py
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Iterable
import matplotlib.pyplot as plt

class FigureBase(ABC):
    """
    Base shape for drawing polygons with optional segmentation by color.
    """

    def _order_coordinates(self, x, y):
        x, y = np.array(x), np.array(y)
        order = np.argsort(np.arctan2(y - y.mean(), x - x.mean()))
        return x[order], y[order]

    def plot(self, ax,
            left_center_pts: List[Tuple[float, float]],
            right_center_pts: List[Tuple[float, float]],
            heights: List[float],
            colors: List[str],
            weights: List[List[float]],   # per-shape weights
            orientation: str = "vertical",
            zorder: int = 0):
        """
        Draw polygons (rectangles or parallelograms) with segmented coloring.

        Parameters
        ----------
        ax : matplotlib Axes
        left_center_pts, right_center_pts : list of (x,y) for centerlines
        heights : list of widths/heights of each shape
        colors : global list of color strings
        weights : list of weight lists, one per shape
                (each list has same length as colors)
        orientation : "vertical" or "horizontal"
        """
        xs, ys = self.get_coordinates(left_center_pts, right_center_pts, heights)

        for (poly_x, poly_y, wts) in zip(xs, ys, weights):
            arr = np.array(wts, dtype=float)
            if arr.sum() <= 0:
                arr = np.ones(len(colors))
            fracs = arr / arr.sum()

            if orientation == "horizontal":
                # edges: left_top→left_bottom, right_top→right_bottom
                left_top  = np.array([poly_x[0], poly_y[0]])
                left_bot  = np.array([poly_x[1], poly_y[1]])
                right_top = np.array([poly_x[2], poly_y[2]])
                right_bot = np.array([poly_x[3], poly_y[3]])

                cum = 0.0
                for f, col in zip(fracs, colors):
                    f0, f1 = cum, cum + f
                    cum = f1
                    lt = left_top  + (left_bot  - left_top)  * f0
                    lb = left_top  + (left_bot  - left_top)  * f1
                    rt = right_top + (right_bot - right_top) * f0
                    rb = right_top + (right_bot - right_top) * f1
                    poly_x_slice = [lt[0], lb[0], rb[0], rt[0]]
                    poly_y_slice = [lt[1], lb[1], rb[1], rt[1]]
                    ax.fill(poly_x_slice, poly_y_slice, color=col, edgecolor=None, zorder=zorder)

            else:  # horizontal split (vertical bars)
                top_left    = np.array([poly_x[0], poly_y[0]])
                bot_left    = np.array([poly_x[1], poly_y[1]])
                top_right   = np.array([poly_x[2], poly_y[2]])
                bot_right   = np.array([poly_x[3], poly_y[3]])

                default_col = colors[0]
                highlight_cols = colors[1:]
                highlight_fracs = fracs[1:]
                cum = 0.0

                # Draw highlights first (bottom-to-top / left side)
                for f, col in zip(highlight_fracs, highlight_cols):
                    f0, f1 = cum, cum + f
                    cum = f1
                    tl = top_left  + (top_right  - top_left)  * f0
                    tr = top_left  + (top_right  - top_left)  * f1
                    bl = bot_left  + (bot_right - bot_left) * f0
                    br = bot_left  + (bot_right - bot_left) * f1
                    poly_x_slice = [tl[0], tr[0], br[0], bl[0]]
                    poly_y_slice = [tl[1], tr[1], br[1], bl[1]]
                    ax.fill(poly_x_slice, poly_y_slice, color=col, edgecolor=None, zorder=zorder)

                # Draw default color last (topmost / right side)
                tl = top_left + (top_right - top_left) * cum
                tr = top_right
                bl = bot_left + (bot_right - bot_left) * cum
                br = bot_right
                poly_x_slice = [tl[0], tr[0], br[0], bl[0]]
                poly_y_slice = [tl[1], tr[1], br[1], bl[1]]
                ax.fill(poly_x_slice, poly_y_slice, color=default_col, edgecolor=None, zorder=zorder)

        return ax



    @abstractmethod
    def get_coordinates(self, left_center_pts, right_center_pts, heights):
        pass


class Parallelogram(FigureBase):
    def get_coordinates(self, left_center_pts, right_center_pts, heights):
        xs, ys = [], []
        for l, r, w in zip(left_center_pts, right_center_pts, heights):
            x, y = np.zeros(4), np.zeros(4)
            alpha = np.arctan(abs(l[1] - r[1]) / abs(l[0] - r[0])) if l[0] != r[0] else np.arctan(np.inf)
            vertical_w = w / np.cos(alpha)

            x[0:2], x[2:4] = l[0], r[0]
            y[0], y[1], y[2], y[3] = l[1] + vertical_w / 2, l[1] - vertical_w / 2, \
                                     r[1] + vertical_w / 2, r[1] - vertical_w / 2

            xs.append(x)
            ys.append(y)

        return xs, ys
    
    # def get_coordinates(self, left_center_pts, right_center_pts, heights):
    #     xs, ys = [], []
    #     for l, r, h in zip(left_center_pts, right_center_pts, heights):
    #         x = np.zeros(4)
    #         y = np.zeros(4)

    #         # left side (x stays l[0], y up/down by h/2)
    #         x[0] = x[1] = l[0]
    #         y[0] = l[1] + h / 2  # top
    #         y[1] = l[1] - h / 2  # bottom

    #         # right side (x stays r[0], y up/down by h/2)
    #         x[2] = x[3] = r[0]
    #         y[2] = r[1] + h / 2  # top
    #         y[3] = r[1] - h / 2  # bottom

    #         xs.append(x)
    #         ys.append(y)

    #     return xs, ys


class Rectangle(FigureBase):
    def get_coordinates(self, left_center_pts, right_center_pts, heights):
        xs, ys = [], []
        for l, r, w in zip(left_center_pts, right_center_pts, heights):
            x, y = np.zeros(4), np.zeros(4)
            alpha = np.arctan(abs(l[1] - r[1]) / abs(l[0] - r[0])) if l[0] != r[0] else np.arctan(np.inf)
            vertical_w = w / np.cos(alpha)
            ax, ay = l[0], l[1] + vertical_w / 2
            cx = ax + w * np.sin(alpha)
            cy = ay - w * np.cos(alpha)
            ex = ax + w * np.sin(alpha) / 2
            ey = ay - w * np.cos(alpha) / 2
            new_ax = ax + l[0] - ex
            new_ay = ay + l[1] - ey
            new_cx = cx + l[0] - ex
            new_cy = cy + l[1] - ey
            x[0], x[1] = new_ax, new_cx
            y[0], y[1] = (new_ay, new_cy) if l[1] <= r[1] else (new_cy, new_ay)
            x[2], x[3] = x[0] + r[0] - l[0], x[1] + r[0] - l[0]
            y[2], y[3] = y[0] + r[1] - l[1], y[1] + r[1] - l[1]
            xs.append(x); ys.append(y)
        return xs, ys