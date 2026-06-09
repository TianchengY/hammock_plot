# value.py
from typing import List, Optional
import numpy as np

class Value:
    def __init__(self, id: str, occurrences = 0, occ_by_colour: Optional[List[int]] = None, dtype = np.str_):
        """
            Each instance of a Value corresponds to a unique value in a unibar.

            Attributes:
                id: the Value label
                dtype: the datatype
                occurrences: how many times the value occurs in the unibar
                occ_by_color: the # occurrences of each of the highlighted groups in the Value
                vert_centre: the vertical coordinate of the Value's centre
                numeric: the numeric value associated with the Value (if it is categorical, there is no numeric value associated.)
        """
        self.dtype = dtype
        self.id = id
        self.occurrences = occurrences
        # occ_by_colour: [non_highlight_count, hi_count_1, hi_count_2, ...]
        self.occ_by_colour = occ_by_colour if occ_by_colour is not None else [self.occurrences]
        self.vert_centre: float = 0.0
        if dtype != np.str_:
            self.numeric = float(id)
        else:
            self.numeric = None

    def set_y(self, centre: float = None):
        """
            Sets the y-coordinate of the Value
        """
        if centre is not None:
            self.vert_centre = float(centre)
            return

    def __repr__(self):
        """
            Debugging statement that print's Value's ID, number of occurrences, and the y-coordinate of the Value
        """
        return f"Value(id={self.id!r}, occ={self.occurrences}, y={self.vert_centre:.2f})"