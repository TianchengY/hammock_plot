# value.py
from typing import Dict, List, Optional
import numpy as np

class Value:
    def __init__(self, id: str, occurrences: int = 0, occ_by_colour: Optional[List[int]] = None, dtype = np.str_):
        self.dtype = dtype
        self.id = id
        self.occurrences = int(occurrences)
        # occ_by_colour: [non_highlight_count, hi_count_1, hi_count_2, ...]
        self.occ_by_colour = occ_by_colour if occ_by_colour is not None else [self.occurrences]
        self.vert_centre: float = 0.0
        self.next: Dict[str, int] = {}
        if dtype != np.str_:
            self.numeric = float(id)
        else:
            self.numeric = None

    def set_y(self, centre: float = None):
        if centre is not None:
            self.vert_centre = float(centre)
            return

    def add_next(self, next_id: str, count: int = 1):
        self.next[next_id] = self.next.get(id, 0) + int(count)

    def set_occurrences(self, total: int, occ_by_colour: Optional[List[int]] = None):
        self.occurrences = int(total)
        if occ_by_colour is not None:
            self.occ_by_colour = [int(x) for x in occ_by_colour]
        else:
            self.occ_by_colour = [int(total)]

    def __repr__(self):
        return f"Value(id={self.id!r}, occ={self.occurrences}, y={self.vert_centre:.2f})"