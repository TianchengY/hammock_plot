# summary.py
from typing import Any, Dict, List, Optional
import numpy as np

WRAP_WIDTH = 72 # column at which comma-separated lists wrap


def _yes_no(flag) -> str:
    return "yes" if flag else "no"


def _wrap_list(label: str, items: List[Any], indent: str) -> List[str]:
    """
        Renders "<indent><label>: a, b, c", wrapping at WRAP_WIDTH with the
        continuation lines aligned under the first item. Breaks only between
        items, so a single name longer than the width overflows rather than
        being split in half.
    """
    prefix = f"{indent}{label}: "
    hang = " " * len(prefix)

    if not items:
        return [prefix + "none"]

    lines = []
    current = prefix
    for item in items:
        piece = str(item)
        if current == prefix:
            current += piece
            continue
        # the comma stays on the line being closed, so count it when deciding
        candidate = f"{current}, {piece}"
        if len(candidate) <= WRAP_WIDTH:
            current = candidate
        else:
            lines.append(current + ",")
            current = hang + piece
    lines.append(current)
    return lines


class VariableSummary:
    """
        The resolved specification for one axis: what the package decided the
        variable was, how it drew it, and whatever ordering or binning applied.

        Attributes:
            name: the variable name, as passed in var
            data_type: "categorical" or "numerical"
            display_type: the display actually drawn, defaults resolved
            num_levels: number of label levels, or "by value" for an axis whose
                labels are drawn per distinct value. None if categorical.
            value_order: the drawn order, but only if the user set one
            type_from_value_order: True if this variable is only categorical
                because value_order converted it
    """
    def __init__(self,
                 name: str,
                 data_type: str,
                 display_type: str,
                 num_levels=None,
                 value_order: Optional[List[Any]] = None,
                 type_from_value_order: bool = False):
        self.name = name
        self.data_type = data_type
        self.display_type = display_type
        self.num_levels = num_levels
        self.value_order = value_order
        self.type_from_value_order = type_from_value_order

    def _lines(self) -> List[str]:
        data_type = self.data_type
        if self.type_from_value_order:
            # a numeric column named in value_order is silently retyped as
            # categorical; without saying so the line reads as a bug
            data_type += " (set by value_order)"

        lines = [f"  {self.name}",
                 f"    Data type: {data_type}",
                 f"    Display type: {self.display_type}"]
        if self.num_levels is not None:
            lines.append(f"    Numerical levels: {self.num_levels}")
        if self.value_order is not None:
            lines.extend(_wrap_list("Value order", self.value_order, "    "))
        return lines

    def __repr__(self):
        return f"VariableSummary(name={self.name!r}, data_type={self.data_type!r}, display_type={self.display_type!r})"


class HammockSummary:
    """
        A readback of the call that produced a figure: the dataframe it was
        given, the arguments it was called with, and what the package resolved
        for each axis. Every value here is a property of the call, so two
        summaries of the same call at different figure sizes are identical.

        Built by Hammock at the end of plot(); the not_plotted constructor
        covers a Hammock that only holds a dataframe so far.
    """
    def __init__(self,
                 n_observations: int,
                 n_variables: int,
                 columns: Optional[List[str]] = None,
                 plotted: bool = False,
                 var: Optional[List[str]] = None,
                 n_plotted: Optional[int] = None,
                 hi_var: Optional[str] = None,
                 hi_values: Optional[List[Any]] = None,
                 hi_box: Optional[str] = None,
                 missing: bool = False,
                 missing_placeholder: Optional[str] = None,
                 same_scale: Optional[List[str]] = None,
                 weights: Optional[str] = None,
                 unibar: bool = True,
                 label: bool = True,
                 violin_bw_method=None,
                 save_path: Optional[str] = None,
                 variables: Optional[List[VariableSummary]] = None):
        self.n_observations = n_observations
        self.n_variables = n_variables
        self.columns = columns
        self.plotted = plotted
        self.var = var
        self.n_plotted = n_plotted
        self.hi_var = hi_var
        self.hi_values = hi_values
        self.hi_box = hi_box
        self.missing = missing
        self.missing_placeholder = missing_placeholder
        self.same_scale = same_scale
        self.weights = weights
        self.unibar = unibar
        self.label = label
        self.violin_bw_method = violin_bw_method
        self.save_path = save_path
        self.variables = variables if variables is not None else []

    @classmethod
    def not_plotted(cls, data_df) -> "HammockSummary":
        """
            Builds the summary for a Hammock that has only been handed a
            dataframe. At this point var, weights and the rest haven't arrived,
            so the columns are listed instead - they're what you need to choose
            var from.
        """
        return cls(
            n_observations=data_df.shape[0],
            n_variables=data_df.shape[1],
            columns=data_df.columns.tolist(),
            plotted=False,
        )

    @property
    def display_types(self) -> Dict[str, str]:
        """
            {variable name: display type}, in axis order.
        """
        return {v.name: v.display_type for v in self.variables}

    @property
    def data_types(self) -> Dict[str, str]:
        """
            {variable name: "categorical" or "numerical"}, in axis order.
        """
        return {v.name: v.data_type for v in self.variables}

    @property
    def value_order(self) -> Dict[str, List[Any]]:
        """
            {variable name: drawn order}, for the variables the user ordered.
        """
        return {v.name: v.value_order for v in self.variables if v.value_order is not None}

    @property
    def numerical_levels(self) -> Dict[str, Any]:
        """
            {variable name: level count or "by value"}, for numerical variables.
        """
        return {v.name: v.num_levels for v in self.variables if v.num_levels is not None}

    def _data_lines(self) -> List[str]:
        lines = ["Data:",
                 f"  Observations: {self.n_observations}",
                 f"  Num Variables: {self.n_variables}"]
        if not self.plotted:
            lines.extend(_wrap_list("Columns", self.columns, "  "))
        return lines

    def _specification_lines(self) -> List[str]:
        lines = ["Plot specification:"]
        lines.extend(_wrap_list("Variables", self.var, "  "))

        # only worth a line when rows actually went missing; under missing=True
        # nothing is ever dropped
        if self.n_plotted is not None and self.n_plotted != self.n_observations:
            lines.append(f"  Observations plotted: {self.n_plotted} of {self.n_observations}")

        lines.append(f"  Highlight variable: {self.hi_var if self.hi_var else 'none'}")
        if self.hi_var:
            # hi_box only describes how a highlight is laid out, so it goes with
            # the highlight rather than printing on every plot
            lines.extend(_wrap_list("Highlight values", self.hi_values, "  "))
            lines.append(f"  Highlight box layout: {self.hi_box}")

        lines.append(f"  Missing values shown: {_yes_no(self.missing)}")
        if self.missing:
            lines.append(f"  Missing placeholder: {self.missing_placeholder}")

        if self.same_scale:
            lines.extend(_wrap_list("Same scale", self.same_scale, "  "))
        else:
            lines.append("  Same scale: none")

        lines.append(f"  Weight variable: {self.weights if self.weights else 'none'}")
        lines.append(f"  Unibars shown: {_yes_no(self.unibar)}")
        lines.append(f"  Labels shown: {_yes_no(self.label)}")

        if self.violin_bw_method is not None:
            lines.append(f"  Violin bandwidth: {self.violin_bw_method}")
        if self.save_path:
            lines.append(f"  Saved to: {self.save_path}")
        return lines

    def _variable_lines(self) -> List[str]:
        lines = ["Variable specifications:"]
        for i, variable in enumerate(self.variables):
            if i > 0:
                lines.append("")
            lines.extend(variable._lines())
        return lines

    def __str__(self):
        blocks = [["Hammock object"], self._data_lines()]
        if self.plotted:
            blocks.append(self._specification_lines())
            blocks.append(self._variable_lines())
        else:
            blocks.append(["Not yet plotted. Call plot(var=[...]) to draw."])
        return "\n\n".join("\n".join(block) for block in blocks)

    def __repr__(self):
        # the readback is the useful repr; a bare name in a notebook should show
        # the same thing print does
        return self.__str__()


def build_variable_summaries(unibars,
                             value_order: Dict[str, List[Any]],
                             user_ordered: set,
                             converted_by_value_order: set) -> List[VariableSummary]:
    """
        Reads the resolved specification off the Unibars that were
        built, in axis order.

        Everything here has to come from the Unibar rather than from the
        arguments: display_type is defaulted in Figure.build_unibars, and
        label_type is rewritten inside Unibar._build_values (a numerical axis
        with fewer than 7 distinct values is labelled per value, never binned).
        Re-deriving either would mean a second copy of a rule that is free to
        drift from the first.

        user_ordered is the set of value_order keys captured before plot()
        rewrote the dict
    """
    summaries = []
    for uni in unibars:
        categorical = uni.val_type == np.str_

        num_levels = None
        if not categorical:
            # levels only apply when the labels are drawn as levels; otherwise
            # num_levels is set but never reached, so printing it would name a
            # binning that didn't happen
            num_levels = uni.num_levels if uni.label_type == "levels" else "by value"

        order = value_order.get(uni.name) if uni.name in user_ordered else None

        summaries.append(VariableSummary(
            name=uni.name,
            data_type="categorical" if categorical else "numerical",
            display_type=uni.display_type,
            num_levels=num_levels,
            value_order=order,
            type_from_value_order=uni.name in converted_by_value_order,
        ))
    return summaries
