from typing import Optional

from trava.ext.split_logic.basic import BasicSplitLogic
from trava.split.logic import SplitLogic


class TimeSplitLogic(SplitLogic):
    """
    Basic time split. It works just by sorting the data
    by time_col and taking the tail of the data.
    Assumes that you work with pandas data frame.

    Init parameters
    ----------
    time_col: str
        What is the column that contains time data.
    need_to_sort:
        Whether your data is already sorted
    group_col:
        Whether your data contains groups and you want
        to split every group separately.
    """

    def __init__(self, time_col: str, need_to_sort=True, group_col: Optional[str] = None):
        self._time_col = time_col
        self._need_to_sort = need_to_sort
        self._split_logic = BasicSplitLogic(shuffle=False, validation_from_test=True, group_col=group_col)

    def split(self, data, test_size: float, valid_size: float, **kwargs) -> tuple:
        sorted_data = data

        if self._need_to_sort:
            sorted_data = data.sort_values(self._time_col)

        return self._split_logic.split(data=sorted_data, test_size=test_size, valid_size=valid_size)
