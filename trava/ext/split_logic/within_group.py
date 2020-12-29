import pandas as pd

from trava.ext.split_logic.basic import BasicSplitLogic
from trava.split.logic import SplitLogic


class WithinGroupSplitLogic(SplitLogic):
    """
    Performs a split for every group in the data and then
    merges them into the single result.
    Assumes that you work with pandas data frame.
    Init parameters

    ----------
    group_col:
        What is column that contains group identifiers.
    group_split_logic: SplitLogic
        Contains logic how to split each group.
    group_split_logic_kwargs: dict
        Any additional parameters you want to pass to group_split_logic.
    """

    def __init__(self, group_col: str, group_split_logic: SplitLogic, group_split_logic_kwargs: dict = None):
        self._group_col = group_col
        self._group_split_logic = group_split_logic or BasicSplitLogic()
        self._group_split_logic_kwargs = group_split_logic_kwargs if group_split_logic_kwargs else {}

    def split(self, data, test_size: float, valid_size: float, **kwargs) -> tuple:
        unique_groups = data[self._group_col].unique()

        all_groups_data = []

        for group in unique_groups:
            group_data = data[data[self._group_col] == group]

            split_group_data = self._group_split_logic.split(
                group_data, test_size=test_size, valid_size=valid_size, **self._group_split_logic_kwargs
            )
            if not valid_size:
                # to safely merge dataframes after split within each group
                split_group_data = split_group_data[:2]
            all_groups_data.append(split_group_data)

        separated_groups_data = list(zip(*all_groups_data))
        merged_groups_data = [pd.concat(dfs, axis=0, copy=False, sort=False) for dfs in separated_groups_data]
        if not valid_size:
            # to conform to expected output
            merged_groups_data.append(None)
        return tuple(merged_groups_data)
