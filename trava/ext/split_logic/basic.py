from typing import Optional
from sklearn.model_selection import train_test_split

from trava.split.logic import SplitLogic


class BasicSplitLogic(SplitLogic):
    """
    Basically just a wrapper around sklearn's train_test_split.
    Can be used in combination with another split logic.
    """

    def __init__(
        self,
        random_state: int = 15,
        shuffle: bool = True,
        validation_from_test: bool = True,
        group_col: Optional[str] = None,
        stratify: bool = None,
    ):
        self._random_state = random_state
        self._shuffle = shuffle
        self._stratify = stratify
        self._valid_from_test = validation_from_test
        self._group_col = group_col

    def split(self, data, test_size: float, valid_size: float = 0.0, **kwargs) -> tuple:
        data_to_split = data

        if self._group_col:
            unique_groups = data[self._group_col].unique()
            data_to_split = unique_groups

        train_data, test_data = train_test_split(
            data_to_split,
            test_size=test_size,
            shuffle=self._shuffle,
            stratify=self._stratify,
            random_state=self._random_state,
            **kwargs
        )

        valid_data = None
        if valid_size > 0.0:
            if self._valid_from_test:
                data_for_valid_split = test_data
            else:
                data_for_valid_split = train_data
            data_left, valid_data = train_test_split(
                data_for_valid_split,
                test_size=valid_size,
                shuffle=self._shuffle,
                stratify=self._stratify,
                random_state=self._random_state,
                **kwargs
            )
            if self._valid_from_test:
                test_data = data_left
            else:
                train_data = data_left

        if self._group_col:
            train_data = data[data[self._group_col].isin(train_data)]
            test_data = data[data[self._group_col].isin(test_data)]
            if valid_data is not None:
                valid_data = data[data[self._group_col].isin(valid_data)]

        return train_data, test_data, valid_data
