from abc import ABC, abstractmethod
from typing import Optional, List

from trava.split.logic import SplitLogic
from trava.split.result import SplitResult


class SplitResultHandler(ABC):
    """
    Can be used to modify SplitResult ( e.g. apply resampling )
    """

    @abstractmethod
    def handle(self, split_result: SplitResult) -> SplitResult:
        pass


class DataSplitConfig:
    """
    Data class containing all the data needed to make a split.
    """

    def __init__(
        self,
        split_logic: SplitLogic,
        target_col_name: str,
        test_size: float,
        valid_size: float = 0.0,
        split_result_handlers: Optional[List[SplitResultHandler]] = None,
        ignore_cols: Optional[List[str]] = None,
    ):
        self.split_logic = split_logic
        self.target_col_name = target_col_name
        self.test_size = test_size
        self.valid_size = valid_size
        self.split_result_handlers = split_result_handlers or []
        self.ignore_cols = ignore_cols or []
