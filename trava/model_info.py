from abc import abstractmethod
from typing import Optional

import numpy as np


class ModelInfo:
    """
    Just an interface for TravaModel to not expose its internals.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @abstractmethod
    def get_model(self, for_train: bool):
        pass

    @abstractmethod
    def raw_model(self):
        pass

    @abstractmethod
    def y(self, for_train: bool) -> np.array:
        pass

    @abstractmethod
    def y_pred(self, for_train: bool) -> np.array:
        pass

    @abstractmethod
    def y_pred_proba(self, for_train: bool) -> np.array:
        pass

    @property
    @abstractmethod
    def fit_time(self) -> Optional[float]:
        pass

    @property
    @abstractmethod
    def predict_time(self) -> Optional[float]:
        pass
