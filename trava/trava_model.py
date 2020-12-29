from time import time  # type: ignore
from typing import Optional, Any, Dict

import numpy as np
import pandas as pd

from trava.model_info import ModelInfo


class _RawModelImitation:
    """
    If TravaModel doesn't have raw model in memory anymore,
    it will be replaced with an object of this class to be able
    to provide results made by the real model.

    Init parameters
    ----------
    y_pred: np.array
        hard labels, predicted by a raw model
    y_pred_proba: np.array
        classes' probabilities, predicted by a raw model
    """

    def __init__(self, y_pred, y_pred_proba: Optional[Any] = None):
        self._y_pred = y_pred
        self._y_pred_proba = y_pred_proba

    def predict(self, X):
        return self._y_pred

    def predict_proba(self, X):
        assert self._y_pred_proba is not None, "You haven't provided y_pred_proba"
        return self._y_pred_proba


class _RawModelWrapper:
    def __init__(self, raw_model, y_pred: np.ndarray, y_pred_proba: np.ndarray):
        self._raw_model = raw_model

        self._y_pred: np.ndarray = y_pred
        self._y_pred_proba: np.ndarray = y_pred_proba

    def __eq__(self, o: object) -> bool:
        return self._raw_model == o

    def __getattr__(self, attr):
        return getattr(self._raw_model, attr)

    def predict(self, *args, **kwargs):
        return self._y_pred

    def predict_proba(self, *args, **kwargs):
        return self._y_pred_proba


class TravaModel(ModelInfo):
    """
    Wrapper of any model you want to use with Trava.
    Stores both true and predicted labels as well as some metadata.

    Init parameters
    ----------
    raw_model: sklearn-style model
        Model that supports fit, predict and predict_proba methods.
    model_id: str
        Model unique identifier, will be used for saving metrics etc
    """

    def __init__(self, raw_model, model_id: str):
        self._raw_model = raw_model
        self._needs_proba = hasattr(raw_model, "predict_proba") and callable(raw_model.predict_proba)
        self._model_id = model_id

        self._y_train = None
        self._y_train_pred = None
        self._y_train_pred_proba = None
        self._y_test = None
        self._y_test_pred = None
        self._y_test_pred_proba = None

        self._fit_params: Dict[str, Any] = {}
        self._predict_params: Dict[str, Any] = {}
        self._fit_time: Optional[float] = None
        self._predict_time: Optional[float] = None

    def copy(
        self, model_id: Optional[str] = None, existing_model: Optional["TravaModel"] = None, only_fit: bool = True
    ):
        """
        Makes a copy of the trava_model, but with another model_id
        """
        if (not model_id and not existing_model) or (model_id and existing_model):
            raise Exception("Pass either model_id or existing_model")

        result = existing_model
        if not result:
            # TODO: ugly
            assert model_id
            result = TravaModel(raw_model=self.raw_model, model_id=model_id)

        result._y_train = np.copy(self._y_train)
        result._y_train_pred = np.copy(self._y_train_pred)
        result._y_train_pred_proba = np.copy(self._y_train_pred_proba)
        if not only_fit:
            result._y_test = np.copy(self._y_test)
            result._y_test_pred = np.copy(self._y_test_pred)
            result._y_test_pred_proba = np.copy(self._y_test_pred_proba)

        if self.fit_params:
            result._fit_params = self.fit_params.copy()
        if not only_fit and self.predict_params:
            result._predict_params = self.predict_params.copy()
        result._fit_time = self.fit_time
        if not only_fit:
            result._predict_time = self.predict_time

        return result

    @property
    def model_id(self) -> str:
        return self._model_id

    def get_model(self, for_train: bool):
        if for_train:
            y_pred = self._y_train_pred
            y_pred_proba = self._y_train_pred_proba
        else:
            y_pred = self._y_test_pred
            y_pred_proba = self._y_test_pred_proba

        if self._raw_model:
            return _RawModelWrapper(raw_model=self._raw_model, y_pred=y_pred, y_pred_proba=y_pred_proba)

        if y_pred is None:
            raise ValueError("y_pred is missing as well as raw model, unexpected behaviour")

        return _RawModelImitation(y_pred=y_pred, y_pred_proba=y_pred_proba)

    @property
    def raw_model(self):
        return self._raw_model

    @property
    def fit_params(self) -> dict:
        return self._fit_params

    @property
    def fit_time(self) -> Optional[float]:
        return self._fit_time

    @property
    def predict_time(self) -> Optional[float]:
        return self._predict_time

    @property
    def predict_params(self) -> dict:
        return self._predict_params

    @property
    def is_classification_model(self) -> bool:
        return self._needs_proba

    def y(self, for_train: bool) -> np.array:
        if for_train:
            return self._y_train
        else:
            return self._y_test

    def y_pred(self, for_train: bool) -> np.array:
        return self._y_train_pred if for_train else self._y_test_pred

    def y_pred_proba(self, for_train: bool) -> np.array:
        return self._y_train_pred_proba if for_train else self._y_test_pred_proba

    def fit(self, X, y, fit_params: Optional[dict] = None, predict_params: Optional[dict] = None):
        """
        Fits the saved model

        Parameters
        ----------
        X: pandas dataframe
            Features for the model
        y: numpy array, pandas series
            Target for the model
        fit_params: dict
            Custom params to use when calling model's fit method
        predict_params: dict
                Custom params to use when calling model's predict method

        """
        self._fit_params = fit_params or {}
        self._predict_params = predict_params or {}

        start = time()
        self._raw_model.fit(X, y, **self._fit_params)
        end = time()
        self._fit_time = end - start

        self._y_train = y.values if isinstance(y, pd.Series) else y
        self._y_train_pred = self._raw_model.predict(X, **self._predict_params)

        if self._needs_proba:
            self._y_train_pred_proba = self._raw_model.predict_proba(X, **self._predict_params)

    def predict(self, X, y):
        """
        Calls predict and (if possible) predict proba on the saved model

        Parameters
        ----------
        X: pandas dataframe
            Features for the model
        """
        start = time()
        self._y_test = y.values if isinstance(y, pd.Series) else y
        predict_params = self._predict_params or {}
        self._y_test_pred = self._raw_model.predict(X, **predict_params)
        end = time()
        self._predict_time = end - start

        if self._needs_proba:
            self._y_test_pred_proba = self._raw_model.predict_proba(X, **predict_params)

    def unload_model(self):
        """
        Deletes fitted model from memory
        """
        del self._raw_model
        self._raw_model = None
