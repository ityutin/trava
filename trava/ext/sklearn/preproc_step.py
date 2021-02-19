# noinspection PyPep8Naming
import pandas as pd
from sklearn.pipeline import Pipeline

from trava.fit_predictor import FitPredictConfigUpdateStep, FitPredictConfig
from trava.split.result import SplitResult
from trava.tracker import Tracker


class PreprocConfigStep(FitPredictConfigUpdateStep):
    """
    Updates split data using sklearn's Pipeline

    Init parameters
    ----------
    preprocessing: Pipeline
        Object that contains all the data transformations you need. ( without a model in it )
    """

    def __init__(self, preprocessing: Pipeline):
        self._preprocessing = preprocessing

    def fit_split_data(self, raw_split_data: SplitResult, config: FitPredictConfig, tracker: Tracker) -> SplitResult:
        X_train = self._preprocessing.fit_transform(X=raw_split_data.X_train, y=raw_split_data.y_train)
        X_test = self._preprocessing.transform(X=raw_split_data.X_test)

        X_valid = raw_split_data.X_valid

        if X_valid is not None:
            X_valid = self._preprocessing.transform(X=raw_split_data.X_valid)

        result = SplitResult(
            X_train=X_train,
            X_test=X_test,
            y_train=raw_split_data.y_train,
            y_test=raw_split_data.y_test,
            X_valid=X_valid,
            y_valid=raw_split_data.y_valid,
        )

        return result

    def fit_params(
        self, fit_params: dict, fit_split_data: SplitResult, config: FitPredictConfig, tracker: Tracker
    ) -> dict:
        """
        Previous steps could already put some data in params.
        """
        return self._find_and_process_df(params=fit_params)

    def predict_params(
        self, predict_params: dict, fit_split_data: SplitResult, config: FitPredictConfig, tracker: Tracker
    ) -> dict:
        return self._find_and_process_df(params=predict_params)

    def _find_and_process_df(self, params: dict) -> dict:
        def process_df(df: pd.DataFrame):
            return self._preprocessing.transform(X=df)

        def try_process_df(maybe_df):
            if isinstance(maybe_df, pd.DataFrame):
                return process_df(df=maybe_df)

            return maybe_df

        def processed_value(value):
            if isinstance(value, (list, tuple)):
                result = []
                for list_value in value:
                    result.append(processed_value(value=list_value))

                return result
            elif isinstance(value, dict):
                for dict_key, dict_value in value.items():
                    dict_value[dict_key] = processed_value(value=dict_value)

                return value

            return try_process_df(maybe_df=value)

        for key, value in params.items():
            params[key] = processed_value(value=value)

        return params
