# noinspection PyPep8Naming
from typing import List

from sklearn.pipeline import Pipeline

from trava.fit_predictor import FitPredictConfigUpdateStep, FitPredictConfig, FitPredictor, RawModelUpdateStep, \
    FinalHandlerStep
from trava.logger import TravaLogger
from trava.split.result import SplitResult


class PreprocessingConfigUpdateStep(FitPredictConfigUpdateStep):
    """
    Updates split data using sklearn's Pipeline

    Init parameters
    ----------
    preprocessing: Pipeline
        Object that contains all the data transformations you need. ( without a model in it )
    """

    def __init__(self, preprocessing: Pipeline = None):
        self._preprocessing = preprocessing

    def fit_split_data(self, raw_split_data: SplitResult, config: FitPredictConfig) -> SplitResult:
        X_train = self._preprocessing.fit_transform(X=raw_split_data.X_train)
        X_test = self._preprocessing.transform(raw_split_data.X_test)
        X_valid = self._preprocessing.transform(raw_split_data.X_valid)

        result = SplitResult(X_train=X_train,
                             y_train=raw_split_data.y_train,
                             X_test=X_test,
                             y_test=raw_split_data.y_test,
                             X_valid=X_valid,
                             y_valid=raw_split_data.y_valid)

        return result


class SkFitPredictor(FitPredictor):
    def __init__(self,
                 preprocessing: Pipeline = None,
                 raw_model_update_steps: List[RawModelUpdateStep] = None,
                 config_update_steps: List[FitPredictConfigUpdateStep] = None,
                 final_steps: List[FinalHandlerStep] = None,
                 logger: TravaLogger = None):
        super().__init__(raw_model_update_steps=raw_model_update_steps,
                         config_update_steps=config_update_steps,
                         final_steps=final_steps,
                         logger=logger)

        if preprocessing:
            self._config_steps += [PreprocessingConfigUpdateStep(preprocessing=preprocessing)]