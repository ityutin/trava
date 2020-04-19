from typing import List, Tuple

from trava.ext.cv.base import CV
from trava.fit_predictor import FitPredictor, RawModelUpdateStep, FitPredictConfigUpdateStep, FinalHandlerStep, \
    FitPredictConfig
from trava.logger import TravaLogger
from trava.trava_model import TravaModel
from trava.split.result import SplitResult


class CVFitPredictor(FitPredictor):
    def __init__(self,
                 cv: CV,
                 ignore_cols: List[str],
                 groups=None,
                 raw_model_update_steps: List[RawModelUpdateStep] = None,
                 config_update_steps: List[FitPredictConfigUpdateStep] = None,
                 final_steps: List[FinalHandlerStep] = None,
                 logger: TravaLogger = None):
        super().__init__(raw_model_update_steps=raw_model_update_steps,
                         config_update_steps=config_update_steps,
                         final_steps=final_steps,
                         logger=logger)

        self._cv = cv
        self._groups = groups
        self._ignore_cols = ignore_cols

    def _models_configs(self, raw_model, config: FitPredictConfig) -> List[Tuple[TravaModel, FitPredictConfig]]:
        result = []

        X = config.raw_dataset.X
        y = config.raw_dataset.y
        X_cleaned = X.drop(self._ignore_cols, axis=1)

        for fold_idx, (train_indices, test_indices) in enumerate(self._cv.split(X=X_cleaned, y=y, groups=self._groups)):
            X_train, y_train = X_cleaned.iloc[train_indices], y.iloc[train_indices]
            X_test, y_test = X_cleaned.iloc[test_indices], y.iloc[test_indices]

            split_result = SplitResult(X_train=X_train,
                                       y_train=y_train,
                                       X_test=X_test,
                                       y_test=y_test)

            fold_model_id = config.model_id + '_fold_{}'.format(fold_idx + 1)
            model_config = FitPredictConfig(raw_split_data=split_result,
                                            raw_model=config.raw_model,
                                            model_init_params=config.model_init_params,
                                            model_id=fold_model_id,
                                            scorers_providers=config.scorers_providers,
                                            serialize_model=config.serialize_model,
                                            fit_params=config.fit_params,
                                            predict_params=config.predict_params)

            trava_model = TravaModel(raw_model=raw_model, model_id=fold_model_id)

            result.append((trava_model, model_config))

        return result