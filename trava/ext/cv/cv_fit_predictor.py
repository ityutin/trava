from typing import List, Tuple

from trava.raw_dataset import RawDataset

from trava.ext.cv.base import CV
from trava.fit_predictor import FitPredictor, FitPredictConfig, FitPredictorSteps
from trava.logger import TravaLogger
from trava.trava_model import TravaModel
from trava.split.result import SplitResult


class CVFitPredictor(FitPredictor):
    def __init__(
        self,
        cv: CV,
        raw_dataset: RawDataset,
        ignore_cols: List[str],
        groups=None,
        steps: FitPredictorSteps = None,
        logger: TravaLogger = None,
    ):
        super().__init__(steps=steps or FitPredictorSteps(), logger=logger)

        self._cv = cv
        self._raw_dataset = raw_dataset
        self._groups = groups
        self._ignore_cols = ignore_cols

    def _models_configs(self, raw_model, config: FitPredictConfig) -> List[Tuple[TravaModel, FitPredictConfig]]:
        result = []

        X = self._raw_dataset.X
        y = self._raw_dataset.y
        X_cleaned = X.drop(self._ignore_cols, axis=1)

        for fold_idx, (train_indices, test_indices) in enumerate(self._cv.split(X=X_cleaned, y=y, groups=self._groups)):
            X_train, y_train = X_cleaned.iloc[train_indices], y.iloc[train_indices]
            X_test, y_test = X_cleaned.iloc[test_indices], y.iloc[test_indices]

            split_result = SplitResult(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

            fold_model_id = config.model_id + "_fold_{}".format(fold_idx + 1)
            model_config = FitPredictConfig(
                raw_split_data=split_result,
                raw_model=config.raw_model,
                model_init_params=config.model_init_params,
                model_id=fold_model_id,
                scorers_providers=config.scorers_providers,
                serializer=config.serializer,
                fit_params=config.fit_params,
                predict_params=config.predict_params,
            )

            trava_model = TravaModel(raw_model=raw_model, model_id=fold_model_id)

            result.append((trava_model, model_config))

        return result
