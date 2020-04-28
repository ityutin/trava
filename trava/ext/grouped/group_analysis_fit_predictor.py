from typing import Tuple, List

from trava.fit_predictor import FitPredictConfig, FitPredictor, RawModelUpdateStep, FitPredictConfigUpdateStep, \
    FinalHandlerStep, FitPredictorSteps
from trava.logger import TravaLogger
from trava.trava_model import TravaModel
from trava.split.result import SplitResult


class GroupAnalysisFitPredictor(FitPredictor):
    """
    If your dataset contains data related to multiple groups,
    you may want to test your model's performance on each group separately.
    See subclasses for details.
    """
    def __init__(self,
                 group_col_name: str,
                 steps: FitPredictorSteps = None,
                 logger: TravaLogger = None):
        super().__init__(steps=steps or FitPredictorSteps(),
                         logger=logger)

        self._group_col_name = group_col_name

    def _group_X_y(self, group, X, y):
        if X is None or y is None:
            return None, None

        cond = X[self._group_col_name] == group
        group_X = X[cond]
        group_y = y[cond]

        return group_X.drop(self._group_col_name, axis=1), group_y


class TrainOnOneTestOnOneFitPredictor(GroupAnalysisFitPredictor):
    """
    Trains model on each group's subset as well as gets predictions using that subset.
    """
    def _models_configs(self, raw_model, config: FitPredictConfig) -> List[Tuple[TravaModel, FitPredictConfig]]:
        unique_groups = sorted(set(config.raw_split_data.X_train[self._group_col_name].values))

        result = []
        for group in unique_groups:
            model_config = self._config_for_group(group=group, config=config)
            group_model_id = model_config.model_id + '_' + str(group)
            trava_model = TravaModel(raw_model=raw_model, model_id=group_model_id)

            result.append((trava_model, model_config))

        return result

    def _config_for_group(self, group, config: FitPredictConfig) -> FitPredictConfig:
        split_result = config.raw_split_data

        group_X_train, group_y_train = self._group_X_y(group=group, X=split_result.X_train, y=split_result.y_train)
        group_X_test, group_y_test = self._group_X_y(group=group, X=split_result.X_test, y=split_result.y_test)
        group_X_valid, group_y_valid = self._group_X_y(group=group, X=split_result.X_valid, y=split_result.y_valid)

        group_split_result = SplitResult(X_train=group_X_train,
                                         y_train=group_y_train,
                                         X_test=group_X_test,
                                         y_test=group_y_test,
                                         X_valid=group_X_valid,
                                         y_valid=group_y_valid)

        result = FitPredictConfig(raw_split_data=group_split_result,
                                  raw_model=config.raw_model,
                                  model_init_params=config.model_init_params,
                                  model_id=config.model_id,
                                  scorers_providers=config.scorers_providers,
                                  serializer=config.serializer,
                                  fit_params=config.fit_params,
                                  predict_params=config.predict_params)

        return result


class TrainOnAllTestOnOneFitPredictor(GroupAnalysisFitPredictor):
    """
    Trains model on a whole dataset, but predictions are made for each group separately.
    """
    def __init__(self, group_col_name: str, logger: TravaLogger = None):
        super().__init__(group_col_name=group_col_name, logger=logger)

        self._model = None

    def _models_configs(self, raw_model, config: FitPredictConfig) -> List[Tuple[TravaModel, FitPredictConfig]]:
        unique_groups = sorted(set(config.raw_split_data.X_train[self._group_col_name].values))

        result = []
        main_model = None
        for group in unique_groups:
            group_config = self._config_for_group(group=group, config=config)
            group_model_id = group_config.model_id + '_' + str(group)
            if main_model:
                result_model = main_model.copy(model_id=group_model_id)
            else:
                main_model = TravaModel(raw_model=raw_model, model_id=group_model_id)
                result_model = main_model

            result.append((result_model, group_config))

        return result

    def _config_for_group(self, group, config: FitPredictConfig) -> FitPredictConfig:
        group_X_test, group_y_test = self._group_X_y(group=group,
                                                     X=config.raw_split_data.X_test,
                                                     y=config.raw_split_data.y_test)
        group_X_valid, group_y_valid = self._group_X_y(group=group,
                                                       X=config.raw_split_data.X_valid,
                                                       y=config.raw_split_data.y_valid)

        group_split_result = SplitResult(X_train=config.raw_split_data.X_train.drop(self._group_col_name, axis=1),
                                         y_train=config.raw_split_data.y_train,
                                         X_test=group_X_test,
                                         y_test=group_y_test,
                                         X_valid=group_X_valid,
                                         y_valid=group_y_valid)

        result = FitPredictConfig(raw_split_data=group_split_result,
                                  raw_model=config.raw_model,
                                  model_init_params=config.model_init_params,
                                  model_id=config.model_id,
                                  scorers_providers=config.scorers_providers,
                                  serializer=config.serializer,
                                  fit_params=config.fit_params,
                                  predict_params=config.predict_params)

        return result

    def _fit(self, trava_model: TravaModel, X, y, fit_params: dict, predict_params: dict):
        if not trava_model.fit_time:
            trava_model.fit(X=X, y=y, fit_params=fit_params, predict_params=predict_params)
