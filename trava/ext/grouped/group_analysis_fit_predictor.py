from typing import Tuple, List

from trava.fit_predictor import FitPredictConfig, FitPredictor, FitPredictorSteps
from trava.logger import TravaLogger
from trava.trava_model import TravaModel
from trava.split.result import SplitResult


class GroupAnalysisFitPredictor(FitPredictor):
    """
    If your dataset contains data related to multiple groups,
    you may want to test your model's performance on each group separately.
    See subclasses for details.
    """

    def __init__(self, group_col_name: str, steps: FitPredictorSteps = None, logger: TravaLogger = None):
        super().__init__(steps=steps or FitPredictorSteps(), logger=logger)

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
        split_result = config.raw_split_data
        assert split_result

        unique_groups = sorted(set(split_result.X_train[self._group_col_name].values))

        result = []
        for group in unique_groups:
            model_config = self._config_for_group(group=group, config=config)
            group_model_id = model_config.model_id + "_" + str(group)
            trava_model = TravaModel(raw_model=raw_model, model_id=group_model_id)

            result.append((trava_model, model_config))

        return result

    def _config_for_group(self, group, config: FitPredictConfig) -> FitPredictConfig:
        split_result = config.raw_split_data
        assert split_result

        group_X_train, group_y_train = self._group_X_y(group=group, X=split_result.X_train, y=split_result.y_train)
        group_X_test, group_y_test = self._group_X_y(group=group, X=split_result.X_test, y=split_result.y_test)
        group_X_valid, group_y_valid = self._group_X_y(group=group, X=split_result.X_valid, y=split_result.y_valid)

        group_split_result = SplitResult(
            X_train=group_X_train,
            y_train=group_y_train,
            X_test=group_X_test,
            y_test=group_y_test,
            X_valid=group_X_valid,
            y_valid=group_y_valid,
        )

        result = FitPredictConfig(
            raw_split_data=group_split_result,
            raw_model=config.raw_model,
            model_init_params=config.model_init_params,
            model_id=config.model_id,
            scorers_providers=config.scorers_providers,
            serializer=config.serializer,
            fit_params=config.fit_params,
            predict_params=config.predict_params,
        )

        return result


class TrainOnAllTestOnOneFitPredictor(GroupAnalysisFitPredictor):
    """
    Trains model on a whole dataset, but predictions are made for each group separately.
    """

    def __init__(self, group_col_name: str, steps: FitPredictorSteps = None, logger: TravaLogger = None):
        super().__init__(group_col_name=group_col_name, steps=steps, logger=logger)
        self._group_models: List[TravaModel] = []
        self._is_raw_model_ready = False

    def _models_configs(self, raw_model, config: FitPredictConfig) -> List[Tuple[TravaModel, FitPredictConfig]]:
        split_result = config.raw_split_data
        assert split_result

        unique_groups = sorted(set(split_result.X_train[self._group_col_name].values))

        result = []
        for group in unique_groups:
            group_config = self._config_for_group(group=group, config=config)
            group_model_id = group_config.model_id + "_" + str(group)
            result_model = TravaModel(raw_model=raw_model, model_id=group_model_id)

            result.append((result_model, group_config))

        self._group_models = [item[0] for item in result]

        return result

    def _config_for_group(self, group, config: FitPredictConfig) -> FitPredictConfig:
        split_result = config.raw_split_data
        assert split_result

        group_X_test, group_y_test = self._group_X_y(group=group, X=split_result.X_test, y=split_result.y_test)
        group_X_valid, group_y_valid = self._group_X_y(group=group, X=split_result.X_valid, y=split_result.y_valid)

        group_split_result = SplitResult(
            X_train=split_result.X_train.drop(self._group_col_name, axis=1),
            y_train=split_result.y_train,
            X_test=group_X_test,
            y_test=group_y_test,
            X_valid=group_X_valid,
            y_valid=group_y_valid,
        )

        result = FitPredictConfig(
            raw_split_data=group_split_result,
            raw_model=config.raw_model,
            model_init_params=config.model_init_params,
            model_id=config.model_id,
            scorers_providers=config.scorers_providers,
            serializer=config.serializer,
            fit_params=config.fit_params,
            predict_params=config.predict_params,
        )

        return result

    def _fit(self, trava_model: TravaModel, X, y, fit_params: dict, predict_params: dict):
        if not self._is_raw_model_ready:
            trava_model.fit(X=X, y=y, fit_params=fit_params, predict_params=predict_params)

            for group_model in self._group_models:
                if group_model != trava_model:
                    trava_model.copy(existing_model=group_model, only_fit=True)

            self._is_raw_model_ready = True
