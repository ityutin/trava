from collections import Counter

from trava.ext.boosting_eval.boosting_logic import CommonBoostingEvalLogic
from trava.ext.boosting_eval.eval_steps import EvalFitSteps
from trava.fit_predictor import FitPredictConfig, FitPredictConfigUpdateStep, FitPredictorSteps
from trava.split.result import SplitResult
from trava.tracker import Tracker


class _GroupConfigUpdateStep(FitPredictConfigUpdateStep):
    def __init__(self, group_col_name: str):
        self._group_col_name = group_col_name

    def fit_split_data(self, raw_split_data: SplitResult, config: FitPredictConfig, tracker: Tracker) -> SplitResult:
        X_valid = None
        if raw_split_data.X_valid is not None:
            X_valid = raw_split_data.X_valid.drop(self._group_col_name, axis=1)

        result = SplitResult(
            X_train=raw_split_data.X_train.drop(self._group_col_name, axis=1),
            y_train=raw_split_data.y_train,
            X_test=raw_split_data.X_test.drop(self._group_col_name, axis=1),
            y_test=raw_split_data.y_test,
            X_valid=X_valid,
            y_valid=raw_split_data.y_valid,
        )

        return result

    def fit_params(
        self, fit_params: dict, fit_split_data: SplitResult, config: FitPredictConfig, tracker: Tracker
    ) -> dict:
        raw_split_data = config.raw_split_data
        assert raw_split_data
        train_counted_groups = self._counted_groups(X=raw_split_data.X_train)
        fit_params["group"] = train_counted_groups

        return fit_params

    def _counted_groups(self, X):
        train_groups = X[self._group_col_name].values
        counted_groups = list(Counter(train_groups).values())
        return counted_groups


class _GroupEvalConfigUpdateStep(_GroupConfigUpdateStep):
    def __init__(self, group_col_name: str):
        super().__init__(group_col_name=group_col_name)

    def fit_params(
        self, fit_params: dict, fit_split_data: SplitResult, config: FitPredictConfig, tracker: Tracker
    ) -> dict:
        fit_params = super().fit_params(
            fit_params=fit_params, fit_split_data=fit_split_data, config=config, tracker=tracker
        )

        raw_split_data = config.raw_split_data
        assert raw_split_data
        assert raw_split_data.X_valid is not None, "X_valid set must be present to run evaluation"

        eval_counted_groups = self._counted_groups(X=raw_split_data.X_valid)
        fit_params["eval_group"] = [fit_params["group"], eval_counted_groups]

        return fit_params

    def _counted_groups(self, X):
        train_groups = X[self._group_col_name].values
        counted_groups = list(Counter(train_groups).values())
        return counted_groups


class GroupFitSteps(FitPredictorSteps):
    """
    Simple extension for problems that are based on groups ( e.g. ranking )
    that provides group parameter for training a model.

    Init parameters
    ----------
    group_col_name: str
        Which column is used to store groups
    """

    def __init__(self, group_col_name: str):
        group_config_step = _GroupConfigUpdateStep(group_col_name=group_col_name)
        super().__init__(config_steps=[group_config_step])


class GroupEvalFitSteps(EvalFitSteps):
    """
    Same as GroupFitSteps, but also adds some modifications to support evaluation.

    Init parameters
    ----------
    eval_logic: Eval
        Contains logic of how to perform evaluation on the model.
    group_col_name: str
        Which column is used to store groups
    """

    def __init__(self, eval_logic: CommonBoostingEvalLogic, group_col_name: str):
        group_eval_config_step = _GroupEvalConfigUpdateStep(group_col_name=group_col_name)

        super().__init__(eval_logic=eval_logic)
        self.config_steps.insert(0, group_eval_config_step)
