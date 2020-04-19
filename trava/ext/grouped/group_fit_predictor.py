from collections import Counter
from typing import List

from trava.ext.boosting_eval.boosting_logic import CommonBoostingEvalLogic
from trava.ext.boosting_eval.eval_fit_predictor import EvalConfigStep, PlotEvalResultsStep
from trava.fit_predictor import FitPredictConfig, FitPredictor, FitPredictConfigUpdateStep, RawModelUpdateStep, \
    FinalHandlerStep
from trava.logger import TravaLogger
from trava.split.result import SplitResult


class _GroupConfigUpdateStep(FitPredictConfigUpdateStep):
    def __init__(self,
                 group_col_name: str,
                 group_param_name: str = 'group'):
        self._group_col_name = group_col_name
        self._group_parameter_name = group_param_name

    def fit_split_data(self, raw_split_data: SplitResult, config: FitPredictConfig) -> SplitResult:
        result = SplitResult(X_train=raw_split_data.X_train.drop(self._group_col_name, axis=1),
                             y_train=raw_split_data.y_train,
                             X_test=raw_split_data.X_test.drop(self._group_col_name, axis=1),
                             y_test=raw_split_data.y_test,
                             X_valid=raw_split_data.X_valid.drop(self._group_col_name, axis=1),
                             y_valid=raw_split_data.y_valid)

        return result

    def fit_params(self, fit_params: dict, fit_split_data: SplitResult, config: FitPredictConfig) -> dict:
        train_counted_groups = self._counted_groups(X=config.raw_split_data.X_train)
        fit_params[self._group_parameter_name] = train_counted_groups

        return fit_params

    def _counted_groups(self, X):
        train_groups = X[self._group_col_name].values
        counted_groups = list(Counter(train_groups).values())
        return counted_groups


class _GroupEvalConfigUpdateStep(_GroupConfigUpdateStep):
    def __init__(self,
                 group_col_name: str,
                 group_param_name: str,
                 eval_group_param_name: str):
        super().__init__(group_col_name=group_col_name,
                         group_param_name=group_param_name)

        self._eval_group_param_name = eval_group_param_name

    def fit_params(self, fit_params: dict, fit_split_data: SplitResult, config: FitPredictConfig) -> dict:
        fit_params = super().fit_params(fit_params=fit_params, fit_split_data=fit_split_data, config=config)

        eval_counted_groups = self._counted_groups(X=config.raw_split_data.X_valid)
        fit_params[self._eval_group_param_name] = [fit_params[self._group_parameter_name], eval_counted_groups]

        return fit_params

    def _counted_groups(self, X):
        train_groups = X[self._group_col_name].values
        counted_groups = list(Counter(train_groups).values())
        return counted_groups


class GroupFitPredictor(FitPredictor):
    """
    Simple extension for problems that are based on groups ( e.g. ranking )
    that provides group parameter for training a model.

    Init parameters
    ----------
    group_col_name: str
        Which column is used to store groups
    """
    def __init__(self,
                 group_col_name: str,
                 group_param_name: str = 'group',
                 raw_model_update_steps: List[RawModelUpdateStep] = None,
                 config_update_steps: List[FitPredictConfigUpdateStep] = None,
                 final_steps: List[FinalHandlerStep] = None,
                 logger: TravaLogger = None):
        raw_model_update_steps = raw_model_update_steps or []
        config_update_steps = config_update_steps or []
        final_steps = final_steps or []

        group_config_step = _GroupConfigUpdateStep(group_col_name=group_col_name,
                                                   group_param_name=group_param_name)

        super().__init__(raw_model_update_steps=raw_model_update_steps,
                         config_update_steps=[group_config_step] + config_update_steps,
                         final_steps=final_steps,
                         logger=logger)


class GroupEvalFitPredictor(FitPredictor):
    """
    Same as GroupFitPredictor, but also adds some modifications to support evaluation.

    Init parameters
    ----------
    group_col_name: str
        Which column is used to store groups
    """
    def __init__(self,
                 group_col_name: str,
                 eval_logic: CommonBoostingEvalLogic,
                 group_param_name: str = 'group',
                 eval_group_param_name: str = 'eval_group',
                 raw_model_update_steps: List[RawModelUpdateStep] = None,
                 config_update_steps: List[FitPredictConfigUpdateStep] = None,
                 final_steps: List[FinalHandlerStep] = None,
                 logger: TravaLogger = None):
        raw_model_update_steps = raw_model_update_steps or []
        config_update_steps = config_update_steps or []
        final_steps = final_steps or []

        group_eval_config_step = _GroupEvalConfigUpdateStep(group_col_name=group_col_name,
                                                            group_param_name=group_param_name,
                                                            eval_group_param_name=eval_group_param_name)
        eval_config_step = EvalConfigStep(eval_logic=eval_logic)
        plot_step = PlotEvalResultsStep(eval_logic=eval_logic)

        config_update_steps += [group_eval_config_step, eval_config_step]
        final_steps += [plot_step]

        super().__init__(raw_model_update_steps=raw_model_update_steps,
                         config_update_steps=config_update_steps,
                         final_steps=final_steps,
                         logger=logger)
