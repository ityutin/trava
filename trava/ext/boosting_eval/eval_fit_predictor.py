from typing import List

from sklearn.pipeline import Pipeline

from trava.ext.boosting_eval.boosting_logic import CommonBoostingEvalLogic
from trava.fit_predictor import FitPredictConfig, FitPredictConfigUpdateStep, FinalHandlerStep, FitPredictor, \
    RawModelUpdateStep
from trava.logger import TravaLogger
from trava.trava_model import TravaModel
from trava.split.result import SplitResult


class EvalConfigStep(FitPredictConfigUpdateStep):
    """
    Is used for training model with evaluation sets.
    Was made for boosting models.

    Init parameters
    ----------
    X_eval: pandas dataframe
            Eval features for the model
    y_eval: numpy array, pandas series
        Eval target for the model
    eval_logic: Eval
        Contains logic of how to perform evaluation on the model.
    """

    def __init__(self,
                 eval_logic: CommonBoostingEvalLogic):
        self._eval_logic = eval_logic

    def fit_params(self, fit_params: dict, fit_split_data: SplitResult, config: FitPredictConfig) -> dict:
        split_result = fit_split_data
        self._assert_eval(X_eval=split_result.X_valid, y_eval=split_result.y_valid)

        fit_params = self._eval_logic.setup_eval(fit_params=fit_params,
                                                 X_train=split_result.X_train,
                                                 y_train=split_result.y_train,
                                                 X_eval=split_result.X_valid,
                                                 y_eval=split_result.y_valid)

        return fit_params

    @staticmethod
    def _assert_eval(X_eval, y_eval):
        assert_text = "Couldn't find eval data in split_result"
        assert X_eval is not None and y_eval is not None, assert_text


class PlotEvalResultsStep(FinalHandlerStep):
    """
    Is used for training model with evaluation sets.
    Was made for boosting models.

    Init parameters
    ----------
    eval_logic: Eval
        Contains logic of how to perform evaluation on the model.
    """

    def __init__(self, eval_logic: CommonBoostingEvalLogic):
        self._eval_logic = eval_logic

    def handle(self, trava_model: TravaModel, config: FitPredictConfig):
        self._eval_logic.plot_if_needed(model_id=trava_model.model_id, model=trava_model.raw_model)


class EvalFitPredictor(FitPredictor):
    """
    Is used for training model with evaluation sets.
    Was made for boosting models.
    Just a wrapper for FitPredictor steps.

    Init parameters
    ----------
    eval_logic: Eval
        Contains logic of how to perform evaluation on the model.
    """
    def __init__(self,
                 eval_logic: CommonBoostingEvalLogic,
                 raw_model_update_steps: List[RawModelUpdateStep] = None,
                 config_update_steps: List[FitPredictConfigUpdateStep] = None,
                 final_steps: List[FinalHandlerStep] = None,
                 logger: TravaLogger = None):
        raw_model_update_steps = raw_model_update_steps or []
        config_update_steps = config_update_steps or []
        final_steps = final_steps or []

        config_step = EvalConfigStep(eval_logic=eval_logic)
        plot_step = PlotEvalResultsStep(eval_logic=eval_logic)

        super().__init__(raw_model_update_steps=raw_model_update_steps,
                         config_update_steps=config_update_steps + [config_step],
                         final_steps=final_steps + [plot_step],
                         logger=logger)