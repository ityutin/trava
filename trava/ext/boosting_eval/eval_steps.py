from trava.ext.boosting_eval.boosting_logic import CommonBoostingEvalLogic
from trava.fit_predictor import FitPredictConfig, FitPredictConfigUpdateStep, FinalHandlerStep, FitPredictorSteps
from trava.tracker import Tracker
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

    def __init__(self, eval_logic: CommonBoostingEvalLogic):
        self._eval_logic = eval_logic

    def fit_params(
        self, fit_params: dict, fit_split_data: SplitResult, config: FitPredictConfig, tracker: Tracker
    ) -> dict:
        split_result = fit_split_data
        self._assert_eval(X_eval=split_result.X_valid, y_eval=split_result.y_valid)

        fit_params = self._eval_logic.setup_eval(
            fit_params=fit_params,
            X_train=split_result.X_train,
            y_train=split_result.y_train,
            X_eval=split_result.X_valid,
            y_eval=split_result.y_valid,
        )

        return fit_params

    @staticmethod
    def _assert_eval(X_eval, y_eval):
        assert_text = "Couldn't find eval data in split_result"
        assert X_eval is not None and y_eval is not None, assert_text


class EvalFinalStep(FinalHandlerStep):
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

    def handle(self, trava_model: TravaModel, config: FitPredictConfig, tracker: Tracker):
        self._eval_logic.plot_if_needed(model_id=trava_model.model_id, model=trava_model.raw_model, tracker=tracker)
        self._eval_logic.track_eval_metrics(model_id=trava_model.model_id, model=trava_model.raw_model, tracker=tracker)


class EvalFitSteps(FitPredictorSteps):
    """
    Is used for training model with evaluation sets.
    Was made for boosting models.

    Init parameters
    ----------
    eval_logic: Eval
        Contains logic of how to perform evaluation on the model.
    """

    def __init__(self, eval_logic: CommonBoostingEvalLogic):
        config_step = EvalConfigStep(eval_logic=eval_logic)
        final_step = EvalFinalStep(eval_logic=eval_logic)

        super().__init__(config_steps=[config_step], final_steps=[final_step])
