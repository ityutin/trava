from typing import Optional

from trava.ext.boosting_eval.boosting_logic import CommonBoostingEvalLogic


class CatBoostEvalLogic(CommonBoostingEvalLogic):
    def __init__(self, needs_plot: bool, eval_metric: str = None, early_stopping_rounds: Optional[int] = 10):
        assert not eval_metric, "Initialize your model with the eval metric instead"

        super().__init__(needs_plot=needs_plot, eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds)

    def _best_iteration(self, model) -> int:
        return model.get_best_iteration()

    def _evals_results(self, model) -> dict:
        return model.get_evals_result()

    def _train_metrics_key(self, model) -> Optional[str]:
        if self._n_eval_sets(model=model) == 1:
            return None

        return "learn"

    def _eval_metrics_key(self, model) -> str:
        return "validation"

    @property
    def _user_train_in_eval(self) -> bool:
        return False
