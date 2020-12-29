from typing import Optional

from trava.ext.boosting_eval.boosting_logic import CommonBoostingEvalLogic


class LightGBMEvalLogic(CommonBoostingEvalLogic):
    def _best_iteration(self, model) -> int:
        return model.best_iteration_

    def _evals_results(self, model) -> dict:
        return model.evals_result_

    def _train_metrics_key(self, model) -> Optional[str]:
        if self._n_eval_sets(model=model) == 1:
            return None

        return "training"

    def _eval_metrics_key(self, model) -> str:
        return "valid_" + str(self._n_eval_sets(model=model) - 1)
