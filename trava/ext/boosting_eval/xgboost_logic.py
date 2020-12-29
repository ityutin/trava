from typing import Optional

from trava.ext.boosting_eval.boosting_logic import CommonBoostingEvalLogic


class XGBoostEvalLogic(CommonBoostingEvalLogic):
    def _best_iteration(self, model) -> int:
        return model.best_iteration

    def _evals_results(self, model) -> dict:
        return model.evals_result()

    def _train_metrics_key(self, model) -> Optional[str]:
        if self._n_eval_sets(model=model) == 1:
            return None

        return self._results_key(idx=0)

    def _eval_metrics_key(self, model) -> str:
        n_eval_sets = self._n_eval_sets(model=model)
        result = self._results_key(idx=n_eval_sets - 1)
        return result

    @staticmethod
    def _results_key(idx: int) -> str:
        return "validation_" + str(idx)


class XGBoostRankerEvalLogic(XGBoostEvalLogic):
    def _evals_results(self, model) -> dict:
        return model.evals_result

    @staticmethod
    def _results_key(idx: int) -> str:
        return "eval_" + str(idx)
