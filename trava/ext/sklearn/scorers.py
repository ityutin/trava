from sklearn.metrics import make_scorer
from typing import Callable

from trava.model_info import ModelInfo
from trava.scorer import Scorer


class SklearnScorer(Scorer):
    """
    Uses standard sklearn's make_scorer.
    """

    def __init__(
        self,
        score_func: Callable,
        greater_is_better=True,
        needs_proba=False,
        needs_threshold=False,
        sample_weight_required=False,
        **metrics_kwargs
    ):
        self._greater_is_better = greater_is_better
        self._needs_threshold = needs_threshold
        self._sample_weight_required = sample_weight_required

        super().__init__(
            score_func=score_func,
            needs_proba=needs_proba,
            requires_raw_model=needs_threshold,
            requires_X_y=False,
            **metrics_kwargs
        )

    def _make_scorer(self, score_func: Callable, **metrics_kwargs) -> Callable:
        def scorer(model, model_info: ModelInfo, for_train: bool, X, X_raw, y):
            if self._sample_weight_required and X is None:
                raise Exception(
                    "Sample weight is required for score ({}) calculation, "
                    "so it must be called with valid X_y data".format(self.func_name)
                )

            if X is None:
                # the last resort. If data and model are unloaded and your metric support interface
                # like (y_true, y_pred, **kwargs), then you will be able to calculate metrics

                y_cached = model_info.y(for_train=for_train)

                if self._needs_proba:
                    y_pred_values = model_info.y_pred_proba(for_train=for_train)
                    if len(set(y_cached)) == 2:
                        y_pred_values = y_pred_values[:, 1]
                else:
                    y_pred_values = model_info.y_pred(for_train=for_train)

                result = score_func(y_cached, y_pred_values, **metrics_kwargs)
            else:
                sk_scorer = self._get_sklearn_scorer(score_func=score_func, **metrics_kwargs)
                sample_weight = self._sample_weight(X=X, X_raw=X_raw)
                result = sk_scorer(model, X, y, sample_weight=sample_weight)
            return result

        return scorer

    def _get_sklearn_scorer(self, score_func: Callable, **metrics_kwargs):
        result = make_scorer(
            score_func=score_func,
            greater_is_better=self._greater_is_better,
            needs_proba=self._needs_proba,
            needs_threshold=self._needs_threshold,
            **metrics_kwargs
        )
        return result

    @staticmethod
    def _sample_weight(X, X_raw):
        """
        Maybe it will help to handle cases when the parameter is needed in a scorer.
        Introduced this template method just to not forget about it.
        """
        pass


# wrappers for sklearn metrics


def sk(score_func: Callable, **kwargs) -> Scorer:
    return _sk(score_func=score_func, needs_proba=False, **kwargs)


def sk_proba(score_func: Callable, **kwargs) -> Scorer:
    return _sk(score_func=score_func, needs_proba=True, **kwargs)


def _sk(score_func: Callable, needs_proba=False, **kwargs) -> Scorer:
    return SklearnScorer(score_func=score_func, needs_proba=needs_proba, **kwargs)
