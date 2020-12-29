from collections import Counter
import typing as t
import numpy as np

from trava.model_info import ModelInfo
from trava.scorer import Scorer


class GroupScorer(Scorer):
    """
    Supports group based metrics ( e.g. ranking ).
    Expects that score_func receives true scores and predicted scores.

    Init parameters
    ----------
    score_func: callable
        See superclass
    group_col_name: str
        Which column in the input data is used for storing group values
    needs_grouped_y: bool
        If True, your score_func will be called with 2d arrays instead of 1d.
        y_true will contain grouped scores for all the groups ( same goes for y_pred )
    metrics_kwargs: dict
        Any additional parameters to pass to score_func
    """

    def __init__(self, score_func: t.Callable, group_col_name: str, needs_grouped_y=False, **metrics_kwargs):
        self._group_col_name = group_col_name
        self._needs_grouped_y = needs_grouped_y

        super().__init__(
            score_func=score_func, needs_proba=False, requires_raw_model=False, requires_X_y=True, **metrics_kwargs
        )

    def _make_scorer(self, score_func: t.Callable, **metrics_kwargs) -> t.Callable:
        def scorer(model, model_info: ModelInfo, for_train: bool, X, X_raw, y):
            y_pred = model_info.y_pred(for_train=for_train)
            groups = X_raw[self._group_col_name].values
            counted_groups = list(Counter(groups).values())
            indices = np.cumsum(counted_groups)[:-1]
            grouped_y = np.split(y, indices)
            grouped_y_pred = np.split(y_pred, indices)

            if self._needs_grouped_y:
                return score_func(grouped_y, grouped_y_pred, **metrics_kwargs)
            else:
                metrics = []
                for y, y_pred in zip(grouped_y, grouped_y_pred):
                    metrics.append(score_func(y, y_pred, **metrics_kwargs))
                return np.mean(metrics)

        return scorer
