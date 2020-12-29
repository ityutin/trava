from abc import ABC, abstractmethod
from typing import Optional, Callable

from trava.trava_model import TravaModel
from trava.model_info import ModelInfo


class Scorer(ABC):
    """
    Wraps the score function and makes scorer in sklearn style.

    Init parameters
    ----------
    score_func: callable
        Function to calculate a metric.
        It must accept true labels as well as predicted labels.
    needs_proba: bool
        Whether a model must calculate probabilities instead of hard labels.
    requires_raw_model: bool, requires_X_y: bool
        These two parameters are required only if you already have evaluated model
        and then want to calculate some new metrics using it.
        By the moment you want to do it both raw model and X_y data may be unloaded,
        yet TravaModel caches y_pred and y_pred_proba results. If more data is required
        for the scorer or it preforms some complex logic, you may restrict using post-evaluation
        using these two parameters.
    name: str
        Custom name for your metric, func.__name__ is taken otherwise.
    metrics_kwargs: dict
        Any additional parameters to pass to score_func
    """

    def __init__(
        self,
        score_func: Callable,
        needs_proba=False,
        requires_raw_model=False,
        requires_X_y=False,
        name: Optional[str] = None,
        **metrics_kwargs
    ):
        self._func_name = name or score_func.__name__
        self._is_other_scorer = issubclass(type(self), OtherScorer)
        self._needs_proba = needs_proba
        self._requires_raw_model = requires_raw_model
        self._requires_X_y = requires_X_y

        self._scorer = self._make_scorer(score_func=score_func, **metrics_kwargs)

    @property
    def is_other_scorer(self):
        return self._is_other_scorer

    @abstractmethod
    def _make_scorer(self, score_func: Callable, **metrics_kwargs) -> Callable:
        """
        Creates a function that calls score_func with provided true labels and predictions.
        """
        pass

    @property
    def func_name(self) -> str:
        return self._func_name

    def __call__(self, trava_model: TravaModel, for_train: bool, X, X_raw, y, **kwargs):
        if self._requires_raw_model and not trava_model.raw_model():
            raise Exception("Cannot perform eval on model {} " "because it was unloaded.".format(trava_model.model_id))

        if self._requires_X_y and (X is None or X_raw is None or y is None):
            raise Exception(
                "Cannot perform eval on model {} "
                "because data is required and was unloaded.".format(trava_model.model_id)
            )

        return self._scorer(
            model=trava_model.get_model(for_train=for_train),
            model_info=trava_model,
            for_train=for_train,
            X=X,
            X_raw=X_raw,
            y=y,
            **kwargs
        )


class OtherScorer(Scorer):
    """
    You may calculate any custom value you want based on model/input data etc.
    Such metrics are separated from a model's performance metrics.
    """

    def _make_scorer(self, score_func: Callable, **metrics_kwargs) -> Callable:
        def scorer(model, model_info: ModelInfo, for_train: bool, X, X_raw, y):
            return score_func(model=model, model_info=model_info, for_train=for_train, X=X, X_raw=X_raw, y=y)

        return scorer
