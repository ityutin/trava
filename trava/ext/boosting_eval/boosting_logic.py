from abc import abstractmethod
from typing import Optional

import matplotlib.pyplot as plt

from trava.tracker import Tracker


class CommonBoostingEvalLogic:
    """
    Some models support evaluation with early-stopping ( like gradient boosting ).
    This class was written with boosting in mind. Probably it can be generalized
    thus it won't be the base class anymore.
    """

    def __init__(self, needs_plot: bool, eval_metric: str = None, early_stopping_rounds: Optional[int] = 10):
        """
        Parameters
        ----------
        needs_plot: bool
            Whether to plot the results of evaluation at the end.
        eval_metric: str
            What metric to use for evaluation
        early_stopping_rounds: int
            Number of rounds to wait before perform early stopping
        """
        self._needs_plot = needs_plot
        self._eval_metric = eval_metric
        self._early_stopping_rounds = early_stopping_rounds

    def setup_eval(self, fit_params, X_train, y_train, X_eval, y_eval) -> dict:
        """
        You must make all the preparations to setup evaluation.

        Parameters
        ----------
        fit_params: dict
            Custom params to use when calling model's fit method
        X_train: pandas dataframe
            Train features for the model
        y_train: numpy array, pandas series
            Train target for the model
        X_eval: pandas dataframe
            Eval features for the model
        y_eval: numpy array, pandas series
            Eval target for the model

        Returns
        -------
        Prepared fit_params
        """
        result = fit_params.copy()

        if self._eval_metric:
            result["eval_metric"] = self._eval_metric

        result["eval_set"] = self._eval_set(X_train=X_train, y_train=y_train, X_eval=X_eval, y_eval=y_eval)

        if self._early_stopping_rounds:
            result["early_stopping_rounds"] = self._early_stopping_rounds

        return result

    def _eval_set(self, X_train, y_train, X_eval, y_eval):
        eval_set = [(X_eval, y_eval)]

        if self._user_train_in_eval:
            eval_set = [(X_train, y_train)] + eval_set

        return eval_set

    def train_set_results(self, model) -> Optional[list]:
        """
        Gets resulting metrics of evaluation for train set.

        Parameters
        ----------
        model: sklearn-style model
            Model that supports fit, predict and predict_proba methods.

        Returns
        -------
        List containing evaluation metrics for train set.
        """
        eval_results = self._evals_results(model=model)
        key = self._train_metrics_key(model=model)

        if not key:
            return None

        return eval_results[key][self._get_metric_name(model=model)]

    def eval_set_results(self, model) -> list:
        """
        Gets resulting metrics of evaluation for eval set.

        Parameters
        ----------
        model: sklearn-style model
            Model that supports fit, predict and predict_proba methods.

        Returns
        -------
        List containing evaluation metrics for eval set.
        """
        eval_results = self._evals_results(model=model)
        key = self._eval_metrics_key(model=model)

        return eval_results[key][self._get_metric_name(model=model)]

    def plot_if_needed(self, model_id: str, model, tracker: Tracker):
        """
        If plot was requested during initialization, learning curves
        will be plotted when evaluation has finished.

        Parameters
        ----------
        model_id: str
            Model unique identifier, will be used for saving metrics etc
        model: sklearn-style model
            Model that supports fit, predict and predict_proba methods.
        tracker: Tracker
            Tracker for saving plots & eval results
        """
        if self._needs_plot:
            metric_to_plot = self._get_metric_name(model=model)

            train_results = self.train_set_results(model=model)
            eval_results = self.eval_set_results(model=model)

            epochs = len(eval_results)
            x_axis = range(0, epochs)

            fig, ax = plt.subplots()
            if train_results is not None:
                ax.plot(x_axis, train_results, label="Train")

            ax.plot(x_axis, eval_results, label="Test")
            ax.legend()
            plt.ylabel(metric_to_plot)
            plt.title("{} {}".format("{} validation results".format(model_id), metric_to_plot))
            plt.show()

            plot_filename = "eval_plot" + "_" + model_id
            tracker.track_plot(model_id=model_id, fig=fig, filename=plot_filename)

    def track_eval_metrics(self, model_id: str, model, tracker: Tracker):
        """
        Tracks eval results

        Parameters
        ----------
        model_id: str
            Model unique identifier, will be used for saving metrics etc
        model: sklearn-style model
            Model that supports fit, predict and predict_proba methods.
        tracker: Tracker
            Tracker for saving plots & eval results
        """
        metric_to_plot = self._get_metric_name(model=model)

        train_results = self.train_set_results(model=model) or []
        eval_results = self.eval_set_results(model=model) or []

        tracker.track_tag(model_id=model_id, tag_key="eval_metric", tag_value=metric_to_plot)
        tracker.track_metric_value(model_id=model_id, name="early_stopping_rounds", value=self._early_stopping_rounds)
        tracker.track_metric_value(model_id=model_id, name="stopped_iteration", value=len(eval_results) - 1)
        tracker.track_metric_value(model_id=model_id, name="best_iteration", value=self._best_iteration(model=model))

        self._track_eval_results(
            model_id=model_id, eval_results=train_results, metric=metric_to_plot, train=True, tracker=tracker
        )

        self._track_eval_results(
            model_id=model_id, eval_results=eval_results, metric=metric_to_plot, train=False, tracker=tracker
        )

    @staticmethod
    def _track_eval_results(model_id: str, eval_results: list, metric: str, train: bool, tracker: Tracker):
        metric_name = ("train" if train else "eval") + "_" + metric

        for idx, value in enumerate(eval_results):
            tracker.track_metric_value(model_id=model_id, name=metric_name, value=value, step=idx)

    @abstractmethod
    def _best_iteration(self, model) -> int:
        pass

    @abstractmethod
    def _evals_results(self, model) -> dict:
        pass

    @abstractmethod
    def _train_metrics_key(self, model) -> Optional[str]:
        pass

    @abstractmethod
    def _eval_metrics_key(self, model) -> str:
        pass

    @property
    def _user_train_in_eval(self) -> bool:
        return True

    def _get_metric_name(self, model) -> str:
        eval_results = self._evals_results(model=model)
        result = list(eval_results[self._eval_metrics_key(model=model)].keys())[0]
        return result

    def _n_eval_sets(self, model) -> int:
        eval_results = self._evals_results(model=model)

        if not isinstance(eval_results, dict):
            raise ValueError("Eval results must be dict")

        n_eval_sets = len(eval_results.keys())
        assert 0 < n_eval_sets <= 2, "only train and test are now supported"

        return n_eval_sets
