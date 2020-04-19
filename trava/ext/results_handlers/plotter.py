from typing import List, Tuple

from trava.logger import TravaLogger
from trava.metric import Metric
from trava.model_results import ModelResult
from trava.results_handler import ResultsHandler
from trava.scorer import Scorer


class PlotHandler(ResultsHandler):
    """
    Plots metrics.
    """
    def __init__(self, scorers: List[Scorer], plot_funcs: List[callable]):
        """
        Parameters
        ----------
        scorers: list
            Scorers for metrics you want to plot.
        plot_funcs:
            Actual plot functions for metrics. Must be in order according to scorers.
            One scorer - one plot function.
        """
        assert len(scorers) == len(plot_funcs), "provide plot_func for each scorer"

        super().__init__(scorers)

        self._plot_funcs = plot_funcs

    def handle(self, results: List[ModelResult], logger: TravaLogger):
        all_train_metrics = []
        all_test_metrics = []
        for model_results in results:
            if not model_results.is_one_fit_result:
                continue
            if model_results.train_metrics(provider=self):
                model_train_metrics = model_results.train_metrics(provider=self)
                if len(model_train_metrics) > 0:
                    all_train_metrics.append(model_train_metrics)

            model_test_metrics = model_results.test_metrics(provider=self)
            if len(model_test_metrics) > 0:
                all_test_metrics.append(model_test_metrics)

        plots_needed = len(all_test_metrics) > 0

        if not plots_needed:
            return

        if len(all_train_metrics) > 0:
            self._plot_metrics(metrics_set=list(zip(*all_train_metrics)), label='Train')

        if plots_needed:
            self._plot_metrics(metrics_set=list(zip(*all_test_metrics)), label='Test')

    def _plot_metrics(self, metrics_set: List[Tuple[Metric]], label: str):
        for metrics, plot_func in zip(metrics_set, self._plot_funcs):
            plot_func(metrics, label)
