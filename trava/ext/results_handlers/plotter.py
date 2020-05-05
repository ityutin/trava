import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Any

from trava.logger import TravaLogger
from trava.metric import Metric
from trava.model_results import ModelResult
from trava.results_handler import ResultsHandler
from trava.scorer import Scorer
from trava.trava_tracker import Tracker


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

    def handle(self, results: List[ModelResult], logger: TravaLogger, tracker: Tracker):
        self._show(results=results, logger=logger, tracker=tracker)
        self._track(results=results, logger=logger, tracker=tracker)

    def _show(self, results: List[ModelResult], logger: TravaLogger, tracker: Tracker):
        self._plot(results=results, logger=logger, show=True, use_one_figure=True, tracker=tracker)

    def _track(self, results: List[ModelResult], logger: TravaLogger, tracker: Tracker):
        self._plot(results=results, logger=logger, show=False, use_one_figure=False, tracker=tracker)

    def _plot(self,
              results: List[ModelResult],
              logger: TravaLogger,
              show: bool,
              tracker: Tracker,
              use_one_figure: bool = False,
              model_id: str = None):
        all_train_metrics = []
        all_test_metrics = []
        for model_result in results:
            if not model_result.is_one_fit_result:
                if show:
                    continue

                many_model_results = []
                for evaluator in model_result.evaluators:
                    many_model_results.append(ModelResult(model_id=evaluator.model_id, evaluators=[evaluator]))

                self._plot(results=many_model_results,
                           logger=logger,
                           show=False,
                           tracker=tracker,
                           use_one_figure=True,
                           model_id=model_result.model_id)

                self._plot(results=many_model_results,
                           logger=logger,
                           show=False,
                           tracker=tracker,
                           use_one_figure=False)
                continue
            if model_result.train_metrics(provider=self):
                model_train_metrics = model_result.train_metrics(provider=self)
                all_train_metrics.append(model_train_metrics)

            model_test_metrics = model_result.test_metrics(provider=self)
            all_test_metrics.append(model_test_metrics)

        if len(all_train_metrics) > 0:
            self._plot_metrics_set(metrics_set=list(zip(*all_train_metrics)),
                                   label='Train',
                                   show=show,
                                   use_one_figure=use_one_figure,
                                   tracker=tracker,
                                   model_id=model_id)

        self._plot_metrics_set(metrics_set=list(zip(*all_test_metrics)),
                               label='Test',
                               show=show,
                               use_one_figure=use_one_figure,
                               tracker=tracker,
                               model_id=model_id)

    def _plot_metrics_set(self,
                          metrics_set: List[Tuple[Metric]],
                          label: str,
                          show: bool,
                          use_one_figure: bool,
                          tracker: Tracker,
                          model_id: str = None):
        if show:
            self._show_metrics_set(metrics_set=metrics_set, label=label, tracker=tracker)
        else:
            self._track_metrics_set(metrics_set=metrics_set,
                                    label=label,
                                    tracker=tracker,
                                    use_one_figure=use_one_figure,
                                    model_id=model_id)

    def _show_metrics_set(self,
                          metrics_set: List[Tuple[Metric]],
                          label: str,
                          tracker: Tracker):
        fig, ax = self._fig_ax()
        for plot_idx, (metrics, plot_func) in self._enumerated_metrics_plots(metrics_set=metrics_set):
            self._plot_metrics(metrics=metrics,
                               plot_func=plot_func,
                               label=label,
                               tracker=tracker,
                               show=True,
                               use_one_figure=True,
                               fig=fig,
                               ax=ax)

    def _track_metrics_set(self,
                           metrics_set: List[Tuple[Metric]],
                           label: str,
                           tracker: Tracker,
                           use_one_figure: bool,
                           model_id: str = None):
        fig, ax = None, None
        if use_one_figure:
            fig, ax = self._fig_ax()

        for plot_idx, (metrics, plot_func) in self._enumerated_metrics_plots(metrics_set=metrics_set):
            self._plot_metrics(metrics=metrics,
                               plot_func=plot_func,
                               label=label,
                               tracker=tracker,
                               show=False,
                               use_one_figure=use_one_figure,
                               model_id=model_id,
                               fig=fig,
                               ax=ax)

    def _plot_metrics(self,
                      metrics: Tuple[Metric],
                      plot_func: callable,
                      label: str,
                      tracker: Tracker,
                      show: bool,
                      use_one_figure: bool,
                      model_id: str = None,
                      fig=None,
                      ax=None):
        def color_for(idx):
            r = lambda: random.randint(0, 255)
            base_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            if idx > len(base_colors) - 1:
                return '#%02X%02X%02X' % (r(), r(), r())

            return base_colors[idx]

        if use_one_figure:
            fig, ax = self._fig_ax(existing_fig=fig)

        for metric_idx, metric in enumerate(metrics):
            if not use_one_figure:
                fig, ax = self._fig_ax(existing_fig=fig)

            plot_func(metric=metric, fig=fig, ax=ax, color=color_for(idx=metric_idx), label=label)
            if not show:
                filename = (label + '_' + metric.name).lower()
                tracker.track_plot(model_id=model_id or metric.model_id,
                                   fig=fig,
                                   filename=filename)

        if show:
            fig.show()
        else:
            plt.close(fig)

    def _enumerated_metrics_plots(self, metrics_set: List[Tuple[Metric]]):
        return enumerate(zip(metrics_set, self._plot_funcs))

    @staticmethod
    def _fig_ax(existing_fig=None) -> tuple:
        if existing_fig:
            plt.close(existing_fig)

        return plt.subplots(figsize=(10, 10))
