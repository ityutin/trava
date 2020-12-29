from typing import List

from trava.logger import TravaLogger
from trava.metric import Metric
from trava.model_results import ModelResult
from trava.results_handler import ResultsHandler
from trava.scorer import Scorer
from trava.trava_tracker import Tracker


class MetricsDictHandler(ResultsHandler):
    """
    Returns all the metrics wrapped in a dictionary.
    """

    def __init__(self, scorers: List[Scorer], include_train_metrics: bool = False):
        super().__init__(scorers)

        self._include_train_metrics = include_train_metrics

    def handle(self, results: List[ModelResult], logger: TravaLogger, tracker: Tracker):
        result = {}

        for model_results in results:
            model_metrics = {}
            if self._include_train_metrics:
                model_metrics["train"] = self._metrics_dict(metrics=model_results.train_metrics(provider=self))

            model_metrics["test"] = self._metrics_dict(metrics=model_results.test_metrics(provider=self))
            any_metrics = model_results.other_metrics(provider=self)
            model_metrics["other"] = self._metrics_dict(metrics=any_metrics)

            result[model_results.model_id] = model_metrics

        return result

    @staticmethod
    def _metrics_dict(metrics: List[Metric]):
        result = {}

        for metric in metrics:
            if metric.is_scalar:
                result[metric.name] = metric.value

        return result
