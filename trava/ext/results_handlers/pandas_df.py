import pandas as pd
from typing import List

from trava.logger import TravaLogger
from trava.metric import Metric
from trava.model_results import ModelResult
from trava.results_handler import ResultsHandler
from trava.trava_tracker import Tracker


class PandasDfHandler(ResultsHandler):
    """
    Returns all the metrics wrapped in a pandas dataframe.
    """

    def handle(self, results: List[ModelResult], logger: TravaLogger, tracker: Tracker):
        result = pd.DataFrame()

        for model_results in results:
            test_metrics = self._metrics_dict(metrics=model_results.test_metrics(provider=self))
            other_metrics = self._metrics_dict(metrics=model_results.other_metrics(provider=self))

            all_metrics = {**test_metrics, **other_metrics}
            all_metrics["model_id"] = model_results.model_id

            result = pd.concat([result, pd.DataFrame(all_metrics, index=[0])], axis=0).reset_index(drop=True)

        return result

    @staticmethod
    def _metrics_dict(metrics: List[Metric]):
        result = {}

        for metric in metrics:
            if metric.is_scalar:
                result[metric.name] = metric.value

        return result
