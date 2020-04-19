from typing import List

from trava.logger import TravaLogger
from trava.scorer import Scorer, OtherScorer
from trava.model_info import ModelInfo
from trava.model_results import ModelResult
from trava.results_handler import ResultsHandler


class TestScorer(Scorer):
    def _make_scorer(self, score_func: callable, **metrics_kwargs) -> callable:
        def scorer(model, model_info: ModelInfo, for_train: bool, X, X_raw, y):
            metric_value = score_func(X['f1'].values)
            return metric_value

        return scorer


class TestAnyScorer(OtherScorer):
    def __init__(self, model_func: callable):
        def score_func(model, model_info: ModelInfo, for_train: bool, X, X_raw, y):
            return model_func(model_info)

        super().__init__(score_func=score_func)


class TestResultsHandler(ResultsHandler):
    def handle(self, results: List[ModelResult], logger: TravaLogger):
        all_train_metrics = []
        all_test_metrics = []
        all_any_metrics = []
        for model_result in results:
            all_train_metrics += model_result.train_metrics(provider=self)
            all_test_metrics += model_result.test_metrics(provider=self)
            all_any_metrics += model_result.other_metrics(provider=self)

        result = [
            [metric.value for metric in all_train_metrics],
            [metric.value for metric in all_test_metrics],
            [metric.value for metric in all_any_metrics] if all_any_metrics else [],
        ]
        return result
