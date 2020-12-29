from typing import List, Callable

from trava.logger import TravaLogger
from trava.model_serializer import ModelSerializer
from trava.scorer import Scorer, OtherScorer
from trava.model_info import ModelInfo
from trava.model_results import ModelResult
from trava.results_handler import ResultsHandler
from trava.trava_tracker import Tracker


class TestScorer(Scorer):
    def _make_scorer(self, score_func: Callable, **metrics_kwargs) -> Callable:
        def scorer(model, model_info: ModelInfo, for_train: bool, X, X_raw, y):
            metric_value = score_func(X["f1"].values)
            return metric_value

        return scorer


class TestAnyScorer(OtherScorer):
    def __init__(self, model_func: Callable):
        def score_func(model, model_info: ModelInfo, for_train: bool, X, X_raw, y):
            return model_func(model_info)

        super().__init__(score_func=score_func)


class TestResultsHandler(ResultsHandler):
    def handle(self, results: List[ModelResult], logger: TravaLogger, tracker: Tracker):
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


class TestModel:
    def __init__(self, required_param_1, required_param_2, def_1=1, def_2="tada", def_3=None, def_4=[], def_5=False):
        self._required_param_1 = required_param_1
        self._required_param_2 = required_param_2
        self._def_1 = def_1
        self._def_2 = def_2
        self._def_3 = def_3
        self._def_4 = def_4
        self._def_5 = def_5


class TestSerializer(ModelSerializer):
    def __init__(self):
        self._loaded = False
        self._saved = False

    def load(self, path: str):
        self._loaded = True

    def save(self, model, path: str):
        self._loaded = False
