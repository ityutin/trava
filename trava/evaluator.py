from typing import List, Dict, Callable

from trava.metric import Metric
from trava.trava_model import TravaModel
from trava.scorers_provider import ScorersProvider
from trava.split.result import SplitResult


# noinspection PyPep8Naming
class Evaluator:
    """
    Calculates metrics for a provided model.

    Init parameters
    ----------
    fit_data: SplitResult
        Data that was used to train a model.
    raw_data: SplitResult
        Raw data we used to make fit_data through transformations.
    trava_model: TravaModel
        Encapsulates a real raw model and operations on it.
    """

    def __init__(self, fit_split_data: SplitResult, raw_split_data: SplitResult, trava_model: TravaModel):
        self._trava_model = trava_model
        self._fit_split_data = fit_split_data
        self._raw_split_data = raw_split_data

        self._train_metrics: Dict[str, List[Metric]] = {}
        self._test_metrics: Dict[str, List[Metric]] = {}
        self._other_metrics: Dict[str, List[Metric]] = {}

    @property
    def model_id(self) -> str:
        return self._trava_model.model_id

    @property
    def trava_model(self):
        return self._trava_model

    @property
    def fit_split_data(self) -> SplitResult:
        return self._fit_split_data

    @property
    def raw_split_data(self) -> SplitResult:
        return self._raw_split_data

    def evaluate(self, scorers_providers: List[ScorersProvider]) -> tuple:
        """
        Calculates metrics using the given scorers providers.
        Metrics will be saved in memory after evaluation.

        Parameters
        ----------
        scorers_providers: List[ScorersProvider]

        Returns
        -------
        Tuple of metrics separated by train, test and other metrics.
        """
        train_metrics = self._metrics_map(for_train=True, scorers_providers=scorers_providers, use_metric_scorers=True)

        test_metrics = self._metrics_map(for_train=False, scorers_providers=scorers_providers, use_metric_scorers=True)

        other_metrics = self._metrics_map(
            for_train=False, scorers_providers=scorers_providers, use_metric_scorers=False
        )

        self._merge_metrics(all_metrics=self._train_metrics, new_metrics=train_metrics)
        self._merge_metrics(all_metrics=self._test_metrics, new_metrics=test_metrics)
        self._merge_metrics(all_metrics=self._other_metrics, new_metrics=other_metrics)

        return train_metrics, test_metrics, other_metrics

    @staticmethod
    def _merge_metrics(all_metrics: dict, new_metrics: dict):
        for provider_id, metrics in new_metrics.items():
            existing_metrics = all_metrics.get(provider_id, [])
            existing_metrics += metrics
            all_metrics[provider_id] = existing_metrics

    def train_metrics(self, provider: ScorersProvider) -> List[Metric]:
        """
        Returns metrics calculated on train set
        """
        return self._train_metrics[provider.provider_id]

    def test_metrics(self, provider: ScorersProvider) -> List[Metric]:
        """
        Returns metrics calculated on test set
        """
        return self._test_metrics[provider.provider_id]

    def other_metrics(self, provider: ScorersProvider) -> List[Metric]:
        """
        Returns custom train/test-independent metrics
        """
        return self._other_metrics[provider.provider_id]

    @staticmethod
    def _metrics_scorers_getter() -> Callable[[ScorersProvider], list]:
        return lambda provider: provider.metric_scorers()

    @staticmethod
    def _other_scorers_getter() -> Callable[[ScorersProvider], list]:
        return lambda provider: provider.other_scorers()

    def _metrics_map(
        self, for_train: bool, scorers_providers: List[ScorersProvider], use_metric_scorers: bool
    ) -> Dict[str, List[Metric]]:
        """
        Calculates metrics using the saved scorers providers
        """
        result = {}
        metrics_cache: Dict[str, Metric] = {}

        if self._fit_split_data is None:
            X = None
            X_raw = None
            y = None
        elif for_train:
            X = self._fit_split_data.X_train
            X_raw = self._raw_split_data.X_train
            y = self._fit_split_data.y_train
        else:
            X = self._fit_split_data.X_test
            X_raw = self._raw_split_data.X_test
            y = self._fit_split_data.y_test

        for provider in scorers_providers:
            if use_metric_scorers:
                scorers = provider.metric_scorers()
            else:
                scorers = provider.other_scorers()

            provider_metrics = []
            for scorer in scorers:
                if metrics_cache.get(scorer.func_name):
                    provider_metrics.append(metrics_cache[scorer.func_name])
                    continue

                metric = self._get_metric(X=X, X_raw=X_raw, y=y, for_train=for_train, scorer=scorer)
                metrics_cache[scorer.func_name] = metric

                provider_metrics.append(metric)

            result[provider.provider_id] = provider_metrics

        return result

    def _get_metric(self, for_train: bool, X, X_raw, y, scorer) -> Metric:
        """
        Calculate metric using the given scorer
        """
        value = scorer(trava_model=self.trava_model, for_train=for_train, X=X, X_raw=X_raw, y=y)
        return Metric(name=scorer.func_name, value=value, model_id=self.model_id)

    def unload_data(self):
        """
        If storing the data is unwanted behavior, you can unload it.
        """
        del self._fit_split_data
        del self._raw_split_data

        self._fit_split_data = None
        self._raw_split_data = None
