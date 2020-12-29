from typing import List, Optional, Tuple, Iterator
import numpy as np

from trava.evaluator import Evaluator
from trava.metric import Metric
from trava.scorers_provider import ScorersProvider


class ModelResult:
    """
    Helps to get metrics for a model based on its evaluators.
    There can be more than one fit per model when you
    train it on different sets of data ( e.g. using CV ).
    Model results contains an evaluator for each fit
    and helps to aggregate metrics from multiple fits.

    Init parameters
    ----------
    model_id: str
        Model unique identifier, will be used for saving metrics etc
    evaluators:
        All the fits that were made with the same model.
    """

    def __init__(self, model_id: str, evaluators: List[Evaluator]):
        assert len(evaluators) > 0, "At least one evaluator is required"

        self._model_id = model_id
        self._evaluators = evaluators

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_one_fit_result(self) -> bool:
        return len(self._evaluators) == 1

    @property
    def raw_models(self) -> Optional[dict]:
        """
        Gets the raw models

        Returns
        -------
        List of trained models
        """
        return dict([(evaluator.model_id, evaluator.trava_model.raw_model) for evaluator in self._evaluators])

    @property
    def evaluators(self) -> List[Evaluator]:
        """
        Gets all the evaluators provided

        Returns
        -------
        List of the trained models
        """
        return self._evaluators

    def train_metrics(self, provider: ScorersProvider) -> List[Metric]:
        """
        Parameters
        ----------
        provider: ScorersProvider
            Provides scorers to calculate metrics

        Returns
        -------
        All metrics calculated on the train set.
        """
        return self._metrics(provider=provider, on="train")

    def test_metrics(self, provider: ScorersProvider) -> List[Metric]:
        """
        Parameters
        ----------
        provider: ScorersProvider
            Provides scorers to calculate metrics

        Returns
        -------
        All metrics calculated on the test set.
        """
        return self._metrics(provider=provider, on="test")

    def other_metrics(self, provider: ScorersProvider) -> List[Metric]:
        """
        Parameters
        ----------
        provider: ScorersProvider
            Provides scorers to calculate metrics

        Returns
        -------
        All other metrics calculated on the test set.
        """
        return self._metrics(provider=provider, on="any")

    def _metrics(self, provider: ScorersProvider, on: str) -> List[Metric]:
        all_metrics = [
            self._get_metrics(provider=provider, evaluator=evaluator, on=on) for evaluator in self._evaluators
        ]
        return self._calculate_metrics(metrics=all_metrics)

    def _calculate_metrics(self, metrics: List[List[Metric]]):
        """
        If there is only one evaluator, returns all metrics as it is.
        If there are many evaluators, than it averages out all scalar metrics
        and ignores all the complex metrics.
        """
        first_evaluator_metrics = metrics[0]

        if first_evaluator_metrics is None:
            return None

        if len(self._evaluators) == 1:
            return first_evaluator_metrics

        result = []

        zipped_all_metrics: Iterator[Tuple[Metric, ...]] = zip(*metrics)
        for zipped_metrics in zipped_all_metrics:
            first_metric: Metric = zipped_metrics[0]

            if not first_metric.is_scalar:
                continue

            mean_value = np.mean([metric.value for metric in zipped_metrics])

            mean_metric = type(first_metric)(name=first_metric.name, value=mean_value, model_id=self._model_id)

            result.append(mean_metric)

        return result

    @staticmethod
    def _get_metrics(provider: ScorersProvider, evaluator: Evaluator, on: str) -> List[Metric]:
        if on == "train":
            return evaluator.train_metrics(provider=provider)
        elif on == "test":
            return evaluator.test_metrics(provider=provider)
        elif on == "any":
            return evaluator.other_metrics(provider=provider)
        else:
            raise ValueError("Unknown source of metrics {}".format(on))
