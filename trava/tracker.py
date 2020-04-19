from functools import wraps
from typing import List, Optional

from trava.logger import TravaLogger
from trava.metric import Metric
from trava.model_results import ModelResult
from trava.results_handler import ResultsHandler
from trava.scorer import Scorer


def track_if_enabled(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracker = args[0]
        if tracker.is_enabled:
            return func(*args, **kwargs)
        return None
    return wrapper


class TravaTracker(ResultsHandler):
    """
    Must be subclassed to have effect. Used for experiment tracking.
    Helps you to track important stuff about a model using the desired tracking framework.
    Made as a subclass of ResultsHandler for convenience integration with Trava internals.

    Init parameters
    ----------
    scorers: List[Scorer]
        Provided scorers will be used to get metrics for tracking.
    """
    def __init__(self, scorers: List[Scorer]):

        super().__init__(scorers=scorers)

        self._enabled = True

    def add_scorers(self, scorers: List[Scorer]):
        self._scorers += scorers

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @track_if_enabled
    def start_tracking(self, model_id: str, track_name: Optional[str] = None, nested: bool = False):
        self._start_tracking(model_id=model_id, track_name=track_name, nested=nested)

    @track_if_enabled
    def end_tracking(self, model_id: str):
        self._end_tracking(model_id=model_id)

    @track_if_enabled
    def track_set_tracking_group(self, group: str):
        """
        Use to group all the following tracking as a whole.
        """
        self._track_set_tracking_group(group=group)

    @track_if_enabled
    def track_model_description(self, model_id: str, description: str):
        self._track_model_description(model_id=model_id, description=description)

    @track_if_enabled
    def track_model_init_params(self, model_id: str, params: dict):
        self._track_model_init_params(model_id=model_id, params=params)

    @track_if_enabled
    def track_fit_params(self, model_id: str, params: dict):
        self._track_fit_params(model_id=model_id, params=params)

    @track_if_enabled
    def track_predict_params(self, model_id: str, params: dict):
        self._track_predict_params(model_id=model_id, params=params)

    @track_if_enabled
    def track_metric(self, model_id: str, metric: Metric):
        self._track_metric(model_id=model_id, metric=metric)

    @track_if_enabled
    def track_model_info(self, model_id: str, model):
        self._track_model_info(model_id=model_id, model=model)

    @track_if_enabled
    def track_model_artifact(self, model_id: str, model):
        self._track_model_artifact(model_id=model_id, model=model)

    @track_if_enabled
    def track(self, model_id: str, *args, **kwargs):
        """
        Just an abstract method to track anything you want
        """
        self._track(model_id=model_id, *args, **kwargs)

    def handle(self, results: List[ModelResult], logger: TravaLogger):
        for model_results in results:
            self.track_model_results(model_results=model_results)

    def track_model_results(self, model_results: ModelResult):
        self._track_metrics(model_id=model_results.model_id,
                            metrics=model_results.test_metrics(provider=self))
        self._track_metrics(model_id=model_results.model_id,
                            metrics=model_results.other_metrics(provider=self))

    def _track_metrics(self, model_id: str, metrics: List[Metric]):
        for metric in metrics:
            if metric.is_scalar:
                self.track_metric(model_id=model_id, metric=metric)

    # TO OVERRIDE

    def _start_tracking(self, model_id: str, track_name: Optional[str] = None, nested: bool = False):
        pass

    def _end_tracking(self, model_id: str):
        pass

    def _track_set_tracking_group(self, group: str):
        pass

    def _track_model_description(self, model_id: str, description: str):
        pass

    def _track_model_init_params(self, model_id: str, params: dict):
        pass

    def _track_fit_params(self, model_id: str, params: dict):
        pass

    def _track_predict_params(self, model_id: str, params: dict):
        pass

    def _track_metric(self, model_id: str, metric: Metric):
        pass

    def _track_model_info(self, model_id: str, model):
        pass

    def _track_model_artifact(self, model_id: str, model):
        pass

    def _track(self, model_id: str, *args, **kwargs):
        pass

    ###

