import os
import matplotlib.pyplot as plt
from tempfile import mkdtemp
from shutil import rmtree
from functools import wraps
from typing import List, Optional, Dict

from trava.logger import TravaLogger
from trava.metric import Metric
from trava.model_results import ModelResult
from trava.model_serializer import ModelSerializer
from trava.tracker import Tracker
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


class TravaTracker(Tracker, ResultsHandler):
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
        self._started_models: Dict[str, bool] = {}

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
        self._started_models[model_id] = True
        self._start_tracking(model_id=model_id, track_name=track_name, nested=nested)

    @track_if_enabled
    def end_tracking(self, model_id: str):
        self._end_tracking(model_id=model_id)
        del self._started_models[model_id]

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
    def track_metric(self, model_id: str, metric: Metric, train: bool, step=None):
        name = f"train_{metric.name}" if train else metric.name
        self.track_metric_value(model_id=model_id, name=name, value=metric.value, step=step)

    @track_if_enabled
    def track_metric_value(self, model_id: str, name: str, value, step=None):
        self._track_metric_value(model_id=model_id, name=name, value=value, step=step)

    @track_if_enabled
    def track_model_info(self, model_id: str, model):
        self._track_model_info(model_id=model_id, model=model)

    @track_if_enabled
    def track_tag(self, model_id: str, tag_key: str, tag_value):
        self._track_tag(model_id=model_id, tag_key=tag_key, tag_value=tag_value)

    @track_if_enabled
    def track_artifact(self, model_id: str, filepath):
        self._track_artifact(model_id=model_id, file_path=filepath)

    @track_if_enabled
    def track_model_artifact(self, model_id: str, model, serializer: ModelSerializer):
        tmpdir = mkdtemp()
        filepath = os.path.join(tmpdir, model_id + "_model")
        serializer.save(model=model, path=filepath)
        self.track_artifact(model_id=model_id, filepath=filepath)
        rmtree(tmpdir)

    @track_if_enabled
    def track(self, model_id: str, *args, **kwargs):
        """
        Just an abstract method to track anything you want
        """
        self._track(model_id=model_id, *args, **kwargs)  # type: ignore

    @track_if_enabled
    def track_plot(self, model_id: str, fig, filename: str):
        should_restart = self._started_models.get(model_id) is None
        # TODO: probably should be checked everywhere
        if should_restart:
            self.start_tracking(model_id=model_id)

        tmpdir = mkdtemp()
        try:
            filepath = os.path.join(tmpdir, filename + ".png")
            fig.savefig(filepath)
            self.track_artifact(model_id=model_id, filepath=filepath)
        finally:
            plt.close(fig)
            rmtree(tmpdir)

            if should_restart:
                self.end_tracking(model_id=model_id)

    def handle(self, results: List[ModelResult], logger: TravaLogger, tracker: Tracker):
        # here we can ignore provided tracker param since we expect that there is only one tracker
        # per trava, so self is equal to tracker. Looks bad yet not critical to refactor it right now.
        assert tracker == self, "Something's wrong, there can be only one tracker instance per Trava"

        for model_results in results:
            self.track_model_results(model_results=model_results)

    def track_model_results(self, model_results: ModelResult):
        self._track_metrics(
            model_id=model_results.model_id, metrics=model_results.train_metrics(provider=self), train=True
        )
        self._track_metrics(
            model_id=model_results.model_id, metrics=model_results.test_metrics(provider=self), train=False
        )
        self._track_metrics(
            model_id=model_results.model_id, metrics=model_results.other_metrics(provider=self), train=False
        )

    def _track_metrics(self, model_id: str, metrics: List[Metric], train: bool):
        for metric in metrics:
            if metric.is_scalar:
                self.track_metric(model_id=model_id, metric=metric, train=train)

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

    def _track_metric_value(self, model_id: str, name: str, value, step=None):
        pass

    def _track_model_info(self, model_id: str, model):
        pass

    def _track_tag(self, model_id: str, tag_key: str, tag_value):
        pass

    def _track_artifact(self, model_id: str, file_path):
        pass

    def _track(self, model_id: str, *args, **kwargs):
        pass

    ###
