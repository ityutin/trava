from typing import Optional

from trava.metric import Metric
from trava.model_results import ModelResult
from trava.model_serializer import ModelSerializer


class Tracker:
    """
    An interface for a tracker object.
    """

    def start_tracking(self, model_id: str, track_name: Optional[str] = None, nested: bool = False):
        pass

    def end_tracking(self, model_id: str):
        pass

    def track_set_tracking_group(self, group: str):
        pass

    def track_model_description(self, model_id: str, description: str):
        pass

    def track_model_init_params(self, model_id: str, params: dict):
        pass

    def track_fit_params(self, model_id: str, params: dict):
        pass

    def track_predict_params(self, model_id: str, params: dict):
        pass

    def track_model_results(self, model_results: ModelResult):
        pass

    def track_metric(self, model_id: str, metric: Metric, train: bool, step=None):
        pass

    def track_metric_value(self, model_id: str, name: str, value, step=None):
        pass

    def track_model_info(self, model_id: str, model):
        pass

    def track_tag(self, model_id: str, tag_key: str, tag_value):
        pass

    def track_artifact(self, model_id: str, filepath):
        pass

    def track_model_artifact(self, model_id: str, model, serializer: ModelSerializer):
        pass

    def track(self, model_id: str, *args, **kwargs):
        pass

    def track_plot(self, model_id: str, fig, filename: str):
        pass
