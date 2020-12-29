from typing import Optional, List

from trava.logger import TravaLogger
from trava.model_serializer import ModelSerializer
from trava.scorer import Scorer
from trava.trava_tracker import TravaTracker


class LogTracker(TravaTracker):
    def __init__(self, scorers: List[Scorer], logger: TravaLogger):
        super().__init__(scorers=scorers)

        self._logger = logger

    def _start_tracking(self, model_id: str, track_name: Optional[str] = None, nested: bool = False):
        self._logger.log("Start tracking: {}".format(model_id))

    def _end_tracking(self, model_id: str):
        self._logger.log("End tracking: {}".format(model_id))

    def _track_set_tracking_group(self, group: str):
        self._logger.log("Set tracking group: {}".format(group))

    def _track_model_description(self, model_id: str, description: str):
        self._logger.log("Track description: {} - {}".format(model_id, description))

    def _track_model_init_params(self, model_id: str, params: dict):
        self._logger.log("Track init params: {} - {}".format(model_id, params))

    def _track_fit_params(self, model_id: str, params: dict):
        self._logger.log("Track fit params: {} - {}".format(model_id, params))

    def _track_predict_params(self, model_id: str, params: dict):
        self._logger.log("Track predict params: {} - {}".format(model_id, params))

    def _track_metric_value(self, model_id: str, name: str, value, step=None):
        self._logger.log("Track metric: {} - {} : {}".format(model_id, name, value))

    def _track_model_info(self, model_id: str, model):
        self._logger.log("Track model: {} - {}".format(model_id, model))

    def _track_tag(self, model_id: str, tag_key: str, tag_value):
        self._logger.log("Track tag: {} - {}".format(model_id, tag_value))

    def track_model_artifact(self, model_id: str, model, serializer: ModelSerializer):
        self._logger.log("Track artifact: {} - {}".format(model_id, model))

    def _track(self, model_id: str, *args, **kwargs):
        self._logger.log("Track something: {} - {}".format(model_id, (args, kwargs)))
