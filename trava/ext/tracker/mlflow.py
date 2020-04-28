from typing import List, Optional

import mlflow
import mlflow.sklearn

from trava.metric import Metric
from trava.model_serializer import ModelSerializer
from trava.scorer import Scorer
from trava.trava_tracker import TravaTracker


class _RunInfo:
    def __init__(self, run_id: str, run_name: str):
        self.run_id = run_id
        self.run_name = run_name


class MLFlowTracker(TravaTracker):
    def __init__(self, scorers: List[Scorer]):
        super().__init__(scorers=scorers)

        self._model_id_run_info_map = {}

    def __getattr__(self, called_method):
        result = getattr(mlflow, called_method)
        if callable(result):
            def wrapper(*args, **kwargs):
                return getattr(mlflow, called_method)(*args, **kwargs)
            return wrapper
        raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, called_method))

    def _start_tracking(self, model_id: str, track_name: Optional[str] = None, nested: bool = False):
        if self._try_start_existing_run(model_id=model_id, nested=nested):
            return

        run_name = track_name or model_id
        mlflow.start_run(run_name=run_name, nested=nested)
        self._save_run_id(run_name=run_name, model_id=model_id)

    def _try_start_existing_run(self, model_id: str, nested: bool):
        maybe_run_info = self._model_id_run_info_map.get(model_id)

        if not maybe_run_info:
            return False

        mlflow.start_run(run_id=maybe_run_info.run_id, nested=nested)

        return True

    def _end_tracking(self, model_id: str):
        mlflow.end_run()

    def _track_set_tracking_group(self, group: str):
        mlflow.set_experiment(group)

    def _track_model_description(self, model_id: str, description: str):
        self.track_tag(model_id=model_id, tag_key='description', tag_value=description)

    def _track_model_init_params(self, model_id: str, params: dict):
        mlflow.log_params(params)

    def _track_fit_params(self, model_id: str, params: dict):
        super()._track_fit_params(model_id, params)

    def _track_predict_params(self, model_id: str, params: dict):
        super()._track_predict_params(model_id, params)

    def _track_metric_value(self, model_id: str, name: str, value, step=None):
        mlflow.log_metric(name, value, step=step)

    def _track_model_info(self, model_id: str, model):
        self.track_tag(model_id=model_id, tag_key='model_type', tag_value=type(model).__name__)

    def _track_tag(self, model_id: str, tag_key: str, tag_value):
        mlflow.set_tag(tag_key, tag_value)

    def _track_artifact(self, model_id: str, file_path):
        mlflow.log_artifact(local_path=file_path)

    def _save_run_id(self, run_name: str, model_id: str):
        run_id = mlflow.active_run().info.run_id
        self._model_id_run_info_map[model_id] = _RunInfo(run_id=run_id, run_name=run_name)
