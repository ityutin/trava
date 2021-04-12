import os
import typing as t

from trava.logger import TravaLogger
from trava.metric import Metric
from trava.model_results import ModelResult
from trava.results_handler import ResultsHandler
from trava.scorer import Scorer
from trava.trava_tracker import Tracker
from trava.utils.tmp import open_tmp_dir


class LoggerHandler(ResultsHandler):
    """
    Just logs all the requested metrics.
    """

    def __init__(self, scorers: t.List[Scorer]):
        super().__init__(scorers=scorers)

        self._run_logs: t.Optional[str] = None

    def handle(self, results: t.List[ModelResult], logger: TravaLogger, tracker: Tracker):
        logger.log("*** Logging: ***")

        for model_result in results:
            self._handle_model_results(model_result=model_result, logger=logger, tracker=tracker, track_only=False)

            if not model_result.is_one_fit_result:
                for evaluator in model_result.evaluators:
                    nested_model_result = ModelResult(model_id=evaluator.model_id, evaluators=[evaluator])
                    self._handle_model_results(
                        model_result=nested_model_result, logger=logger, tracker=tracker, track_only=True
                    )

        logger.log("*** END ***\n")

    def _handle_model_results(self, model_result: ModelResult, logger: TravaLogger, tracker: Tracker, track_only: bool):
        model_id = model_result.model_id
        self._start_run(model_id=model_id, tracker=tracker)
        self._handle_run(text="* Results for {} model *".format(model_id), logger=logger, track_only=track_only)

        if model_result.train_metrics(provider=self):
            self._handle_run(text="Train metrics:", logger=logger, track_only=track_only)
            self._handle_metrics(
                metrics=model_result.train_metrics(provider=self), logger=logger, track_only=track_only
            )

        self._handle_run(text="Test metrics:", logger=logger, track_only=track_only)
        self._handle_metrics(metrics=model_result.test_metrics(provider=self), logger=logger, track_only=track_only)

        self._handle_run(text="Other metrics:", logger=logger, track_only=track_only)
        self._handle_metrics(metrics=model_result.other_metrics(provider=self), logger=logger, track_only=track_only)
        self._track_run(model_id=model_id, tracker=tracker)
        self._stop_run(model_id=model_id, tracker=tracker)

    def _handle_metrics(self, metrics: t.List[Metric], logger: TravaLogger, track_only: bool):
        for metric in metrics:
            self._handle_run(text=str(metric.name) + ":", logger=logger, track_only=track_only)
            self._handle_run(text=str(metric.value), logger=logger, track_only=track_only)

        self._handle_run(text="\n", logger=logger, track_only=track_only)

    def _start_run(self, model_id: str, tracker: Tracker):
        tracker.start_tracking(model_id=model_id)
        self._run_logs = ""

    def _handle_run(self, text: str, logger: TravaLogger, track_only: bool):
        if self._run_logs is None:
            raise Exception("Something went wrong, you must start a run first")

        self._run_logs += f"\n{text}"

        if not track_only:
            logger.log(text)

    def _track_run(self, model_id: str, tracker: Tracker):
        with open_tmp_dir() as tmp_dir:
            logs_path = os.path.join(tmp_dir, "run_logs.txt")
            with open(logs_path, "w") as file:
                file.write(self._run_logs or "")

            tracker.track_artifact(model_id=model_id, filepath=logs_path)

    def _stop_run(self, model_id: str, tracker: Tracker):
        tracker.end_tracking(model_id=model_id)
        self._run_logs = None
