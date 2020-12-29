from typing import List

from trava.logger import TravaLogger
from trava.metric import Metric
from trava.model_results import ModelResult
from trava.results_handler import ResultsHandler
from trava.trava_tracker import Tracker


class LoggerHandler(ResultsHandler):
    """
    Just logs all the requested metrics.
    """

    def handle(self, results: List[ModelResult], logger: TravaLogger, tracker: Tracker):
        logger.log("*** Logging: ***")
        for model_results in results:
            logger.log("* Results for {} model *".format(model_results.model_id))

            if model_results.train_metrics(provider=self):
                logger.log("Train metrics:")
                LoggerHandler._log_metrics(metrics=model_results.train_metrics(provider=self), logger=logger)

            logger.log("Test metrics:")
            LoggerHandler._log_metrics(metrics=model_results.test_metrics(provider=self), logger=logger)

            logger.log("Other metrics:")
            LoggerHandler._log_metrics(metrics=model_results.other_metrics(provider=self), logger=logger)
        logger.log("*** END ***\n")

    @staticmethod
    def _log_metrics(metrics: List[Metric], logger: TravaLogger):
        for metric in metrics:
            logger.log(str(metric.name) + ":")
            logger.log(str(metric.value))

        logger.log("\n")
