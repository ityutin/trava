from abc import ABC
from typing import Optional, List, Any, Dict

from trava.logger import TravaLogger
from trava.model_results import ModelResult
from trava.results_handler import ResultsHandler
from trava.trava_tracker import TravaTracker
from trava.utils.model_params_filter import merge_given_params_with_default


class _TravaBase(ABC):
    """
    Trava stands for TrainValidation.

    It will help you to gather metrics from different models you try
    and present them in any way that is convenient for you.

    Init parameters
    ----------
    logger: TravaLogger
        Encapsulates logic of work with native python logger
    tracker: Tracker
        Is used to track your experiments
    results_handlers: list
        Each results handler provides logic how to present the metrics to a user.
    """

    def __init__(
        self,
        logger: Optional[TravaLogger] = None,
        tracker: Optional[TravaTracker] = None,
        results_handlers: List[ResultsHandler] = None,
    ):
        self._logger = logger or TravaLogger()
        self._tracker: TravaTracker = tracker or TravaTracker(scorers=[])
        self._results_handlers = results_handlers or []

        self._results: Dict[str, ModelResult] = {}

    @property
    def results(self) -> list:
        """
        Calls every results handler to present all the metrics
        for all the fits that were made so far.

        Returns
        -------
        List of outputs for every results handler that returns not None result
        """
        all_model_results = list(self._results.values())
        return self._results_for(results=all_model_results, results_handlers=self._results_handlers)

    def _results_for(self, results: List[ModelResult], results_handlers: List[ResultsHandler]) -> list:
        """
        Applies all the handlers and returns their result

        Returns
        -------
        List of outputs for every results handler that returns not None result
        """
        result = []
        for handler in results_handlers:
            handler_result = handler.handle(results=results, logger=self._logger, tracker=self._tracker)
            if handler_result is not None:
                result.append(handler_result)

        return result

    def raw_models_for(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get raw models associated with the given model_id

        Parameters
        ----------
        model_id: str
            Unique identifier of the requested model

        Returns
        -------
        List of trained models if exists
        """
        model_results = self._model_results(model_id=model_id)
        return model_results.raw_models

    def _model_results(self, model_id: str) -> ModelResult:
        result = self._results.get(model_id)
        assert result, "You haven't fit a model with such model_id: {}".format(model_id)
        return result

    def _assert_existing_model(self, model_id: str):
        assert self._results.get(model_id) is None, "you tried to save a model with the already existing id"

    def _create_raw_model(self, model_type, model_init_params: dict) -> tuple:
        """
        Creates model object with the provided params.

        model_type: type of sklearn-style model
            Type of model that supports fit, predict and predict_proba methods
        model_init_params: dict
            Parameters to use to initialize model_type
        :return:
        Initialized model + trackable init parameters, including default ones.
        See _trackable_init_params_types for supported param types.
        """
        raw_model = model_type(**model_init_params)
        all_init_params = merge_given_params_with_default(object_type=model_type, params=model_init_params)

        return raw_model, all_init_params
