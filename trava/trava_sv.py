import copy
from typing import Optional, List

from trava.evaluator import Evaluator
from trava.fit_predictor import FitPredictor, FitPredictConfig
from trava.model_results import ModelResult
from trava.raw_dataset import RawDataset
from trava.results_handler import ResultsHandler
from trava.scorer import Scorer
from trava.split.result import SplitResult
from trava.tracker import TravaTracker
from trava.trava_base import _TravaBase


# noinspection PyPep8Naming
class TravaSV(_TravaBase):
    """
    Made for working with supervised problems.
    """
    def fit_predict(self,
                    model_id: str,
                    model_type: Optional[type],
                    description: Optional[str] = None,
                    model_init_params: Optional[dict] = None,
                    raw_split_data: Optional[SplitResult] = None,
                    raw_dataset: Optional[RawDataset] = None,
                    fit_predictor: FitPredictor = None,
                    fit_params: dict = None,
                    predict_params: dict = None,
                    keep_models_in_memory: bool = True,
                    keep_data_in_memory: bool = True,
                    serialize_model: bool = False):
        """
        Calls model's fit and predict with the data provided, calculates metrics and stores them.
        model_type and model_init_params were separated for the sake of easy parameters tracking.

        Parameters
        ----------
        model_id: str
            Model unique identifier, will be used for saving metrics etc
        model_type: type of sklearn-style model
            Type of model that supports fit, predict and predict_proba methods
            You should provide either model or model_type + model_init_params.
        description: str
            Describe the fit. It will be tracked if you set up tracker.
        model_init_params: dict
            Parameters to use to initialize model if model_type is present
        raw_split_data: SplitResult
             Already split train/test sets
        raw_dataset: RawDataset
            Raw data before split
        fit_predictor: FitPredictor
            Object responsible for performing fit and predict on a model
        fit_params: dict
            Custom params to use when calling model's fit method
        predict_params: dict
            Custom params to use when calling model's predict method
        keep_models_in_memory: bool
            Whether it's needed to store models in memory after the fit
        keep_data_in_memory: bool
            Whether it's needed to store the provided data in memory after the fit
        serialize_model: bool
            Whether it's needed to serialize fitted model

        Returns
        -------
        List of outputs for every results handler that returns not None result
        """
        assert raw_split_data or raw_dataset, "Provide either split result or raw_dataset"

        fit_params = fit_params or {}
        predict_params = predict_params or {}
        model_init_params = model_init_params or {}
        fit_predictor = fit_predictor or FitPredictor(logger=self._logger)

        model = model_type(**model_init_params)

        evaluators = self._fit_predict(model=model,
                                       model_id=model_id,
                                       model_init_params=model_init_params,
                                       fit_predictor=fit_predictor,
                                       fit_params=fit_params,
                                       predict_params=predict_params,
                                       serialize_model=serialize_model,
                                       split_result=raw_split_data,
                                       raw_dataset=raw_dataset,
                                       description=description)
        result = self._results_for_evaluators(evaluators=evaluators,
                                              save_model_results=True,
                                              main_model_id=model_id)

        if not keep_models_in_memory:
            [evaluator.trava_model.unload_model() for evaluator in evaluators]

        if not keep_data_in_memory:
            [evaluator.unload_data() for evaluator in evaluators]
        return result

    def evaluate(self,
                 model_id: str,
                 results_handlers: Optional[List[ResultsHandler]],
                 save_results: bool) -> list:
        """
        Make possible to evaluate an existing model using new results handlers.
        Made in case you forgot to pass all the handlers in init
        or if you've come up with the new ideas.

        Note: in some cases it's impossible, see requires_raw_model/requires_X_y parameters in Scorer class.

        Parameters
        ----------
        model_id: str
            Unique identifier of the requested model
        results_handlers: list
            Each results handler provides logic how to present the metrics to a user.
        save_results: bool
            Whether you want to add the new metrics to the rest of them or just want to take a look.

        Returns
        -------
        List of outputs for every results handler that returns not None result
        """
        temp_evaluators = []

        for evaluator in self._evaluators(model_id=model_id):
            if save_results:
                evaluator.evaluate(scorers_providers=results_handlers)

            # copy of evaluator just to get the results
            temp_evaluator = Evaluator(trava_model=evaluator.trava_model,
                                       fit_split_data=evaluator.fit_split_data,
                                       raw_split_data=evaluator.raw_split_data)
            temp_evaluator.evaluate(scorers_providers=results_handlers)
            temp_evaluators.append(temp_evaluator)

        result = self._results_for_evaluators(evaluators=temp_evaluators,
                                              results_handlers=results_handlers,
                                              main_model_id=model_id)

        return result

    def evaluate_track(self,
                       model_id: str,
                       scorers: List[Scorer]):
        """
        Similar to evaluate method, but this time it allows you to track new metrics

        Note: in some cases it's impossible, see requires_raw_model/requires_X_y parameters in Scorer class.

        Parameters
        ----------
        model_id: str
            Unique identifier of the requested model
        scorers: List[Scorer]
            List of scorers you want to use to track new metrics.
        """
        # we don't want to modify scorers in existing tracker
        tracker = self._tracker_copy()
        tracker.add_scorers(scorers=scorers)

        evaluators = self._evaluators(model_id=model_id)
        for evaluator in evaluators:
            evaluator.evaluate(scorers_providers=[tracker])

            self._track_evaluators(evaluators=[evaluator],
                                   model_id=evaluator.model_id,
                                   tracker=tracker)

        if len(evaluators) > 1:
            self._track_evaluators(evaluators=evaluators,
                                   model_id=model_id,
                                   tracker=tracker)

    def _tracker_copy(self) -> TravaTracker:
        """
        Note: in some cases it's impossible, see requires_raw_model/requires_X_y parameters in Scorer class.
        """
        return copy.deepcopy(self._tracker)

    @staticmethod
    def _track_evaluators(evaluators: List[Evaluator], model_id: str, tracker: TravaTracker):
        tracker.start_tracking(model_id=model_id)
        main_model_results = ModelResult(model_id=model_id, evaluators=evaluators)
        tracker.track_model_results(model_results=main_model_results)
        tracker.end_tracking(model_id=model_id)

    def _evaluators(self, model_id: str):
        model_results = self._model_results(model_id=model_id)
        return model_results.evaluators

    def detailed_results_for(self, model_id: str) -> Optional[list]:
        """
        Presents results for specific model.
        It may be a single fit as well as multiple CV fits.
        In case of multiple fits every fit will be shown separately.

        Parameters
        ----------
        model_id: str
            Unique identifier of the requested model

        Returns
        -------
        List of outputs for every results handler that returns not None result
        """
        maybe_model_results = self._results.get(model_id)

        if not maybe_model_results:
            raise AttributeError('Model with id {} is not found.'.format(model_id))

        return self._results_for_evaluators(evaluators=maybe_model_results.evaluators)

    def _results_for_evaluators(self,
                                evaluators: List[Evaluator],
                                save_model_results: bool = False,
                                results_handlers: List[ResultsHandler] = None,
                                main_model_id: Optional[str] = None) -> list:
        # if main_model_id is provided it means that all evaluators
        # are related to the one run and must be averaged
        is_agg_result = len(evaluators) > 1 and main_model_id is not None

        evaluators_results = []
        for evaluator in evaluators:
            model_results = ModelResult(model_id=evaluator.model_id,
                                        evaluators=[evaluator])
            evaluators_results.append(model_results)

            if not is_agg_result and save_model_results:
                self._results[model_results.model_id] = model_results

        if is_agg_result:
            main_model_result = ModelResult(model_id=main_model_id,
                                            evaluators=evaluators)
            evaluators_results = [main_model_result]
            if save_model_results:
                self._results[main_model_id] = main_model_result

        all_results_handlers = results_handlers or self._results_handlers
        return self._results_for(results=evaluators_results, results_handlers=all_results_handlers)

    def _fit_predict(self,
                     model_id: str,
                     model,
                     model_init_params: Optional[dict],
                     fit_predictor: FitPredictor,
                     fit_params: dict,
                     predict_params: dict,
                     serialize_model: bool,
                     split_result: Optional[SplitResult] = None,
                     raw_dataset: Optional[RawDataset] = None,
                     description: Optional[str] = None):
        all_results_handlers = self._results_handlers + [self._tracker]

        config = FitPredictConfig(raw_split_data=split_result,
                                  raw_dataset=raw_dataset,
                                  raw_model=model,
                                  model_init_params=model_init_params,
                                  model_id=model_id,
                                  scorers_providers=all_results_handlers,
                                  serialize_model=serialize_model,
                                  fit_params=fit_params,
                                  predict_params=predict_params,
                                  description=description)

        evaluators = fit_predictor.fit_predict(config=config, tracker=self._tracker)

        return evaluators