from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Tuple, Optional

from trava.evaluator import Evaluator
from trava.logger import TravaLogger
from trava.model_serializer import ModelSerializer
from trava.trava_model import TravaModel
from trava.model_results import ModelResult
from trava.scorers_provider import ScorersProvider
from trava.split.result import SplitResult
from trava.tracker import Tracker


class FitPredictConfig:
    """
    Data class, containing all the
    information needed for fit&predict process.
    """

    def __init__(
        self,
        raw_model,
        model_init_params: Optional[dict],
        model_id: str,
        scorers_providers: List[ScorersProvider],
        serializer: Optional[ModelSerializer],
        raw_split_data: Optional[SplitResult] = None,
        fit_params: dict = None,
        predict_params: dict = None,
        description: Optional[str] = None,
    ):
        self.raw_model = raw_model
        self.model_init_params = model_init_params or {}
        self.model_id = model_id
        self.scorers_providers = scorers_providers
        self.serializer = serializer
        self.raw_split_data = raw_split_data
        self.fit_params = fit_params or {}
        self.predict_params = predict_params or {}
        self.description = description

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FitPredictConfig):
            return False

        model_check = self.raw_model == other.raw_model
        init_params_check = self.model_init_params == other.model_init_params
        model_id_check = self.model_id == other.model_id
        scorers_providers_check = self.scorers_providers == other.scorers_providers
        serializer_check = self.serializer == other.serializer
        raw_split_data_check = self.raw_split_data == other.raw_split_data
        fit_params_check = self.fit_params == other.fit_params
        predict_params_check = self.predict_params == other.predict_params
        description_check = self.description == other.description

        result = (
            model_check
            and init_params_check
            and model_id_check
            and scorers_providers_check
            and serializer_check
            and raw_split_data_check
            and fit_params_check
            and predict_params_check
            and description_check
        )  # noqa: E127

        return result


class RawModelUpdateStep(ABC):
    """
    Using this step you can update properties of a raw model object.
    """

    @abstractmethod
    def update_model(self, raw_model, config: FitPredictConfig):
        """
        Updates the given raw model.

        Parameters
        ----------
        raw_model: sklearn-style model
            Model that supports fit, predict and predict_proba methods.
        config: FitPredictConfig
            Initial config object

        Returns
        -------
        Updated raw model
        """
        pass


class FitPredictConfigUpdateStep:
    """
    Using this step you can update parameters containing
    in FitPredictConfig to then use them in fit&predict process.
    All methods will be called in the presented order.
    """

    def fit_split_data(self, raw_split_data: SplitResult, config: FitPredictConfig, tracker: Tracker) -> SplitResult:
        """
        Prepares split data to fit a model with.

        Parameters
        ----------
        raw_split_data: SplitResult
            Current split_data state before applying any steps or just after another step
        config: FitPredictConfig:
            Model's config object
        tracker: Tracker
            Tracker object

        Returns
        -------
        Updated split data
        """
        return raw_split_data

    def fit_params(
        self, fit_params: dict, fit_split_data: SplitResult, config: FitPredictConfig, tracker: Tracker
    ) -> dict:
        """
        Prepares fit parameters to be used with a model.

        Parameters
        ----------
        fit_params: dict
            Current fit_params state before applying any steps or just after another step.
        fit_split_data: SplitResult
            Final split data that will be used to fit the model.
        config: FitPredictConfig
            Model's config object
        tracker: Tracker
            Tracker object

        Returns
        -------
        Updated fit parameters
        """
        return fit_params

    def predict_params(
        self, predict_params: dict, fit_split_data: SplitResult, config: FitPredictConfig, tracker: Tracker
    ) -> dict:
        """
        Prepares predict parameters to be used with a model.

        Parameters
        ----------
        predict_params: dict
            Current predict_params state before applying any steps or just after another step.
        fit_split_data: SplitResult
            Final split data that will be used to fit the model.
        config: FitPredictConfig
            Model's config object
        tracker: Tracker
            Tracker object

        Returns
        -------
        Updated predict parameters
        """
        return predict_params


class FinalHandlerStep(ABC):
    """
    Using this step you can do whatever you want after fit&predict process
    using a resulting TravaModel object.
    """

    @abstractmethod
    def handle(self, trava_model: TravaModel, config: FitPredictConfig, tracker: Tracker):
        pass


class FitPredictorSteps:
    def __init__(
        self,
        raw_model_steps: List[RawModelUpdateStep] = None,
        config_steps: List[FitPredictConfigUpdateStep] = None,
        final_steps: List[FinalHandlerStep] = None,
    ):
        self.raw_model_steps = raw_model_steps or []
        self.config_steps = config_steps or []
        self.final_steps = final_steps or []

    def __add__(self, other):
        raw_model_steps = self.raw_model_steps + other.raw_model_steps
        config_steps = self.config_steps + other.config_steps
        final_steps = self.final_steps + other.final_steps

        result = FitPredictorSteps(raw_model_steps=raw_model_steps, config_steps=config_steps, final_steps=final_steps)

        return result


# noinspection PyMethodMayBeStatic
class FitPredictor(ABC):
    """
    Responsible for fit&predict&evaluation processes.

    Init parameters
    ----------
    raw_model_update_steps: List[RawModelUpdateStep]:
        Provide subclasses of RawModelUpdateStep to update your raw model
    config_update_steps: List[FitPredictConfigUpdateStep]
        Provide subclasses of FitPredictConfigUpdateStep to update different parameters
        in fit predict config
    final_steps: List[FinalHandlerStep]:
        Provide subclasses of FinalHandlerStep to perform any action using a resulting TravaModel
    logger: TravaLogger
        Logger object to inform you about the flow.
    """

    def __init__(self, steps: FitPredictorSteps = None, logger: TravaLogger = None):
        self._steps = steps or FitPredictorSteps()

        self._logger = logger or TravaLogger()

    def fit_predict(self, config: FitPredictConfig, tracker: Tracker) -> List[Evaluator]:
        """
        Prepares the data, performs one or many fit&predict, evaluates the results, tracks everything along the way.

        Parameters
        ----------
        config: FitPredictConfig
            Initial config object
        tracker: TravaTracker
            Helps with tracking things

        Returns
        -------
        List of Evaluator objects, one evaluator per fit.
        """
        raw_model = config.raw_model

        for updater in self._steps.raw_model_steps:
            raw_model = updater.update_model(raw_model=raw_model, config=config)

        # here will be stored evaluators for models, one for each model
        evaluators = []

        # each fit predictor can decide how many fits should we perform.
        # so subclasses prepare TravaModel and FitPredictorConfig for each fit.
        # yet using the same raw_model
        models_configs = self._models_configs(raw_model=raw_model, config=config)
        multiple_models_tracking = len(models_configs) > 1

        if multiple_models_tracking:
            # if we are going to train the raw model multiple times, we should group those fits
            # So we first track the "root" model, it will contain averaged results between all fits.
            self._start_tracking(
                config=config, raw_model=raw_model, model_id=config.model_id, tracker=tracker, nested=False
            )

        for trava_model, model_config in models_configs:
            # now fitting the model as many times as subclass wants
            model_id = trava_model.model_id
            split_result_fit = model_config.raw_split_data
            assert split_result_fit

            # here we start tracking one of the real fits.
            self._start_tracking(
                config=config, raw_model=raw_model, model_id=model_id, tracker=tracker, nested=multiple_models_tracking
            )

            fit_params = model_config.fit_params
            predict_params = model_config.predict_params
            # preparing data and parameters for fit&predict
            for config_updater in self._steps.config_steps:
                split_result_fit = config_updater.fit_split_data(
                    raw_split_data=split_result_fit, config=model_config, tracker=tracker
                )

                fit_params = config_updater.fit_params(
                    fit_params=fit_params, fit_split_data=split_result_fit, config=model_config, tracker=tracker
                )
                predict_params = config_updater.predict_params(
                    predict_params=predict_params, fit_split_data=split_result_fit, config=model_config, tracker=tracker
                )

            tracker.track_fit_params(model_id=trava_model.model_id, params=fit_params)
            tracker.track_predict_params(model_id=trava_model.model_id, params=predict_params)

            self._log_fit_start(trava_model=trava_model, split_result=split_result_fit)

            self._fit(
                trava_model=trava_model,
                X=split_result_fit.X_train,
                y=split_result_fit.y_train,
                fit_params=fit_params,
                predict_params=predict_params,
            )

            self._predict(trava_model=trava_model, X=split_result_fit.X_test, y=split_result_fit.y_test)

            # performing custom final steps provided in init
            for step in self._steps.final_steps:
                step.handle(trava_model=trava_model, config=model_config, tracker=tracker)

            self._logger.log(msg=f"Model evaluation {model_id}")

            # calculating metrics for the model
            evaluator = self._evaluator(model_config=model_config, split_result=split_result_fit, model=trava_model)
            evaluator.evaluate(scorers_providers=model_config.scorers_providers)

            self._track_metrics(model_id=model_id, evaluators=[evaluator], tracker=tracker)

            if model_config.serializer:
                self._logger.log(msg=f"Model serialization {model_id}")
                tracker.track_model_artifact(
                    model_id=model_id, model=trava_model.raw_model, serializer=model_config.serializer
                )

            evaluators.append(evaluator)
            tracker.end_tracking(model_id=model_id)

        if multiple_models_tracking:
            # just as we started tracking the "root" model in the beginning, here we have to end tracking
            self._track_metrics(model_id=config.model_id, evaluators=evaluators, tracker=tracker)
            tracker.end_tracking(model_id=config.model_id)

        return evaluators

    def _log_fit_start(self, trava_model: TravaModel, split_result: SplitResult):
        msg = f"Model fit start {trava_model.model_id}:\n"
        msg += f"train set shape ({split_result.X_train.shape})\n"
        if trava_model.is_classification_model:
            msg += f"target distribution ({dict(Counter(split_result.y_train))})\n"
        msg += f"test set shape ({split_result.X_test.shape})\n"
        if trava_model.is_classification_model:
            msg += f"target distribution ({dict(Counter(split_result.y_test))})\n"

        if split_result.X_valid is not None:
            msg += f"validation set shape ({split_result.X_valid.shape})\n"
            if trava_model.is_classification_model:
                msg += f"target distribution ({dict(Counter(split_result.y_valid))})\n"
        self._logger.log(msg=msg)

    def _evaluator(self, model_config: FitPredictConfig, split_result: SplitResult, model: TravaModel):
        """
        The method was extracted for unit-test purposes. :]
        """
        evaluator = Evaluator(
            trava_model=model, fit_split_data=split_result, raw_split_data=model_config.raw_split_data
        )
        return evaluator

    def _start_tracking(self, config: FitPredictConfig, raw_model, model_id: str, tracker: Tracker, nested: bool):
        tracker.start_tracking(model_id=model_id, nested=nested)
        self._track_model(
            model=raw_model,
            model_init_params=config.model_init_params,
            model_id=model_id,
            tracker=tracker,
            description=config.description,
        )

    def _track_metrics(self, model_id: str, evaluators: List[Evaluator], tracker: Tracker):
        model_results = ModelResult(model_id=model_id, evaluators=evaluators)
        tracker.track_model_results(model_results=model_results)

    def _track_model(self, model, model_init_params: dict, model_id: str, description: Optional[str], tracker: Tracker):
        tracker.track_model_init_params(model_id=model_id, params=model_init_params)
        tracker.track_model_info(model_id=model_id, model=model)
        if description:
            tracker.track_model_description(model_id=model_id, description=description)

    # TO OVERRIDE IF NEEDED

    def _models_configs(self, raw_model, config: FitPredictConfig) -> List[Tuple[TravaModel, FitPredictConfig]]:
        """
        If you want to run multiple fits on the same raw_model,
        just configure TravaModels and configs for them in your subclass.
        """
        return [(TravaModel(raw_model=raw_model, model_id=config.model_id), config)]

    def _fit(self, trava_model: TravaModel, X, y, fit_params: dict, predict_params: dict):
        """
        If you want to control the fit process
        """
        trava_model.fit(X=X, y=y, fit_params=fit_params, predict_params=predict_params)

    def _predict(self, trava_model: TravaModel, X, y):
        """
        If you want to control the predict process
        """
        trava_model.predict(X=X, y=y)
