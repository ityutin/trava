from trava.model_info import ModelInfo
from trava.scorer import OtherScorer


class FitTimeScorer(OtherScorer):
    """
    Returns the time was required to train a model.
    """

    def __init__(self):
        def fit_time(model, model_info: ModelInfo, for_train: bool, X, X_raw, y):
            return model_info.fit_time

        super().__init__(score_func=fit_time)


class PredictTimeScorer(OtherScorer):
    """
    Returns the time was required to predict an output using a model.
    """

    def __init__(self):
        def predict_time(model, model_info: ModelInfo, for_train: bool, X, X_raw, y):
            return model_info.predict_time

        super().__init__(score_func=predict_time)
