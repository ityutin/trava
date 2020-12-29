import pytest
import numpy as np
from trava.ext.sklearn.scorers import SklearnScorer


@pytest.fixture()
def metrics_kwargs():
    return {
        "param1": 1,
        "param2": "2",
    }


@pytest.fixture()
def score_func(mocker):
    score_func = mocker.Mock()
    score_func.__name__ = "test_scorer_func"
    return score_func


@pytest.mark.parametrize("needs_proba", [True, False], ids=["proba", "no_proba"])
@pytest.mark.parametrize("for_train", [True, False], ids=["train", "no_train"])
@pytest.mark.parametrize("needs_threshold", [True, False], ids=["thresh", "no_thresh"])
@pytest.mark.parametrize("greater_is_better", [True, False], ids=["greater", "no_greater"])
def test_model_data_present(
    mocker, score_func, metrics_kwargs, needs_proba, for_train, needs_threshold, greater_is_better
):
    scorer = SklearnScorer(
        score_func=score_func,
        greater_is_better=greater_is_better,
        needs_proba=needs_proba,
        needs_threshold=needs_threshold,
        **metrics_kwargs
    )
    sklearn_scorer = mocker.Mock()

    sklearn_scorer_func = mocker.patch.object(scorer, "_get_sklearn_scorer")
    sklearn_scorer_func.return_value = sklearn_scorer

    scorer_sample_weight_func = mocker.patch.object(scorer, "_sample_weight")
    sample_weight = mocker.Mock()
    scorer_sample_weight_func.return_value = sample_weight

    trava_model = mocker.Mock()
    train_raw_model = mocker.Mock()
    test_raw_model = mocker.Mock()

    def get_model(for_train: bool):
        if for_train:
            raw_model = train_raw_model
        else:
            raw_model = test_raw_model
        return raw_model

    trava_model.get_model = mocker.MagicMock(side_effect=get_model)
    X = mocker.Mock()
    X_raw = mocker.Mock()
    y = mocker.Mock()
    scorer(trava_model=trava_model, for_train=for_train, X=X, X_raw=X_raw, y=y)

    sklearn_scorer_func.assert_called_with(score_func=score_func, **metrics_kwargs)
    scorer_sample_weight_func.assert_called_with(X=X, X_raw=X_raw)
    if for_train:
        raw_model = train_raw_model
    else:
        raw_model = test_raw_model
    sklearn_scorer.assert_called_with(raw_model, X, y, sample_weight=sample_weight)


@pytest.mark.parametrize("needs_proba", [True, False], ids=["proba", "no_proba"])
@pytest.mark.parametrize("for_train", [True, False], ids=["train", "no_train"])
@pytest.mark.parametrize("needs_threshold", [True, False], ids=["thresh", "no_thresh"])
@pytest.mark.parametrize("greater_is_better", [True, False], ids=["greater", "no_greater"])
@pytest.mark.parametrize("multiclass", [True, False], ids=["multiclass", "no_multiclass"])
def test_no_model_no_data_present(
    mocker, score_func, metrics_kwargs, needs_proba, needs_threshold, greater_is_better, multiclass, for_train
):
    scorer = SklearnScorer(
        score_func=score_func,
        greater_is_better=greater_is_better,
        needs_proba=needs_proba,
        needs_threshold=needs_threshold,
        **metrics_kwargs
    )
    trava_model = mocker.Mock()
    train_y_pred = mocker.MagicMock()
    test_y_pred = mocker.MagicMock()
    train_y_pred_proba = mocker.MagicMock()
    test_y_pred_proba = mocker.MagicMock()
    if multiclass:
        train_y = [1, 2, 3]
        test_y = [1, 2, 3]
    else:
        train_y = [1, 2]
        test_y = [1, 2]

        if needs_proba:
            train_y_pred_proba = np.array([[0.1, 0.9]])
            test_y_pred_proba = np.array([[0.1, 0.9]])

    def y_pred(for_train: bool):
        if for_train:
            return train_y_pred
        else:
            return test_y_pred

    def y_pred_proba(for_train: bool):
        if for_train:
            return train_y_pred_proba
        else:
            return test_y_pred_proba

    def y(for_train: bool):
        if for_train:
            return train_y
        else:
            return test_y

    trava_model.y_pred = mocker.MagicMock(side_effect=y_pred)
    trava_model.y_pred_proba = mocker.MagicMock(side_effect=y_pred_proba)
    trava_model.y = mocker.MagicMock(side_effect=y)

    scorer(trava_model=trava_model, for_train=for_train, X=None, X_raw=None, y=None)

    if needs_proba:
        if for_train:
            y_pred_for_scoring = train_y_pred_proba if multiclass else train_y_pred_proba[:, 1]
        else:
            y_pred_for_scoring = test_y_pred_proba if multiclass else train_y_pred_proba[:, 1]
    else:
        if for_train:
            y_pred_for_scoring = train_y_pred
        else:
            y_pred_for_scoring = test_y_pred

    if for_train:
        y_true_for_scoring = train_y
    else:
        y_true_for_scoring = test_y

    score_func.assert_called_with(y_true_for_scoring, y_pred_for_scoring, **metrics_kwargs)
