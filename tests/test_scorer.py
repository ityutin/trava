from typing import Optional, Dict, Any

import pytest

from tests.objects_for_tests import TestScorer, TestAnyScorer
from trava.model_info import ModelInfo
from trava.scorer import Scorer


@pytest.fixture
def score_func(mocker, score_func_name):
    result = mocker.Mock()
    result.__name__ = score_func_name

    return result


@pytest.fixture(scope="class")
def score_func_name():
    return "test_score_func"


def test_is_any_scorer(mocker, score_func):
    scorer = TestScorer(score_func=score_func)
    assert not scorer.is_other_scorer
    any_scorer = TestAnyScorer(model_func=score_func)
    assert any_scorer.is_other_scorer


def test_func_name(mocker, score_func, score_func_name):
    scorer = TestScorer(score_func=score_func)
    assert scorer._func_name == score_func_name


def test_require_raw_model(mocker, score_func):
    scorer = TestScorer(score_func=score_func, requires_raw_model=True)
    trava_model = mocker.Mock()
    trava_model.raw_model.return_value = None

    X = mocker.MagicMock()
    X_raw = mocker.MagicMock()
    y = mocker.MagicMock()

    with pytest.raises(Exception):
        scorer(trava_model=trava_model, for_train=True, X=X, X_raw=X_raw, y=y)

    trava_model.raw_model.return_value = mocker.MagicMock()

    scorer(trava_model=trava_model, for_train=True, X=X, X_raw=X_raw, y=y)


def test_require_X_y(mocker, score_func):
    scorer = TestScorer(score_func=score_func, requires_X_y=True)
    trava_model = mocker.Mock()
    trava_model.raw_model.return_value = mocker.MagicMock()

    X = None
    X_raw = mocker.MagicMock()
    y = mocker.MagicMock()

    with pytest.raises(Exception):
        scorer(trava_model=trava_model, for_train=True, X=X, X_raw=X_raw, y=y)

    X = mocker.MagicMock()
    X_raw = None
    y = mocker.MagicMock()

    with pytest.raises(Exception):
        scorer(trava_model=trava_model, for_train=True, X=X, X_raw=X_raw, y=y)

    X = mocker.MagicMock()
    X_raw = mocker.MagicMock()
    y = None

    with pytest.raises(Exception):
        scorer(trava_model=trava_model, for_train=True, X=X, X_raw=X_raw, y=y)

    X = mocker.MagicMock()
    X_raw = mocker.MagicMock()
    y = mocker.MagicMock()
    scorer(trava_model=trava_model, for_train=True, X=X, X_raw=X_raw, y=y)


@pytest.mark.parametrize("for_train", [True, False])
def test_scorer_func_input(mocker, score_func, for_train):
    kwargs = {"a": 3, "b": "3fff"}
    scorer = TestScorer(score_func=score_func, **kwargs)
    internal_scorer = mocker.patch.object(scorer, "_scorer")

    trava_model = mocker.Mock()
    trava_model.raw_model.return_value = mocker.MagicMock()
    raw_model = mocker.Mock()
    trava_model.get_model.return_value = raw_model

    X = mocker.MagicMock()
    X_raw = mocker.MagicMock()
    y = mocker.MagicMock()

    scorer(trava_model=trava_model, for_train=for_train, X=X, X_raw=X_raw, y=y, **kwargs)

    internal_scorer.assert_called_once_with(
        model=raw_model, model_info=trava_model, for_train=for_train, X=X, X_raw=X_raw, y=y, **kwargs
    )


def test_make_scorer(mocker, score_func):
    class LocalTestScorer(Scorer):
        def __init__(
            self,
            score_func: callable,
            needs_proba=False,
            requires_raw_model=False,
            requires_X_y=False,
            name: Optional[str] = None,
            **metrics_kwargs
        ):
            self._make_scorer_score_func = None
            self._make_scorer_metrics_kwargs: Dict[str, Any] = {}
            super().__init__(score_func, needs_proba, requires_raw_model, requires_X_y, name, **metrics_kwargs)

        def _make_scorer(self, score_func: callable, **metrics_kwargs) -> callable:
            self._make_scorer_score_func = score_func
            self._make_scorer_metrics_kwargs = metrics_kwargs

            def scorer(model, model_info: ModelInfo, for_train: bool, X, X_raw, y):
                return 1

            return scorer

    kwargs = {"a": 3, "b": "3fff"}
    scorer = LocalTestScorer(score_func=score_func, **kwargs)

    assert scorer._make_scorer_score_func == score_func
    assert scorer._make_scorer_metrics_kwargs == kwargs
