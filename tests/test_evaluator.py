import pytest
import pandas as pd
import numpy as np
import random
from copy import copy

from typing import List

from trava.evaluator import Evaluator
from trava.split.result import SplitResult


@pytest.fixture(scope="class")
def model_id():
    return "test_model_id"


@pytest.fixture()
def model(mocker, model_id):
    result = mocker.Mock()
    result.model_id = model_id
    return result


@pytest.fixture(scope="class")
def X_y_train():
    X_train = pd.DataFrame({"f1": [1, 2, 3]})
    y_train = np.array([0, 0, 0])

    return X_train, y_train


@pytest.fixture(scope="class")
def X_y_test():
    X_test = pd.DataFrame({"f1": [4, 5, 6]})
    y_test = np.array([1, 1, 1])

    return X_test, y_test


@pytest.fixture(scope="class")
def split_result(X_y_train, X_y_test):
    X_train, y_train = X_y_train
    X_test, y_test = X_y_test

    return SplitResult(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def _evaluator(model, split_result, raw_split_result):
    evaluator = Evaluator(fit_split_data=split_result, raw_split_data=raw_split_result, trava_model=model)

    return evaluator


def test_properties(model, model_id, split_result):
    raw_split_result = copy(split_result)
    evaluator = _evaluator(model=model, split_result=split_result, raw_split_result=raw_split_result)

    assert evaluator.model_id == model_id
    assert evaluator.trava_model == model
    assert evaluator.fit_split_data == split_result
    assert evaluator.raw_split_data == raw_split_result


def test_evaluate_calls_scorer(mocker, model, model_id, split_result):
    raw_split_result = copy(split_result)
    evaluator = _evaluator(model=model, split_result=split_result, raw_split_result=raw_split_result)

    results_handler = mocker.MagicMock()
    scorer = mocker.Mock()
    results_handler.metric_scorers.return_value = [scorer]
    results_handler.other_scorers.return_value = [scorer]
    results_handler.provider_id = "TestResultsHandler"
    evaluator.evaluate(scorers_providers=[results_handler])

    results_handler.assert_has_calls(
        [
            mocker.call.metric_scorers(),  # train
            mocker.call.metric_scorers(),  # test
            mocker.call.other_scorers(),  # any
        ]
    )

    def _assert(called_args, for_train: bool):
        assert called_args["trava_model"] == model
        assert called_args["for_train"] == for_train

        X = split_result.X_train if for_train else split_result.X_test
        y = split_result.y_train if for_train else split_result.y_test
        X_raw = raw_split_result.X_train if for_train else raw_split_result.X_test

        assert called_args["X"].equals(X)
        assert called_args["X_raw"].equals(X_raw)
        assert np.array_equal(called_args["y"], y)

    _assert(called_args=scorer.call_args_list[0][1], for_train=True)
    _assert(called_args=scorer.call_args_list[1][1], for_train=False)
    _assert(called_args=scorer.call_args_list[2][1], for_train=False)


def test_empty_metrics_fail(mocker, model, model_id, split_result):
    raw_split_result = copy(split_result)
    evaluator = _evaluator(model=model, split_result=split_result, raw_split_result=raw_split_result)

    provider = mocker.Mock()
    provider.provider_id = "TestProvider"
    with pytest.raises(Exception):
        evaluator.train_metrics(provider=provider)
    with pytest.raises(Exception):
        evaluator.test_metrics(provider=provider)
    with pytest.raises(Exception):
        evaluator.other_metrics(provider=provider)


@pytest.mark.parametrize("n_scorers", [1, 3])
def test_all_metrics(mocker, model, model_id, split_result, n_scorers):
    raw_split_result = copy(split_result)
    evaluator = _evaluator(model=model, split_result=split_result, raw_split_result=raw_split_result)

    results_handler = mocker.MagicMock()

    scorers = []
    train_values = []
    test_and_any_values = []
    for i in range(n_scorers):
        scorer_train_value = random.randint(0, 1000)
        scorer_test_meta_values = [random.randint(0, 1000), random.randint(0, 1000)]
        test_any_metrics_values_copy = scorer_test_meta_values.copy()

        def _build_func(train_value: int, test_meta_values: List[int]):
            def _get_metric(trava_model, for_train: bool, X, X_raw, y):
                if for_train:
                    return train_value

                # test and any metrics both use test set
                result = test_meta_values[0]
                test_meta_values.remove(result)
                return result

            return _get_metric

        scorer = mocker.MagicMock(
            side_effect=_build_func(train_value=scorer_train_value, test_meta_values=scorer_test_meta_values)
        )
        train_values.append(scorer_train_value)
        test_and_any_values.append(test_any_metrics_values_copy)
        scorers.append(scorer)

    results_handler.metric_scorers.return_value = scorers
    results_handler.other_scorers.return_value = scorers
    results_handler.provider_id = "TestResultsHandler"
    evaluator.evaluate(scorers_providers=[results_handler])

    train_metrics = evaluator.train_metrics(provider=results_handler)
    test_metrics = evaluator.test_metrics(provider=results_handler)
    any_metrics = evaluator.other_metrics(provider=results_handler)

    for idx, metric in enumerate(train_metrics):
        assert metric.value == train_values[idx]

    for idx, metric in enumerate(test_metrics):
        assert metric.value == test_and_any_values[idx][0]

    for idx, metric in enumerate(any_metrics):
        assert metric.value == test_and_any_values[idx][1]


def test_unload_data():
    raw_split_result = copy(split_result)
    evaluator = _evaluator(model=model, split_result=split_result, raw_split_result=raw_split_result)
    evaluator.unload_data()

    assert not evaluator.fit_split_data
    assert not evaluator.raw_split_data


def test_metrics_caching(mocker, model, model_id, split_result):
    raw_split_result = copy(split_result)
    evaluator = _evaluator(model=model, split_result=split_result, raw_split_result=raw_split_result)

    scorer_name = "test_scorer_name"
    scorers_provider = mocker.MagicMock()

    scorer_1 = mocker.Mock()
    scorer_1.func_name = scorer_name

    scorer_2 = mocker.Mock()
    scorer_2.func_name = scorer_name

    scorer_3 = mocker.Mock()
    scorer_3.func_name = "another_scorer"

    scorers = [
        scorer_1,
        scorer_2,
        scorer_3,
    ]

    scorers_provider.metric_scorers.return_value = scorers
    scorers_provider.other_scorers.return_value = []

    evaluator.evaluate(scorers_providers=[scorers_provider])

    scorer_1.assert_called()
    scorer_2.assert_not_called()
    scorer_3.assert_called()
