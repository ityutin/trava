import pytest
import numpy as np
import random

from trava.metric import Metric
from trava.model_results import ModelResult


@pytest.fixture(scope="class")
def model_id():
    return "test_model_id"


def test_evaluators(mocker, model_id):
    evaluators = [mocker.Mock(), mocker.Mock()]
    model_result = ModelResult(model_id=model_id, evaluators=evaluators)
    assert model_result.evaluators == evaluators


def test_model_id(mocker, model_id):
    evaluators = [mocker.Mock()]
    model_result = ModelResult(model_id=model_id, evaluators=evaluators)
    assert model_result.model_id == model_id


def test_evaluators_required(mocker, model_id):
    evaluators = []
    with pytest.raises(Exception):
        ModelResult(model_id=model_id, evaluators=evaluators)
    with pytest.raises(Exception):
        ModelResult(model_id=model_id, evaluators=None)


@pytest.mark.parametrize("n_evaluators", [1, 3])
def test_is_one_fit_result(mocker, model_id, n_evaluators):
    evaluators = [mocker.Mock() for _ in range(n_evaluators)]
    model_result = ModelResult(model_id=model_id, evaluators=evaluators)

    if n_evaluators > 1:
        assert not model_result.is_one_fit_result
    else:
        assert model_result.is_one_fit_result


@pytest.mark.parametrize("n_evaluators", [1, 3])
def test_models(mocker, model_id, n_evaluators):
    raw_models = {}
    evaluators = []
    for idx in range(n_evaluators):
        evaluator = mocker.Mock()
        trava_model = mocker.Mock()
        raw_model = mocker.Mock()
        trava_model.raw_model = raw_model
        evaluator.model_id = model_id + "_" + str(idx)
        evaluator.trava_model = trava_model

        evaluators.append(evaluator)
        raw_models[evaluator.model_id] = raw_model

    model_result = ModelResult(model_id=model_id, evaluators=evaluators)

    assert model_result.raw_models == raw_models


@pytest.mark.parametrize("n_evaluators", [1, 3])
def test_train_metrics(mocker, n_evaluators):
    scorers_provider = mocker.Mock()

    main_model_id = "main_model_id"
    all_metrics = []
    evaluators = []
    for idx in range(n_evaluators):
        evaluator = mocker.Mock()
        if n_evaluators > 1:
            evaluator_model_id = "model_" + str(idx)
        else:
            evaluator_model_id = main_model_id

        evaluator.model_id = evaluator_model_id
        metrics = [
            Metric(name="test_metric_1", value=random.randint(0, 1000), model_id=evaluator_model_id),
            Metric(name="test_metric_2", value="blahblah", model_id=evaluator_model_id),
        ]
        all_metrics.append(metrics)

        def _wrapper(metrics):
            def _train_metrics(provider):
                if provider != scorers_provider:
                    raise ValueError
                return metrics.copy()

            return _train_metrics

        evaluator.train_metrics = mocker.MagicMock(side_effect=_wrapper(metrics=metrics))
        evaluators.append(evaluator)

    model_result = ModelResult(model_id=model_id, evaluators=evaluators)

    if n_evaluators > 1:
        filtered_all_metrics = []
        for i in range(len(all_metrics)):
            filtered_all_metrics.append([metric for metric in all_metrics[i] if metric.is_scalar])
        all_metrics = filtered_all_metrics

    for metric, eval_metrics in zip(model_result.train_metrics(provider=scorers_provider), zip(*all_metrics)):
        first_eval_metric = eval_metrics[0]

        assert metric.name == first_eval_metric.name
        if n_evaluators > 1:
            if not metric.is_scalar:
                continue
            eval_metrics_value = np.mean([metric.value for metric in eval_metrics])
        else:
            eval_metrics_value = metric.value

        assert metric.value == eval_metrics_value


@pytest.mark.parametrize("n_evaluators", [1, 3])
def test_test_metrics(mocker, n_evaluators):
    scorers_provider = mocker.Mock()

    main_model_id = "main_model_id"
    all_metrics = []
    evaluators = []
    for idx in range(n_evaluators):
        evaluator = mocker.Mock()
        if n_evaluators > 1:
            evaluator_model_id = "model_" + str(idx)
        else:
            evaluator_model_id = main_model_id

        evaluator.model_id = evaluator_model_id
        metrics = [
            Metric(name="test_metric_1", value=random.randint(0, 1000), model_id=evaluator_model_id),
            Metric(name="test_metric_2", value="blahblah", model_id=evaluator_model_id),
        ]
        all_metrics.append(metrics)

        def _wrapper(metrics):
            def _train_metrics(provider):
                if provider != scorers_provider:
                    raise ValueError
                return metrics.copy()

            return _train_metrics

        evaluator.test_metrics = mocker.MagicMock(side_effect=_wrapper(metrics=metrics))
        evaluators.append(evaluator)

    model_result = ModelResult(model_id=model_id, evaluators=evaluators)

    if n_evaluators > 1:
        filtered_all_metrics = []
        for i in range(len(all_metrics)):
            filtered_all_metrics.append([metric for metric in all_metrics[i] if metric.is_scalar])
        all_metrics = filtered_all_metrics

    for metric, eval_metrics in zip(model_result.test_metrics(provider=scorers_provider), zip(*all_metrics)):
        first_eval_metric = eval_metrics[0]

        assert metric.name == first_eval_metric.name
        if n_evaluators > 1:
            if not metric.is_scalar:
                continue
            eval_metrics_value = np.mean([metric.value for metric in eval_metrics])
        else:
            eval_metrics_value = metric.value

        assert metric.value == eval_metrics_value


@pytest.mark.parametrize("n_evaluators", [1, 3])
def test_any_metrics(mocker, n_evaluators):
    scorers_provider = mocker.Mock()

    main_model_id = "main_model_id"
    all_metrics = []
    evaluators = []
    for idx in range(n_evaluators):
        evaluator = mocker.Mock()
        if n_evaluators > 1:
            evaluator_model_id = "model_" + str(idx)
        else:
            evaluator_model_id = main_model_id

        evaluator.model_id = evaluator_model_id
        metrics = [
            Metric(name="test_metric_1", value=random.randint(0, 1000), model_id=evaluator_model_id),
            Metric(name="test_metric_2", value="blahblah", model_id=evaluator_model_id),
        ]
        all_metrics.append(metrics)

        def _wrapper(metrics):
            def _train_metrics(provider):
                if provider != scorers_provider:
                    raise ValueError
                return metrics.copy()

            return _train_metrics

        evaluator.other_metrics = mocker.MagicMock(side_effect=_wrapper(metrics=metrics))
        evaluators.append(evaluator)

    model_result = ModelResult(model_id=model_id, evaluators=evaluators)

    if n_evaluators > 1:
        filtered_all_metrics = []
        for i in range(len(all_metrics)):
            filtered_all_metrics.append([metric for metric in all_metrics[i] if metric.is_scalar])
        all_metrics = filtered_all_metrics

    for metric, eval_metrics in zip(model_result.other_metrics(provider=scorers_provider), zip(*all_metrics)):
        first_eval_metric = eval_metrics[0]

        assert metric.name == first_eval_metric.name
        if n_evaluators > 1:
            if not metric.is_scalar:
                continue
            eval_metrics_value = np.mean([metric.value for metric in eval_metrics])
        else:
            eval_metrics_value = metric.value

        assert metric.value == eval_metrics_value
