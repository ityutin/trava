import pytest
import numpy as np
import pandas as pd

from tests.objects_for_tests import TestResultsHandler, TestScorer, TestAnyScorer, TestModel
from trava.evaluator import Evaluator
from trava.fit_predictor import FitPredictConfig
from trava.metric import Metric
from trava.raw_dataset import RawDataset
from trava.split.result import SplitResult
from trava.trava_tracker import TravaTracker
from trava.trava_sv import TravaSV


@pytest.fixture(scope="class")
def trava_tracker():
    return TravaTracker(scorers=[])


@pytest.fixture(scope="class")
def test_result_handlers():
    return [TestResultsHandler(scorers=[TestScorer(score_func=np.sum)])]


@pytest.fixture
def trava(test_result_handlers, trava_tracker):
    return TravaSV(results_handlers=test_result_handlers, tracker=trava_tracker)


@pytest.fixture(scope="class")
def split_result(X_y_train, X_y_test):
    X_train, y_train = X_y_train
    X_test, y_test = X_y_test

    return SplitResult(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


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
def raw_dataset():
    result = RawDataset(df=pd.DataFrame({"df": [1, 2, 3], "test_target": [0, 0, 1]}), target_col_name="test_target")
    return result


@pytest.fixture(scope="class")
def model_init_params():
    model_init_params = {
        "test1": 1,
        "test2": "second",
    }
    return model_init_params


@pytest.fixture(scope="class")
def model_id():
    return "test_model_id"


@pytest.fixture(scope="class")
def main_model_id(model_id):
    return "main_" + model_id


@pytest.fixture()
def raw_model(mocker):
    model_mock = mocker.Mock()
    return model_mock


@pytest.fixture()
def trava_model(mocker, raw_model):
    model_mock = mocker.Mock()
    trava_model.raw_model = raw_model
    return model_mock


@pytest.fixture
def model_type(mocker, raw_model):
    model_type = mocker.stub()
    model_type.return_value = raw_model

    return model_type


def _fit_predictor(mocker, evaluators):
    fit_predictor = mocker.Mock()
    fit_predictor.fit_predict.return_value = evaluators
    return fit_predictor


def _mock_evaluator(
    mocker, model_id, train_values=None, test_values=None, any_values=None, trava_model=None, split_result=None
):
    evaluator = mocker.Mock()
    evaluator.model_id = model_id

    if train_values:
        train_metrics = [_metric(value=value, model_id=model_id) for value in train_values]
        evaluator.train_metrics.return_value = train_metrics

    if test_values:
        test_metrics = [_metric(value=value, model_id=model_id) for value in test_values]
        evaluator.test_metrics.return_value = test_metrics

    if any_values:
        any_metrics = [_metric(value=value, model_id=model_id) for value in any_values]
        evaluator.other_metrics.return_value = any_metrics

    if trava_model:
        evaluator.trava_model = trava_model

    if split_result:
        evaluator.fit_split_data = split_result

    if raw_dataset:
        evaluator.raw_split_data = split_result

    return evaluator


def _true_evaluator(model, split_result):
    return Evaluator(trava_model=model, fit_split_data=split_result, raw_split_data=split_result)


def _metric(model_id, value):
    return Metric(name=model_id + "_" + str(value), value=value, model_id=model_id)


def _fit_with_n_evaluators(
    mocker,
    trava_model,
    model_id,
    main_model_id,
    model_type,
    split_result,
    trava,
    n_evaluators,
    test_result_handlers,
    only_calculate_metrics: bool = False,
    keep_data_in_memory: bool = True,
    mock_evaluators: bool = True,
):
    evaluators = []
    for i in range(n_evaluators):
        indexed_model_id = model_id + "_" + str(i)
        eval_model_id = indexed_model_id if n_evaluators > 1 else model_id
        if mock_evaluators:
            evaluator = _mock_evaluator(
                mocker,
                model_id=eval_model_id,
                train_values=mocker.MagicMock(),
                test_values=mocker.MagicMock(),
                any_values=mocker.MagicMock(),
                trava_model=trava_model,
                split_result=split_result,
            )
        else:
            trava_model.model_id = eval_model_id
            evaluator = _true_evaluator(model=trava_model, split_result=split_result)
            # since fit predictor is mocked I need to evaluate the true evaluator manually
            evaluator.evaluate(scorers_providers=test_result_handlers)
        evaluators.append(evaluator)

    fit_predictor = _fit_predictor(mocker=mocker, evaluators=evaluators)
    trava.fit_predict(
        raw_split_data=split_result,
        model_id=main_model_id if n_evaluators > 1 else model_id,
        model_type=model_type,
        fit_predictor=fit_predictor,
        only_calculate_metrics=only_calculate_metrics,
        keep_data_in_memory=keep_data_in_memory,
    )
    return evaluators


# TESTS


def test_model_init(trava: TravaSV, split_result: SplitResult, model_init_params, model_type, model_id):
    trava.fit_predict(
        raw_split_data=split_result, model_id=model_id, model_type=model_type, model_init_params=model_init_params
    )

    model_type.assert_called_once_with(**model_init_params)


def test_fit_predictor(
    mocker,
    raw_model,
    trava: TravaSV,
    raw_dataset: RawDataset,
    split_result: SplitResult,
    model_init_params,
    model_type,
    model_id,
    test_result_handlers,
    trava_tracker,
):
    fit_predictor = _fit_predictor(mocker=mocker, evaluators=[])

    keep_models_in_memory = True
    serializer = mocker.Mock()
    fit_params = {"a": 1, "b": 2}
    predict_params = {"c": 3, "d": 4}
    description = "test_description"
    trava.fit_predict(
        raw_split_data=split_result,
        model_id=model_id,
        model_type=model_type,
        description=description,
        model_init_params=model_init_params,
        fit_predictor=fit_predictor,
        fit_params=fit_params,
        predict_params=predict_params,
        keep_models_in_memory=keep_models_in_memory,
        serializer=serializer,
    )

    fit_predictor.fit_predict.assert_called_once()

    test_fit_config = FitPredictConfig(
        raw_split_data=split_result,
        raw_model=raw_model,
        model_init_params=model_init_params,
        model_id=model_id,
        scorers_providers=test_result_handlers + [trava_tracker],
        serializer=serializer,
        fit_params=fit_params,
        predict_params=predict_params,
        description=description,
    )

    fit_predictor.fit_predict.assert_called_once_with(config=test_fit_config, tracker=trava_tracker)


@pytest.mark.parametrize(
    "evaluator_values",
    [
        [
            ([1.0, 3.0, {}, "train_string"], [4.0, 4.0, [], "test_string"], [10, pd.DataFrame, []])  # train  # test
        ],  # model meta
        [
            (
                [1.0, 3.0, {}, "train_string"],  # train
                [4.0, 4.0, [], "test_string"],  # test
                [10, pd.DataFrame, []],
            ),  # model meta
            (
                [9.0, 5.0, {}, "train_string"],  # train
                [16.0, 8.0, [], "test_string"],  # test
                [20, pd.DataFrame, []],
            ),  # model meta
        ],
    ],
)
def test_fit_predict_many_result(
    mocker, trava: TravaSV, split_result: SplitResult, model_type, model_id, evaluator_values
):
    evaluators = []

    for values in evaluator_values:
        evaluator = _mock_evaluator(
            mocker, model_id=model_id, train_values=values[0], test_values=values[1], any_values=values[2]
        )

        evaluators.append(evaluator)

    fit_predictor = _fit_predictor(mocker=mocker, evaluators=evaluators)

    result = trava.fit_predict(
        raw_split_data=split_result, model_id=model_id, model_type=model_type, fit_predictor=fit_predictor
    )
    if len(evaluator_values) == 1:
        assert result == [[evaluator_values[0][0], evaluator_values[0][1], evaluator_values[0][2]]]
    else:
        assert result == [[[5.0, 4.0], [10.0, 6.0], [15.0]]]

    assert trava.detailed_results_for(model_id=model_id)


@pytest.mark.parametrize("n_evaluators", [1, 3])
def test_evaluate_no_save(
    mocker, trava, model_id, main_model_id, model_type, trava_model, split_result, test_result_handlers, n_evaluators
):
    evaluators = _fit_with_n_evaluators(
        mocker=mocker,
        trava_model=trava_model,
        model_id=model_id,
        main_model_id=main_model_id,
        model_type=model_type,
        split_result=split_result,
        trava=trava,
        n_evaluators=n_evaluators,
        test_result_handlers=test_result_handlers,
    )

    trava.evaluate(
        model_id=main_model_id if n_evaluators > 1 else model_id,
        results_handlers=test_result_handlers,
        save_results=False,
    )

    for evaluator in evaluators:
        evaluator.evaluate.assert_not_called()


# noinspection PyPep8Naming
@pytest.mark.parametrize("n_evaluators", [1, 3])
def test_evaluate_save(
    mocker,
    trava,
    model_id,
    main_model_id,
    raw_model,
    trava_model,
    model_type,
    split_result,
    X_y_train,
    X_y_test,
    n_evaluators,
):
    any_and_test_data = 1233
    trava_model.get_model.return_value = raw_model
    trava_model.any_and_test_data = any_and_test_data

    result_handlers = [
        TestResultsHandler(
            scorers=[
                TestScorer(score_func=np.sum),
                TestAnyScorer(model_func=lambda scorer_model: scorer_model.any_and_test_data),
            ]
        )
    ]

    evaluators = _fit_with_n_evaluators(
        mocker=mocker,
        trava_model=trava_model,
        model_id=model_id,
        main_model_id=main_model_id,
        model_type=model_type,
        split_result=split_result,
        trava=trava,
        n_evaluators=n_evaluators,
        test_result_handlers=result_handlers,
    )

    result = trava.evaluate(
        model_id=main_model_id if n_evaluators > 1 else model_id, results_handlers=result_handlers, save_results=True
    )

    for evaluator in evaluators:
        evaluator.evaluate.assert_called_once_with(scorers_providers=result_handlers)

    X_train, _ = X_y_train
    X_test, _ = X_y_test

    f1_train_sum = np.sum(X_train["f1"])
    f1_test_sum = np.sum(X_test["f1"])

    assert result == [[[f1_train_sum], [f1_test_sum], [any_and_test_data]]]


@pytest.mark.parametrize("n_evaluators", [1, 3])
def test_evaluate_track(
    mocker,
    trava,
    model_id,
    main_model_id,
    model_type,
    trava_model,
    split_result,
    test_result_handlers,
    X_y_train,
    X_y_test,
    n_evaluators,
):
    evaluators = _fit_with_n_evaluators(
        mocker=mocker,
        trava_model=trava_model,
        model_id=model_id,
        main_model_id=main_model_id,
        model_type=model_type,
        split_result=split_result,
        trava=trava,
        n_evaluators=n_evaluators,
        test_result_handlers=test_result_handlers,
    )

    tracker_mock = mocker.Mock()
    tracker_copy_func = mocker.patch.object(trava, "_tracker_copy")
    tracker_copy_func.return_value = tracker_mock

    scorer_1 = mocker.Mock()
    scorer_2 = mocker.Mock()

    scorers = [scorer_1, scorer_2]

    trava.evaluate_track(model_id=main_model_id if n_evaluators > 1 else model_id, scorers=scorers)
    calls = [mocker.call.add_scorers(scorers=scorers)]
    for idx, evaluator in enumerate(evaluators):
        calls += [
            mocker.call.start_tracking(model_id=evaluator.model_id),
            mocker.call.track_model_results(model_results=mocker.ANY),
            mocker.call.end_tracking(model_id=evaluator.model_id),
        ]

        track_model_results_kwargs = tracker_mock.track_model_results.call_args_list[idx][1]
        model_results = track_model_results_kwargs["model_results"]

        assert (
            model_results.model_id == evaluator.model_id
            and len(model_results.evaluators) == 1
            and model_results.evaluators == [evaluator]
        )

    if n_evaluators > 1:
        calls += [
            mocker.call.start_tracking(model_id=main_model_id),
            mocker.call.track_model_results(model_results=mocker.ANY),
            mocker.call.end_tracking(model_id=main_model_id),
        ]

        track_model_results_kwargs = tracker_mock.track_model_results.call_args_list[n_evaluators][1]
        model_results = track_model_results_kwargs["model_results"]

        assert (
            model_results.model_id == main_model_id
            and len(model_results.evaluators) == n_evaluators
            and model_results.evaluators == evaluators
        )

    tracker_mock.assert_has_calls(calls)


def test_detailed_results_no_model(trava):
    with pytest.raises(AttributeError):
        trava.detailed_results_for(model_id="arbitrary_id")


@pytest.mark.parametrize("n_evaluators", [1, 3])
def test_detailed_results(
    mocker,
    trava,
    model_id,
    main_model_id,
    model_type,
    trava_model,
    split_result,
    test_result_handlers,
    X_y_train,
    X_y_test,
    n_evaluators,
):
    test_model_id = main_model_id if n_evaluators > 1 else model_id

    meta_test_data = 1233
    trava_model.meta_test_data = meta_test_data

    result_handler = TestResultsHandler(
        scorers=[
            TestScorer(score_func=np.sum),
            TestAnyScorer(model_func=lambda scorer_model: scorer_model.meta_test_data),
        ]
    )
    _fit_with_n_evaluators(
        mocker=mocker,
        trava_model=trava_model,
        model_id=model_id,
        main_model_id=main_model_id,
        model_type=model_type,
        split_result=split_result,
        trava=trava,
        n_evaluators=n_evaluators,
        test_result_handlers=[result_handler],
        mock_evaluators=False,
    )

    result = trava.detailed_results_for(model_id=test_model_id)

    X_train, _ = X_y_train
    X_test, _ = X_y_test

    f1_train_sum = np.sum(X_train["f1"])
    f1_test_sum = np.sum(X_test["f1"])

    assert result == [[[f1_train_sum] * n_evaluators, [f1_test_sum] * n_evaluators, [meta_test_data] * n_evaluators]]


# TODO: move to trava_base tests
@pytest.mark.parametrize("n_fits", [1, 2])
def test_results(
    mocker,
    trava,
    model_id,
    main_model_id,
    model_type,
    trava_model,
    split_result,
    test_result_handlers,
    X_y_train,
    X_y_test,
    n_fits,
):
    score_funcs = [np.sum, np.mean]

    X_train, _ = X_y_train
    X_test, _ = X_y_test

    meta_test_data = 1233
    trava_model.meta_test_data = meta_test_data

    metrics_values = []
    for i in range(n_fits):
        score_func = score_funcs[i]
        result_handler = TestResultsHandler(
            scorers=[
                TestScorer(score_func=score_func),
                TestAnyScorer(model_func=lambda scorer_model: scorer_model.meta_test_data),
            ]
        )
        _fit_with_n_evaluators(
            mocker=mocker,
            trava_model=trava_model,
            model_id=model_id + "_" + str(i),
            main_model_id=main_model_id,
            model_type=model_type,
            split_result=split_result,
            trava=trava,
            n_evaluators=1,
            test_result_handlers=[result_handler],
            mock_evaluators=False,
        )

        metrics_values.append((score_func(X_train["f1"]), score_func(X_test["f1"]), meta_test_data))

    results = trava.results

    test_metrics_values = [list(tup) for tup in list(zip(*metrics_values))]
    # test_metrics_values.append([])
    assert results == [test_metrics_values]


@pytest.mark.parametrize("n_evaluators", [1, 3])
def test_evaluate_dont_keep_data(
    mocker, trava, model_id, main_model_id, model_type, trava_model, split_result, test_result_handlers, n_evaluators
):
    evaluators = _fit_with_n_evaluators(
        mocker=mocker,
        trava_model=trava_model,
        model_id=model_id,
        main_model_id=main_model_id,
        model_type=model_type,
        split_result=split_result,
        trava=trava,
        n_evaluators=n_evaluators,
        test_result_handlers=test_result_handlers,
        keep_data_in_memory=True,
    )

    for evaluator in evaluators:
        evaluator.unload_data.assert_not_called()

    evaluators = _fit_with_n_evaluators(
        mocker=mocker,
        trava_model=trava_model,
        model_id=model_id,
        main_model_id=main_model_id,
        model_type=model_type,
        split_result=split_result,
        trava=trava,
        n_evaluators=n_evaluators,
        test_result_handlers=test_result_handlers,
        keep_data_in_memory=False,
    )

    for evaluator in evaluators:
        evaluator.unload_data.assert_called()


@pytest.mark.parametrize("n_evaluators", [1, 3])
def test_raw_models(
    mocker,
    trava,
    model_id,
    main_model_id,
    model_type,
    raw_model,
    trava_model,
    split_result,
    test_result_handlers,
    n_evaluators,
):
    evaluators = _fit_with_n_evaluators(
        mocker=mocker,
        trava_model=trava_model,
        model_id=model_id,
        main_model_id=main_model_id,
        model_type=model_type,
        split_result=split_result,
        trava=trava,
        n_evaluators=n_evaluators,
        test_result_handlers=test_result_handlers,
        keep_data_in_memory=False,
        mock_evaluators=False,
    )

    target_model_id = model_id if n_evaluators == 1 else main_model_id
    raw_models = trava.raw_models_for(model_id=target_model_id)

    test_raw_models = dict([(evaluator.model_id, evaluator.trava_model.raw_model) for evaluator in evaluators])

    assert raw_models == test_raw_models


def test_create_raw_model(mocker, trava):
    model_type = TestModel
    def_1_value = "custom_value"
    req_param_value = 111
    model_init_params = {"required_param_1": mocker.Mock(), "required_param_2": req_param_value, "def_1": def_1_value}
    model, result_init_params = trava._create_raw_model(model_type=model_type, model_init_params=model_init_params)

    test_init_params = {"required_param_2": req_param_value, "def_1": def_1_value, "def_2": "tada", "def_5": False}

    assert isinstance(model, model_type)
    assert result_init_params == test_init_params


@pytest.mark.parametrize("n_evaluators", [1, 3])
def test_only_calculate_metrics(mocker, model_id, main_model_id, model_type, trava_model, split_result, n_evaluators):
    results_handler = mocker.Mock()
    results_handlers = [results_handler]
    trava = TravaSV(results_handlers=results_handlers, tracker=trava_tracker)
    _fit_with_n_evaluators(
        mocker=mocker,
        trava_model=trava_model,
        model_id=model_id,
        main_model_id=main_model_id,
        model_type=model_type,
        split_result=split_result,
        trava=trava,
        n_evaluators=n_evaluators,
        only_calculate_metrics=True,
        test_result_handlers=results_handlers,
        keep_data_in_memory=True,
    )

    results_handler.handle.assert_not_called()
