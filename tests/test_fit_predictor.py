import pytest
from copy import copy

from tests.objects_for_tests import TestSerializer
from trava.fit_predictor import FitPredictor, FitPredictorSteps
from trava.split.result import SplitResult


@pytest.fixture(scope="class")
def model_id():
    return "test_model_id"


@pytest.fixture()
def raw_model(mocker, model_id):
    result = mocker.Mock()
    result.model_id = model_id
    return result


@pytest.fixture
def X_y_train(mocker):
    X_train = mocker.MagicMock()
    X_train["f1"] = [1, 2, 3]
    y_train = [0, 0, 0]

    return X_train, y_train


@pytest.fixture
def X_y_test(mocker):
    X_test = mocker.MagicMock()
    X_test["f1"] = [4, 5, 6]
    y_test = [1, 1, 1]

    return X_test, y_test


@pytest.fixture
def split_result(X_y_train, X_y_test):
    X_train, y_train = X_y_train
    X_test, y_test = X_y_test

    return SplitResult(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


@pytest.mark.parametrize("n_steps", [1, 3])
def test_model_update_steps(mocker, n_steps, raw_model):
    config = mocker.MagicMock()
    config.fit_params = mocker.MagicMock()
    config.predict_params = mocker.MagicMock()
    config.raw_model = raw_model
    config.scorers_providers = mocker.MagicMock()

    steps = []
    for _ in range(n_steps):
        step = mocker.Mock()
        step.update_model.return_value = raw_model
        steps.append(step)

    fp_steps = FitPredictorSteps(raw_model_steps=steps)
    fp = FitPredictor(steps=fp_steps)

    fp.fit_predict(config=config, tracker=mocker.Mock())

    for step in steps:
        step.update_model.assert_called_with(raw_model=raw_model, config=config)


def test_models_configs(mocker, raw_model, model_id):
    fp = FitPredictor()
    models_configs_func = mocker.patch.object(fp, "_models_configs")
    models_configs = mocker.MagicMock()
    models_configs_func.return_value = models_configs

    config = mocker.Mock()
    config.raw_model = raw_model
    config.scorers_providers = mocker.MagicMock()

    fp.fit_predict(config=config, tracker=mocker.Mock())

    models_configs_func.assert_called_once_with(raw_model=raw_model, config=config)
    models_configs.__iter__.assert_called()


@pytest.mark.parametrize("n_steps", [1, 3])
def test_config_update_steps(mocker, n_steps, split_result, raw_model):
    fit_params = {"c": 2, "d": 3}
    predict_params = {"_": 4, "+": 5}

    config = mocker.Mock()
    config.raw_split_data = split_result
    config.raw_model = raw_model
    config.fit_params = fit_params
    config.predict_params = predict_params
    config.scorers_providers = mocker.MagicMock()

    steps = []
    for _ in range(n_steps):
        step = mocker.Mock()
        step.fit_split_data.return_value = split_result
        step.fit_params.return_value = fit_params
        step.predict_params.return_value = predict_params
        steps.append(step)
    fp_steps = FitPredictorSteps(config_steps=steps)
    fp = FitPredictor(steps=fp_steps)

    models_configs_func = mocker.patch.object(fp, "_models_configs")
    models_configs_func.return_value = [(mocker.Mock(), config)]

    tracker = mocker.Mock()
    fp.fit_predict(config=config, tracker=tracker)

    for step in steps:
        step.fit_split_data.assert_called_with(raw_split_data=split_result, config=config, tracker=tracker)
        step.fit_params.assert_called_with(
            fit_params=fit_params, fit_split_data=split_result, config=config, tracker=tracker
        )
        step.predict_params.assert_called_with(
            predict_params=predict_params, fit_split_data=split_result, config=config, tracker=tracker
        )


@pytest.mark.parametrize("n_steps", [1, 3])
def test_final_steps(mocker, split_result, raw_model, n_steps):
    fit_params = {"c": 2, "d": 3}
    predict_params = {"_": 4, "+": 5}

    config = mocker.Mock()
    config.raw_split_data = split_result
    config.raw_model = raw_model
    config.fit_params = fit_params
    config.predict_params = predict_params
    config.scorers_providers = mocker.MagicMock()

    steps = []
    for _ in range(n_steps):
        step = mocker.Mock()
        step.fit_split_data.return_value = split_result
        step.fit_params.return_value = fit_params
        step.predict_params.return_value = predict_params
        steps.append(step)
    fp_steps = FitPredictorSteps(final_steps=steps)
    fp = FitPredictor(steps=fp_steps)

    models_configs_func = mocker.patch.object(fp, "_models_configs")
    trava_model = mocker.Mock()
    models_configs_func.return_value = [(trava_model, config)]

    tracker = mocker.Mock()
    fp.fit_predict(config=config, tracker=tracker)

    for step in steps:
        step.handle.assert_called_with(trava_model=trava_model, config=config, tracker=tracker)


@pytest.mark.parametrize("n_models", [1, 3])
@pytest.mark.parametrize("has_description", [True, False], ids=["has_descr", "no_descr"])
@pytest.mark.parametrize("serializer", [TestSerializer(), None], ids=["serial", "no_serial"])
def test_tracking(mocker, model_id, split_result, raw_model, n_models, has_description, serializer):
    config = mocker.Mock()
    config.model_id = model_id
    config.raw_split_data = split_result
    config.serializer = serializer
    config.raw_model = raw_model
    model_init_params = mocker.Mock()

    description = "descr"
    if has_description:
        config.description = description
    else:
        config.description = None

    config.model_init_params = model_init_params
    fit_params = mocker.MagicMock()
    config.fit_params = fit_params
    predict_params = mocker.MagicMock()
    config.predict_params = predict_params
    config.scorers_providers = mocker.MagicMock()
    fp = FitPredictor()

    models_configs = []
    for idx in range(n_models):
        nested_model = mocker.Mock()
        nested_model.raw_model = raw_model
        nested_model_id = model_id + "_" + str(idx)
        nested_model.model_id = nested_model_id
        model_config = copy(config)
        model_config.model_id = nested_model_id
        models_configs.append((nested_model, model_config))

    models_configs_func = mocker.patch.object(fp, "_models_configs")
    models_configs_func.return_value = models_configs

    tracker = mocker.Mock()
    fp.fit_predict(config=config, tracker=tracker)

    expected_calls = []

    is_multiple_models = n_models > 1
    if is_multiple_models:
        expected_calls += _start_tracking_check_get_calls(
            mocker=mocker,
            model_id=model_id,
            raw_model=raw_model,
            tracker=tracker,
            description=description,
            model_init_params=model_init_params,
            has_description=has_description,
            nested=False,
        )

    models = [model_config[0] for model_config in models_configs]
    for idx, model in enumerate(models):
        expected_calls += _start_tracking_check_get_calls(
            mocker=mocker,
            model_id=model.model_id,
            raw_model=raw_model,
            tracker=tracker,
            description=description,
            model_init_params=model_init_params,
            has_description=has_description,
            nested=is_multiple_models,
        )

        expected_calls += [
            mocker.call.track_fit_params(model_id=model.model_id, params=fit_params),
            mocker.call.track_predict_params(model_id=model.model_id, params=predict_params),
        ]

        model_results_kwargs = tracker.track_model_results.call_args_list[idx][1]
        model_results = model_results_kwargs["model_results"]
        assert model_results.model_id == model.model_id

        expected_calls += [mocker.call.track_model_results(model_results=model_results)]

        if serializer:
            expected_calls += [
                mocker.call.track_model_artifact(model_id=model.model_id, model=raw_model, serializer=serializer)
            ]

        expected_calls += [mocker.call.end_tracking(model_id=model.model_id)]

    if is_multiple_models:
        model_results_kwargs = tracker.track_model_results.call_args_list[n_models][1]
        model_results = model_results_kwargs["model_results"]
        assert model_results.model_id == model_id

        expected_calls += [
            mocker.call.track_model_results(model_results=model_results),
            mocker.call.end_tracking(model_id=model_id),
        ]

    tracker.assert_has_calls(expected_calls)


def _start_tracking_check_get_calls(
    mocker, model_id, raw_model, tracker, description, model_init_params, has_description: bool, nested: bool
) -> list:
    expected_calls = [
        mocker.call.start_tracking(model_id=model_id, nested=nested),
        mocker.call.track_model_init_params(model_id=model_id, params=model_init_params),
        mocker.call.track_model_info(model_id=model_id, model=raw_model),
    ]

    if has_description:
        expected_calls.append(mocker.call.track_model_description(model_id=model_id, description=description))
    else:
        tracker.track_model_description.assert_not_called()

    return expected_calls


def test_has_any_log_calls(mocker, split_result, raw_model):
    config = mocker.Mock()
    config.raw_split_data = split_result
    config.raw_model = raw_model
    config.fit_params = mocker.MagicMock()
    config.predict_params = mocker.MagicMock()
    config.scorers_providers = mocker.MagicMock()

    logger = mocker.Mock()
    fp = FitPredictor(logger=logger)
    fp.fit_predict(config=config, tracker=mocker.Mock())
    logger.log.assert_called()


@pytest.mark.parametrize("n_models", [1, 3])
def test_fit_predict(mocker, model_id, split_result, raw_model, n_models):
    config = mocker.Mock()
    config.model_id = model_id
    config.raw_split_data = split_result
    config.raw_model = raw_model
    model_init_params = mocker.Mock()

    config.model_init_params = model_init_params
    fit_params = mocker.MagicMock()
    config.fit_params = fit_params
    predict_params = mocker.MagicMock()
    config.predict_params = predict_params
    scorers_providers = mocker.MagicMock()
    config.scorers_providers = scorers_providers
    fp = FitPredictor()
    fit_mock = mocker.patch.object(fp, "_fit")
    fit_mock.return_value = None
    predict_mock = mocker.patch.object(fp, "_predict")
    predict_mock.return_value = None

    models_configs = []
    for idx in range(n_models):
        nested_model = mocker.Mock()
        nested_model.raw_model = raw_model
        nested_model_id = model_id + "_" + str(idx)
        nested_model.model_id = nested_model_id

        model_config = copy(config)

        model_config.model_id = nested_model_id
        models_configs.append((nested_model, model_config))

    models_configs_func = mocker.patch.object(fp, "_models_configs")
    models_configs_func.return_value = models_configs

    tracker = mocker.Mock()
    fp.fit_predict(config=config, tracker=tracker)

    is_multiple_models = n_models > 1
    if is_multiple_models:
        for idx, (model, config) in enumerate(models_configs):
            fit_call_args = fit_mock.call_args_list[idx][1]
            assert fit_call_args["trava_model"] == model
            assert fit_call_args["X"] == config.raw_split_data.X_train
            assert fit_call_args["y"] == config.raw_split_data.y_train
            assert fit_call_args["fit_params"] == fit_params
            assert fit_call_args["predict_params"] == predict_params

            predict_call_args = predict_mock.call_args_list[idx][1]

            assert predict_call_args["trava_model"] == model
            assert predict_call_args["X"] == config.raw_split_data.X_test
            assert predict_call_args["y"] == config.raw_split_data.y_test
    else:
        fit_mock.assert_called_once_with(
            trava_model=models_configs[0][0],
            X=split_result.X_train,
            y=split_result.y_train,
            fit_params=fit_params,
            predict_params=predict_params,
        )

        predict_mock.assert_called_once_with(
            trava_model=models_configs[0][0], X=split_result.X_test, y=split_result.y_test
        )


@pytest.mark.parametrize("n_models", [1, 3])
def test_return_evaluators(mocker, model_id, split_result, raw_model, n_models):
    config = mocker.Mock()
    config.model_id = model_id
    config.raw_split_data = split_result
    config.raw_model = raw_model
    model_init_params = mocker.Mock()

    config.model_init_params = model_init_params
    fit_params = mocker.MagicMock()
    config.fit_params = fit_params
    predict_params = mocker.MagicMock()
    config.predict_params = predict_params
    config.scorers_providers = mocker.MagicMock()
    fp = FitPredictor()

    models_configs = []
    for idx in range(n_models):
        nested_model = mocker.Mock()
        nested_model.raw_model = raw_model
        nested_model_id = model_id + "_" + str(idx)
        nested_model.model_id = nested_model_id
        model_config = copy(config)
        model_config.model_id = nested_model_id
        models_configs.append((nested_model, model_config))

    models_configs_func = mocker.patch.object(fp, "_models_configs")
    models_configs_func.return_value = models_configs

    tracker = mocker.Mock()
    evaluators = fp.fit_predict(config=config, tracker=tracker)

    assert len(evaluators) == n_models

    is_multiple_models = n_models > 1
    if is_multiple_models:
        models = [model_config[0] for model_config in models_configs]

        for model, evaluator in zip(models, evaluators):
            assert model.model_id == evaluator.model_id
            assert model == evaluator.trava_model
            assert split_result == evaluator.fit_split_data
            assert split_result == evaluator.raw_split_data
    else:
        model = models_configs[0][0]
        evaluator = evaluators[0]

        assert model.model_id == evaluator.model_id
        assert model == evaluator.trava_model
        assert split_result == evaluator.fit_split_data
        assert split_result == evaluator.raw_split_data


@pytest.mark.parametrize("n_models", [1, 3])
def test_evaluators_get_called(mocker, model_id, split_result, raw_model, n_models):
    config = mocker.Mock()
    config.model_id = model_id
    config.raw_split_data = split_result
    config.raw_model = raw_model
    model_init_params = mocker.Mock()

    config.model_init_params = model_init_params
    fit_params = mocker.MagicMock()
    config.fit_params = fit_params
    predict_params = mocker.MagicMock()
    config.predict_params = predict_params
    scorers_providers = mocker.MagicMock()
    config.scorers_providers = scorers_providers
    fp = FitPredictor()

    models_configs = []
    for idx in range(n_models):
        nested_model = mocker.Mock()
        nested_model.raw_model = raw_model
        nested_model_id = model_id + "_" + str(idx)
        nested_model.model_id = nested_model_id
        model_config = copy(config)
        model_config.model_id = nested_model_id
        models_configs.append((nested_model, model_config))

    models_configs_func = mocker.patch.object(fp, "_models_configs")
    models_configs_func.return_value = models_configs

    evaluator_func = mocker.patch.object(fp, "_evaluator")
    evaluator_func.return_value = mocker.Mock()

    evaluators = fp.fit_predict(config=config, tracker=mocker.Mock())

    is_multiple_models = n_models > 1
    if is_multiple_models:
        for idx in range(n_models):
            model = models_configs[idx][0]
            config = models_configs[idx][1]
            evaluator = evaluators[idx]

            eval_func_args = evaluator_func.call_args_list[idx][1]
            assert eval_func_args["model_config"] == config
            assert eval_func_args["split_result"] == config.raw_split_data
            assert eval_func_args["model"] == model

            evaluator.evaluate.assert_called_with(scorers_providers=scorers_providers)
    else:
        model = models_configs[0][0]
        config = models_configs[0][1]
        evaluator = evaluators[0]

        evaluator_func.assert_called_with(model_config=config, split_result=split_result, model=model)
        evaluator.evaluate.assert_called_with(scorers_providers=scorers_providers)
