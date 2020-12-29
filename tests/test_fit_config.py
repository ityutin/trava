import pytest

from trava.fit_predictor import FitPredictConfig


@pytest.fixture(scope="class")
def model_id():
    return "test_model_id"


@pytest.fixture()
def model(mocker, model_id):
    result = mocker.Mock()
    result.model_id = model_id
    return result


def test_equality(mocker, model, model_id):
    model_init_params = {"a": 1, "b": 2}
    scorers_providers = mocker.MagicMock()
    serializer = mocker.Mock()
    split_result = mocker.Mock()
    fit_params = {"c": 3, "d": 4}
    predict_params = {"_": 11, "+": 43}
    description = "descr"

    first_config = FitPredictConfig(
        raw_model=model,
        model_init_params=model_init_params,
        model_id=model_id,
        scorers_providers=scorers_providers,
        serializer=serializer,
        raw_split_data=split_result,
        fit_params=fit_params,
        predict_params=predict_params,
        description=description,
    )

    second_config = FitPredictConfig(
        raw_model=model,
        model_init_params=model_init_params,
        model_id=model_id,
        scorers_providers=scorers_providers,
        serializer=serializer,
        raw_split_data=split_result,
        fit_params=fit_params,
        predict_params=predict_params,
        description=description,
    )

    assert first_config == second_config
    second_config.serializer = mocker.Mock()
    assert first_config != second_config
    assert first_config != "str"

    none_config = FitPredictConfig(
        raw_model=None,
        model_init_params=None,
        model_id="123",
        scorers_providers=[],
        serializer=None,
        raw_split_data=mocker.Mock(),
    )
    assert first_config != none_config
