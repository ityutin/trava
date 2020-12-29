import pytest
import numpy as np

from trava.trava_model import TravaModel


@pytest.fixture(scope="class")
def model_id():
    return "test_model_id"


@pytest.fixture
def raw_model(mocker):
    return mocker.Mock()


@pytest.fixture
def X(mocker):
    return mocker.Mock()


@pytest.fixture
def y(mocker):
    return mocker.Mock()


@pytest.fixture
def fit_params(mocker):
    return {"fit_param": 111}


@pytest.fixture
def predict_params(mocker):
    return {"predict_param": 222}


@pytest.mark.parametrize("use_existing_model", [True, False], ids=["existing model", "no existing model"])
@pytest.mark.parametrize("only_fit", [True, False], ids=["only fit", "no only fit"])
def test_copy(mocker, model_id, use_existing_model, only_fit):
    raw_model = mocker.Mock()
    model = TravaModel(raw_model=raw_model, model_id=model_id)

    y_train = np.array([0, 0, 1])
    y_train_pred = np.array([1, 2, 3])
    y_train_pred_proba = np.array([3, 4, 5])
    y_test = np.array([-2, 3, 5])
    y_test_pred = np.array([6, 7, 8])
    y_test_pred_proba = np.array([9, 10, 11])

    fit_params = {"1": 2}
    predict_params = {"2": 3}
    fit_time = 123
    predict_time = 434

    model._y_train = y_train
    model._y_train_pred = y_train_pred
    model._y_train_pred_proba = y_train_pred_proba

    model._y_test = y_test
    model._y_test_pred = y_test_pred
    model._y_test_pred_proba = y_test_pred_proba

    model._fit_params = fit_params
    model._predict_params = predict_params
    model._fit_time = fit_time
    model._predict_time = predict_time

    model_copy_id = model_id + "_copy"
    existing_model = None
    existing_model_id = "existing_model"
    # what a mess... but should work
    if use_existing_model:
        existing_model = TravaModel(raw_model=raw_model, model_id=existing_model_id)
        model_copy = model.copy(existing_model=existing_model, only_fit=only_fit)
    else:
        model_copy = model.copy(model_id=model_copy_id, only_fit=only_fit)

    if use_existing_model:
        assert model_copy.model_id == existing_model_id
    else:
        assert model_copy.model_id == model_copy_id

    y = model.y(for_train=True)
    copy_y = model_copy.y(for_train=True)
    assert np.array_equal(y, copy_y)
    assert model.fit_params == model_copy.fit_params
    assert model.fit_time == model_copy.fit_time

    if use_existing_model:
        assert existing_model == model_copy
    else:
        assert existing_model != model_copy

    if only_fit:
        assert model_copy.y(for_train=False) is None
        assert model_copy.y_pred(for_train=False) is None
        assert model_copy.y_pred_proba(for_train=False) is None
        assert model_copy.predict_time is None
        assert model_copy.predict_params == {}
    else:
        assert np.array_equal(model.y(for_train=False), model_copy.y(for_train=False))
        assert np.array_equal(model.y_pred(for_train=False), model_copy.y_pred(for_train=False))
        assert np.array_equal(model.y_pred_proba(for_train=False), model_copy.y_pred_proba(for_train=False))
        assert model.predict_time == model_copy.predict_time
        assert model.predict_params == model_copy.predict_params


def test_model_id(mocker, model_id):
    raw_model = mocker.Mock()
    model = TravaModel(raw_model=raw_model, model_id=model_id)

    assert model.model_id == model_id


@pytest.mark.parametrize("needs_proba", [True, False], ids=["proba", "no_proba"])
def test_get_model(mocker, raw_model, X, y, needs_proba):
    model = TravaModel(raw_model=raw_model, model_id=model_id)

    assert model.get_model(for_train=True) == raw_model
    assert model.get_model(for_train=False) == raw_model

    y_predict_proba = mocker.Mock()
    if needs_proba:
        raw_model.predict_proba.return_value = y_predict_proba

    y_pred = mocker.Mock()
    raw_model.predict.return_value = y_pred

    model.fit(X=X, y=y)
    model.predict(X=X, y=y)

    model.unload_model()

    train_cached_model = model.get_model(for_train=True)
    test_cached_model = model.get_model(for_train=False)

    assert train_cached_model != raw_model
    assert test_cached_model != raw_model

    assert train_cached_model.predict(X) == y_pred
    if needs_proba:
        assert train_cached_model.predict_proba(X) == y_predict_proba


@pytest.mark.parametrize("for_train", [True, False], ids=["for_train", "for_test"])
def test_get_model_unload(mocker, raw_model, for_train):
    trava_model = TravaModel(raw_model=raw_model, model_id=model_id)
    trava_model.unload_model()

    with pytest.raises(ValueError):
        trava_model.get_model(for_train=for_train)

    if for_train:
        y_pred_key = "_y_train_pred"
    else:
        y_pred_key = "_y_test_pred"

    y_pred_mock = mocker.Mock()
    mocker.patch.object(trava_model, y_pred_key, y_pred_mock)

    assert trava_model.get_model(for_train=for_train).predict(X=None) == y_pred_mock


def test_raw_model(mocker, raw_model, model_id):
    model = TravaModel(raw_model=raw_model, model_id=model_id)

    assert model.raw_model == raw_model


def test_fit_params(mocker, raw_model, model_id, X, y, fit_params, predict_params):
    model = TravaModel(raw_model=raw_model, model_id=model_id)
    model.fit(X=X, y=y, fit_params=fit_params, predict_params=predict_params)

    assert model.fit_params == fit_params


def test_predict_params(mocker, raw_model, model_id, X, y, fit_params, predict_params):
    model = TravaModel(raw_model=raw_model, model_id=model_id)
    model.fit(X=X, y=y, fit_params=fit_params, predict_params=predict_params)

    assert model.predict_params == predict_params


def test_fit_time(mocker, raw_model, model_id, X, y):
    model = TravaModel(raw_model=raw_model, model_id=model_id)
    assert not model.fit_time
    model.fit(X=X, y=y)
    assert model.fit_time


def test_predict_time(mocker, raw_model, model_id, X, y):
    model = TravaModel(raw_model=raw_model, model_id=model_id)
    assert not model.predict_time
    model.predict(X=X, y=y)
    assert model.predict_time


@pytest.mark.parametrize("needs_proba", [True, False], ids=["proba", "no_proba"])
def test_fit(mocker, raw_model, model_id, X, y, fit_params, predict_params, needs_proba):
    if needs_proba:
        predict_proba = mocker.Mock()
        raw_model.predict_proba.return_value = predict_proba

    y_pred = mocker.Mock()
    raw_model.predict.return_value = y_pred

    model = TravaModel(raw_model=raw_model, model_id=model_id)
    model.fit(X=X, y=y, fit_params=fit_params, predict_params=predict_params)

    raw_model.fit.assert_called_once_with(X, y, **fit_params)
    raw_model.predict.assert_called_once_with(X, **predict_params)

    if needs_proba:
        raw_model.predict_proba.assert_called_with(X, **predict_params)

    assert model.fit_time


@pytest.mark.parametrize("needs_proba", [True, False], ids=["proba", "no_proba"])
def test_predict(mocker, raw_model, model_id, X, y, fit_params, needs_proba):
    if needs_proba:
        predict_proba = mocker.Mock()
        raw_model.predict_proba.return_value = predict_proba

    y_pred = mocker.Mock()
    raw_model.predict.return_value = y_pred

    model = TravaModel(raw_model=raw_model, model_id=model_id)
    model.predict(X=X, y=y)

    raw_model.predict.assert_called_once_with(X)

    if needs_proba:
        raw_model.predict_proba.assert_called_with(X)

    assert model.predict_time


@pytest.mark.parametrize("is_classification", [True, False], ids=["is_classification", "no_classification"])
def test_is_classification(mocker, raw_model, model_id, X, y, fit_params, is_classification):
    predict_proba = mocker.Mock()

    if is_classification:
        raw_model.predict_proba.return_value = predict_proba
    else:
        raw_model.predict_proba = None

    model = TravaModel(raw_model=raw_model, model_id=model_id)

    if is_classification:
        assert model.is_classification_model
    else:
        assert not model.is_classification_model


@pytest.mark.parametrize("needs_proba", [True, False], ids=["proba", "no_proba"])
def test_all_y(mocker, raw_model, model_id, X, y, fit_params, predict_params, needs_proba):
    model = TravaModel(raw_model=raw_model, model_id=model_id)

    predict_proba_train = mocker.Mock()
    if needs_proba:
        raw_model.predict_proba.return_value = predict_proba_train
    y_pred_train = mocker.Mock()
    raw_model.predict.return_value = y_pred_train
    model.fit(X=X, y=y, fit_params=fit_params, predict_params=predict_params)

    predict_proba_test = mocker.Mock()
    if needs_proba:
        raw_model.predict_proba.return_value = predict_proba_test
    y_pred_test = mocker.Mock()
    raw_model.predict.return_value = y_pred_test
    X_test = mocker.Mock()
    y_test = mocker.Mock()
    model.predict(X=X_test, y=y_test)

    assert model.y_pred(for_train=True) == y_pred_train
    assert model.y_pred(for_train=False) == y_pred_test
    assert model.y(for_train=True) == y
    assert model.y(for_train=False) == y_test
    if needs_proba:
        assert model.y_pred_proba(for_train=True) == predict_proba_train
        assert model.y_pred_proba(for_train=False) == predict_proba_test


def test_unload(mocker, model_id):
    raw_model = mocker.Mock()
    model = TravaModel(raw_model=raw_model, model_id=model_id)
    model.unload_model()

    assert not model.raw_model
