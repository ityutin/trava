import pytest

from trava.trava_tracker import TravaTracker


@pytest.fixture(scope="class")
def model_id():
    return "test_model_id"


def _mocks_for_subclass_calls(mocker, tracker, model_id):
    method_names = list(_parameters_for_methods(mocker=mocker, model_id=model_id).keys())
    subclass_method_names = ["_" + method_name for method_name in method_names]

    mocks = [mocker.patch.object(tracker, method_name) for method_name in subclass_method_names]

    return mocks


def _parameters_for_methods(mocker, model_id):
    return {
        "start_tracking": dict(model_id=model_id),
        "end_tracking": dict(model_id=model_id),
        "track_set_tracking_group": dict(group="group"),
        "track_model_description": dict(model_id=model_id, description="desc"),
        "track_model_init_params": dict(model_id=model_id, params=mocker.Mock()),
        "track_fit_params": dict(model_id=model_id, params=mocker.Mock()),
        "track_predict_params": dict(model_id=model_id, params=mocker.Mock()),
        "track_metric_value": dict(model_id=model_id, name=mocker.Mock(), value=mocker.Mock()),
        "track_model_info": dict(model_id=model_id, model=mocker.Mock()),
        "track_tag": dict(model_id=model_id, tag_key=mocker.Mock(), tag_value=mocker.Mock()),
        "track": dict(model_id=model_id, privet="poka"),
    }


def test_is_enabled():
    tracker = TravaTracker(scorers=[])
    assert tracker.is_enabled
    tracker.disable()
    assert not tracker.is_enabled
    tracker.enable()
    assert tracker.is_enabled


def test_add_scorers(mocker):
    initial_scorers = [mocker.Mock()]
    tracker = TravaTracker(scorers=initial_scorers)
    new_scorers = [mocker.Mock(), mocker.Mock()]
    tracker.add_scorers(scorers=new_scorers)

    assert tracker.scorers == (initial_scorers + new_scorers)


def test_if_enabled_true(mocker, model_id):
    tracker = TravaTracker(scorers=[])

    mocks = _mocks_for_subclass_calls(mocker=mocker, tracker=tracker, model_id=model_id)
    parameters = _parameters_for_methods(mocker=mocker, model_id=model_id)

    calls = [getattr(tracker, method_name)(**params) for method_name, params in parameters.items()]

    assert len(mocks) == len(calls)
    [mock.assert_called_once() for mock in mocks]


def test_if_enabled_false(mocker, model_id):
    tracker = TravaTracker(scorers=[])
    tracker.disable()

    mocks = _mocks_for_subclass_calls(mocker=mocker, tracker=tracker, model_id=model_id)
    parameters = _parameters_for_methods(mocker=mocker, model_id=model_id)

    calls = [getattr(tracker, method_name)(**params) for method_name, params in parameters.items()]

    assert len(mocks) == len(calls)
    [mock.assert_not_called() for mock in mocks]


def test_track_model_artifact(mocker, model_id):
    serializer = mocker.Mock()
    tracker = TravaTracker(scorers=[])
    model = mocker.Mock()
    tracker.track_model_artifact(model_id=model_id, model=model, serializer=serializer)

    serializer.save.assert_called_once_with(model=model, path=mocker.ANY)
