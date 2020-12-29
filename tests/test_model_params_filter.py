import numbers

from trava.utils.model_params_filter import merge_given_params_with_default, filter_params, _allowed_params_types


def test_merge_given_params():
    class TestObject:
        def __init__(self, a: bool, b: str = "preved", c: int = 10, d: tuple = ()):
            pass

    params = {"a": True, "c": 999}

    result = merge_given_params_with_default(object_type=TestObject, params=params)

    true_result = {"a": True, "b": "preved", "c": 999}

    assert result == true_result


def test_filter_params():
    params = {"a": 1, "b": "b", "c": [1, 2, 3], "d": Exception, "e": test_filter_params, "f": 123.0}

    result = filter_params(params=params)

    true_params = {"a": 1, "b": "b", "f": 123.0}

    assert result == true_params


def test_allowed_init_params_types():
    test_init_params_types = (
        numbers.Number,
        str,
        bool,
    )
    assert _allowed_params_types() == test_init_params_types
