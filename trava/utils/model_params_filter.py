import inspect
import numbers


def merge_given_params_with_default(object_type, params: dict) -> dict:
    """
    Merges default params for object_type with params given by a user
    """
    all_params = dict(params)
    for key, param in inspect.signature(object_type.__init__).parameters.items():
        if not params.get(key) and param.default != param.empty:
            all_params[key] = param.default

    result = filter_params(params=all_params)
    return result


def filter_params(params: dict) -> dict:
    result = {}
    for key, value in params.items():
        if isinstance(value, _allowed_params_types()):
            result[key] = value

    return result


def _allowed_params_types() -> tuple:
    """
    Types for a model's init params that Trava is able to track.
    """
    return (
        numbers.Number,
        str,
        bool,
    )
