import pytest

from trava.split.config import DataSplitConfig
from trava.split.splitter import Splitter


def _config(mocker, valid_size=0.0):
    split_logic = mocker.Mock()
    split_logic.split.return_value = [mocker.MagicMock(), mocker.MagicMock(), mocker.MagicMock()]
    config = DataSplitConfig(
        split_logic=split_logic,
        target_col_name="target",
        test_size=0.3,
        valid_size=valid_size,
        split_result_handlers=[],
        ignore_cols=["col1", "col2"],
    )
    return config


@pytest.fixture(scope="class")
def additional_params():
    additional_params = {"param1": "value", "param2": 1}
    return additional_params


def test_split_logic(mocker, additional_params):
    df = mocker.Mock()
    config = _config(mocker=mocker)
    Splitter.split(df=df, config=config, **additional_params)

    config.split_logic.split.assert_called_once_with(
        data=df, test_size=config.test_size, valid_size=config.valid_size, **additional_params
    )


@pytest.mark.parametrize("n_handlers", [3])
def test_result_handlers(mocker, n_handlers):
    result_handlers = []
    config = _config(mocker=mocker)
    split_result = mocker.Mock()

    for _ in range(n_handlers):
        handler = mocker.Mock()
        handler.handle.return_value = split_result
        result_handlers.append(handler)

    config.split_result_handlers = result_handlers

    Splitter.split(df=mocker.Mock(), config=config)

    [handler.handle.assert_called_once() for handler in result_handlers]


@pytest.mark.parametrize("valid_size", [0.0, 0.2])
def test_split_X_y(mocker, additional_params, valid_size):
    df = mocker.Mock()
    config = _config(mocker=mocker, valid_size=valid_size)
    Splitter.split(df=df, config=config, **additional_params)

    train_df, test_df, valid_df = config.split_logic.split()
    train_df.drop.assert_called_once_with(config.ignore_cols + [config.target_col_name], axis=1)

    if valid_size != 0.0:
        valid_df.drop.assert_called_once_with(config.ignore_cols + [config.target_col_name], axis=1)
