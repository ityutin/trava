from typing import List

from trava.split.config import DataSplitConfig
from trava.split.result import SplitResult


# noinspection PyPep8Naming
class Splitter:
    @staticmethod
    def split(df, config: DataSplitConfig, **kwargs):
        split_logic = config.split_logic

        train_df, test_df, valid_df = split_logic.split(
            data=df, test_size=config.test_size, valid_size=config.valid_size, **kwargs
        )

        def split_X_y(source_df):
            return Splitter._split_X_y(
                df=source_df, ignore_cols=config.ignore_cols, target_col_name=config.target_col_name
            )

        # TODO: change cols to indices
        X_train, y_train = split_X_y(source_df=train_df)

        X_valid, y_eval = None, None
        if valid_df is not None:
            X_valid, y_eval = split_X_y(source_df=valid_df)

        X_test, y_test = split_X_y(source_df=test_df)

        result = SplitResult(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, X_valid=X_valid, y_valid=y_eval
        )

        for handler in config.split_result_handlers:
            result = handler.handle(split_result=result)

        return result

    @staticmethod
    def _split_X_y(df, ignore_cols: List[str], target_col_name: str):
        X, y = df.drop(ignore_cols + [target_col_name], axis=1), df[target_col_name]
        return X, y
