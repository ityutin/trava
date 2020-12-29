from logging import Logger, INFO
from typing import Optional


class TravaLogger:
    """
    Wrapper of built-in python logger to have a control over it in future.
    """

    def __init__(self, logger: Optional[Logger] = None):
        self._logger = logger

    def log(self, msg: str, level=INFO):
        if self._logger:
            self._logger.log(msg=msg, level=level)
