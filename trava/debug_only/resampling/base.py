from abc import abstractmethod

from trava.logger import TravaLogger
from trava.split.config import SplitResultHandler
from trava.split.result import SplitResult


# noinspection PyPep8Naming
class Resampler(SplitResultHandler):
    """
    Basic interface for resampling logic.
    """
    def __init__(self, logger: TravaLogger):
        self._logger = logger

    def handle(self, split_result: SplitResult) -> SplitResult:
        X_train, y_train = self.resample(X=split_result.X_train, y=split_result.y_train)

        split_result.X_train = X_train
        split_result.y_train = y_train

        return split_result

    def resample(self, X, y):
        """
        Parameters
        ----------
        X: pandas dataframe
            Features for the model
        y: numpy array, pandas series
            Target for the model

        Returns
        -------
        Resamples X and y
        """
        self._logger.log(msg='Before resampling')
        self._log_data(y_train=y)
        self._logger.log(msg='\nAfter resampling')
        X, y = self._resample(X_train=X,
                              y_train=y)
        self._log_data(y_train=y)

        return X, y

    def _log_data(self, y_train):
        unique_labels = set(y_train)
        self._logger.log(msg='Number of unique labels: {}\n'.format(len(unique_labels)))

        for label in unique_labels:
            n_rows_for_label = str(len(y_train[y_train == label]))
            self._logger.log(msg='Number of rows for label {}: {}'.format(label, n_rows_for_label))

    @abstractmethod
    def _resample(self, X_train, y_train):
        pass
