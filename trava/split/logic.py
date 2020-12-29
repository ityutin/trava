from abc import ABC, abstractmethod


class SplitLogic(ABC):
    """
    Encapsulates logic of how to split data.
    """

    @abstractmethod
    def split(self, data, test_size: float, valid_size: float, **kwargs) -> tuple:
        """
        Splits the data

        Parameters
        ----------
        data: any data type your split logic supports
        test_size: float
            fraction of data to use as test set.
        valid_size: float
            fraction of data to use as validation set. Pass 0.0 if you don't want validation set.
        kwargs: dict
            Any additional parameters your class works with. Note: Init injection is more preferable.

        Returns
        -------
        Tuple of (train, test, validation) data
        """
        pass
