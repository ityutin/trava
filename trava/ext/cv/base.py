from abc import ABC, abstractmethod


class CV(ABC):
    """
    You may want to write custom CV subclass for your specific needs.
    Or just use SklearnCV right away with the scikit-learn CV of your choice.
    """

    @abstractmethod
    def split(self, X, y=None, groups=None):
        """
        Scikit-learn style split method. Should be self-explanatory.
        """
        pass
