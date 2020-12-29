from trava.ext.cv.base import CV


class SklearnCV(CV):
    """
    Dummy wrapper for sklearn cross validation.
    """

    def __init__(self, sklearn_cv):
        self._sklearn_cv = sklearn_cv

    def split(self, X, y=None, groups=None):
        return self._sklearn_cv.split(X=X, y=y, groups=groups)
