import numpy as np
import pandas as pd

from trava.debug_only.resampling.base import Resampler


class BasicUnderSampler(Resampler):
    """
    Very basic implemenation of random resampling. Not recommended for use.
    """
    def _resample(self, X_train, y_train, random_state: int = 42):
        y_unique = list(set(y_train))
        assert len(y_unique) == 2, "Only binary cases are currently supported"

        y_class_one = y_unique[0]
        y_class_two = y_unique[1]

        class_one_mask = y_train == y_class_one
        class_two_mask = y_train == y_class_two

        X_one = X_train[class_one_mask]
        y_one = y_train[class_one_mask]
        X_two = X_train[class_two_mask]
        y_two = y_train[class_two_mask]

        n_class_one = len(X_one)
        n_class_two = len(X_two)

        if n_class_one > n_class_two:
            X_majority, y_majority, X_minority, y_minority = X_one, y_one, X_two, y_two
        else:
            X_majority, y_majority, X_minority, y_minority = X_two, y_two, X_one, y_one

        def sample(data, n: int, random_state: int):
            if isinstance(data, np.ndarray):
                np.random.seed(random_state)
                result = np.random.choice(data, n)
            elif isinstance(X_train, pd.DataFrame):
                result = data.sample(n, random_state=random_state)
            else:
                raise ValueError('Unknown data type {}'.format(type(data).__name__))

            return result

        def concat(data: list):
            if isinstance(data[0], np.ndarray):
                result = np.hstack(data)
            else:
                result = pd.concat(data, axis=0, copy=False, sort=False)

            return result

        X_sampled = sample(data=X_majority, n=len(X_minority), random_state=random_state)
        y_sampled = sample(data=y_majority, n=len(X_minority), random_state=random_state)

        X_train = concat([X_minority, X_sampled])
        y_train = concat([y_minority, y_sampled])

        return X_train, y_train
