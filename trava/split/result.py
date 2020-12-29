class SplitResult:
    def __init__(self, X_train, X_test, y_train, y_test, X_valid=None, y_valid=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_valid = X_valid
        self.y_valid = y_valid
