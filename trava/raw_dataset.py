class RawDataset:
    """
    Data class representing the whole raw dataset
    that should be split somehow in the future.
    """

    def __init__(self, df, target_col_name: str):
        self.df = df
        self.target_col_name = target_col_name

        self._X = self.df.drop(self.target_col_name, axis=1)
        self._y = self.df[target_col_name]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y
