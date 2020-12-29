import numbers


class Metric:
    """
    Data class that contains result of some scorer's calculation.
    """

    def __init__(self, name: str, value, model_id: str):
        """
        Parameters
        ----------
        name: str
            What is the name of metric's function
        value:
            Value that we got after calling a scorer.
        model_id: str
            Model unique identifier, will be used for saving metrics etc
        """
        self.name = name
        self.is_scalar = isinstance(value, numbers.Number)
        self.value = value
        self.model_id = model_id
