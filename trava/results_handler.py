from abc import ABC, abstractmethod
from typing import List

from trava.logger import TravaLogger
from trava.model_results import ModelResult
from trava.scorer import Scorer
from trava.scorers_provider import ScorersProvider
from trava.tracker import Tracker


class ResultsHandler(ABC, ScorersProvider):
    """
    Provides logic how to process and show calculated metrics to a user.
    """

    def __init__(self, scorers: List[Scorer]):
        """
        Parameters
        ----------
        scorers: list
            Scorers list we want to process and show.
        """
        self._scorers = scorers.copy()

    @property
    def scorers(self) -> List[Scorer]:
        return self._scorers

    @abstractmethod
    def handle(self, results: List[ModelResult], logger: TravaLogger, tracker: Tracker):
        """
        Implement the logic here.

        Parameters
        ----------
        results: list of ModelResult
            Contains metrics for all the models that were trained using TraVa so far.
        logger: TravaLogger
            If you want to log something along the way
        tracker: Tracker
            If you want to tracker something along the way

        Returns
        -------
        Your handler is able to return anything you want.
        """
        pass
