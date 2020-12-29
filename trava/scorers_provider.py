from abc import abstractmethod
from typing import List

from trava.scorer import Scorer


class ScorersProvider:
    """
    Just an interface for a source of scorers.
    """

    @property
    def provider_id(self) -> str:
        return type(self).__name__

    @property
    @abstractmethod
    def scorers(self) -> List[Scorer]:
        pass

    def metric_scorers(self) -> List[Scorer]:
        return [scorer for scorer in self.scorers if not scorer.is_other_scorer]

    def other_scorers(self) -> List[Scorer]:
        return [scorer for scorer in self.scorers if scorer.is_other_scorer]
