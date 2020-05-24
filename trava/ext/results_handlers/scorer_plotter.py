from abc import ABC, abstractmethod

from trava.metric import Metric


class ScorerPlotter(ABC):
    @abstractmethod
    def plot(self, metric: Metric, fig, ax, color: str, label: str):
        pass
