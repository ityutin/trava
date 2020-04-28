from abc import ABC, abstractmethod


class ModelSerializer(ABC):
    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def save(self, model, path: str):
        pass
