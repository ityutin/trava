from pickle import load, dump  # type: ignore
from trava.model_serializer import ModelSerializer


class PickleModelSerializer(ModelSerializer):
    def load(self, path: str):
        with open(path, "r") as file:
            return load(file)

    def save(self, model, path: str):
        with open(path, "wb") as file:
            return dump(model, file)
