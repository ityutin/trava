import pickle
from trava.model_serializer import ModelSerializer


class PickleModelSerializer(ModelSerializer):
    def load(self, path: str):
        with open(path, 'r') as file:
            return pickle.load(file)

    def save(self, model, path: str):
        with open(path, 'wb') as file:
            return pickle.dump(model, file)
