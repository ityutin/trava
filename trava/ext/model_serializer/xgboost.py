import xgboost as xgb
import os

from trava.model_serializer import ModelSerializer


class XGBoostModelSerializer(ModelSerializer):
    def load(self, path: str):
        model = xgb.Booster()
        model.load_model(os.path.abspath(path))

    def save(self, model, path: str):
        model.save_model(path)
