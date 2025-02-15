import json
import os

import bentoml
import mlflow
import numpy as np
from dynaconf import settings
from ultralytics import YOLO

# env
os.environ["MLFLOW_TRACKING_URI"] = settings.MLFLOW_TRACKING_URI
os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = settings.MLFLOW_DEFAULT_ARTIFACT_ROOT
os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"] = settings.AWS_DEFAULT_REGION


mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)  # pour avoir mlflow


@bentoml.service(resources={"gpu": 1})
class YOLOService:
    def __init__(self) -> None:
        self.client = mlflow.tracking.MlflowClient()
        self.model = YOLO(self.__get_champion_uri__())

    def __get_champion_uri__(self) -> str:
        """This function is used to get the champion model URI from the MLFlow server

        Returns:
            str: the champion model URI
        """
        champion_models = self.client.search_registered_models(
            filter_string="'Champion' IN aliases"
        )

        model_name = champion_models[-1].name
        model_version = champion_models[-1].aliases["Champion"].version

        download_uri = (
            self.client.get_model_version_download_uri(model_name, model_version)
            + "/weights/best.pt"
        )
        return download_uri

    @bentoml.api
    def predict_image(self, image: np.ndarray) -> list[dict]:
        """This function is used to predict the image

        Args:
            image : image to predict

        Returns:
            list[dict]: the prediction
        """
        result = self.model.predict(image)
        return json.loads(result.tojson())
