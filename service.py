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
        pass

    def __get_champion_uri__(self) -> str:
        """This function is used to get the champion model URI from the MLFlow server

        Returns:
            str: the champion model URI
        """
        return "pass"

    def __get_champion_model__(self) -> YOLO:
        """This function is used to get the champion modelfrom the MLFlow server

        Returns:
            YOLO: the champion model
        """
        model = YOLO(self.__get_champion_uri__())
        return model

    @bentoml.api
    def predict_image(self, image: np.ndarray) -> list[list[dict]]:
        """This function is used to predict the image

        Args:
            image : image to predict

        Returns:
            np.ndarray: the image with the prediction
        """
        model = self.__get_champion_model__()
        result = model.predict(image)
        return result

    @bentoml.api(batchable=True)
    def predict_video(self, video: list[np.ndarray]) -> list[list[dict]]:
        """This function is used to predict the video

        Args:
            video : video to predict

        Returns:
            np.ndarray: the video with the prediction
        """
        return [[{"pass": "pass"}]]

    @bentoml.api
    def predict_webcam(self) -> list[list[dict]]:
        """This function is used to predict the webcam

        Returns:
            np.ndarray: the webcam with the prediction
        """
        return [[{"pass": "pass"}]]

    @bentoml.api
    def render_image(
        self, image: np.ndarray, prediction: list[list[dict]]
    ) -> np.ndarray:
        """This function is used to render the image

        Args:
            image : image to render
            prediction : prediction to render

        Returns:
            np.ndarray: the image with the prediction
        """
        return [[{"pass": "pass"}]]

    @bentoml.api
    def render_video(
        self, video: list[np.ndarray], prediction: list[list[dict]]
    ) -> np.ndarray:
        """This function is used to render the video

        Args:
            video : video to render
            prediction : prediction to render

        Returns:
            np.ndarray: the video with the prediction
        """
        return np.ndarray()

    @bentoml.api
    def render_webcam(self, prediction: list[list[dict]]) -> np.ndarray:
        """This function is used to render the webcam

        Args:
            prediction : prediction to render

        Returns:
            np.ndarray: the webcam with the prediction
        """
        return np.ndarray()
