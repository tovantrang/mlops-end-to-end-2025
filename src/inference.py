from config import settings
from mlflow import MlflowClient
from ultralytics import YOLO


class Inference:
    def __init__(self, model_name: str) -> None:
        """
        This class is used to create inferences

        Args:
            model_name : name of the model to use for the inference
        """
        self.client = MlflowClient()

        registered_models = self.client.search_registered_models()
        aliases = [res.aliases for res in registered_models][0]
        if "Champion" not in aliases:
            print("no available model for inference")
            return

        champion_id = aliases["Champion"]

        download_uri = (
            self.client.get_model_version_download_uri(model_name, champion_id)
            + "/weights/best.pt"
        )

        self.model = YOLO(download_uri)

    def infer(self, type: str, path: str = ""):
        """
        This function is used to train the model

        Args:
            type : string used to select the inference mode (IMAGE, VIDEO or WEBCAM)
            path : in the case of a IMAGE or VIDEO inference, the path to the obect to use for the inference
        """
        if type not in ["IMAGE", "VIDEO", "WEBCAM"]:
            print("unreconised inference type, must be IMAGE, VIDEO or WEBCAM")
            return

        if type != "IMAGE":
            stream = True
        else:
            stream = False

        if type == "WEBCAM":
            results = self.model(0, stream=stream)
        else:
            results = self.model(path, stream=stream)

        return results


if __name__ == "__main__":
    inf = Inference(model_name=settings.model_name)

    inf.infer(type="IMAGE", path="../assets/20250108_151716.jpg")
