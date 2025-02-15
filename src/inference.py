from mlflow import MlflowClient
from ultralytics import YOLO


class Inference:
    def __init__(self):
        """
        This class is used to create inferences

        Args:
        """
        self.client = MlflowClient()

        champion_models = self.client.search_registered_models(
            filter_string="'Champion' IN aliases"
        )

        if champion_models == []:
            print("no available model for inference")
            return

        model_name = champion_models[-1].name
        model_version = champion_models[-1].aliases["Champion"].version

        download_uri = (
            self.client.get_model_version_download_uri(model_name, model_version)
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
    inf = Inference()

    inf.infer(type="IMAGE", path="../assets/20250108_151716.jpg")
