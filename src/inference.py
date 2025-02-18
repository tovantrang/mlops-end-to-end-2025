import click
import cv2
import mlflow
from config import settings
from dotenv import load_dotenv
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

        download_uri = self.client.get_model_version_download_uri(
            model_name, champion_id
        )
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=download_uri, dst_path="."
        )
        print(local_path)
        self.model = YOLO("best.pt")

    def infer(self, mode: str, path: str = ""):
        """
        This function is used to train the model

        Args:
            mode : string used to select the inference mode (IMAGE, VIDEO or WEBCAM)
            path : in the case of a IMAGE or VIDEO inference, the path to the obect to use for the inference
        """
        if mode not in ["IMAGE", "VIDEO", "WEBCAM"]:
            print("unreconised inference mode, must be IMAGE, VIDEO or WEBCAM")
            return

        if mode != "IMAGE":
            stream = True
        else:
            stream = False

        if mode == "WEBCAM":
            results = self.model.predict(0, stream=stream)
            for result in results:
                im_rgb = result.plot()  # Récupère l'image annotée
                cv2.imshow("YOLO Detection", im_rgb)  # Affiche dans la même fenêtre
                if cv2.waitKey(1) & 0xFF == ord("q"):  # Quitter avec 'q'
                    break
            cv2.destroyAllWindows()

        else:
            results = self.model.predict(path, stream=stream)
            results[0].show()
        return results


@click.command()
@click.option(
    "--mode", default="IMAGE", help="mode of the inference    (IMAGE, VIDEO, WEBCAM)"
)
@click.option("--input", default="test/test.jpg", help="Path of the input")
def main(mode, input):
    load_dotenv("src/config/.local.env")
    inf = Inference(model_name=settings.model_name)

    inf.infer(mode=mode, path=input)


if __name__ == "__main__":
    main()
