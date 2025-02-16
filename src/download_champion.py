import mlflow
from config import settings
from mlflow import MlflowClient


def download_weights():
    """Download the weights of the champion model"""
    client = MlflowClient()

    registered_models = client.search_registered_models()
    aliases = [res.aliases for res in registered_models][0]
    if "Champion" not in aliases:
        print("no available model for inference")
        return

    champion_id = aliases["Champion"]

    download_uri = client.get_model_version_download_uri(
        settings.model_name, champion_id
    )
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=download_uri, dst_path="."
    )
    print(local_path)


if __name__ == "__main__":
    download_weights()
