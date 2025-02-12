import os
import shutil

from config import settings
from picsellia import Client
from tqdm import tqdm


class DataRetriver:
    def __init__(self, api_token: str, dataset_version_id: str) -> None:
        """
        This class is used to load the data from picsellia

        Args:
            api_token (str) : The identifier for the picsellia account
            dataset_version_id (str) : The identifier of the dataset
        """
        self.dataset_ver = Client(api_token=api_token).get_dataset_version_by_id(
            id=dataset_version_id
        )

    def load(self, annotation_folder: str, asset_folder: str) -> None:
        """
        This function is used to load and save localy the data (images + annotations)

        Args:
            annotations_folder (str) : The path where the annotations are stored
            assets_folder (str) : The path where the assets are stored
        """
        print("[INFO] Downloading annotations...")
        annotation_zip_path = self.dataset_ver.export_annotation_file("YOLO", "./")
        shutil.unpack_archive(annotation_zip_path, annotation_folder)

        print("[INFO] Fetching asset list...")
        assets = list(
            self.dataset_ver.list_assets()
        )  # Liste des fichiers à télécharger
        total_assets = len(assets)

        os.makedirs(asset_folder, exist_ok=True)

        print(f"[INFO] Downloading {total_assets} assets...")
        for asset in tqdm(assets, desc="Downloading assets", unit="file"):
            asset.download(asset_folder)

        print("[INFO] Download complete!")


if __name__ == "__main__":
    data_retriver = DataRetriver(
        api_token=settings.api_token,
        dataset_version_id=settings.dataset_version_id,
    )
    data_retriver.load(
        annotation_folder=settings.annotation_folder, asset_folder=settings.asset_folder
    )
