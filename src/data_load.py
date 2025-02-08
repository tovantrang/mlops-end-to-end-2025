import shutil

from picsellia import Client

from src.config.config import settings


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
        annotation_zip_path = self.dataset_ver.export_annotation_file("YOLO", "./")
        shutil.unpack_archive(annotation_zip_path, annotation_folder)

        self.dataset_ver.download(asset_folder)


if __name__ == "__main__":
    data_retriver = DataRetriver(
        api_token=settings.api_token,
        dataset_version_id=settings.dataset_version_id,
    )
    data_retriver.load(
        annotation_folder=settings.annotation_folder, asset_folder=settings.asset_folder
    )
