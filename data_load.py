import shutil
from config import settings
from picsellia import Client


class DataRetriver:
    def __init__(
        self,
        api_token: str,
        dataset_version_id: str,
        annotations_folder: str,
        assets_folder: str,
    ) -> None:
        """
        This class is used to retrive the data from the Picsellia platform

        Args:
            api_token (str) : The api token of the user
            dataset_version_id (str) : The dataset version id
            annotations_folder (str) : The path where the annotations will be stored
            assets_folder (str) : The path where the assets will be stored
        """
        self.dataset_ver = Client(api_token=api_token).get_dataset_version_by_id(
            id=dataset_version_id
        )

        annotation_zip_path = self.dataset_ver.export_annotation_file("YOLO", "./")
        shutil.unpack_archive(annotation_zip_path, annotations_folder)

        self.dataset_ver.download(assets_folder)


if __name__ == "__main__":
    data_retriver = DataRetriver(
        api_token=settings.api_token,
        dataset_version_id=settings.dataset_version_id,
        annotations_folder="./annotations",
        assets_folder="./assets",
    )
