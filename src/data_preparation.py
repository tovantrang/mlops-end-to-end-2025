import os
import random
import shutil

import yaml

from src.config.config import settings


class DataPreparator:
    def __init__(
        self, annotations_folder: str, assets_folder: str, dataset_folder: str
    ) -> None:
        """
        This class is used to prepare the data

        Args:
            annotations_folder (str) : The path where the annotations are stored
            assets_folder (str) : The path where the assets are stored
            dataset_folder (str) : The path where the dataset is stored
        """
        self.annotations_folder = annotations_folder
        self.assets_folder = assets_folder
        self.dataset_folder = dataset_folder

    def prepare(self) -> None:
        """
        This function is used to prepare the data
        """
        self.__structure_creation__()
        self.__prepare_yaml_file__()
        self.__split_data__()

    def __structure_creation__(self) -> None:
        """
        This function is used to create the structure of the dataset
        """
        os.makedirs(self.dataset_folder, exist_ok=True)
        for folder in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.dataset_folder, folder), exist_ok=True)
            os.makedirs(
                os.path.join(self.dataset_folder, folder, "images"), exist_ok=True
            )
            os.makedirs(
                os.path.join(self.dataset_folder, folder, "labels"), exist_ok=True
            )

    def __prepare_yaml_file__(self) -> None:
        """
        This function is used to prepare the yaml file for yolo
        """
        with open(os.path.join(self.annotations_folder, "data.yaml"), "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            nc = data["nc"]
            names = data["names"]

        with open(os.path.join(self.dataset_folder, "data.yaml"), "w") as file:
            yaml.dump(
                {
                    "nc": nc,
                    "names": names,
                    "train": "train",
                    "val": "val",
                    "test": "test",
                },
                file,
            )

    def __split_data__(self) -> None:
        """
        This function is used to split the data into train, val and test.

        Seed 42 is used for reproducibility.
        """
        VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

        images = os.listdir(self.assets_folder)
        images = [image for image in images if image.lower().endswith(VALID_EXTENSIONS)]

        random.seed(42)
        random.shuffle(images)

        train_images = images[: 70 * len(images) // 100]
        val_images = images[70 * len(images) // 100 : 85 * len(images) // 100]
        test_images = images[85 * len(images) // 100 :]

        for image in train_images:
            shutil.copy2(
                os.path.join(self.assets_folder, image),
                os.path.join(self.dataset_folder, "train", "images", image),
            )
            shutil.copy2(
                os.path.join(
                    self.annotations_folder, os.path.splitext(image)[0] + ".txt"
                ),
                os.path.join(
                    self.dataset_folder,
                    "train",
                    "labels",
                    os.path.splitext(image)[0] + ".txt",
                ),
            )

        for image in val_images:
            shutil.copy2(
                os.path.join(self.assets_folder, image),
                os.path.join(self.dataset_folder, "val", "images", image),
            )
            shutil.copy2(
                os.path.join(
                    self.annotations_folder, os.path.splitext(image)[0] + ".txt"
                ),
                os.path.join(
                    self.dataset_folder,
                    "val",
                    "labels",
                    os.path.splitext(image)[0] + ".txt",
                ),
            )

        for image in test_images:
            shutil.copy2(
                os.path.join(self.assets_folder, image),
                os.path.join(self.dataset_folder, "test", "images", image),
            )
            shutil.copy2(
                os.path.join(
                    self.annotations_folder, os.path.splitext(image)[0] + ".txt"
                ),
                os.path.join(
                    self.dataset_folder,
                    "test",
                    "labels",
                    os.path.splitext(image)[0] + ".txt",
                ),
            )


if __name__ == "__main__":
    data_preparator = DataPreparator(
        annotations_folder=settings.annotation_folder,
        assets_folder=settings.asset_folder,
        dataset_folder=settings.dataset_folder,
    )
    data_preparator.prepare()
