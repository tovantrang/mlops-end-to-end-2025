import os

import yaml
from config import settings
from PIL import Image
from tqdm import tqdm


class DataValidator:
    def __init__(self, annotations_folder: str, assets_folder: str) -> None:
        """
        This class is used to validate the data

        Args:
            annotations_folder (str) : The path where the annotations are stored
            assets_folder (str) : The path where the assets are stored
        """
        self.annotations_folder = annotations_folder
        self.assets_folder = assets_folder
        self.max_class_index = self.__get_num_classes__()

    def validate(self):
        """This function is used to validate the data

        Raises:
            FileNotFoundError: If the annotations folder is not found
            FileNotFoundError: If the assets folder is not found
            ValueError: If the box is out of bounds
            ValueError: If the label is out of bounds
        """
        if not os.path.exists(self.annotations_folder):
            raise FileNotFoundError(
                f"Annotations folder not found at {self.annotations_folder}"
            )

        if not os.path.exists(self.assets_folder):
            raise FileNotFoundError(f"Assets folder not found at {self.assets_folder}")

        images_names = os.listdir(self.assets_folder)
        for image_name in tqdm(images_names):
            if not image_name.endswith(".jpg"):
                continue
            image = Image.open(os.path.join(self.assets_folder, image_name))
            annotation_name = image_name.replace(".jpg", ".txt")
            boxes, labels = self.__get_boxes_and_labels__(annotation_name)
            for box in boxes:
                if not self.__box_in_image__(box, image):
                    raise ValueError(
                        f"Box {box} in image {image_name} is out of bounds"
                    )
            for label in labels:
                if label < 0 or label >= self.max_class_index:
                    raise ValueError(
                        f"Label {label} in image {image_name} is out of bounds"
                    )
        print("Data validation successful")

    def __get_boxes_and_labels__(
        self, annotation_name: str
    ) -> tuple[list[list[float]], list[int]]:
        """This function is used to get the boxes and labels of the image

        Args:
            annotation_name (str) : The name of the image

        Returns:
            tuple[list[list[float]], list[int]] : The boxes and labels of the image
        """
        with open(os.path.join(self.annotations_folder, annotation_name), "r") as f:
            lines = f.readlines()
            split_lines = [line.strip().split() for line in lines]
            boxes = []
            labels = []
            for line in split_lines:
                labels.append(int(line[0]))
                boxes.append([float(x) for x in line[1:]])
            return boxes, labels

    def __box_in_image__(self, box: list, image: Image) -> bool:
        """This function is used to check if the box is in the image

        Args:
            box (list) : The box
            image (Image) : The image

        Returns:
            bool : If the box is in the image
        """
        width, height = image.size
        x_center, y_center, box_width, box_height = box
        x_min = x_center - box_width / 2
        x_max = x_center + box_width / 2
        y_min = y_center - box_height / 2
        y_max = y_center + box_height / 2
        if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
            return False
        return True

    def __get_num_classes__(self) -> int:
        """This function is used to get the number of classes

        Returns:
            int : The number of classes
        """
        with open(os.path.join(self.annotations_folder, "data.yaml"), "r") as f:
            data = yaml.safe_load(f)
            return data["nc"]


if __name__ == "__main__":
    data_validator = DataValidator(
        annotations_folder=settings.annotation_folder,
        assets_folder=settings.asset_folder,
    )
    data_validator.validate()
