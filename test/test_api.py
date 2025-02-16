from pathlib import Path

import bentoml
import cv2
import numpy as np


def render_result(image: np.ndarray, result: dict) -> np.ndarray:
    """Renders the result on the image

    Args:
        image (np.ndarray): The image to render the result on
        result (dict): The result of the prediction

    Returns:
        np.ndarray: The image with the result rendered on it
    """
    for box in result["boxes"]:
        x1, y1, x2, y2 = map(int, box["xyxy"])
        label = box["class"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    return image


if __name__ == "__main__":
    url = input("Entrez l'URL du service : ")
    with bentoml.SyncHTTPClient(url) as client:
        result = client.predict(
            image=Path("test.jpg"),
        )
        print(result)
        image = cv2.imread("test.jpg")
        image = render_result(image, result)
        cv2.imshow(image)
