import bentoml
from PIL import Image
from ultralytics import YOLO


@bentoml.service()
class YOLOService:
    def __init__(self) -> None:
        self.model = YOLO("best.pt")

    @bentoml.api
    async def predict(self, image: Image.Image) -> dict:
        """Predicts the class of the input image

        Args:
            image (Image.Image): _description_

        Returns:
            dict: _description_
        """
        results = self.model(image)
        result = results[0]

        class_names = self.model.model.names

        boxes = []
        for box in result.boxes:
            boxes.append(
                {"xyxy": box.xyxy[0].tolist(), "class": class_names[int(box.cls[0])]}
            )

        return {"boxes": boxes, "inference_time": float(result.speed["inference"])}
