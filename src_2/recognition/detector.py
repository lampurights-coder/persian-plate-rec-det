import cv2
import os
from typing import List
import numpy as np
from pathlib import Path
import openvino.runtime as ov
from utilis.open_vino_utilis import postprocess, preprocess_image

# Initialize OpenVINO core once
core = ov.Core()

class Model:
    """
    Wrapper for OpenVINO model inference.
    """
    def __init__(self, model_xml_path: str, device: str = 'AUTO'):
        model_path = Path(model_xml_path)
        ov_model = core.read_model(model_path)
        # Optionally reshape inputs here if needed
        self.compiled_model = core.compile_model(ov_model, device)

    def __call__(self, image: np.ndarray) -> dict:
        # Preprocess image for model input
        input_tensor = preprocess_image(image)
        # Run inference
        results = self.compiled_model(input_tensor)
        # Extract raw boxes from first output
        raw_boxes = results[self.compiled_model.output(0)]

        input_hw = input_tensor.shape[2:]
        # Postprocess into detection dict
        detections = postprocess(
            pred_boxes=raw_boxes,
            input_hw=input_hw,
            orig_img=image,
            number_cls=2
        )
        return detections

class LicensePlateDetector:
    """
    Detects and crops license plates from images using an OpenVINO YOLO model.
    """
    def __init__(
        self,
        model_path: str,
        output_dir: str = './cropped_plates',
        plate_label: int = 1
    ):
        self.detector = Model(model_path)
        self.plate_label = plate_label
        self.output_dir = Path(output_dir)
        

    def detect_plate(
        self,
        image_path: str,
        thresh: float = 0.3
    ) -> List[str]:
        """
        Detect license plates in an image, save each crop to disk, and return list of file paths.

        :param image_path: Path to input image
        :param thresh: Confidence threshold for detections
        :return: List of saved cropped image file paths
        :raises FileNotFoundError: If input image is missing
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Run detection
        results = self.detector(img)
        detections = results[0].get('det', [])  

        saved_paths: List[str] = []
        base_name = Path(image_path).stem

        for idx, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            if cls_id == self.plate_label and conf >= thresh:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                crop = img[y1:y2, x1:x2]
                save_name = f"{base_name}_plate_{idx}.jpg"
                save_path = self.output_dir / save_name
                cv2.imwrite(str(save_path), crop)
                saved_paths.append(str(save_path))

        return saved_paths


if __name__ == '__main__':

    detector = LicensePlateDetector(
        model_path='./models/yolo11n_openvino_model/best.xml',
        output_dir='./cropped_plates'
    )

    img_path = './plates/ye/rasamotor-gac-empow-12-1024x768.jpg'
    saved = detector.detect_plate(img_path, thresh=0.3)
    print(f"Cropped plates saved for {img_path}: {saved}")
    