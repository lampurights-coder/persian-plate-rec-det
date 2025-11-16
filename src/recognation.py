
import math
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple

from utilis.preprocess_utilis import DecodeImage, RecResizeImg, KeepKeys
from utilis.postprocess_utilis import postprocess


class LicensePlateRecognition:
    """
    A unified class for license plate detection and recognition pipeline:
      - Preprocessing: decode image, resize, keep only necessary keys
      - Inference: run ONNX model for recognition
      - Postprocessing: decode output logits into text and confidence

    Usage:
        lpr = LicensePlateRecognition(model_path, char_dict_path)
        results = lpr.process_images(["plate1.jpg", "plate2.png"])
        # results -> [("ABC123", 0.98), ("XYZ789", 0.95)]
    """

    def __init__(
        self,
        model_path: str,
        character_dict_path: str,
        image_shape: List[int] = [3, 48, 320],
        providers: List[str] = ['CPUExecutionProvider']
    ):
        # Setup preprocessing operations
        self.ops = [
            DecodeImage(),
            RecResizeImg(image_shape=image_shape),
            KeepKeys(['image'])
        ]
        # Initialize ONNX inference session
        self.session = ort.InferenceSession(model_path, providers=providers)
        # Path for character dictionary
        self.character_dict_path = character_dict_path

    def _load_image(self, image_path: str) -> dict:
        """Read raw bytes and wrap into a dict key 'image'."""
        with open(image_path, 'rb') as f:
            content = f.read()
        return {"image": content}

    def _preprocess(self, image_paths: List[str]) -> List[np.ndarray]:
        """Apply preprocessing ops on raw image bytes."""
        preprocessed = []
        for path in image_paths:
            data = self._load_image(path)
            for op in self.ops:
                data = op(data)
            # op should output array under 'image'
            img_arr = np.array(data)
            preprocessed.append(img_arr)
        return preprocessed

    def _infer(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Run ONNX model inference for each preprocessed image."""
        outputs = []
        for arr in inputs:
            # assuming model expects input name 'x'
            res = self.session.run(None, {'x': arr})
            outputs.append(np.array(res[0]))
        return outputs

    def _postprocess(self, logits: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Decode logits into text and confidence score."""
        results = []
        for score_map in logits:
            text, conf = postprocess(score_map, self.character_dict_path)
            results.append((text, conf))
        return results

    def process_images(self, image_paths: List[str]) -> List[Tuple[str, float]]:
        """
        Complete end-to-end pipeline: preprocess, infer, postprocess.

        :param image_paths: list of file paths to plate images
        :return: list of tuples (recognized_text, confidence)
        """
        pre = self._preprocess(image_paths)
        logits = self._infer(pre)
        results = self._postprocess(logits)
        return results

if __name__ == '__main__':

    lpr = LicensePlateRecognition('/home/arshia/Downloads/projects/Paddle_conver_onnx/garbage/H/final_result_h/inference.onnx', '/home/arshia/Downloads/projects/Paddle_conver_onnx/PaddleOCR/ppocr/utils/dict/ppocrv5_dict.txt')
    for img in ['/home/arshia/Pictures/Screenshots/Screenshot from 2025-08-07 14-23-41.png']:
        text, conf = lpr.process_images([img])[0]
        print(f"{img}: {text} (confidence: {conf:.4f})")
