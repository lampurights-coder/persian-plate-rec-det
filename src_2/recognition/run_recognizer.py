import argparse
import os
import shutil
import logging
import cv2
import numpy as np

from src_2.recognition.detector import LicensePlateDetector
from src_2.recognition.recognizer import LicensePlateRecognition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plate_system")


class LicensePlateSystem():
    def __init__(self, detector_model, recognition_model):
        self.plate_detector = LicensePlateDetector(
            model_path=detector_model,
            output_dir='./cropped_plates'
        )
        self.plate_recognition = LicensePlateRecognition(
            model_path=recognition_model,
            character_dict_path="./dictionary_plate/dict90.txt"
        )
        self.results = []

    def __call__(self, car_img):
        """
        car_img can be:
          - path to image (str)
          - numpy array (BGR as read by cv2)
        Returns list of recognition results.
        """
        # Let the detector accept either path or numpy array (depends on implementation)
        cropped_plates = self.plate_detector.detect_plate(car_img)

        # Normalize output: detector may return file paths (strings) OR numpy arrays.
        plate_images = []
        for cp in cropped_plates:
            if isinstance(cp, str):
                if not os.path.exists(cp):
                    logger.warning("Cropped plate path does not exist: %s", cp)
                    continue
                img = cv2.imread(cp, cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning("Failed to read cropped plate image: %s", cp)
                    continue
                plate_images.append(img)
            elif isinstance(cp, np.ndarray):
                plate_images.append(cp)
            else:
                logger.warning("Unknown cropped plate type: %s (skipped)", type(cp))

        if not plate_images:
            logger.info("No valid cropped plate images to recognize.")
            return []

        # The recognizer expects a list of numpy images
        self.results = [res for res in self.plate_recognition.process_images(plate_images)]
        return self.results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector_model', default='./models/yolo11n_openvino_model/best.xml',
                        help='Path to detector model')
    parser.add_argument('--recognition_model', default='./models/rb_scaner.onnx',
                        help='Path to recognition model')
    parser.add_argument('--image_path', default='/home/arshia/Downloads/plate_images/dataset_free/recognition/images/1_01_R_20250101100000_frame12145_jpg.rf.8e926bcb098c9066764f2624d6cb5a0c.jpg',
                        help='path to image')
    args = parser.parse_args()

    plate_system = LicensePlateSystem(args.detector_model, args.recognition_model)

    try:
        # pass either a path or a loaded image - detector implementation decides
        ocr_text_results = plate_system(args.image_path)
        print("OCR results:", ocr_text_results)
    finally:
        # remove the temporary cropped_plates folder only if it exists
        out_dir = './cropped_plates'
        if os.path.exists(out_dir) and os.path.isdir(out_dir):
            try:
                shutil.rmtree(out_dir)
                logger.info("Removed temporary folder %s", out_dir)
            except Exception as e:
                logger.warning("Failed to remove %s: %s", out_dir, e)
