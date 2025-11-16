from src.detector import LicensePlateDetector
from src.recognation import LicensePlateRecognition
import argparse
import shutil

class LicensePlateSystem():
    def __init__(self, 
                 detector_model,
                 recogantion_model,
                 ):
        self.plate_detector = LicensePlateDetector(
            model_path = detector_model,          
            output_dir='./cropped_plates'
        )

        self.plate_recogantion =LicensePlateRecognition(
            model_path = recogantion_model,
            character_dict_path="./dictionary_plate/plate_dictionary.txt"
        )

        self.results = []

    def __call__(self, car_img):

        crooped_plates = self.plate_detector.detect_plate(car_img)

        self.results = [self.plate_recogantion.process_images([cp])[0] for cp in crooped_plates]

        return self.results
           

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument('--detector_model', default='./models/yolo11n_openvino_model/best.xml', 
                        help='Path to detector model')
    
    parser.add_argument('--recognation_model', default='./models/rec_ppocrv5.onnx', 
                        help='Path to recoagntion model')
    
    parser.add_argument('--image_path', default='/home/arshia/Downloads/projects/cars/e9dfb202-dfe6-4614-83a3-984750b89684.jpeg', 
                        help='path to image')
    
    args = parser.parse_args()

    plate_det_rec = LicensePlateSystem(args.detector_model, args.recognation_model)
    
    ocr_text_results = plate_det_rec(args.image_path)

    shutil.rmtree('./cropped_plates')

    print(ocr_text_results)
