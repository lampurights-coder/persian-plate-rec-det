<p align="center">
  <img src="ChatGPT Image Aug 6, 2025, 05_17_04 PM.png" width="500" />
</p>

Absolutely! Below is a **complete, professional-grade README** for your License Plate Recognition project, including:

* Project Overview
* Configuration
* Pipeline Explanation
* Quick Start
* API Usage
* Local Script
* Folder Structure
* Requirements
* License

---

# 🚘 License Plate Recognition System

A full pipeline for detecting and recognizing license plates from car images using:

* **YOLOv5 (OpenVINO)** for fast and lightweight detection
* **PaddleOCR PPOCRv5 (ONNX)** for accurate plate text recognition
* **FastAPI** backend for real-time inference over HTTP

---

## 📌 Features

✅ YOLO-based license plate detection (OpenVINO)
✅ PPOCRv5 ONNX model for text recognition
✅ Easy-to-use FastAPI server
✅ Modular design (detector, recognizer, system wrapper)
✅ Clean image preprocessing and postprocessing utilities

---

## ⚙️ Configuration

Place your models and dictionary as follows:

```
project/
├── models/
│   ├── yolo11n_openvino_model/
│   │   └── best.xml         # OpenVINO YOLOv5 detection model
│   └── rec_ppocrv5.onnx     # ONNX PPOCRv5 recognition model
├── dictionary_plate/
│   └── plate_dictionary.txt # Character dictionary used for decoding text
```

You can use your own models if trained differently, just make sure:

* Detection model outputs bounding boxes in YOLO format
* Recognition model accepts resized license plate crops and outputs logits
* Dictionary matches the characters the recognizer was trained on

---

## 🔁 Project Pipeline

```
            Input Image (.jpg, .png)
                      ↓
       [1] License Plate Detection (YOLOv5, OpenVINO)
                      ↓
          Cropped License Plate Images
                      ↓
    [2] Plate Text Recognition (PPOCRv5, ONNX)
                      ↓
      Output: Text + Confidence per Plate
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```
opencv-python
numpy
onnxruntime
openvino
fastapi
uvicorn
```

### 2. Run the API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 9090 --reload
```

---

## 🧪 Local Inference (No API)

You can also run the system locally from a script:

```bash
python src/system.py \
  --image_path /path/to/car.jpg \
  --detector_model ./models/yolo11n_openvino_model/best.xml \
  --recognation_model ./models/rec_ppocrv5.onnx
```

---

## 🌐 API Usage

### Endpoint

```
POST /process-image
```

### Request (via cURL or Postman)

```bash
curl -X POST http://localhost:9090/process-image \
  -F "file=@/path/to/car.jpg"
```

### Example JSON Response

```json
{
  "results": [
    {
      "text": "IR123XYZ",
      "confidence": 0.9823
    }
  ]
}
```

---

## 📁 Project Structure

```
.
├── app.py                            # FastAPI server
├── src/
│   ├── system.py                     # Full detection + recognition pipeline
│   ├── detector.py                   # OpenVINO-based plate detector
│   ├── recognation.py                # ONNX-based text recognizer
├── utilis/
│   ├── preprocess_utilis.py          # Image decoding, resizing, etc.
│   ├── postprocess_utilis.py         # Text decoding and score mapping
├── models/                           # YOLO + OCR models
├── dictionary_plate/                 # Character dictionary for recognition
├── cropped_plates/                   # Auto-generated during detection
├── requirements.txt
```

---

## 🧾 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Arshia** — Made with ❤️ for real-time intelligent plate reading applications.

---

Would you like me to generate the `requirements.txt` file or a sample `.env` if you're planning on deploying this?
