from fastapi import FastAPI, UploadFile, File, HTTPException
import asyncio
import os
import shutil
import re
from src_2.recognition.run_recognizer import LicensePlateSystem

app = FastAPI(title="License Plate Recognition API", version="1.0")

# Allowed single characters in the middle of the plate
# ALLOWED_CHARS = "BJDSṢṬQLMVHNYAPTṮZŽŠOFKG$&#"

# Compile regex for performance: 
# Matches 2 digits, then one allowed char, then 5 digits
# PATTERN = re.compile(rf"^\d{{2}}[{ALLOWED_CHARS}]\d{{5}}$")

@app.on_event("startup")
async def startup_event():
    app.state.plate_det_rec = LicensePlateSystem(
        "./models/yolo11n_openvino_model/best.xml",
        "./models/rb_scaner.onnx"  
    )

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        results = await asyncio.to_thread(app.state.plate_det_rec, temp_path)

        # Clean up files
        os.remove(temp_path)
        shutil.rmtree('./cropped_plates', ignore_errors=True)

        # Filter results based on pattern match
        filtered_results = []
        for text, conf in results:
            # if PATTERN.match(text):
            if conf >= 0.8:
                filtered_results.append({"text": text, "confidence": float(conf)})

        return {"results": filtered_results}

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=9090, reload=True)
