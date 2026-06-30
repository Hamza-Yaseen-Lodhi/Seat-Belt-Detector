import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model

app = FastAPI(
    title="Seat Belt Detection API",
    description=(
        "A FastAPI backend service for CNN-based Seat Belt vs No Seat Belt "
        "image classification. Upload a driver or passenger image and receive "
        "the prediction result with confidence score in JSON format."
    ),
    version="1.0.0"
)

MODEL_PATH = os.path.join("..", "Models", "cnn_model.h5")
IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)

class_indices = {
    "Seatbelt": 0,
    "NoSeatbelt": 1
}

class_labels = {v: k for k, v in class_indices.items()}


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.get("/", tags=["Home"])
def home():
    return {
        "message": "Seat Belt Detection API is running successfully.",
        "documentation": "Open /docs to test the prediction endpoint.",
        "model_type": "CNN Image Classification",
        "classes": ["Seatbelt", "NoSeatbelt"]
    }


@app.post("/predict", tags=["Prediction"])
async def predict_seatbelt(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    preds = model.predict(img_array, verbose=0)
    preds = np.squeeze(preds)

    if preds.ndim > 0 and len(preds) > 1:
        class_id = int(np.argmax(preds))
        prediction = class_labels[class_id]
        confidence = round(float(preds[class_id]) * 100, 2)
    else:
        prob = float(preds)
        if prob >= 0.5:
            prediction = "NoSeatbelt"
            confidence = round(prob * 100, 2)
        else:
            prediction = "Seatbelt"
            confidence = round((1 - prob) * 100, 2)

    return JSONResponse({
        "filename": file.filename,
        "prediction": prediction,
        "confidence": confidence
    })