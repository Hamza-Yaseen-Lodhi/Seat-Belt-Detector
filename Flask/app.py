import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Model path: Flask folder ke bahar Models folder hai
MODEL_PATH = os.path.join("..", "Models", "cnn_model.h5")
IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)

# Same class order as training
class_indices = {
    "Seatbelt": 0,
    "NoSeatbelt": 1
}

class_labels = {v: k for k, v in class_indices.items()}


def predict_image(file):
    image = Image.open(file).convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)

    # Convert prediction to 1D array
    preds = np.squeeze(preds)

    # Case 1: Softmax output like [0.95, 0.05]
    if preds.ndim > 0 and len(preds) > 1:
        class_id = int(np.argmax(preds))
        confidence = round(float(preds[class_id]) * 100, 2)
        prediction = class_labels[class_id]

    # Case 2: Sigmoid output like 0.85
    else:
        prob = float(preds)
        if prob >= 0.5:
            prediction = "NoSeatbelt"
            confidence = round(prob * 100, 2)
        else:
            prediction = "Seatbelt"
            confidence = round((1 - prob) * 100, 2)

    return prediction, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            prediction, confidence = predict_image(file)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)