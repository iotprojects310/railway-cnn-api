from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# -----------------------
# FastAPI App
# -----------------------
app = FastAPI(title="Railway CNN Safety API")

# -----------------------
# Load Model & Scaler
# -----------------------
MODEL_PATH = "cnn_safe_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

scaler_mean = np.load("scaler_mean.npy")
scaler_scale = np.load("scaler_scale.npy")

WINDOW_SIZE = 60
THRESHOLD = 0.32519  # your trained anomaly threshold

# -----------------------
# Request Schema
# -----------------------
class SensorData(BaseModel):
    AccX: float
    AccY: float
    AccZ: float

# -----------------------
# Health Check
# -----------------------
@app.get("/")
def health():
    return {"status": "Railway CNN Safety API running"}

# -----------------------
# Prediction Endpoint
# -----------------------
@app.post("/predict")
def predict(data: SensorData):
    x = np.array([[data.AccX, data.AccY, data.AccZ]])
    x = (x - scaler_mean) / scaler_scale

    # Repeat to fake a 60-step window
    x = np.repeat(x, WINDOW_SIZE, axis=0)
    x = x.reshape(1, WINDOW_SIZE, 3)

    reconstructed = model.predict(x, verbose=0)
    mse = np.mean(np.square(x - reconstructed))

    return {
        "reconstruction_error": float(mse),
        "status": "danger" if mse > THRESHOLD else "safe"
    }
