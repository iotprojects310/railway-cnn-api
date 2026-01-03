from fastapi import FastAPI, HTTPException, status
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
# Lazy-load model and scaler files. If files are missing or invalid,
# keep `model`/`scaler_*` as None so the app can still start and
# serve the health endpoint.
model = None
scaler_mean = None
scaler_scale = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Warning: could not load model: {e}")

try:
    scaler_mean = np.load("scaler_mean.npy")
    scaler_scale = np.load("scaler_scale.npy")
except Exception as e:
    print(f"Warning: could not load scaler files: {e}")

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
    if model is None or scaler_mean is None or scaler_scale is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Model or scaler not loaded")

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
