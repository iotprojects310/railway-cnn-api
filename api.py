from fastapi import FastAPI
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load model and scaler
model = tf.keras.models.load_model("cnn_safe_model.keras")
scaler_mean = np.load("scaler_mean.npy")
scaler_scale = np.load("scaler_scale.npy")

@app.get("/")
def root():
    return {"status": "CNN Railway Safety API is running"}

@app.post("/predict")
def predict(data: dict):
    x = np.array([[data["AccX"], data["AccY"], data["AccZ"]]])
    x = (x - scaler_mean) / scaler_scale
    x = x.reshape(1, 1, 3)

    score = model.predict(x)[0][0]
    return {
        "score": float(score),
        "status": "danger" if score > 0.5 else "safe"
    }

