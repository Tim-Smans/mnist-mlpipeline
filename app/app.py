import torch
import torch.nn as nn
import torch.optim as optim
import flask
from flask import request, jsonify
import numpy as np
import boto3
import os
from PIL import Image
import torchvision.transforms as transforms


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://9.223.80.153")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
MINIO_BUCKET = "ml-models"
MODEL_FILENAME = "trained_model.pth"
MODEL_LOCAL_PATH = f"{MODEL_FILENAME}"

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)

if not os.path.exists(MODEL_LOCAL_PATH):
    print(f"Downloading model from MinIO: {MODEL_FILENAME}...")
    s3.download_file(MINIO_BUCKET, MODEL_FILENAME, MODEL_LOCAL_PATH)
    print("Model downloaded successfully.")



s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)
    
# Model laden
model = MNISTModel()
model.load_state_dict(torch.load(MODEL_LOCAL_PATH, map_location=torch.device('cpu')))
model.eval()

app = flask.Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Pak het bestand uit de request
        file = request.files["file"]
        image = Image.open(file).convert("L")  # Converteer naar grayscale (MNIST is zwart-wit)

        # Transformeer de afbeelding
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = transform(image).unsqueeze(0)  # Voeg batch-dimensie toe

        # Voorspel met het model
        outputs = model(image)
        prediction = torch.argmax(outputs, dim=1).item()

        return jsonify({"prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
