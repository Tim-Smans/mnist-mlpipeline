import streamlit as st
import numpy as np
import requests
import json
from PIL import Image, ImageOps

# MLflow API Endpoint
API_URL = "http://127.0.0.1/mlflow-model/invocations"  # Change this for Kubernetes

st.title("MNIST Digit Classifier")
st.write("Upload a digit (0-9) to classify it using the MLflow Model.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    """Preprocesses the image: Resize, Grayscale, Normalize, Invert"""
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to MNIST format
    image_array = np.array(image) / 255.0  # Normalize (0-1)
    return image_array.tolist()

if st.button("Predict"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        input_data = preprocess_image(image)
        payload = json.dumps({"instances": [input_data]})

        # Send request to MLflow API
        try:
            response = requests.post(API_URL, data=payload, headers={"Content-Type": "application/json", "Host": "mlflow-model.local"})
            prediction = response.json()
            
            # Extract prediction
            logits = prediction.get("predictions", [[]])[0]  # Get first prediction
            predicted_digit = np.argmax(logits)

            st.success(f"Predicted Digit: {predicted_digit}")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please upload an image first!")
