import numpy as np
import requests
import json
from PIL import Image

# Load the image (ensure it's 28x28 grayscale)
image_path = "mnist_image.jpg"  # Change to your actual image path
img = Image.open(image_path).convert("L")  # Convert to grayscale
img = img.resize((28, 28))  # Resize to 28x28

# Convert to numpy array and normalize (0-1 range)
img_array = np.array(img) / 255.0

# Convert to list and wrap in JSON format
payload = json.dumps({"instances": [img_array.tolist()]})

# Send the request
url = "http://127.0.0.1/mlflow-model/invocations"
headers = {"Content-Type": "application/json", "Host": "mlflow-model.local" }
response = requests.post(url, data=payload, headers=headers)

# Print response
print("Model Prediction:", response.json())


logits = [-0.22823435068130493, -4.134037017822266, -2.0648386478424072, 5.149596691131592, -11.72005558013916, 8.128214836120605, -4.749321460723877, -5.237892150878906, 1.4965994358062744, 0.16873927414417267]

# De voorspelde digit is de index met de hoogste waarde
predicted_digit = np.argmax(logits)

print("Model voorspelling:", predicted_digit)