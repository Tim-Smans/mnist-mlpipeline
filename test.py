import requests
import json
import numpy as np

# API URL
API_URL = "http://9.223.144.137/predict"

# Simuleer een MNIST afbeelding (28x28 pixels, genormaliseerd)
mnist_sample = np.random.rand(28, 28).astype(np.float32).tolist()

# JSON payload
payload = json.dumps({"inputs": [mnist_sample]})

# Headers
headers = {"Content-Type": "application/json"}

# API Call
response = requests.post(API_URL, data=payload, headers=headers)

# Output resultaat
if response.status_code == 200:
    print("✅ Voorspelling ontvangen:", response.json())
else:
    print("❌ Fout:", response.status_code, response.text)
