import boto3
import os

# MinIO settings
MINIO_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://istio-ingressgateway.istio-system.svc.cluster.local")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minio")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
BUCKET_NAME = "ml-models"  # Pas aan naar jouw bucket
MODEL_PREFIX = "latest/"  # Dit is de map waar je modelbestanden in zitten

# Maak een S3 client aan
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_URL,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

# Download de model directory bestand per bestand
local_model_dir = "/app/model"
os.makedirs(local_model_dir, exist_ok=True)

# List all objects in the model directory
objects = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=MODEL_PREFIX)

if "Contents" not in objects:
    print(f"❌ No files found in MinIO under {BUCKET_NAME}/{MODEL_PREFIX}")
    exit(1)

print(f"📥 Downloading model from MinIO: {BUCKET_NAME}/{MODEL_PREFIX}")

for obj in objects["Contents"]:
    file_key = obj["Key"]
    file_name = file_key.replace(MODEL_PREFIX, "", 1)  # Strip the prefix
    local_file_path = os.path.join(local_model_dir, file_name)

    # Create subdirectories if needed
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

    print(f"⬇️ Downloading {file_key} -> {local_file_path}")
    s3.download_file(BUCKET_NAME, file_key, local_file_path)

print("✅ Model ready at /app/model")
