import boto3
import mlflow
from mlflow.tracking import MlflowClient

# MinIO config
s3 = boto3.client(
    "s3",
    endpoint_url="http://istio-ingressgateway.istio-system.svc.cluster.local",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123"
)
# Get the latest version of the mnist model in MLFlow model registry
MLFLOW_TRACKING_URI = "http://istio-ingressgateway.istio-system.svc.cluster.local/mlflow/"
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
model_name = "mnist-model"

latest_version = max(
    [v.version for v in client.search_model_versions(f"name='{model_name}'")]
)

latest_path = client.get_model_version_download_uri(model_name, latest_version)
latest_uuid = latest_path.split("/")[3] 

print(f"🔍 Latest path {latest_path}")

print(f"🔍 Latest uuid {latest_uuid}")

# MinIO Path for the newest version
source_prefix = f"{latest_uuid}/artifacts/mnist_model/"
latest_prefix = "latest/"

print(f"🔍 Expected Key in MinIO: ml-models/{latest_uuid}/artifacts/mnist_model/")


# Delete the old 'latest' directory
objects = s3.list_objects_v2(Bucket="ml-models", Prefix=latest_prefix)
if "Contents" in objects:
    for obj in objects["Contents"]:
        s3.delete_object(Bucket="ml-models", Key=obj["Key"])

# Copy all files of the latest version to the 'latest' directory
objects = s3.list_objects_v2(Bucket="ml-models", Prefix=source_prefix)
if "Contents" in objects:
    for obj in objects["Contents"]:
        src_key = obj["Key"]
        dest_key = src_key.replace(source_prefix, latest_prefix)
        print(f"Copying {src_key} → {dest_key}")
        s3.copy_object(Bucket="ml-models", CopySource={"Bucket": "ml-models", "Key": src_key}, Key=dest_key)


print(f"'latest' now points to {latest_uuid}")
