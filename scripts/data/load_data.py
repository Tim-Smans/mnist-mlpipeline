import argparse
import tensorflow as tf
import numpy as np
import boto3
import os

# Parsing arguments to output two datasets: one for training and one for testing
parser = argparse.ArgumentParser()
parser.add_argument('--output_dataset_train', type=str, required=True)
parser.add_argument('--output_dataset_test', type=str, required=True)
args = parser.parse_args()

# Set environment variables for MinIO access
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MINIO_ENDPOINT", "http://istio-ingressgateway.istio-system.svc.cluster.local")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("MINIO_ACCESS_KEY", "minio")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("MINIO_SECRET_KEY", "minio123")

# Configure MinIO Client
s3 = boto3.client(
    "s3",
    endpoint_url="http://istio-ingressgateway.istio-system.svc.cluster.local",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123",
    config=boto3.session.Config(signature_version='s3v4')
)

# Download files from MinIO
s3.download_file("ml-data", "mnist/train.npz", args.output_dataset_train)
s3.download_file("ml-data", "mnist/test.npz", args.output_dataset_test)

# Load dataset from NPZ format
def load_npz_dataset(filepath):
    with np.load(filepath) as data:
        images = data['images']
        labels = data['labels']
    return images, labels

train_images, train_labels = load_npz_dataset(args.output_dataset_train)
test_images, test_labels = load_npz_dataset(args.output_dataset_test)

# Normalize images to range [-1, 1]
train_images = (train_images.astype(np.float32) / 255.0) * 2 - 1
test_images = (test_images.astype(np.float32) / 255.0) * 2 - 1

# Save datasets using TensorFlow format
np.savez(args.output_dataset_train, images=train_images, labels=train_labels)
np.savez(args.output_dataset_test, images=test_images, labels=test_labels)

print("Datasets saved successfully in TensorFlow format")
