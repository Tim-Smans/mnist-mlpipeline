import argparse
import torch
import torchvision.transforms as transforms
import boto3
import os

# Parsing arguments this will output 2 datasets, 1 for training and 1 for testing
parser = argparse.ArgumentParser()
parser.add_argument('--output_dataset_train', type=str, required=True)
parser.add_argument('--output_dataset_test', type=str, required=True)
args = parser.parse_args()

# Configure MinIO Client
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://9.223.80.153/"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minio"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minio123"),
    config=boto3.session.Config(signature_version='s3v4')
)

# Download files from minio
s3.download_file("ml-data", "mnist/train.pt", args.output_dataset_train)
s3.download_file("ml-data", "mnist/test.pt", args.output_dataset_test)

# Apply transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load and transform data
trainset = torch.load(args.output_dataset_train, weights_only=False)
testset = torch.load(args.output_dataset_test, weights_only=False)

trainset.transform = transform
testset.transform = transform

# Save transformed datasets, using the arguments so that it will be sent to the next step in the pipeline.
torch.save(trainset, args.output_dataset_train)
torch.save(testset, args.output_dataset_test)