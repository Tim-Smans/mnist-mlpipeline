import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the dataset
testset = torch.load("/app/test.pt")

# Select an image
image, label = testset[101]

# Denormalize the image
image = image * 0.5 + 0.5
image = transforms.ToPILImage()(image)

# Save the image
image.save("/app/mnist_sample.png")

# Upload to MinIO
import boto3
s3 = boto3.client(
    "s3",
    endpoint_url="http://9.223.154.101:9000",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123"
)

s3.upload_file("/app/mnist_sample.png", "ml-data", "mnist_sample.png")

print("Saved and uploaded sample image")
