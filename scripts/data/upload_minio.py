import boto3

# Configuring the MinIO client
s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123"
)

# Uploading the files
# line 1 -> Where the file is located locally
# line 2 -> The name of the bucket to use in MinIO
# line 3 -> The name of the directory/file in MinIO
s3.upload_file(
    "./data/saves/trainset.npz", 
    "ml-data",
    "mnist/train.npz" 
)

s3.upload_file(
    "./data/saves/testset.npz",
    "ml-data",
    "mnist/test.npz" 
)

print("Successfully uploaded TensorFlow-compatible dataset to MinIO")
