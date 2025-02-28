import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--test_data', type=str, required=True)
parser.add_argument('--model_input', type=str, required=True)
parser.add_argument('--loss_input', type=str, required=True)
parser.add_argument('--optimizer_input', type=str, required=True)
parser.add_argument('--trained_model', type=str, required=True)
args = parser.parse_args()

# Load inputs
model = torch.load(args.model_input, weights_only=False)
criterion = torch.load(args.loss_input, weights_only=False)
optimizer_state = torch.load(args.optimizer_input, weights_only=False)

trainset = torch.load(args.train_data, weights_only=False)
testset = torch.load(args.test_data, weights_only=False)

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer.load_state_dict(optimizer_state)

# Create DataLoaders
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Training loop
for epoch in range(5):  # Example: 5 epochs
    for data, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save the trained model
torch.save(model.state_dict(), "/app/trained_model.pth")

# Upload the trained model to MinIO
import boto3
s3 = boto3.client(
    "s3",
    endpoint_url="http://9.223.154.101:9000",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123"
)

s3.upload_file("/app/trained_model.pth", "ml-models", "trained_model.pth")

print("Completed model training and uploaded to MinIO")