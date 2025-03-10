import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import mlflow
import mlflow.pytorch
import mlflow.tracking
import os

# Configuring the enviromental variables for MinIO
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MINIO_ENDPOINT", "http://9.223.80.153/")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("MINIO_ACCESS_KEY", "minio")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("MINIO_SECRET_KEY", "minio123")

# Configure MLFlow
MLFLOW_TRACKING_URI = "http://9.223.80.153/mlflow/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Creating an experiment and setting it as the active experiment, also making sure that we are using the right bucket by setting the artifact location.
experiment_name = "mnist-pipeline"
try:
    mlflow.create_experiment(
        name=experiment_name,
        artifact_location="s3://test/"
    )
except mlflow.exceptions.MlflowException:
    pass


print("MLflow Tracking URI:", mlflow.get_tracking_uri())

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--test_data', type=str, required=True)
parser.add_argument('--model_input', type=str, required=True)
parser.add_argument('--loss_input', type=str, required=True)
parser.add_argument('--optimizer_input', type=str, required=True)
parser.add_argument('--trained_model', type=str, required=True)
args = parser.parse_args()


# Loading all our inputs out of the arguments
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


# Testing if mlflow is connected
client = mlflow.tracking.MlflowClient()
print("Active Experiment ID:", client.get_experiment_by_name("mnist-pipeline"))


# Defining our training loop
# TODO: Adding epochs as a parameter
def train(model, train_data, test_data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 5
    best_accuracy = 0

    if mlflow.active_run():
        print(f"Ending existing run {mlflow.active_run().info.run_id}")
        mlflow.end_run()


    with mlflow.start_run() as run:
        print(f"Started MLflow run with ID: {run.info.run_id} with {num_epochs} epochs") 
        print("Active runs:", mlflow.search_runs())
        try:
            print("Started MLflow run..")
            mlflow.log_param("learning_rate", 0.001)
            mlflow.log_param("epochs", num_epochs)

            for epoch in range(num_epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                for images, labels in train_data:
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                mlflow.log_metric("train_loss", running_loss, step=epoch)
                mlflow.log_metric("accuracy", accuracy, step=epoch)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model, args.trained_model)



            print(f"Final Accuracy: {best_accuracy}")
            mlflow.log_metric("final_accuracy", best_accuracy)
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="mnist_model",
                code_paths=["model.py"],
                registered_model_name="mnist-model",
            )
            
        except Exception as e:
            print(f"MLflow logging error: {e}")


# Call the train function with the appropriate parameters
train(model, trainloader, testloader)

# Save the trained model
# TODO: Add a unique name every time (timestamped)
torch.save(model.state_dict(), "/app/trained_model.pth")

# Upload the trained model to MinIO
import boto3
s3 = boto3.client(
    "s3",
    endpoint_url="http://9.223.80.153/",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123"
)

s3.upload_file("/app/trained_model.pth", "ml-models", "trained_model.pth")

print("Completed model training and uploaded to MinIO")