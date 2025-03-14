import argparse
import tensorflow as tf
import numpy as np
import mlflow
import mlflow.tensorflow
import mlflow.tracking
import os
import boto3
from prometheus_client import start_http_server, Gauge
import time

# Prometheus
# Start prometheus metrics server
start_http_server(8000)

# Define metrics for prometheus
loss_metric = Gauge("training_loss", "Training loss per epoch")
accuracy_metric = Gauge("training_accuracy", "Training accuracy per epoch")


# Configuring the environmental variables for MinIO
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MINIO_ENDPOINT", "http://istio-ingressgateway.istio-system.svc.cluster.local")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("MINIO_ACCESS_KEY", "minio")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("MINIO_SECRET_KEY", "minio123")

# Configure MLFlow
MLFLOW_TRACKING_URI = "http://istio-ingressgateway.istio-system.svc.cluster.local/mlflow/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Creating an experiment
experiment_name = "mnist-pipeline"
try:
    mlflow.create_experiment(
        name=experiment_name,
        artifact_location="s3://ml-models/"
    )
except mlflow.exceptions.MlflowException:
    pass

print("MLflow Tracking URI:", mlflow.get_tracking_uri())

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--test_data', type=str, required=True)
parser.add_argument('--trained_model', type=str, required=True)
parser.add_argument('--epochs', type=int, default=5)
args = parser.parse_args()

# Load datasets
def load_npz_dataset(filepath):
    with np.load(filepath) as data:
        images = data['images']
        labels = data['labels']
    return images, labels

train_images, train_labels = load_npz_dataset(args.train_data)
test_images, test_labels = load_npz_dataset(args.test_data)

# Define the model (merged from define_model.py)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


# Define loss and optimizer (merged from define_loss.py)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Define training function
def train_model():
    num_epochs = 5
    batch_size = 32

    if mlflow.active_run():
        print(f"Ending existing run {mlflow.active_run().info.run_id}")
        mlflow.end_run()

    with mlflow.start_run(experiment_id="1") as run:
        print(f"Started MLflow run with ID: {run.info.run_id} for {num_epochs} epochs")
        print("Active runs:", mlflow.search_runs())

        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("epochs", num_epochs)

        history = model.fit(
            train_images, train_labels,
            validation_data=(test_images, test_labels),
            epochs=num_epochs,
            batch_size=batch_size
        )

        for epoch in range(num_epochs):
            # MLFlow logging
            mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
            mlflow.log_metric("accuracy", history.history["accuracy"][epoch], step=epoch)
            
            # Prometheus logging
            loss_metric.set(history.history["loss"][epoch])  # Update Prometheus metric
            accuracy_metric.set(history.history["accuracy"][epoch])  # Update Prometheus metric

        model.save("mnist-model.keras")
        
        tf.saved_model.save(model, 'app/mnist_model_saved')

        mlflow.log_metric("final_accuracy", history.history["accuracy"][-1])
        mlflow.tensorflow.log_model(
            model,
            artifact_path="mnist_model",
            registered_model_name="mnist-model"
        )


def upload_directory_to_s3(local_directory, bucket_name, s3_prefix):
    """
    Upload een hele map naar S3.
    
    :param local_directory: Pad naar de lokale map.
    :param bucket_name: Naam van de S3 bucket.
    :param s3_prefix: Prefix (map) in S3 waar de bestanden naartoe worden geüpload.
    """
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Relatief pad behouden ten opzichte van de lokale map
            relative_path = os.path.relpath(local_file_path, local_directory)
            s3_key = os.path.join(s3_prefix, relative_path)
            
            try:
                print(f"Uploading {local_file_path} to {s3_key}")
                s3.upload_file(local_file_path, bucket_name, s3_key)
            except NameError:
                print("Credentials niet gevonden of ongeldig.")
                return False
    return True

train_model()

s3 = boto3.client(
    "s3",
    endpoint_url="http://istio-ingressgateway.istio-system.svc.cluster.local",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123",
    config=boto3.session.Config(signature_version='s3v4')
)

local_directory = "./app/mnist_model_saved"
bucket_name = "ml-models"
s3_prefix = "latest/saved/mnist_model_saved"

upload_directory_to_s3(local_directory, bucket_name, s3_prefix)

print("Completed model training and uploaded to MinIO")
