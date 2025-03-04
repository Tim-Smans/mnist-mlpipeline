import json
import os
import boto3
import mlflow

# Configureer MinIO/Mlflow
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://9.223.80.153"  # Trailing slash
os.environ['AWS_ACCESS_KEY_ID'] = "minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio123"


# Boto3-client voor bucket checks
s3 = boto3.client(
    "s3",
    endpoint_url="http://9.223.80.153/",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123",
    config=boto3.session.Config(signature_version='s3v4')
)

# MLflow config
MLFLOW_TRACKING_URI = "http://9.223.80.153/mlflow/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = mlflow.tracking.MlflowClient()

runs = client.search_runs(experiment_ids=["1"])
if runs:
    run_id = runs[0].info.run_id
    artifacts = client.list_artifacts(run_id, path="test_artifacts")  
    print(f"Artifacts in run {run_id}: {artifacts}")
else:
    print("Geen runs gevonden.")


# Maak experiment aan met expliciete artifact locatie
experiment_name = "test-experiment"
try:
    mlflow.create_experiment(
        name=experiment_name,
        artifact_location="s3://test/"
    )
except mlflow.exceptions.MlflowException:
    pass  # Experiment bestaat al

mlflow.set_experiment(experiment_name)

# Test artifact upload
with mlflow.start_run() as run:
    try:
        # Maak testbestand
        test_path = "/tmp/test.txt"
        with open(test_path, "w") as f:
            f.write("TEST CONTENT")
            
        # Upload
        mlflow.log_artifact(test_path, artifact_path="test_artifacts")
        print("✅ Upload succesvol!")
        
    except Exception as e:
        print(f"❌ Fout: {e}")
        raise