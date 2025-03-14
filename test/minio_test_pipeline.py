from kfp import dsl
from kfp.dsl import container_component, Output, Dataset

@dsl.container_component
def test_minio_connection(output_log: Output[Dataset]):
    return dsl.ContainerSpec(
        image="amazonlinux",  # A lightweight container
        command=["sh", "-c"],
        args=[
            "yum install -y python3-pip && pip3 install boto3 && "
            "python3 -c '"
            "import boto3; "
            "s3 = boto3.client(\"s3\", endpoint_url=\"http://istio-ingressgateway.istio-system.svc.cluster.local\", "
            "aws_access_key_id=\"minio\", aws_secret_access_key=\"minio123\"); "
            "print(s3.list_buckets())'"
        ],
    )

from kfp import compiler, dsl

@dsl.pipeline(name="minio-connection-test", description="Pipeline to test MinIO connectivity")
def minio_pipeline():
    test_minio_connection()

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=minio_pipeline,
        package_path="minio_test_pipeline.yaml"
    )
