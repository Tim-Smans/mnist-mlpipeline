FROM python:3.9

# Installeer benodigde pakketten
RUN pip install mlflow boto3 gunicorn numpy pandas tensorflow

# Stel de werkdirectory in
WORKDIR /app

# Zet de MinIO credentials als omgevingsvariabelen
ENV MLFLOW_S3_ENDPOINT_URL=http://istio-ingressgateway.istio-system.svc.cluster.local
ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=minio123

# Kopieer het startup-script naar de container
COPY download_model.py /app/download_model.py

# Expose de MLflow API-poort
EXPOSE 5001

# Start script dat het model ophaalt en serveert
CMD ["bash", "-c", "python /app/download_model.py && mlflow models serve -m /app/model --host 0.0.0.0 --port 5001 --no-conda"]
