FROM python:3.9

WORKDIR /app

COPY app.py /app/

RUN pip install flask torch torchvision boto3 pillow

CMD ["python", "app.py"]
