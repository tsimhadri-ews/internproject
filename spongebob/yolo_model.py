from ultralytics import YOLO
import boto3
import os 
import pandas as pd
import onnx
from onnx2pytorch import ConvertModel
import torch


model = YOLO("yolov8n.yaml")

training = model.train(data="/home/ubuntu/objectdetection/data.yaml", epochs=20, batch=16, conf=.4, imgsz=640, device=0, plots=True)

metrics = model.val()
map50_95 = metrics.box.map
map50 = metrics.box.map50
map75 = metrics.box.map75
maps = metrics.box.maps

version = 1 

# Export the model to ONNX format
path = model.export() #onnx
print("model format is " + path)

bucket_name="spongebobpipeline"
role_arn = 'arn:aws:iam::533267059960:role/aws-s3-access'
session_name = 'kubeflow-pipeline-session'
sts_client = boto3.client('sts')
response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
credentials = response['Credentials']
    
# Configure AWS SDK with temporary credentials
s3_client = boto3.client('s3',
                      aws_access_key_id=credentials['AccessKeyId'],
                      aws_secret_access_key=credentials['SecretAccessKey'],
                      aws_session_token=credentials['SessionToken'])




metrics = pd.DataFrame(columns=["Version", "yolo","mAP50-95", "mAP50","mAP75", "mAPs" ])
models_path = './tmp/spongebob/models'


if not os.path.exists(models_path):
    os.makedirs(models_path)
    print(f"Folder '{models_path}' created successfully.")
else:
    print(f"Folder '{models_path}' already exists.")


metrics.loc[len(metrics.index)] = [version, path, map50_95, map50, map75,maps]
with open('./tmp/spongebob/models/best.onnx', 'wb') as f:
    s3_client.upload_file("tmp/spongebob/models/best.onnx", bucket_name, "yolo/best.onnx")

