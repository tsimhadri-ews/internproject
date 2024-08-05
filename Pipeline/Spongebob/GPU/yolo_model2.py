from ultralytics import YOLO
import boto3
import os
import pandas as pd
import onnx
from onnx2pytorch import ConvertModel
import torch
import json

import psycopg2
from psycopg2 import sql

from sqlalchemy import create_engine
from sqlalchemy import create_engine, Table, Column, Float, Integer, String, MetaData, ARRAY
from sqlalchemy import select, desc, insert, text

from botocore.exceptions import ClientError

def get_secret():
    secret_name = "DBCreds"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    secret = get_secret_value_response['SecretString']

    # Parse the secret string to get the credentials
    secret_dict = json.loads(secret)
    username = secret_dict['username']
    password = secret_dict['password']
    host = secret_dict['host']
    port = secret_dict['port']
    dbname = secret_dict['dbname']

    return username, password, host, port, dbname

# Retrieve database credentials
(user, pswd, host, port, db) = get_secret()

model = YOLO("yolov8n.yaml")

training = model.train(data="/home/ubuntu/objectdetection/data.yaml", epochs=20, batch=16, conf=.4, imgsz=640, device=0, plots=True, save=True, project="models", name="results")

metrics = model.val()
map50_95 = metrics.box.map
map50 = metrics.box.map50
map75 = metrics.box.map75
maps = metrics.box.maps

version = 1

results_path = training.save_dir

print(f"Results saved to {results_path}")

# Export the model to ONNX format
#path = model.save() #onnx
#print("model format is " + path)

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

metrics = pd.DataFrame(columns=["version", "map50_95", "map50","map75" ])
models_path = './tmp/spongebob/models'


if not os.path.exists(models_path):
    os.makedirs(models_path)
    print(f"Folder '{models_path}' created successfully.")
else:
    print(f"Folder '{models_path}' already exists.")

db_details = {
    'dbname': db,
    'user': user,
    'password': pswd,
    'host': host,
    'port': port
}

print(metrics)

engine = create_engine(f'postgresql+psycopg2://{db_details["user"]}:{db_details["password"]}@{db_details["host"]}:{db_details["port"]}/{db_details["dbname"]}')

try:
    with engine.connect() as conn:
        query = text('SELECT * FROM metadata_table_spongebob ORDER BY version DESC LIMIT 1;')
        data = pd.read_sql_query(query, conn)
        version = data['version'].iloc[0]
        print(version)
except Exception as e:
    version = 1


insert_query = """
    INSERT INTO spongebob_model_metrics (name, version, uri, in_use, map50_95, map50, map75)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (name, version) DO NOTHING;
"""
try:
    conn = psycopg2.connect(**db_details)
    cursor = conn.cursor()
    print("Connected to PostgreSQL successfully.")

    metrics.loc[len(metrics.index)] = [version, map50_95, map50, map75]
    s3_client.upload_file(f"./{results_path}/weights/best.pt", bucket_name, f"yolo/version{version}/model.pt")

    # Iterate through DataFrame rows and insert into the table
    for index, row in metrics.iterrows():
        cursor.execute(insert_query, (
            "yolo",
            row['version'],
            f"./yolo/version{version}/model.pt",
            False,
            row['map50_95'],
            row['map50'],
            row['map75']
        ))
    conn.commit()
    print("Data inserted successfully.")

    cursor.close()
    conn.close()
    print("PostgreSQL connection closed.")
except Exception as e:
    print(f"Failed to connect to PostgreSQL or insert data: {e}")

