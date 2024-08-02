import argparse
import boto3
import os
import cv2
from ultralytics import YOLO
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

import json
import psycopg2

from PIL import Image
import base64
import io

import boto3
from botocore.exceptions import ClientError
import json

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

def base64_to_image(base64_str):
    """Convert a base64-encoded string to a PIL Image."""
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))

    image.save("./image.jpg", format='JPEG')

    return image

def download_model_from_s3(version):
    print(version)
    bucket_name="spongebobpipeline"
    role_arn = 'arn:aws:iam::533267059960:role/aws-s3-access'
    session_name = 'kubeflow-pipeline-session'
    sts_client = boto3.client('sts')
    response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
    credentials = response['Credentials']
    # Configure AWS SDK with temporary credentials
    s3 = boto3.client('s3',
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken'])

    file_path = f'yolo/version{version}/model.pt'
    local_file_path = './model.pt'

    try:
        s3.download_file(bucket_name, file_path, local_file_path)
        print(f"Model downloaded from s3://{bucket_name}/{file_path} to {local_file_path}")
    except (NoCredentialsError, PartialCredentialsError) as e:
        print("Error with AWS credentials:", e)
        raise
    except Exception as e:
        print("Error downloading the model from S3:", e)
        raise

def outcome_to_database(uid, outcome):
    (user,pswd,host,port,db) = get_secret()

    db_details = {
        'dbname': db,
        'user': user,
        'password': pswd,
        'host': host,
        'port': port
    }
    # Data to insert
    outcome_data = {
        'uid': uid,  # Generating a unique ID
        'outcome': outcome,  # This can be 0 or 1
        'confirmed': 2
    }

# Connect to PostgreSQL
    try:
        conn = psycopg2.connect(**db_details)
        cursor = conn.cursor()
        print("Connected to PostgreSQL successfully.")
    except Exception as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        exit()
    # Insert data into the table
    # max_retries = 3
    # for attempt in range(max_retries):
    #     try:
    insert_query = """
    INSERT INTO spongebob_outcomes (uid, outcome,confirmed)
    VALUES (%s, %s,%s)
    """
    cursor.execute(insert_query, (outcome_data['uid'], outcome_data['outcome'],outcome_data['confirmed']))

    conn.commit()

    if cursor:
        cursor.close()
    if conn:
        conn.close()
    print("PostgreSQL connection closed.")


def run_model_on_image(uid, model_path):
        # Load the YOLO model
    model = YOLO(model_path)

    # Run inference on the input image
    results = model("./image.jpg")

    # Read the input image
    img = cv2.imread("./image.jpg")

    outcome = 0

    # Draw bounding boxes on the image
    for r in results:
        for box in r.boxes:
            coordinates = (box.xyxy).tolist()[0]
            left, top, right, bottom = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
            outcome = 1
        r.save_txt(f'prediction/{uid}.txt')



    outcome_to_database(uid, outcome)

    # Save the image with bounding boxes
    cv2.imwrite(f"./prediction/{uid}.jpg", img)
    print(f"Image saved with bounding boxes to ./Result/{uid}.jpg")

def main():

    parser = argparse.ArgumentParser(description="Run YOLO model on an input image and save the output with bounding boxes.")
    parser.add_argument('uid', type=str, help="uid of image")
    parser.add_argument('version', type=int, help="version of model")
    parser.add_argument('image_path', type=str, help="Path to the input image")

    args = parser.parse_args()
    download_model_from_s3(args.version)

    model_path = './model.pt'

    image = base64_to_image(args.image_path)

    # Run the model on the provided image
    run_model_on_image(args.uid, model_path)

if __name__ == "__main__":
    main()