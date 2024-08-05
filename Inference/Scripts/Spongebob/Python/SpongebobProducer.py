import streamlit as st
from PIL import Image

import boto3
from botocore.exceptions import ClientError
import json

import logging
import requests
import pandas as pd
from kafka import KafkaProducer
import json
import psycopg2
from sqlalchemy import create_engine, text
from scipy.special import boxcox
import numpy as np
import uuid
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import base64
import io

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

def image_to_base64(image):
    """Convert a PIL Image to a base64-encoded string."""
    buffer = io.BytesIO()
    image.save(buffer, format=image.format)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str


def main():
    st.title("Image Upload Example")

    (user,pswd,host,port,db) = get_secret()
    brokers = [f'{host}:9092']

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    db_config = {
        'dbname': db,
        'user': user,
        'password': pswd,
        'host': host,
        'port': port
    }

    # PostgreSQL connection setup
    logger.info("Connecting to PostgreSQL...")
    try:
        engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}")
        logger.info("PostgreSQL connection established successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        exit()

    postgres_topic = "spongebobTopic"


    if uploaded_file is not None:
        # Open the uploaded image file
        image = Image.open(uploaded_file)

        uid = str(uuid.uuid4())
        
        # Display the image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to bytes or another serializable format
        image_bytes = image_to_base64(image)  # Get the raw bytes of the image

        message = {
            'uid': uid,
            'image': image_bytes
        }

        logger.info("Creating Kafka producer...")
        try:
            postgres_producer = KafkaProducer(
                bootstrap_servers=brokers,
                value_serializer=lambda message: json.dumps(message).encode('utf-8'),
            )

            # Send image data to Kafka topic
            postgres_producer.send(postgres_topic, message)
            logger.info("Image data sent to s3.")
        except Exception as e:
            logger.error(f"Failed to send data to Kafka: {e}")

if __name__ == "__main__":
    main()
    