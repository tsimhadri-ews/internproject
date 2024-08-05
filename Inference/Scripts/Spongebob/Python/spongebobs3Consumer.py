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


(user,pswd,host,port,db) = get_secret()

def base64_to_image(base64_str):
    """Convert a base64-encoded string to a PIL Image."""
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    
    image.save("./image.jpg", format='JPEG')
    
    return image


import json
from kafka import KafkaConsumer
import psycopg2
from psycopg2.extras import execute_values
from PIL import Image
import base64
import io
from io import StringIO
import paramiko

import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine

secret_name = "key"
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

original_string = secret_dict['private']
wrapped_key_content = '\n'.join(original_string[i:i+64] for i in range(0, len(original_string), 64))
formatted_key = f"-----BEGIN RSA PRIVATE KEY-----\n{wrapped_key_content}\n-----END RSA PRIVATE KEY-----"

# PostgreSQL database connection details
db_config = {
    'dbname': db,
    'user': user,
    'password': pswd,
    'host': host,
    'port': port
}

# Kafka topic
postgres_topic = "spongebobTopic"
brokers = [f"{host}:9092"]

# Create a Kafka consumer
consumer = KafkaConsumer(
    postgres_topic,
    bootstrap_servers=brokers,
    value_deserializer=lambda message: json.loads(message.decode('utf-8'))
)

# Connect to PostgreSQL
try:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    print("Connected to PostgreSQL successfully.")
except Exception as e:
    print(f"Failed to connect to PostgreSQL: {e}")
    exit()

try:
    print("Starting to consume messages.")
    for message in consumer:
        data = message.value
        print(f"uid = {data.get('uid')}")
        image = base64_to_image(data.get('image'))
        image.show()

        bucket_name="spongebobpipeline"
        role_arn = 'arn:aws:iam::533267059960:role/aws-s3-access'
        session_name = 'kubeflow-pipeline-session'
        sts_client = boto3.client('sts')
        response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
        credentials = response['Credentials']

        # Configure AWS SDK with temporary credentials
#         s3_client = boto3.client('s3',
#                               aws_access_key_id=credentials['AccessKeyId'],
#                               aws_secret_access_key=credentials['SecretAccessKey'],
#                               aws_session_token=credentials['SessionToken'])
        
#         buffer = io.BytesIO()
#         image.save(buffer, format="JPEG")
#         buffer.seek(0)
#         s3_client.upload_fileobj(buffer, bucket_name, f"combined/new/{str(data.get('uid'))}.jpg")
        
        hostname = 'ec2-54-167-24-42.compute-1.amazonaws.com'
        port = 22
        username = 'ubuntu'

        key = formatted_key
        key_file = StringIO(key)
        # Load the private key
        private_key = paramiko.RSAKey.from_private_key(key_file)

        # Establish SSH connection
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname, port, username, pkey=private_key)

        venv = 'yolo/bin/activate'
        remote_directory = '/home/ubuntu/objectdetection'
        
        script_test = 'predict2.py'

        engine = create_engine(f'postgresql+psycopg2://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{db_config["dbname"]}')
        try:
            with engine.connect() as conn:
                query = text('SELECT * FROM spongebob_model_metrics where in_use is true LIMIT 1;')
                data = pd.read_sql_query(query, conn)
                version = data['version'].iloc[0]
                print(version)
        except Exception as e:
            version = 1
        
        command = f"cd {remote_directory} && source {venv} && python3 {script_test} {data.get('uid')} {2} {data.get('image')}"
        stdin, stdout, stderr = ssh_client.exec_command(command)
        #print(stdout.read().decode())
        #print(stderr.read().decode())
        while True:
            # Read from stdout and stderr
            line_stdout = stdout.readline()
            line_stderr = stderr.readline()

            # Print the output
            if line_stdout:
                print(line_stdout, end='')  # Print stdout line
            if line_stderr:
                print(line_stderr, end='')  # Print stderr line

            # Check if the process is finished
            if stdout.channel.exit_status_ready():
                break

        # Print any remaining output
        print(stdout.read().decode())
        print(stderr.read().decode())
except KeyboardInterrupt:
    print("Consumer interrupted.")
finally:
    # Close PostgreSQL connection
    if cursor:
        cursor.close()
    if conn:
        conn.close()
        print("PostgreSQL connection closed.")
