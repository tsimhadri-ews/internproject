import boto3
from botocore.exceptions import ClientError
import json

import logging
from kafka import KafkaConsumer
import requests
import json
import psycopg2
import uuid
import pandas as pd
import time

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



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

resp = []

# Replace with your Kafka broker address(es)
brokers = [f"{host}:9092"]

# Replace with your topic name
topic = "phd-kserve"


db_details = {
    'dbname': db,
    'user': user,
    'password': pswd,
    'host': host,
    'port': port
}

try:
    conn = psycopg2.connect(**db_details)
    cursor = conn.cursor()
    print("Connected to PostgreSQL successfully.")
except Exception as e:
    print(f"Failed to connect to PostgreSQL: {e}")
    exit()
    
try:
    fetch_query = "SELECT * FROM phishing_model_metrics where in_use is true LIMIT 1;"
    model = pd.read_sql(fetch_query, conn)
except Exception as e:
    print(f"Failed to fetch data: {e}")
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()

# KServe inference service URL
kserve_url = f"http://{model['name'][0]}-version{model['version'][0]}-phd.kubeflow-user-example-com.svc.cluster.local/v1/models/{model['name'][0]}-version{model['version'][0]}-phd:predict"

# Create a Kafka consumer
logger.info("Creating Kafka consumer...")
consumer = KafkaConsumer(
    topic,
    bootstrap_servers=brokers,
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group2',
    value_deserializer=lambda v: v.decode('utf-8')
)

logger.info(f"Subscribed to topic: {topic}")


# Function to send data to KServe
def send_to_kserve(data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(kserve_url, headers=headers, data=json.dumps(data))
    return response.json()

def outcome_to_database(values):
    db_details = {
        'dbname': db,
        'user': user,
        'password': pswd,
        'host': host,
        'port': port
    }
    # Data to insert
    outcome_data = {
        'uid': values[0],  # Generating a unique ID
        'outcome': values[1]  # This can be 0 or 1
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
    INSERT INTO phishing_outcomes (uid, outcome)
    VALUES (%s, %s)
    """
    cursor.execute(insert_query, (outcome_data['uid'], outcome_data['outcome']))

    update_query = """
    UPDATE phishing_data
    SET outcome = %s
    WHERE uid = %s
    """
    #cursor.execute(update_query, (outcome_data['outcome'], outcome_data['uid']))
    conn.commit()
        #     break
        #     print("Data inserted and updated successfully.")
        # except Exception as e:
        #     if attempt < max_retries - 1:
        #         print(f"Attempt {attempt + 1} failed: {e}. Retrying in 5 seconds...")
        #         time.sleep(5)
        #     else:
        #         logger.error(f"Failed after {max_retries} attempts: {e}")
        #         conn.rollback()
        #         raise
    
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    print("PostgreSQL connection closed.")

        
        

try:
    for message in consumer:
        try:
            resp = []
            message = json.loads(message.value)  # Message value is already a dict
            print(f"Received message: {message}")
            uid = message.pop('uid', None)
            resp.append(uid)
            #print(f"new message: {message}")
            # Prepare data for KServe
            kserve_payload = {
                "instances": [list(message.values())]  # Reshape the data to 2D array
            }

            # Send data to KServe
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    # Send data to KServe
                    kserve_response = send_to_kserve(kserve_payload)
                    if kserve_response:
                        resp.append(kserve_response['predictions'][0])
                        print(f"KServe response: {kserve_response}")
                        print(resp)
                        outcome_to_database(resp)
                        break  # Exit the loop if the response is received
                    else:
                        raise ValueError("No response from KServe")
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
        except json.JSONDecodeError:
            print(f"Received non-JSON message: {message.value}")
        except Exception as e:
            logger.error(f"Error sending to KServe: {e}")
        #logger.info(f"Partition: {message.partition}, Offset: {message.offset}")
except KeyboardInterrupt:
    logger.info("Consumer stopped manually.")
finally:
    consumer.close()
    logger.info("Consumer closed.")
