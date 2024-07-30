import boto3
from botocore.exceptions import ClientError
import json
import json
from kafka import KafkaConsumer
import psycopg2
from psycopg2.extras import execute_values

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

# PostgreSQL database connection details
db_config = {
    'dbname': db,
    'user': user,
    'password': pswd,
    'host': host,
    'port': port
}

# Kafka topic
postgres_topic = "cyd-postgresql"
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
        #print(f"Received message: {data['outcome']}")
        uid = data.pop('uid')
        print(f"Received {uid} message: {data['outcome']}")
        data.pop('outcome')
        try:
            insert_query = """
            INSERT INTO cyber_data (uid, features)
            VALUES (%s, %s)
            """
            cursor.execute(insert_query, (uid, json.dumps(data)))
            conn.commit()
            print("Data inserted successfully.")
        except Exception as e:
            print(f"Failed to insert data: {e}")
            conn.rollback()
        
        
except KeyboardInterrupt:
    print("Consumer interrupted.")
finally:
    # Close PostgreSQL connection
    if cursor:
        cursor.close()
    if conn:
        conn.close()
        print("PostgreSQL connection closed.")
