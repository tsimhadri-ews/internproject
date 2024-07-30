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

import pandas as pd
from sqlalchemy import create_engine, text
import time
import logging
import requests
import psycopg2
from scipy.special import boxcox


hostname = host
port = port
username = user
password = pswd
database = db

conn = psycopg2.connect(
    host=hostname,
    port=port,
    user=username,
    password=password,
    database=database,
    connect_timeout=5  # Adjust timeout as needed
)
# Start timer
start_time = time.time()

# Create the table with a primary key using raw SQL
table_name = 'malware_data'

create_table_query = f"""
CREATE TABLE IF NOT EXISTS intrusion_data (
    uid VARCHAR PRIMARY KEY,
    features JSONB,
    outcome Integer default 2,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=hostname,
        port=port,
        user=username,
        password=password,
        database=database,
        connect_timeout=5  # Adjust timeout as needed
    )

    # Create a cursor object using the connection
    cursor = conn.cursor()

    # Execute the SQL statement to create the table
    cursor.execute(create_table_query)

    # Commit the transaction
    conn.commit()
    print("Table 'test_table' created successfully.")

except psycopg2.Error as e:
    print(f"Error creating table: {e}")

# End timer
end_time = time.time()
duration = end_time - start_time

print(f"Time taken to create the table with column names: {duration} seconds")
