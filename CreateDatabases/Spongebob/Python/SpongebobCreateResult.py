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

import psycopg2
from psycopg2 import sql

# Database details
db_details = {
    'dbname': db,
    'user': user,
    'password': pswd,
    'host': host,
    'port': port
}

# Connect to PostgreSQL
try:
    conn = psycopg2.connect(**db_details)
    cursor = conn.cursor()
    print("Connected to PostgreSQL successfully.")
except Exception as e:
    print(f"Failed to connect to PostgreSQL: {e}")
    exit()

# Create the table
try:
    create_table_query = sql.SQL("""
    CREATE TABLE IF NOT EXISTS spongebob_outcomes (
        uid text PRIMARY KEY,
        outcome smallint default NULL,
        confirmed smallint
    )
    """)

    cursor.execute(create_table_query)
    conn.commit()
    print("Table 'outcomes' created successfully.")
except Exception as e:
    print(f"Failed to create table: {e}")
    conn.rollback()
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    print("PostgreSQL connection closed.")
