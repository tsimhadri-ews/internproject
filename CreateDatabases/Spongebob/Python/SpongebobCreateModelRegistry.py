import boto3
from botocore.exceptions import ClientError
import json
import psycopg2
from psycopg2 import sql

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

try:
    create_table_query = sql.SQL("""
    CREATE TABLE spongebob_model_metrics (
        name VARCHAR(100) NOT NULL,
        version INTEGER NOT NULL,
        URI VARCHAR(255),
        in_use BOOLEAN,
        map50_95 DOUBLE PRECISION,
        map50 DOUBLE PRECISION,
        map75 DOUBLE PRECISION,
        PRIMARY KEY (name, version),
        FOREIGN KEY(version) REFERENCES metadata_table_spongebob(version),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute(create_table_query)
    conn.commit()
    print("Table 'spongebob_model_metrics' created successfully.")
except Exception as e:
    print(f"Failed to create table: {e}")
    conn.rollback()
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    print("PostgreSQL connection closed.")