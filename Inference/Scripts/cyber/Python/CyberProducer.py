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
from io import BytesIO
from zipfile import ZipFile
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

#Perform normalization
def zscore_normalization(df, name, preprocess_info):
    pair = preprocess_info[name.lower()]
    df[name] = (df[name] - pair[0]) / pair[1]
        
#Encode text 
def encode_text(df, name, preprocess_info):
    import base64
    import pickle
    from sklearn.preprocessing import OrdinalEncoder
    
    encoded_string = preprocess_info[name.lower()]
    decoded_bytes = base64.b64decode(encoded_string)
    enc = pickle.loads(decoded_bytes)
    
    data = enc.fit_transform(df[name].values.reshape(-1,1))
    df[name] = data.flatten()
        
#Data preprocessing
def preprocess(df):
    with engine.connect() as conn: 
        query = text('SELECT * FROM metadata_table_cyber ORDER BY version DESC LIMIT 1')
        data = pd.read_sql_query(query, conn)
        row = data.iloc[0]
        factors = row['factor']
    
    for i in df.columns:
        t = (df[i].dtype)
        if i != 'outcome' and factors[i.lower()] == None:
            df.drop(columns=i, inplace=True)
        elif i != 'outcome':
            if t == 'int64' or t == 'float64':
                df[i] = boxcox(df[i], 0.5)
                zscore_normalization(df, i, factors)
               
            else:
                df[i] = df[i].astype(str)
                encode_text(df, i, factors)
    return df

# Initialize logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Broker address
brokers = [f'{host}:9092']

# GitHub raw URL for the CSV file

github_url = 'https://github.com/tsimhadri-ews/internproject/raw/main/Data/UNSW_NB15_testing-set.csv.zip'


# PostgreSQL database connection details
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

# Kafka topics
postgres_topic = "cyd-postgresql"
kserve_topic = "cyd-kserve"

# Download the CSV file from GitHub
logger.info("Downloading CSV file from GitHub...")
response = requests.get(github_url)
zipfile = ZipFile(BytesIO(response.content))

csv_filename = zipfile.namelist()[0]

response = requests.get(github_url)
df = pd.DataFrame()
# Check if the request was successful
if response.status_code == 200:
    logger.info("CSV file downloaded successfully.")
    # Read the data into a pandas DataFrame

    df = pd.read_csv(zipfile.open(csv_filename), index_col=False)
    encoding_dict = {
        'Analysis': 0.0,
        'Backdoor': 1.0,
        'DoS': 2.0,
        'Exploits': 3.0,
        'Fuzzers': 4.0,
        'Generic': 5.0,
        'Normal': 6.0,
        'Reconnaissance': 7.0,
        'Shellcode': 8.0,
        'Worms': 9.0
    }

    df = df.sample(frac=1).reset_index(drop=True)
    
    df['outcome'] = df['attack_cat'].map(encoding_dict)
    df = df.drop(columns=['attack_cat'])
    unprocess_df = df.copy()
    df = preprocess(df)
    
    #print(df['outcome'].loc[5000:5010])
    #print(unprocess_df['outcome'].loc[5000:5010])
    
    df = df.drop(columns=['outcome'])
    logger.info("CSV file processed successfully.")
else:
    logger.error(f"Failed to download file: {response.status_code}")
    exit()

df['uid'] = [str(uuid.uuid4()) for _ in range(len(df))]
unprocess_df['uid'] = df['uid']
# Create Kafka producers
logger.info("Creating Kafka producers...")
postgres_producer = KafkaProducer(
    bootstrap_servers=brokers,
    value_serializer=lambda message: json.dumps({k: v for k, v in message.items()}).encode('utf-8'),
)

kserve_producer = KafkaProducer(
    bootstrap_servers=brokers,
    value_serializer=lambda message: json.dumps(message).encode('utf-8'),
)

df.columns = df.columns.str.lower()
#print(df.columns)

try:
    with engine.connect() as conn: 
        query = text("SELECT * FROM metadata_table_cyber ORDER BY date DESC LIMIT 1;")
        order = pd.read_sql_query(query, conn)
        order = order['factor'][0]
        order.pop('version')
        order = {key: value for key, value in order.items() if value is not None}
        new_order = list(order.keys())
        #print(new_order)
except Exception as e:
    print(f"Failed to fetch data: {e}")
new_order.append('uid')
df = df[new_order]

# Send data to KServe and PostgreSQL
first_5_rows = df.loc[0:5]
for index, row in first_5_rows.iterrows():
    data = row.to_dict()
    # Send to KServe topic
    kserve_producer.send(kserve_topic, data)
    logger.info(f"Sent row {index + 1} to KServe Kafka topic")

kserve_producer.flush()
kserve_producer.close()
# Flush data and close the producers
unprocess_df.columns = unprocess_df.columns.str.lower()
first_5_unprocess = unprocess_df.loc[0:5]
for index, row in first_5_unprocess.iterrows():
    data = row.to_dict()
    # Send to PostgreSQL topic
    postgres_producer.send(postgres_topic, data)
    logger.info(f"Sent row {index + 1} to PostgreSQL Kafka topic")
# Flush data and close the producers
postgres_producer.flush()
postgres_producer.close()

logger.info(f"First 5 rows sent to PostgreSQL Kafka topic: {postgres_topic} and KServe Kafka topic: {kserve_topic}")
