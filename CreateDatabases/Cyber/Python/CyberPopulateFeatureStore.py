def FeatureStoreToDatabase() -> None:
    import requests
    import os
    import pandas as pd
    import numpy as np
    from minio import Minio
    from scipy.special import boxcox
    from sklearn.model_selection import train_test_split
    import boto3
    import json
    from io import BytesIO
    from zipfile import ZipFile
    import psycopg2
    from psycopg2 import sql
    from sqlalchemy import create_engine, text
    import datetime
    
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
    

    github_url = 'https://github.com/tsimhadri-ews/internproject/raw/main/Data/UNSW_NB15_training-set.csv.zip'
    response = requests.get(github_url)
    zipfile = ZipFile(BytesIO(response.content))

    csv_filename = zipfile.namelist()[0]

    db_details = {
        'dbname': db,
        'user': user,
        'password': pswd,
        'host': host,
        'port': port
    }


        
    # Connect to PostgreSQL
    engine = create_engine(f'postgresql+psycopg2://{db_details["user"]}:{db_details["password"]}@{db_details["host"]}:{db_details["port"]}/{db_details["dbname"]}')
    
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

    chunk_size = 10000
    uid_start = 1

    
    for chunk in pd.read_csv(zipfile.open(csv_filename), chunksize=chunk_size):
        df = pd.DataFrame()
        df['outcome'] = chunk['attack_cat'].map(encoding_dict)
        chunk = chunk.drop(columns=['attack_cat'])
        chunk.columns = chunk.columns.str.lower()
        df['features'] = chunk.apply(lambda row: row.to_json(), axis=1)
        df['uid'] = range(uid_start, uid_start + len(chunk))
        uid_start += len(chunk)

        df.columns = df.columns.str.lower()
        df.to_sql("cyber_data", engine, if_exists='append', index=False)
        
    github_url1 = 'https://github.com/tsimhadri-ews/internproject/raw/main/Data/UNSW_NB15_testing-set.csv.zip'
    response1 = requests.get(github_url1)
    zipfile1 = ZipFile(BytesIO(response1.content))

    csv_filename1 = zipfile1.namelist()[0]
    for chunk in pd.read_csv(zipfile1.open(csv_filename1), chunksize=chunk_size):
        df = pd.DataFrame()
        df['outcome'] = chunk['attack_cat'].map(encoding_dict)
        chunk = chunk.drop(columns=['attack_cat'])
        chunk.columns = chunk.columns.str.lower()
        df['features'] = chunk.apply(lambda row: row.to_json(), axis=1)
        df['uid'] = range(uid_start, uid_start + len(chunk))
        uid_start += len(chunk)

        df.columns = df.columns.str.lower()
        df.to_sql("cyber_data", engine, if_exists='append', index=False)

FeatureStoreToDatabase()
