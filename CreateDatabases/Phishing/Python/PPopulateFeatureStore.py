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
    
    db_details = {
        'dbname': db,
        'user': user,
        'password': pswd,
        'host': host,
        'port': port
    }



    # Connect to PostgreSQL
    engine = create_engine(f'postgresql+psycopg2://{db_details["user"]}:{db_details["password"]}@{db_details["host"]}:{db_details["port"]}/{db_details["dbname"]}')
    
    
    chunksize = 10000
    uid_start = 1

    df = pd.DataFrame()
    file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/main/Data/dataset_B_05_2020.csv'

    
    reader = pd.read_csv(file_path, chunksize=chunksize)
    for chunk in reader:
        mapping = {'legitimate':0, 'phishing':1}

        chunk['status'] = chunk['status'].map(mapping)
        df = pd.DataFrame()
        chunk.columns = chunk.columns.str.lower()
        df['outcome'] = chunk['status']
        chunk = chunk.drop(columns=['status'])
        df['features'] = chunk.apply(lambda row: row.to_json(), axis=1)
        df['uid'] = range(uid_start, uid_start + len(chunk))
        uid_start += len(chunk)

        df.columns = df.columns.str.lower()
        df.to_sql("phishing_data", engine, if_exists='append', index=False)

FeatureStoreToDatabase()