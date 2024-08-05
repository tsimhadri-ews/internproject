def model_eval_deploy() -> None:
    import pickle
    import pandas as pd
    import numpy as np 
    import json
    import os 
    import time
    import tensorflow as tf
    
    import boto3
    
    import psycopg2
    from psycopg2 import sql
    from sqlalchemy import create_engine
    
    from kubernetes import client 
    from kserve import KServeClient
    from kserve import constants
    from kserve import utils
    from kserve import V1beta1InferenceService
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1PredictorSpec
    from kserve import V1beta1SKLearnSpec
    
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
    try:
        conn = psycopg2.connect(**db_details)
        cursor = conn.cursor()
        print("Connected to PostgreSQL successfully.")
    except Exception as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        exit()
    
    try:
        fetch_query = f"SELECT * FROM spongebob_model_metrics order by map50 desc limit 1;"
        model = pd.read_sql(fetch_query, conn)
        accuracy = model['map50'][0]
    except Exception as e:
        print(f"Failed to fetch data: {e}")
    old_version = -1
    
    try:
        fetch_query = "SELECT * FROM spongebob_model_metrics where in_use is true LIMIT 1;"
        old_model = pd.read_sql(fetch_query, conn)
        # old_version = old_model['version'][0]
    except Exception as e:
        print(f"a Failed to fetch data: {e}") 
    
    print(accuracy)
    
    if old_version == -1 or old_version != model['version'][0]:
        print("hello")
        if accuracy >= .15:
            # Query to fetch data from the table
            update_query_new = """
                UPDATE spongebob_model_metrics
                SET in_use = true
                WHERE name = %s and version = %s;
            """

            update_query_old = """
                UPDATE spongebob_model_metrics
                SET in_use = false
                WHERE name = %s and version = %s;
            """
            try:
                cursor.execute(update_query_new, (model['name'][0], int(model['version'][0])))
                if(not old_model.empty):
                    cursor.execute(update_query_old, (old_model['name'][0], int(old_model['version'][0])))
                conn.commit()
            except Exception as e:
                print(f"Failed to fetch data: {e}")
        else:
            print("Bad Accuracy: Email Dovelopers!")
    
    cursor.close()
    conn.close()