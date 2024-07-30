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
        fetch_query = "SELECT * FROM phishing_model_metrics ORDER BY created_at desc LIMIT 1;"
        df = pd.read_sql(fetch_query, conn)
    except Exception as e:
        print(f"Failed to fetch data: {e}") 
    
    if(not df.empty):
        version = df['version'][0]
    else:
        version = 1
    
    try:
        fetch_query = f"SELECT * FROM phishing_model_metrics where version={version} order by accuracy desc limit 1;"
        model = pd.read_sql(fetch_query, conn)
        accuracy = model['accuracy'][0]
    except Exception as e:
        print(f"Failed to fetch data: {e}")
    
    try:
        fetch_query = "SELECT * FROM phishing_model_metrics where in_use is true LIMIT 1;"
        old_model = pd.read_sql(fetch_query, conn)
    except Exception as e:
        print(f"Failed to fetch data: {e}") 
    
    
    if accuracy >= .85:
        # Query to fetch data from the table

        name = f"{model['name'][0]}-version{version}-phd"
        print(name)
        namespace = utils.get_default_target_namespace()
        kserve_version='v1beta1'
        api_version = constants.KSERVE_GROUP + '/' + kserve_version

        isvc2 = V1beta1InferenceService(
            api_version=api_version,
            kind=constants.KSERVE_KIND,
            metadata=client.V1ObjectMeta(
                name=name,
                namespace=namespace,
                annotations={'sidecar.istio.io/inject': 'false'}
            ),
            spec=V1beta1InferenceServiceSpec(
                predictor=V1beta1PredictorSpec(
                    service_account_name="s3-service-account",
                    sklearn=V1beta1SKLearnSpec(
                        storage_uri=model['uri'][0]
                    )
                )
            )
        )


        KServe = KServeClient()
        KServe.create(isvc2)



        update_query_new = """
            UPDATE phishing_model_metrics
            SET in_use = true
            WHERE name = %s and version = %s;
        """

        update_query_old = """
            UPDATE phishing_model_metrics
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

        if(not old_model.empty):
            del_name = f"{old_model['name'][0]}-version{old_model['version'][0]}-phd"
            namespace = utils.get_default_target_namespace()

            # Initialize the KServe client
            KServe = KServeClient()

            # Delete the inference service
            KServe.delete(del_name, namespace)
    else:
        print("Bad Accuracy: Email Dovelopers!")
    
    cursor.close()
    conn.close()