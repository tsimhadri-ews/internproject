def train_op() -> None:
    import pickle
    import pandas as pd
    import numpy as np
    import json
    import os
    import time
    import tensorflow as tf
    import boto3
    from minio import Minio
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import LogisticRegression

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sqlalchemy import create_engine
    from sqlalchemy import create_engine, Table, Column, Float, Integer, String, MetaData, ARRAY
    from sqlalchemy import select, desc, insert, text
    from io import BytesIO
    
    import psycopg2
    from psycopg2 import sql
    from sqlalchemy import create_engine
    
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
    
    bucket_name="multiclasspipeline"
    role_arn = 'arn:aws:iam::533267059960:role/aws-s3-access'
    session_name = 'kubeflow-pipeline-session'
    sts_client = boto3.client('sts')
    response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
    credentials = response['Credentials']
    
    # Configure AWS SDK with temporary credentials
    s3_client = boto3.client('s3',
                      aws_access_key_id=credentials['AccessKeyId'],
                      aws_secret_access_key=credentials['SecretAccessKey'],
                      aws_session_token=credentials['SessionToken'])
    
    
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
        
    # Query to fetch data from the table
    try:
        fetch_query = "SELECT * FROM metadata_table_cyber ORDER BY date DESC LIMIT 1;"
        df = pd.read_sql(fetch_query, conn)
    except Exception as e:
        print(f"Failed to fetch data: {e}")
    
    if(not df.empty):
        version = df['version'][0]
    else:
        version = 1
        
    folder_path = f"version{version}"
    
    cursor.close()
    conn.close()
    
    print(f"version{version}/X_train.npy")
    
    response = s3_client.get_object(Bucket=bucket_name, Key=f"version{version}/X_train.npy")
    data = response['Body'].read()
    X_train = np.load(BytesIO(data))
    X_train = pd.DataFrame(X_train)
    
    response = s3_client.get_object(Bucket=bucket_name, Key=f"version{version}/y_train.npy")
    data = response['Body'].read()
    y_train = np.load(BytesIO(data))

    
    response = s3_client.get_object(Bucket=bucket_name, Key=f"version{version}/X_test.npy")
    data = response['Body'].read()
    X_test = np.load(BytesIO(data))
    X_test = pd.DataFrame(X_test)
    
    response = s3_client.get_object(Bucket=bucket_name, Key=f"version{version}/y_test.npy")
    data = response['Body'].read()
    y_test = np.load(BytesIO(data))
    
    
    # Define dataframe to store model metrics
    metrics = pd.DataFrame(columns=["Version", "Model", "Accuracy", "F1", "Precision", "Recall", "Train_Time", "Test_Time"])
    models_path = './tmp/cyber/models'
    
    
    if not os.path.exists(models_path):
        os.makedirs(models_path)
        print(f"Folder '{models_path}' created successfully.")
    else:
        print(f"Folder '{models_path}' already exists.")
        
    #Random Forest Classifier
    start_train = time.time()
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    end_train = time.time()

    start_test = time.time()
    y_pred2=rfc.predict(X_test)
    end_test = time.time()

    accuracy = accuracy_score(y_test, y_pred2)
    
    precision = precision_score(y_test, y_pred2, average='macro')
    recall = recall_score(y_test, y_pred2, average='macro')
    f1 = f1_score(y_test, y_pred2, average="macro")

    print("Precision:", precision)
    print("Recall:", recall)
    
    metrics.loc[len(metrics.index)] = [version, 'rfc', accuracy, f1, precision, recall, end_train-start_train, end_test-start_test]
    with open('./tmp/cyber/models/rfc.pkl', 'wb') as f:
        pickle.dump(rfc, f)
    s3_client.upload_file("tmp/cyber/models/rfc.pkl", bucket_name, f"{folder_path}/rfc/model.pkl")


    # Decision Tree

    start_train = time.time()
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    end_train = time.time()

    start_test = time.time()
    y_pred3=dtc.predict(X_test)
    end_test = time.time()

    accuracy = accuracy_score(y_test, y_pred3)
    f1 = f1_score(y_test, y_pred3, average="macro")
    precision = precision_score(y_test, y_pred3, average="macro")
    recall = recall_score(y_test, y_pred3, average="macro")

    metrics.loc[len(metrics.index)] = [version, 'dtc', accuracy, f1, precision, recall, end_train-start_train, end_test-start_test]
    with open('./tmp/cyber/models/dtc.pkl', 'wb') as f:
        pickle.dump(rfc, f)
    s3_client.upload_file("tmp/cyber/models/dtc.pkl", bucket_name, f"{folder_path}/dtc/model.pkl")


    #KNN

    start_train = time.time()
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)
    end_train = time.time()

    start_test = time.time()
    y_pred4=knn.predict(X_test)
    end_test = time.time()

    accuracy = accuracy_score(y_test, y_pred4)
    f1 = f1_score(y_test, y_pred4, average="macro")
    precision = precision_score(y_test, y_pred4, average="macro")
    recall = recall_score(y_test, y_pred4, average="macro")

    metrics.loc[len(metrics.index)] = [version, 'knn', accuracy, f1, precision, recall, end_train-start_train, end_test-start_test]
    with open('./tmp/cyber/models/knn.pkl', 'wb') as f:
        pickle.dump(rfc, f)
    s3_client.upload_file("tmp/cyber/models/knn.pkl", bucket_name, f"{folder_path}/knn/model.pkl")

    #SGD

    start_train = time.time()
    sgd = SGDClassifier(max_iter=1000, tol=1e-3)
    sgd.fit(X_train, y_train)
    end_train = time.time()

    start_test = time.time()
    y_pred5=sgd.predict(X_test)
    end_test = time.time()

    accuracy = accuracy_score(y_test, y_pred5)
    f1 = f1_score(y_test, y_pred5, average="macro")
    precision = precision_score(y_test, y_pred5, average="macro")
    recall = recall_score(y_test, y_pred5, average="macro")

    metrics.loc[len(metrics.index)] = [version, 'sgd', accuracy, f1, precision, recall, end_train-start_train, end_test-start_test]
    with open('./tmp/cyber/models/sgd.pkl', 'wb') as f:
        pickle.dump(rfc, f)
    s3_client.upload_file("tmp/cyber/models/sgd.pkl", bucket_name, f"{folder_path}/sgd/model.pkl")

    #Logistic Regression

    # start_train = time.time()
    # lrc = LogisticRegression(random_state=0, max_iter=1000)
    # lrc.fit(X_train, y_train)
    # end_train = time.time()

    # start_test = time.time()
    # y_pred6=lrc.predict(X_test)
    # end_test = time.time()

    # accuracy = accuracy_score(y_test, y_pred6)
    # f1 = f1_score(y_test, y_pred6)
    # precision = precision_score(y_test, y_pred6)
    # recall = recall_score(y_test, y_pred6)

    # metrics.loc[len(metrics.index)] = [version, 'lrc', accuracy, f1, precision, recall, end_train-start_train, end_test-start_test]
    # with open('./tmp/cyber/models/lrc.pkl', 'wb') as f:
    #     pickle.dump(rfc, f)
    # s3_client.upload_file("tmp/cyber/models/lrc.pkl", bucket_name, f"{folder_path}/lrc/model.pkl")

    db_details = {
        'dbname': db,
        'user': user,
        'password': pswd,
        'host': host,
        'port': port
    }
        
    insert_query = """
        INSERT INTO cyber_model_metrics (name, version, URI, in_use, accuracy, f1, precision, recall, train_time, test_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (name, version) DO NOTHING;
    """
    try:
        conn = psycopg2.connect(**db_details)
        cursor = conn.cursor()
        print("Connected to PostgreSQL successfully.")

        # Iterate through DataFrame rows and insert into the table
        for index, row in metrics.iterrows():
            cursor.execute(insert_query, (
                row['Model'], 
                row['Version'], 
                f"s3://cyberpipeline/version{version}/{row['Model']}/model.pkl", 
                False, 
                row['Accuracy'], 
                row['F1'], 
                row['Precision'], 
                row['Recall'], 
                row['Train_Time'], 
                row['Test_Time']
            ))
    
        conn.commit()
        print("Data inserted successfully.")

        cursor.close()
        conn.close()
        print("PostgreSQL connection closed.")
    except Exception as e:
        print(f"Failed to connect to PostgreSQL or insert data: {e}")