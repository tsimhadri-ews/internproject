def read_file() -> None:
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
    preprocess_df = {'version':1}
    
    def zscore_normalization(df, name):
        mean = df[name].mean()
        sd = df[name].std()
        df[name] = (df[name] - mean) / sd
        preprocess_df[name] = (mean, sd)
    def preprocess(df):
        df = df.drop(columns=['url'])
        
        for c in df.columns:
            if len(df[c].unique()) == 1:
                df.drop(columns=[c], inplace=True)
        
        corr_matrix = df.corr()
        target_corr = corr_matrix['outcome']
        threshold=0.1
        drop_features = target_corr[abs(target_corr)<=threshold].index.tolist()
        df.drop(columns=drop_features, inplace=True)
        
        for i in df.columns:
            if i != 'outcome':
                zscore_normalization(df, i)
                
        return df

    db_details = {
        'dbname': db,
        'user': user,
        'password': pswd,
        'host': host,
        'port': port
    }

    
    engine = create_engine(f'postgresql+psycopg2://{db_details["user"]}:{db_details["password"]}@{db_details["host"]}:{db_details["port"]}/{db_details["dbname"]}')

    df = pd.DataFrame()
            
    try:
        with engine.connect() as conn:
            query = text('SELECT * FROM phishing_data WHERE outcome != 2;')
            chunksize = 10000 

            chunks = pd.read_sql_query(query, conn, chunksize=chunksize)

            features_list = []

            for chunk in chunks:
                features_df = pd.json_normalize(chunk['features'])
                features_df['outcome'] = chunk['outcome']
                
                df = pd.concat([df, features_df], ignore_index=True)

    except Exception as e:
        print(f"Failed to fetch data: {e}")


    df = preprocess(df)
    
    X = df.drop(columns=['outcome'])
    y = df['outcome']
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
    bucket_name="phishingpipeline"
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
    
    print(s3_client)
    
    folder_path = './tmp/phishing'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")
        
    try:
        with engine.connect() as conn:
            query = text('SELECT * FROM metadata_table_phishing ORDER BY version DESC LIMIT 1;')
            data = pd.read_sql_query(query, conn)
            version = data['version'].iloc[0] + 1
            print(version)
    except Exception as e:
        version = 1
    
    df.to_csv("./tmp/phishing/phishing_data.csv")
    s3_client.upload_file("./tmp/phishing/phishing_data.csv", bucket_name, f"version{version}/phishing_dataset.csv")
    np.save("./tmp/phishing/X_train.npy",X_train)
    s3_client.upload_file("./tmp/phishing/X_train.npy", bucket_name, f"version{version}/X_train.npy")
    np.save("./tmp/phishing/y_train.npy",y_train)
    s3_client.upload_file("./tmp/phishing/y_train.npy", bucket_name, f"version{version}/y_train.npy")
    np.save("./tmp/phishing/X_test.npy",X_test)
    s3_client.upload_file("./tmp/phishing/X_test.npy", bucket_name, f"version{version}/X_test.npy")
    np.save("./tmp/phishing/y_test.npy",y_test)
    s3_client.upload_file("./tmp/phishing/y_test.npy", bucket_name, f"version{version}/y_test.npy")
        

    preprocess_df['version'] = version
    mean_df = pd.DataFrame([preprocess_df])
    meta_df = pd.DataFrame(data = [[version, datetime.datetime.now(), len(X.columns), json.dumps(df.dtypes.astype(str).to_dict()),mean_df.iloc[0].to_json()]], columns = ['version', 'date', 'features', 'types','factor'])
    meta_df.to_sql("metadata_table_phishing", engine, if_exists='append', index=False)

#make some changes to the file 