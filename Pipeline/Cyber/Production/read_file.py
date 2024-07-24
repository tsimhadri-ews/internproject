def read_file() -> None:
    import os
    import pandas as pd
    import numpy as np
    from minio import Minio
    from scipy.special import boxcox
    from sklearn.model_selection import train_test_split
    import boto3
    import json
    import pickle
    
    import base64
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

    def encode_text(df, name):
        from sklearn.preprocessing import OrdinalEncoder
        enc = OrdinalEncoder()
        data = enc.fit_transform(df[name].values.reshape(-1,1))
        df[name] = data.flatten()
        preprocess_df[name] = base64.b64encode(pickle.dumps(enc)).decode('utf-8')

        
    def preprocess(df):        
        for c in df.columns:
            if len(df[c].unique()) == 1:
                df.drop(columns=[c], inplace=True)
        
        for col in df.columns:
            print("not empty")
            t = (df[col].dtype)
            if t == int or t == float:
                df[col] = boxcox(df[col], 0.5)
                zscore_normalization(df, col)
            else:
                encode_text(df, col)

        df.drop(columns=["label"], inplace=True)

        corr_matrix = df.corr()
        target_corr = corr_matrix['attack_cat']
        threshold=0.05
        drop_features = target_corr[abs(target_corr)<=threshold].index.tolist()
        df.drop(columns=drop_features, inplace=True)
                
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
            query = text('SELECT * FROM cyber_data WHERE outcome != NULL;')
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
    
    X = df.drop(columns=["attack_cat"])
    y = df["attack_cat"]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
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
    
    print(s3_client)
    
    folder_path = './tmp/cyber'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")
        
    try:
        with engine.connect() as conn:
            query = text('SELECT * FROM metadata_table_cyber ORDER BY version DESC LIMIT 1;')
            data = pd.read_sql_query(query, conn)
            version = data['version'].iloc[0] + 1
            print(version)
    except Exception as e:
        version = 1
    
    df.to_csv("./tmp/cyber/cyber_data.csv")
    s3_client.upload_file("./tmp/cyber/cyber_data.csv", bucket_name, f"version{version}/cyber_dataset.csv")
    np.save("./tmp/cyber/X_train.npy",X_train)
    s3_client.upload_file("./tmp/cyber/X_train.npy", bucket_name, f"version{version}/X_train.npy")
    np.save("./tmp/cyber/y_train.npy",y_train)
    s3_client.upload_file("./tmp/cyber/y_train.npy", bucket_name, f"version{version}/y_train.npy")
    np.save("./tmp/cyber/X_test.npy",X_test)
    s3_client.upload_file("./tmp/cyber/X_test.npy", bucket_name, f"version{version}/X_test.npy")
    np.save("./tmp/cyber/y_test.npy",y_test)
    s3_client.upload_file("./tmp/cyber/y_test.npy", bucket_name, f"version{version}/y_test.npy")
        

    preprocess_df['version'] = version
    mean_df = pd.DataFrame([preprocess_df])
    meta_df = pd.DataFrame(data = [[version, datetime.datetime.now(), len(X.columns), json.dumps(df.dtypes.astype(str).to_dict()),mean_df.iloc[0].to_json()]], columns = ['version', 'date', 'features', 'types','factor'])
    meta_df.to_sql("metadata_table_cyber", engine, if_exists='append', index=False)

#make some changes to the file 
#run pipeline pls