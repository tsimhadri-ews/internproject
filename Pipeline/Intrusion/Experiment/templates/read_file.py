def read_file(is_experiment: bool = False) -> None:
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
    
    # Dictionary to save mean, sd, and, encoder
    preprocess_df = {'version':1}
    
    #Perform normalization
    def zscore_normalization(df, name):
        mean = df[name].mean()
        sd = df[name].std()
        df[name] = (df[name] - mean) / sd
        preprocess_df[name] = (mean, sd)
        
    #Encode text 
    def encode_text(df, name):
        from sklearn.preprocessing import OrdinalEncoder
        enc = OrdinalEncoder()
        data = enc.fit_transform(df[name].values.reshape(-1,1))
        df[name] = data.flatten()
        preprocess_df[name] = base64.b64encode(pickle.dumps(enc)).decode('utf-8')
        # pickle.loads(a.encode('latin1'))
        
    #Data preprocessing
    def preprocess(df):

        for col in df.columns:
            t = (df[col].dtype)
            if col != 'outcome':
                if (t == int or t == float):
                    zscore_normalization(df, col)
               
                else:
                    df[col] = df[col].astype(str)
                    encode_text(df, col)
                            
        for col in df.columns:
            if len(df[col].unique()) == 1:
                df.drop(col, inplace=True,axis=1)
                preprocess_df[col] = None


        correlation = df.corrwith(df["outcome"])
      
        
        row = 0 
        for num in correlation:
            if num >= -0.05 and num <= 0.05:
                preprocess_df[df.columns[row]] = None
                df.drop(df.columns[row], axis=1, inplace=True)
                row -= 1
            row += 1
        return df
    
    db_details = {
        'dbname': db,
        'user': user,
        'password': pswd,
        'host': host,
        'port': port
    }
    
    # Connect to PostgreSQL
    engine = create_engine(f'postgresql+psycopg2://{db_details["user"]}:{db_details["password"]}@{db_details["host"]}:{db_details["port"]}/{db_details["dbname"]}', connect_args={'connect_timeout': 60})
    
    df = pd.DataFrame()
    
    try:
        with engine.connect() as conn:
            query = text('SELECT * FROM intrusion_data WHERE outcome != 2;')
            chunksize = 10000  # Adjust chunksize as per your memory and performance needs
            chunks = pd.read_sql_query(query, conn, chunksize=chunksize)
            i = 1
            for chunk in chunks:
                # print(f"chunk{i}")
                i = i + 1
                features_df = pd.json_normalize(chunk['features'])
                features_df['outcome'] = chunk['outcome']
                df = pd.concat([df, features_df], ignore_index=True)


                
                
                
    except Exception as e:
            print(f"Failed to fetch data: {e}")

    
    #df = df.drop(columns=['timestamp','uid'])
    df = preprocess(df)
    X = df.drop(columns=['outcome'])
    y = df['outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    bucket_name="intrusionpipeline"
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
    
    
    
    folder_path = './tmp/intrusion'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")
        

    
        
    df.to_csv("./tmp/intrusion/intrusion_data.csv")
    np.save("./tmp/intrusion/X_train.npy",X_train)
    np.save("./tmp/intrusion/y_train.npy",y_train)  
    np.save("./tmp/intrusion/X_test.npy",X_test)
    np.save("./tmp/intrusion/y_test.npy",y_test)
    
      
    if(not is_experiment):
        
        try:
            with engine.connect() as conn:
                query = text('SELECT * FROM metadata_table_intrusion ORDER BY version DESC LIMIT 1;')
                data = pd.read_sql_query(query, conn)
                version = data['version'].iloc[0] + 1
        except Exception as e:
            version = 1
        
        s3_client.upload_file("./tmp/intrusion/intrusion_data.csv", bucket_name, f"version{version}/intrusion_dataset.csv")
        s3_client.upload_file("./tmp/intrusion/X_train.npy", bucket_name, f"version{version}/X_train.npy")
        s3_client.upload_file("./tmp/intrusion/y_train.npy", bucket_name, f"version{version}/y_train.npy")
        s3_client.upload_file("./tmp/intrusion/X_test.npy", bucket_name, f"version{version}/X_test.npy")
        s3_client.upload_file("./tmp/intrusion/y_test.npy", bucket_name, f"version{version}/y_test.npy")
        
        preprocess_df['version'] = version
        mean_df = pd.DataFrame([preprocess_df])
        meta_df = pd.DataFrame(data = [[version, datetime.datetime.now(), len(X.columns), json.dumps(df.dtypes.astype(str).to_dict()),mean_df.iloc[0].to_json()]], columns = ['version', 'date', 'features', 'types','factor'])
        meta_df.to_sql("metadata_table_intrusion", engine, if_exists='append', index=False)
    
    else:
        s3_client.upload_file("./tmp/intrusion/intrusion_data.csv", bucket_name, f"experiment/intrusion_dataset.csv")
        s3_client.upload_file("./tmp/intrusion/X_train.npy", bucket_name, f"experiment/X_train.npy")
        s3_client.upload_file("./tmp/intrusion/y_train.npy", bucket_name, f"experiment/y_train.npy")
        s3_client.upload_file("./tmp/intrusion/X_test.npy", bucket_name, f"experiment/X_test.npy")
        s3_client.upload_file("./tmp/intrusion/y_test.npy", bucket_name, f"experiment/y_test.npy")