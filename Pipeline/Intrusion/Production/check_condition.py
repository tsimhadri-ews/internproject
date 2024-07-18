def check_condition() -> bool:
    import os
    import pandas as pd
    from sqlalchemy import create_engine, text
    import boto3
    import json

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
    engine = create_engine(f'postgresql+psycopg2://{db_details["user"]}:{db_details["password"]}@{db_details["host"]}:{db_details["port"]}/{db_details["dbname"]}', connect_args={'connect_timeout': 60})

    try:
        with engine.connect() as conn:
            query = text('SELECT count(*) FROM intrusion_outcomes;')
            data = pd.read_sql_query(query, conn)
            count = data.iloc[0]['count']
    except Exception as e:
        print("error")
    
    try:
        with engine.connect() as conn:
            query = text('SELECT count(*) FROM metadata_table_intrusion;')
            data = pd.read_sql_query(query, conn)
            meta_count = data.iloc[0]['count']
    except Exception as e:
        print("error")

    if (count % 5 == 0 and count != 0) or meta_count == 0:
        return True
    else:
        return False