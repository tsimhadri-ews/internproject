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
            query = text('select count(*) from phishing_data as phd join phishing_outcomes as pho on pho.uid = phd.uid where phd.outcome!=2;')
            data = pd.read_sql_query(query, conn)
            count = data.iloc[0]['count']
    except Exception as e:
        print("error")
    
    try:
        with engine.connect() as conn:
            query = text('SELECT count(*) FROM metadata_table_phishing;')
            data = pd.read_sql_query(query, conn)
            meta_count = data.iloc[0]['count']
    except Exception as e:
        print("error")

    try:
        with engine.connect() as conn:
            query = text('select count(*) from phishing_data as phd join phishing_outcomes as pho on pho.uid = phd.uid where pho.outcome!=phd.outcome and phd.outcome!=2;')
            data = pd.read_sql_query(query, conn)
            amount_incorrect = data.iloc[0]['count']
    except Exception as e:
        print("error")
        
    pct = amount_incorrect/count
    if count >= 10 or amount_incorrect > 2:
        try:
            with engine.connect() as conn:
                delete_query = text("DELETE FROM phishing_outcomes pho USING phishing_data phd WHERE pho.uid = phd.uid;")
                result = conn.execute(delete_query)
                conn.commit()
        except Exception as e:
            print("error")
        
    if (count >= 10 and count != 0) or meta_count == 0 or amount_incorrect > 2:
        return True
    else:
        return False