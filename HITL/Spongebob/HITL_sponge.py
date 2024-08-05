import os
import cv2
import streamlit as st
import boto3
import psycopg2
import json
from botocore.exceptions import ClientError
import streamlit as st
from streamlit_autorefresh import st_autorefresh

def get_secret():
    secret_name = "DBCreds"
    region_name = "us-east-1"

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
    secret_dict = json.loads(secret)
    username = secret_dict['username']
    password = secret_dict['password']
    host = secret_dict['host']
    port = secret_dict['port']
    dbname = secret_dict['dbname']

    return username, password, host, port, dbname

(user, pswd, host, port, db) = get_secret()

bucket_name = "spongebobpipeline"
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

# Directory paths
image_directory = '/home/ubuntu/objectdetection/prediction'
upload_path = './combined/train'

def upload_to_s3(image_path, txt_file_path, bucket_name, upload_path):
    # Define the key with the specified path
    image_key = os.path.join(upload_path, os.path.basename(image_path))
    text_key = os.path.join(upload_path, os.path.basename(txt_file_path)) if txt_file_path else None

    # Upload the image file to S3
    with open(image_path, 'rb') as image_file:
        s3_client.upload_fileobj(image_file, bucket_name, image_key)
    print(f"Image file uploaded to S3 as: {image_key}")

    # Upload the text file to S3
    if txt_file_path:
        with open(txt_file_path, 'rb') as text_file:
            s3_client.upload_fileobj(text_file, bucket_name, text_key)
        print(f"Text file uploaded to S3 as: {text_key}")

# Streamlit app
st.title("Find Spongebob")

# Use the session state for the current image index
if 'current_image_index' not in st.session_state:
    st.session_state.current_image_index = 0
if 'correct_count' not in st.session_state:
    st.session_state.correct_count = 0
if 'incorrect_count' not in st.session_state:
    st.session_state.incorrect_count = 0

test = 0 


# Display the current image
try:
    conn = psycopg2.connect(
        dbname=db,
        user=user,
        password=pswd,
        host=host,
        port=port
    )
    cursor = conn.cursor()
    print("connected to database")

    # Fetch the UIDs and outcomes from the database
    select_query = "SELECT uid, outcome FROM spongebob_outcomes WHERE confirmed = 2 limit 1"
    cursor.execute(select_query)
    rows = cursor.fetchall()
    if test < len(rows):
        for row in rows:
            uid, outcome = row
            image_name = f"{uid}.jpg"
            text_name = f"{uid}.txt"
            print(image_name)

            # Load and display the image
            image_path = os.path.join(image_directory, image_name)
            image = cv2.imread(image_path)

            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, caption=f"UID: {uid} (Outcome: {outcome})")

                st.write("Is the prediction correct?")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Yes", key=f"yes_button_{st.session_state.current_image_index}"):
                        st.session_state.correct_count += 1
                        test += 1 
                        st.session_state.current_image_index += 1
                        update_query = "UPDATE spongebob_outcomes SET confirmed = %s WHERE uid = %s"
                        cursor.execute(update_query, (outcome, uid))
                        conn.commit()
                        text_path = os.path.join(image_directory, text_name)
                        if os.path.exists(text_path):
                            upload_to_s3(image_path, text_path, bucket_name, upload_path)
                        else:
                            upload_to_s3(image_path, None, bucket_name, upload_path)
                        
                        

                with col2:
                    if st.button("No", key=f"no_button_{st.session_state.current_image_index}"):
                        st.session_state.incorrect_count += 1
                        st.session_state.current_image_index += 1
                        test += 1 
                        update_query = "UPDATE spongebob_outcomes SET confirmed = %s WHERE uid = %s"
                        cursor.execute(update_query, (1 - outcome, uid))
                        conn.commit()
                        
                        

                st.write(f"Correct predictions: {st.session_state.correct_count}")
                st.write(f"Incorrect predictions: {st.session_state.incorrect_count}")
                st.write(f"Total predictions: {st.session_state.incorrect_count + st.session_state.correct_count}")
            else:
                st.write(f"Image not found for UID: {uid}")
                test += 1 
                st.session_state.current_image_index += 1

        else:
            st.write("No more images available.")

except Exception as e:
    conn.rollback()
    st.write(f"An error occurred: {str(e)}")
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
