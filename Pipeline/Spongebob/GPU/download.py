
import boto3
import os 



bucket_name="spongebobpipeline"
role_arn = 'arn:aws:iam::533267059960:role/aws-s3-access'
session_name = 'kubeflow-pipeline-session'
sts_client = boto3.client('sts')
response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
credentials = response['Credentials']
# Configure AWS SDK with temporary credentials
s3 = boto3.client('s3',
                      aws_access_key_id=credentials['AccessKeyId'],
                      aws_secret_access_key=credentials['SecretAccessKey'],
                      aws_session_token=credentials['SessionToken'])


folder_path_test = 'combined/test'
if not os.path.exists(folder_path_test):
    os.makedirs(folder_path_test)
    print(f"Folder '{folder_path_test}' created successfully.")
else:
    print(f"Folder '{folder_path_test}' already exists.")

folder_path_train = 'combined/train'
if not os.path.exists(folder_path_train):
    os.makedirs(folder_path_train)
    print(f"Folder '{folder_path_train}' created successfully.")
else:
    print(f"Folder '{folder_path_train}' already exists.")



response_test = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path_test)
response_train = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path_train)
# local_folder_path = "./tmp/spongebob/train"
local_folder_path = './combined/train'
train_dir = './combined/train'
test_dir = './combined/test'

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Check if the folder contains any objects
if 'Contents' in response_train:
    for obj in response_train['Contents']:
        file_path = obj['Key']
        print(file_path)
        local_file_path = os.path.join(local_folder_path, os.path.basename(file_path))
        print(local_file_path)
        # Download the file from S3
        s3.download_file(bucket_name, file_path, local_file_path)
        print(f"Downloaded {file_path} to {local_file_path}")
else:
    print(f"No objects found in s3://{bucket_name}/{folder_path_train}")

local_folder_path = './combined/test'

if 'Contents' in response_test:
    for obj in response_test['Contents']:
        file_path = obj['Key']
        print(file_path)
        local_file_path = os.path.join(local_folder_path, os.path.basename(file_path))
        print(local_file_path)
        # Download the file from S3
        s3.download_file(bucket_name, file_path, local_file_path)
        print(f"Downloaded {file_path} to {local_file_path}")
else:
    print(f"No objects found in s3://{bucket_name}/{folder_path_test}")