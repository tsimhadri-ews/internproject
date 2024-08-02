def read_file():
    import boto3
    from botocore.exceptions import ClientError
    import json 
    from io import StringIO
    import paramiko 
    import os
    
    secret_name = "key"
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

    original_string = secret_dict['private']
    wrapped_key_content = '\n'.join(original_string[i:i+64] for i in range(0, len(original_string), 64))
    formatted_key = f"-----BEGIN RSA PRIVATE KEY-----\n{wrapped_key_content}\n-----END RSA PRIVATE KEY-----"

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

    # SSH connection details
    hostname = 'ec2-54-226-169-192.compute-1.amazonaws.com'
    port = 22
    username = 'ubuntu'

    key = formatted_key
    key_file = StringIO(key)
    # Load the private key
    private_key = paramiko.RSAKey.from_private_key(key_file)

    # Establish SSH connection
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, port, username, pkey=private_key)
    
    venv = 'yolo/bin/activate'
    remote_directory = '/home/ubuntu/objectdetection'
    yolo = 'yolo_code'

    script_test = 'download.py'

    command = f'cd {remote_directory} && source {venv} && cd {yolo} && python {script_test}'
    stdin, stdout, stderr = ssh_client.exec_command(command)
    #print(stdout.read().decode())
    #print(stderr.read().decode())
    while True:
        # Read from stdout and stderr
        line_stdout = stdout.readline()
        line_stderr = stderr.readline()

        # Print the output
        if line_stdout:
            print(line_stdout, end='')  # Print stdout line
        if line_stderr:
            print(line_stderr, end='')  # Print stderr line

        # Check if the process is finished
        if stdout.channel.exit_status_ready():
            break

    # Print any remaining output
    print(stdout.read().decode())
    print(stderr.read().decode())

    # Close the connection
    ssh_client.close()