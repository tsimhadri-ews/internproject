import requests
from kfp import Client
from kfp_server_api.configuration import Configuration
from kfp_server_api.api_client import ApiClient
import os

# Load credentials from environment variables
KUBEFLOW_HOST = "http://127.0.0.1:30"
KUBEFLOW_USERNAME = os.getenv('USER')
KUBEFLOW_PASSWORD = os.getenv('PASSWORD')

# Authenticate and get session cookie
session = requests.Session()
login_url = f"{KUBEFLOW_HOST}/dex/auth/local/login?back=&state=y27bwdhq72jkijoltkspi7t32"
response = session.get(login_url)
assert response.status_code == 200, f"Failed to access login page: {response.status_code}"

login_data = {
    'login': KUBEFLOW_USERNAME,
    'password': KUBEFLOW_PASSWORD,
}
response = session.post(login_url, data=login_data)
assert response.status_code == 200, f"Failed to log in: {response.status_code}"

# Extract session cookie
session_cookie = session.cookies.get_dict()

# Configure kfp client
client = Client(host=f"{KUBEFLOW_HOST}/pipeline", cookies=session_cookie)

# Define the path to the pipeline YAML file
pipeline_file = 'Pipeline/Intrusion/Production/intrusion_pipeline.yaml'

# Define the pipeline name and experiment name
pipeline_name = 'Intrusion Detection Pipeline'
experiment_name = 'Intrusion Detection Experiment'

# Upload the pipeline
pipeline = client.upload_pipeline(pipeline_file, pipeline_name)

# Create an experiment
experiment = client.create_experiment(experiment_name)

# Run the pipeline
run = client.run_pipeline(experiment.id, 'Intrusion Detection Run', pipeline.id)

print(f'Pipeline {pipeline_name} is running with run ID: {run.id}')

