import kfp
from kfp import Client
import os 

# Replace with your Kubeflow Pipelines endpoint and credentials
KUBEFLOW_PIPELINES_URL = 'http://127.0.0.1:30'

# Kubeflow Pipelines URL




# Create a client to interact with Kubeflow Pipelines
client = Client(host=KUBEFLOW_PIPELINES_URL)
print(client)

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
