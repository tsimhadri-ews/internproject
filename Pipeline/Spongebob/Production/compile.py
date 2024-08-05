import kfp
from kfp import dsl
from kfp import components
from read_file import read_file
from train_op import train_op
from check_condition import check_condition
from model_eval_deploy import model_eval_deploy
import os 
import requests
from bs4 import BeautifulSoup
from kfp_server_api.exceptions import ApiException
from datetime import datetime




check_condition_op = components.func_to_container_op(func=check_condition, base_image='python:3.7', packages_to_install=['pandas==1.1.5', 'sqlalchemy==1.4.45', 'boto3', 'psycopg2-binary','paramiko'])
read_csv_op = components.func_to_container_op(func=read_file, output_component_file='preprocess.yaml', base_image='python:3.7', packages_to_install=['pandas==1.1.5','scikit-learn==1.0.1', 'kfp', 'numpy', 'minio', 'psycopg2-binary', 'sqlalchemy==1.4.45','boto3','paramiko'])
train_op = components.func_to_container_op(func=train_op, output_component_file='train.yaml', base_image='python:3.7', packages_to_install=['pandas', 'scikit-learn==1.0.1','numpy','minio', 'tensorflow', 'psycopg2-binary', 'sqlalchemy','boto3','paramiko'])
eval_deploy = components.func_to_container_op(func=model_eval_deploy, output_component_file='eval_deploy.yaml', base_image='python:3.7', packages_to_install=['pandas', 'scikit-learn==1.0.1','numpy','minio', 'tensorflow', 'psycopg2-binary', 'sqlalchemy','boto3','kubernetes','kserve','paramiko'])

read_data_op = kfp.components.load_component_from_file('preprocess.yaml')
train_op = kfp.components.load_component_from_file('train.yaml')
eval_deploy_op = kfp.components.load_component_from_file('eval_deploy.yaml')



def ml_pipeline():
    check_condition = check_condition_op()
    check_condition.execution_options.caching_strategy.max_cache_staleness = "P0D"
    with dsl.Condition(check_condition.output == 'True'):
        preprocess = read_csv_op()
        preprocess.execution_options.caching_strategy.max_cache_staleness = "P0D"
        train = train_op().after(preprocess)
        train.execution_options.caching_strategy.max_cache_staleness = "P0D"
        eval_deploy = eval_deploy_op().after(train)
        eval_deploy.execution_options.caching_strategy.max_cache_staleness = "P0D"
        
print("compiling pipeline")

def run_pipeline(unique_id, yaml_file):

    # Get the credentials from Kubeflow secrets 
    KUBEFLOW_HOST = 'http://acc85673e1f094914a006f330bb51cb8-353421018.us-east-1.elb.amazonaws.com'
    KUBEFLOW_USERNAME = os.getenv('KUBEFLOW_USERNAME') 
    KUBEFLOW_PASSWORD = os.getenv('KUBEFLOW_PASSWORD') 
    KUBEFLOW_TOKEN = os.getenv('KUBEFLOW_TOKEN')

    print("user", KUBEFLOW_USERNAME)
    print("password", KUBEFLOW_PASSWORD)
    print("token", KUBEFLOW_TOKEN)

    # Create a new session for Kubeflow 
    session = requests.Session()
    login_url = f"{KUBEFLOW_HOST}" 
    response = session.get(login_url)
    assert response.status_code == 200, f"Failed to access login page: {response.status_code}"
    print("Accessed login page")

    # Scrape to get the session id 
    soup = BeautifulSoup(response.text, 'html.parser')
    login_form = soup.find('form')
    login_action = login_form['action']
    hidden_inputs = login_form.find_all("input", type="hidden")

    login_data = {
        'login': KUBEFLOW_USERNAME,
        'password': KUBEFLOW_PASSWORD,
    }
    for hidden_input in hidden_inputs:
        login_data[hidden_input['name']] = hidden_input['value']


    login_post_url = f"{KUBEFLOW_HOST}{login_action}"
    response = session.post(login_post_url, data=login_data, allow_redirects=True)


    assert response.status_code == 200, f"Failed to log in: {response.status_code}"
    print("Logged in successfully")


    session_cookie = session.cookies.get_dict()
    print(f"Session cookie: {session_cookie}")


    if not session_cookie:
        raise ValueError("Session cookie is empty. Login failed.")

    cookie_str = '; '.join([f"{key}={value}" for key, value in session_cookie.items()])
    print(f"Formatted cookie string: {cookie_str}")

    api_endpoint = f"{KUBEFLOW_HOST}/pipeline"
    namespace = "kubeflow-user-example-com"
    client = kfp.Client(host=api_endpoint, cookies=cookie_str, namespace=namespace, existing_token=KUBEFLOW_TOKEN )

    
    experiment_name = f'Test_Experiment_{unique_id}'
    pipeline_name = f'test_pipeline_{unique_id}'
    
    run_name = f'Find_Spongebob_Run_{unique_id}'
    pipeline_file = yaml_file
    
    # Create an experiment
    try:
        experiment = client.create_experiment(name=experiment_name, namespace=namespace)
        print(f'Experiment {experiment_name} created with ID: {experiment.id}')
    except ApiException as e:
        print(f"Exception when creating experiment: {e}")
        print(f"Status: {e.status}")
        print(f"Reason: {e.reason}")
        print(f"Headers: {e.headers}")
        print(f"Body: {e.body}")


    # Upload the pipeline
    pipeline = client.upload_pipeline(pipeline_file, pipeline_name=pipeline_name)

    print(f'Pipeline {pipeline_name} uploaded successfully with ID: {pipeline.id}')
    
    arguments = {} 

    # Run the pipeline 
    try:
        run = client.run_pipeline(experiment.id, run_name, pipeline_file, arguments)
        print(f'Pipeline run {run_name} started with ID: {run.id}')
    except ApiException as e:
        print(f"Exception when running pipeline: {e}")
        print(f"Status: {e.status}")
        print(f"Reason: {e.reason}")
        print(f"Headers: {e.headers}")
        print(f"Body: {e.body}")


unique_id = datetime.now().strftime("%Y%m%d%H%M%S")


# Compile the pipeline
yaml_file = f'spongebob_pipeline_{unique_id}.yaml'
kfp.compiler.Compiler().compile(ml_pipeline, yaml_file)
print(f"Compiled the pipeline to: {os.path.abspath(yaml_file)}")
print("Compiled the pipeline")
run_pipeline(unique_id, yaml_file)
