import kfp
from kfp import dsl
from kfp import components
from check_condition import check_condition
from read_file import read_file
from train_op import train_op
from model_eval_deploy import model_eval_deploy
import os 
import requests
from bs4 import BeautifulSoup
from kfp_server_api.exceptions import ApiException





print("running file")

check_condition_op = components.func_to_container_op(func=check_condition, base_image='python:3.7', packages_to_install=['pandas==1.1.5', 'sqlalchemy==1.4.45', 'boto3', 'psycopg2-binary'])

read_csv_op = components.func_to_container_op(func=read_file, output_component_file='preprocess.yaml', base_image='python:3.7', packages_to_install=['pandas==1.1.5','scikit-learn==1.0.1', 'kfp', 'numpy', 'minio', 'psycopg2-binary', 'sqlalchemy==1.4.45','boto3'])

train_op = components.func_to_container_op(func=train_op, output_component_file='train.yaml', base_image='python:3.7', packages_to_install=['pandas', 'scikit-learn==1.0.1','numpy','minio', 'tensorflow', 'psycopg2-binary', 'sqlalchemy','boto3'])

eval_deploy = components.func_to_container_op(func=model_eval_deploy, output_component_file='eval_deploy.yaml', base_image='python:3.7', packages_to_install=['pandas', 'scikit-learn==1.0.1','numpy','minio', 'tensorflow', 'psycopg2-binary', 'sqlalchemy','boto3','kubernetes','kserve'])

read_data_op = kfp.components.load_component_from_file('preprocess.yaml')
train_op = kfp.components.load_component_from_file('train.yaml')
eval_deploy_op = kfp.components.load_component_from_file('eval_deploy.yaml')

def ml_pipeline():
    print("running pipeline")
    check_condition = check_condition_op()
    check_condition.execution_options.caching_strategy.max_cache_staleness = "P0D"
    with dsl.Condition(check_condition.output == 'True'):
        print("running condition")
        preprocess = read_csv_op()
        preprocess.execution_options.caching_strategy.max_cache_staleness = "P0D"
        train = train_op().after(preprocess)
        train.execution_options.caching_strategy.max_cache_staleness = "P0D"
        eval_deploy = eval_deploy_op().after(train)
        eval_deploy.execution_options.caching_strategy.max_cache_staleness = "P0D"
print("compiling pipeline")

def run_pipeline(yaml_file):
    KUBEFLOW_HOST = 'http://acc85673e1f094914a006f330bb51cb8-353421018.us-east-1.elb.amazonaws.com'
    KUBEFLOW_USERNAME = os.getenv('KUBEFLOW_USERNAME') #runner
    KUBEFLOW_PASSWORD = os.getenv('KUBEFLOW_PASSWORD') #none

    print("user", KUBEFLOW_USERNAME)
    print("password", KUBEFLOW_PASSWORD)


    session = requests.Session()
    login_url = f"{KUBEFLOW_HOST}" 
    response = session.get(login_url)
    assert response.status_code == 200, f"Failed to access login page: {response.status_code}"
    print("Accessed login page")

    #apparently kubeflow doesnt want you to log onto their webpage 
    #kubeflow sucks
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
    namespace = "kubeflow"
    client = kfp.Client(host=api_endpoint, cookies=cookie_str, namespace=namespace)


    experiment_name = 'Test Experiment2'

    try:
        experiment = client.create_experiment(name=experiment_name, namespace=namespace)
        print(f'Experiment {experiment_name} created with ID: {experiment.id}')
    except ApiException as e:
        print(f"Exception when creating experiment: {e}")
        print(f"Status: {e.status}")
        print(f"Reason: {e.reason}")
        print(f"Headers: {e.headers}")
        print(f"Body: {e.body}")
    print(client.list_experiments())

    pipeline_file = yaml_file 

    # Define the pipeline name
    pipeline_name = str(pipeline_file)

    # Upload the pipeline
    pipeline = client.upload_pipeline(pipeline_file, pipeline_name=pipeline_name)

    print(f'Pipeline {pipeline_name} uploaded successfully with ID: {pipeline.id}')
    run_name = 'Intrusion Detection Run9' #change 
    arguments = {} 

    try:
        run = client.run_pipeline(experiment.id, run_name, pipeline_file, arguments)
        print(f'Pipeline run {run_name} started with ID: {run.id}')
    except ApiException as e:
        print(f"Exception when running pipeline: {e}")
        print(f"Status: {e.status}")
        print(f"Reason: {e.reason}")
        print(f"Headers: {e.headers}")
        print(f"Body: {e.body}")





# Compile the pipeline
#kfp.compiler.Compiler().compile(ml_pipeline, 'intrusion_pipeline.yaml')
kfp.compiler.Compiler().compile(ml_pipeline, 'intrusion_pipeline.yaml')
print(f"Compiled the pipeline to: {os.path.abspath('intrusion_pipeline.yaml')}")
print("Compiled the pipeline")
run_pipeline('intrusion_pipeline.yaml')


