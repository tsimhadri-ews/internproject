import kfp
from kfp import dsl
from kfp import components
from check_condition import check_condition
from read_file import read_file
from train_op import train_op
from model_eval_deploy import model_eval_deploy
import os 

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
# Compile the pipeline
#kfp.compiler.Compiler().compile(ml_pipeline, 'intrusion_pipeline.yaml')
kfp.compiler.Compiler().compile(ml_pipeline, 'intrusion_pipeline.yaml')
print(f"Compiled the pipeline to: {os.path.abspath('intrusion_pipeline.yaml')}")
print("Compiled the pipeline")