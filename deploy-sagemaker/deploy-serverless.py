import boto3
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig
import os
import time

region = os.getenv('REGION')
role_name = os.getenv("SM_ROLE_NAME")
model_path = os.getenv('SM_MODEL_PATH')
role = boto3.client('iam').get_role(RoleName=role_name)['Role']['Arn']

model = HuggingFaceModel(
    py_version = "py39",
    #entry_point = "inference_code.py", # this will cause filenotfound error, do not uncomment
    transformers_version  = "4.26", # transformer version
    pytorch_version = "1.13", 
    model_data = model_path,
    role = role,
)

predictor = model.deploy(
    serverless_inference_config = ServerlessInferenceConfig(
        memory_size_in_mb=3072, 
        max_concurrency=5,
    ),
    endpoint_name = "text-summarizer-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
)
print("Succesfully deployed to Serverless Endpoint: ", predictor.endpoint_name)