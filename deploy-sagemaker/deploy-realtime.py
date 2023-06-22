import boto3
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel
import os
import time

region = os.getenv('REGION')
role_name = os.getenv("SM_ROLE_NAME")
model_path = os.getenv('SM_MODEL_PATH')
role = boto3.client('iam').get_role(RoleName=role_name)['Role']['Arn']

model = HuggingFaceModel(
    py_version = "py39",
    entry_point = "inference_code.py",
    transformers_version  = "4.26", # transformer version
    pytorch_version = "1.13",
    model_data = model_path,
    role = role,
)
# deploy model to SageMaker Inference
predictor = model.deploy(
   initial_instance_count=1,
   instance_type="ml.m5.xlarge",
   endpoint_name = "text-summarizer-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
)
print("Succesfully deployed to Realtime Endpoint: ", predictor.endpoint_name)