import json
import sagemaker
import argparse

parser = argparse.ArgumentParser()
# model configs
parser.add_argument('--endpoint', required=True, type=str)
parser.add_argument('--text', required=True, type=str)
args = parser.parse_args()

sm = sagemaker.Session().sagemaker_runtime_client

prompt = {
  "articles": [args.text]
}

response = sm.invoke_endpoint(
    EndpointName=args.endpoint, 
    Body=json.dumps(prompt), 
    ContentType="application/json"
)

r = response["Body"].read()
r = json.loads(r)
print(r[0])