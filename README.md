# Project Architecture

![MLE Project.jpg](images/MLE_Project.jpg)

[https://miro.com/app/board/uXjVM9PubHU=/?share_link_id=646530040677](https://miro.com/app/board/uXjVM9PubHU=/?share_link_id=646530040677)

# Quick Start

All services and components of this project are packaged individually and will be run in separate Docker Containers, so you can use Docker Compose to orchestate those services. First clone this project:

```bash
git clone https://github.com/luntaixiax/text-summarizer-deploy.git
```

## 0. Prerequisite

**Mandatory**

You need `Docker` and `Docker-compose` installed, it is recommanded to use Mac or Linux operating system to run this project.

**Optional**

Although not required, it is still recommanded to have

1. **AWS** account ready (ideally with free tier not expired) if you need to explore the `SageMaker` and `Lambda/API Gateway` Deployment services
2. **Arize** account ready if you want to explore the Monitoring service. You can register an Arize account [here](https://app.arize.com/auth/join) and get your API key and Space Key here:
    
    ![https://storage.googleapis.com/arize-assets/fixtures/copy-keys.png](https://storage.googleapis.com/arize-assets/fixtures/copy-keys.png)
    

## 1. Config the project

### Set secret file

create a file named `secrets.toml` under the project root folder, copy the following content and replace the necessary fields:

```toml
[sagemaker]
aws_access_key_id = "YOUR-AWS-ACCESS-KEY"
aws_secret_access_key = "YOUR-AWS-ACCESS-KEY"
region = "YOUR-AWS-REGION"
sm_endpoint_name = "text-summarizer-2023-06-22-18-40-55"

[services]
local_service_name = "backend-standalone"
mlflow_service_name = "backend-mlflow"
lambda_endpoint = "YOUR-AWS-APIGATEWAY-URI"

[mysql]
ip = "mysql"
port = 3306
username = "root"
password = "luntaixia"
db = "summarizer"

[arize]
SPACE_KEY = "YOUR-ARIZE-SPACE-KEY"  
API_KEY = "YOUR-ARIZE-API-KEY"
MODEL_ID = "cnn-summarizer"

[monitoring]
ip = "monitoring"
port = 9020
```

<aside>
💡 ***Notes***: It is not mandatory to change the `sagemaker` section, `lamda_endpoint` and `arize` section if you do not have one. Just leave as it is, DO NOT DELETE ANY PART, the only downside is that you will not be able to use the corresponding services (AWS, Arize).

</aside>

### Set env file (Optional)

If you want to change the volume mapping path, in the project root folder, create a file called `.env` and put the following into the file:

```toml
VOLUME_MAPPING_PATH="YOUR-VOLUME-MAPPING-FOLDER"
MYSQL_ROOT_USER="YOUR-MYSQL-ROOT-USER"
MYSQL_ROOT_PASSWORD="YOUR-MYSQL-ROOT-PASSWORD"
```

## 2. Start all the services

Use `docker-compose` to start all the services. In the project root, execute `start-app.sh`:

```bash
# you can either create volume mapping dir in current folder
# or specify a preferred location to VOLUME_MAPPING_PATH
mkdir -p volume-mapping;
export VOLUME_MAPPING_PATH=${PWD}/volume-mapping;

# start the service
docker-compose up -d;
```

Wait for some minutes, and you should be able to access the UI from `localhost:8501`

![Untitled](images/Untitled.png)

## 3. Initialize the database

Just simply click on the button `Database Initialization` button on the homepage, weit for a second, until you see an info message appear at the top of the page:

![Untitled](images/Untitled%201.png)

Switch to the `Prediction History` subpage from the sidebar, you will be able to see the historical prediction data:

![Untitled](images/Untitled%202.png)

## 4. Setup the Grafana Dashboard

Grafana is listening on port `3000`, so go to [http://localhost:3000/login](http://localhost:3000/login) and use default username/password `admin` to login:

![Untitled](images/Untitled%203.png)

And it will ask you to change the password, do that and you will land on the main page.

### Setup the MySQL connection

1. From the sidebar, click `Home` > `Connections` > `Add new connection` and search for `mysql` in the search box.
2. click `Create a MySQL data source` on the top right corner
3. fill in the connection details (Host, Database, User, Password) as below:
    
    ![Untitled](images/Untitled%204.png)
    
    you need to use the password from your `secrets.toml` under the `mysql` section
    
4. leave other fields as default and click `Save & test` at the bottom

### Import the Dashboard

1. Go back to home page, and click `Home` > `Dashboards`
2. Click `New` > `Import`
3. Upload the `grafana-dashboard.json` from the project root folder and select MySQL connection we just established from the dropdown list:
    
    ![Untitled](images/Untitled%205.png)
    
4. click Import and you will see the Dashboard!
    
    ![Untitled](images/Untitled%206.png)
    

To this point, the main part of the services have been configured and launched.

Please explore the UI at [http://localhost:8501/](http://localhost:8501/) and the grafana dashboard, have fun!

## 5. Kickoff the Batch Prediction pipeline

### Start Airflow

This step is optional, this is a daily ETL pipeline that is scheduled by Airflow. To start the service, head into the `pipeline-batch` folder:

```bash
cd pipeline-batch;
```

Create some necessary folders for Airflow:

```bash
mkdir -p ./dags ./logs ./plugins ./config;
```

initialize and start airflow:

```bash
# initialize airflow database
docker-compose -f airflow-docker-compose.yaml up airflow-init;
# after the above compose is down, launch the below one
docker-compose -f airflow-docker-compose.yaml up -d;
```

wait for a few minutes for Airflow to load the DAG, and head over to [localhost8080/home](http://localhost:8080/login/), and login using default username/password `airflow`

![Untitled](images/Untitled%207.png)

### Configuration

1. Add a new Variable from UI: go to `Admin` > `Variables`, add one:
    
    ![Untitled](images/Untitled%208.png)
    
2. Add a new connection for mysql: go to `Admin` > `Connections`, add one:
    
    ![Untitled](images/Untitled%209.png)
    
3. Add a slack channel notification connection (optional): go to `Admin` > `Connections`, add one:
    
    ![Untitled](images/Untitled%2010.png)
    
    you can get one API key from slack API webhook section
    
    <aside>
    ⚠️ note you need to copy the webhook URL, blue part in the Host field and pink part in the password field! : https://hooks.slack.com/services[/x](https://hooks.slack.com/services/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)xxxxxx
    
    </aside>
    

### Run the pipeline

Head over to the Dag named `cnn_daily`, and run:

![Untitled](images/Untitled%2011.png)

# Model Part Architecture

## 1. Model and Training

The model involved will be a text summarizer which highlights the content of an article.

| Model Usage | Text summarization, i.e., summarize key content from an article |
| --- | --- |
| Model Input | A piece of article (in string text format) |
| Model Output | A piece of summarization (in string text format) |
| Training Dataset | https://huggingface.co/datasets/cnn_dailymail |
| Pretrained Model | https://huggingface.co/t5-small |
| Performance Metric | Rogue Score |

### Tools Involved

| Tool | Functionality |
| --- | --- |
| Huggingface | Pretrained transformers and training/testing Dataset |
| Mlflow | Tracking for model metrics, Versioning and logging |
| Ray | Distributed Training Engine, need GPU to accelarate |

### Retrain the Model

Training is already done, this project focus on deploying the models. If you would like to retrain the model, you can find the relevant code in the `train` folder, and execute `train.sh`:

```bash
# use train_ray.py if you need distributed training
python train.py \
  --BATCH_SIZE=16 \
  --EPOCHS=1 \
  --OUTPUT_DIR='YOUR-OUTPUT-DIR-HERE/cmm-summarizer' \
  --HF_TOKEN='YOUR-HF-TOKEN-HERE' \
  --HF_HUB_REPO='HF-ACCT-NAME/HF-REPO-NAME-HERE'
```

### Output of this step

After training, you will find:

1. The **model artifact** is uploaded to [luntaixia/cnn-summarizer · Hugging Face](https://huggingface.co/luntaixia/cnn-summarizer)
2. **Mlflow run logs** stored in your training folder, and the experiment name is set to `cnn-summarizer-fine-tune`, as specified in the `train.py`:
    
    ```python
    # MLflow tracking settings
    os.environ['MLFLOW_EXPERIMENT_NAME'] = 'cnn-summarizer-fine-tune'
    os.environ['MLFLOW_FLATTEN_PARAMS'] = '1'
    os.environ['HF_MLFLOW_LOG_ARTIFACTS '] = 'True' # whether to copy artifacts
    ```
    
    You can check the mlflow logs by login to the mlflow UI. Execute the following command in your training folder:
    
    ```bash
    mlflow ui
    ```
    
    ![Untitled](images/Untitled%2012.png)
    
    you will see the server running up on [localhost:5000](http://localhost:5000)
    
    ![Untitled](images/Untitled%2013.png)
    
    ![Untitled](images/Untitled%2014.png)
    

### References

[Fine-tune a 🤗 Transformers model — Ray 2.5.1](https://docs.ray.io/en/latest/ray-air/examples/huggingface_text_classification.html)

[A Full Guide to Finetuning T5 for Text2Text and Building a Demo with Streamlit](https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887)

[Log with MLflow and Hugging Face Transformers](https://www.youtube.com/watch?v=vmfaDZjeB4M)

## 2. Model Serving

After the Model get trained, the model will be deployed and served in four different environments:

| Env | Type | Code Location |
| --- | --- | --- |
| Mlflow Serve | Local - Realtime | ./deploy-mlflow and ./backend-fastapi |
| Local Serve | Local - Realtime | ./deploy-fastapi |
| AWS Lambda | Serverless | ./deploy-lambda |
| AWS SageMaker | Serverless | ./deploy-sagemaker |

![Untitled](images/Untitled%2015.png)

### Mlflow Serving

code is located in two folders (and thus two separate services):

| Code Folder | Note | Endpoint |
| --- | --- | --- |
| deploy-mlflow | use mlflow serve to host the model | localhost:5003/invocations |
| backend-fastapi | use fastapi for serving | localhost:5000/docs |

to start the Mlflow serve manually, try:

```bash
mlflow models serve \
	-h 0.0.0.0 \
	-p 5003 \
	--model-uri cnn-summarizer_mlflow_pyfunc \
	--env-manager local
```

**Sample Requests and Responce**

`/article/summary`

```bash
curl -X 'POST' \
  'http://localhost:5000/article/summarize' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "article": "Article goes here"
}'
```

```json
"Summary goes here"
```

`/article/summarize-batch`

```bash
curl -X 'POST' \
  'http://localhost:5000/article/summarize_batch' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "articles": [
    "Article 1 goes here",
    "Article 2 goes here"
  ]
}'
```

```json
[
  "Summary 1 goes here",
  "Summary 2 goes here"
]
```

`/model/score`

```bash
curl -X 'POST' \
  'http://localhost:5000/model/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "articles": [
    "Article 1 goes here",
    "Article 2 goes here"
  ],
  "targets": [
    "Target 1 goes here",
    "Target 2 goes here"
  ]
}'
```

```json
{
  "rouge1": 0.4807692307692308,
  "rouge2": 0.38181818181818183,
  "rougeL": 0.4807692307692308,
  "rougeLsum": 0.4807692307692308
}
```

### Local Serving

code is located in folder `deploy-fastapi`

Similar to Mlflow serving, the difference is the model artifact will be loaded by the fastapi web service directly, without the help of Mlflow serving. After starting the local serving service, the swagger UI can be accessed at `localhost:8000/docs`

**Sample Requests and Responce**

`/article/summary`

```bash
curl -X 'POST' \
  'http://localhost:8000/article/summarize' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "article": {
    "article": "Article goes here"
  },
  "config": {
    "num_beans": 8,
    "temperature": 1
  }
}'
```

```json
"Summary goes here"
```

`/article/summarize-batch`

```bash
curl -X 'POST' \
  'http://localhost:8000/article/summarize_batch' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "articles": {
    "articles": [
      "Article 1 goes here",
	    "Article 2 goes here"
    ]
  },
  "config": {
    "num_beans": 8,
    "temperature": 1
  }
}'
```

```json
[
  "Summary 1 goes here",
  "Summary 2 goes here"
]
```

`/model/score`

```bash
curl -X 'POST' \
  'http://localhost:8000/model/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "pairs": {
    "articles": [
      "Article 1 goes here",
      "Article 2 goes here"
    ],
    "targets": [
      "Target 1 goes here",
      "Target 2 goes here"
    ]
  },
  "config": {
    "num_beans": 8,
    "temperature": 1
  }
}'
```

```json
{
  "rouge1": 0.46153846153846156,
  "rouge2": 0.36363636363636365,
  "rougeL": 0.46153846153846156,
  "rougeLsum": 0.46153846153846156
}
```

### AWS Lambda

code is located in folder `deploy-lambda`

1. model will be dockerized to AWS ECR, refer to `aws-deploy-ecr.sh`
2. AWS Lambda will trigger the serving script, refer to `aws-deploy-lambda.sh`
3. request will be routed by AWS API Gateway

*The AWS Lambda Details*

![Untitled](images/Untitled%2016.png)

*The API Gateway Details*

![Untitled](images/Untitled%2017.png)

![Untitled](images/Untitled%2018.png)

**Sample Requests and Responce**

`/batch-predict`

```bash
curl  -X POST \
  'https://<YOUR-APIGATEWAY-URI-HERE>/batch-predict' \
  --header 'Accept: */*' \
  --header 'User-Agent: Thunder Client (https://www.thunderclient.com)' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "articles": [
    "Article 1 goes here",
    "Article 2 goes here"
  ],
  "num_beans": 10,
  "temperature": 1.0
}'
```

```json
[
  "Summary 1 goes here",
  "Summary 2 goes here"
]
```

`/online-predict`

```bash
curl  -X POST \
  'https://<YOUR-APIGATEWAY-URI-HERE>/online-predict' \
  --header 'Accept: */*' \
  --header 'User-Agent: Thunder Client (https://www.thunderclient.com)' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "article": "Article goes here",
  "num_beans": 10,
  "temperature": 1.0
}'
```

```json
"Summary goes here"
```

`/score`

```bash
curl  -X POST \
  'https://tnr120ix54.execute-api.ca-central-1.amazonaws.com/score' \
  --header 'Accept: */*' \
  --header 'User-Agent: Thunder Client (https://www.thunderclient.com)' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "articles": [
    "Article 1 goes here",
    "Article 2 goes here"
  ],
  "targets": [
    "Target 1 goes here",
    "Target 2 goes here"
  ]
  "num_beans": 10,
  "temperature": 1.0
}'
```

```json
{
  "rouge1": 0.27999999999999997,
  "rouge2": 0.041666666666666664,
  "rougeL": 0.24000000000000002,
  "rougeLsum": 0.24000000000000002
}
```

### AWS SageMaker

code is located in folder `deploy-sagemaker`, and the entry point is `deploy.sh`:

```bash
mkdir model;
cd model;
mkdir model code;

# copy code files
cd ..;
cp inference_code.py model/code/inference.py;
cp requirements.txt model/code/requirements.txt;
cp prepare.py model/prepare.py;

# download pretrained model to local
cd model;
python prepare.py;

# zip the code and model together
tar -czvf model.tar.gz model code;

# upload to s3
export SM_BUCKET='luntai-sagemaker-learning';
export SM_MODEL_PATH="s3://${SM_BUCKET}/text-summarizer/model.tar.gz";
aws s3 cp model.tar.gz ${SM_MODEL_PATH};

# delete local files
cd ..;
rm -r model;

# deploy to sagemaker
export REGION='ca-central-1';
export SM_ROLE_NAME='sagemaker-fullaccess';
python deploy-serverless.py; # use this if you want serverless deploy
#python deploy-realtime.py; # use this if you want realtime endpoint
```

1. model artifact and serving script (`inference_code.py`) will be packaged and uploaded to AWS S3 bucket
2. SageMaker Huggingface Model will host the model
3. requests will be sent via boto3 client, or can be sent by authorized HTTP request

<aside>
💡 Note that there are two deployment scripts:

| Script | When to use | Note |
| --- | --- | --- |
| deploy-realtime.py | if you want realtime endpoint | cost money |
| deploy-serverless.py | if you want serverless | need cold start |
</aside>

After executing the `deploy.sh` script, you can see the model and endpoint from AWS console UI:

![Untitled](images/Untitled%2019.png)

**References**

[Deploy models to Amazon SageMaker](https://huggingface.co/docs/sagemaker/inference#deploy-a-model-from-the-hub)

[Host a Pretrained Model on SageMaker — Amazon SageMaker Examples 1.0.0 documentation](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-script-mode/pytorch_bert/deploy_bert_outputs.html)

[Deployment Of A Locally Trained Model In Cloud Using AWS SageMaker](https://medium.com/geekculture/84af8989d065)

# Model Monitoring and UI

This part deals with model serving logs, performance logs and UI application. Involved services are:

| Service Tool | Functionality | Note |
| --- | --- | --- |
| Streamlit | central Web UI controlling everything | can do online prediction, monitoring and logging |
| MySQL | central database for logging and data persistency | will only be accessed by monitoring serving API and batch ETL pipeline |
| FastAPI | web server for monitoring API request handler | communicate with MySQL and Arize client |
| Arize.AI | log and monitor model performance (score metrics) | Arize will calculating embeddings and scores, and presumably send to arize server for dashboarding |
| Grafana | realtime dashboard for monitoring metrics | will look at performance, traffic, channel and input/output length |

# CI/CD with Github Actions

All components, except sagemaker deployment and batch pipeline part, need to rebuild Docker containers whenever there is a code change. The process is tedious if you do it manually each time. Github Actions can help automate the docker building and pushing jobs. The jobs will be triggered whenever new code is pushed to the master branch of each component.

| Automation pipeline | Descriptive Script | Process | Trigger when push to |
| --- | --- | --- | --- |
| build text summarizer model with mlflow | build_deploy_mlflow.yaml | CI | ./deploy-mlflow |
| text summarizer (hosted by mlflow) backend server with local serving | build_deploy_backend.yaml | CI | ./backend-fastapi |
| build text summarizer model with local serving | build_deploy_local.yaml | CI | ./deploy-fastapi |
| build text summarizer model with aws lambda | build_deploy_lambda.yaml | CI/CD | ./deploy-lambda |
| build frontend streamlit server with local serving | build_deploy_frontend.yaml | CI | ./frontend-streamlit |
| build monitoring image with local serving | build_monitoring.yaml | CI | ./monitoring |