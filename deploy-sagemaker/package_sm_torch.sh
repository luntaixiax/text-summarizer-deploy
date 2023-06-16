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
python deploy-sagemaker.py;