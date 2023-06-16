mkdir model;

# copy code files
cp prepare.py model/prepare.py;

# download pretrained model to local
cd model;
python prepare.py;

# zip the code and model together
cd model;
tar -czvf model.tar.gz *;

# upload to s3
aws s3 cp model.tar.gz s3://luntai-sagemaker-learning/;

# delete local files
cd ..;
cd ..;
rmdir model;