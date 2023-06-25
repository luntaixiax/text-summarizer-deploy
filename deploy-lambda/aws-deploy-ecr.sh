AWS_REGION=$(aws configure get region)
AWS_ACCT_ID=$(aws sts get-caller-identity | jq '.UserId') # need to install jq: sudo apt-get install jq
REPO_NAME="cnn-summarizer"
DOCKER_IMAGE_NAME="luntaixia/cnn-summarizer-lambda"
ECR_URI="$AWS_ACCT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
ECR_REPO_URI="$ECR_URI/$REPO_NAME"

docker build -t $DOCKER_IMAGE_NAME .

# login to AWS ECR
aws ecr get-login-password \
  --region $AWS_REGION | docker login \
  --username AWS \
  --password-stdin \
  $ECR_URI

# create a new repo
aws ecr create-repository \
  --repository-name $REPO_NAME \
  --image-scanning-configuration scanOnPush=true \
  --image-tag-mutability MUTABLE # allow tag to be overwritten by new image

# tag the latest docker image the same name on ECR
docker tag \
  $DOCKER_IMAGE_NAME:latest \
  $ECR_REPO_URI:latest

# push to ECR
docker push $ECR_REPO_URI:latest