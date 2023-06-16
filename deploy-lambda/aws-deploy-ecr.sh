AWS_REGION=ca-central-1
AWS_ACCT_ID=981401473042
REPO_NAME="cnn-summarizer"
DOCKER_IMAGE_NAME="luntaixia/cnn-summarizer-lambda"

docker build -t $DOCKER_IMAGE_NAME .

# login to AWS ECR
aws ecr get-login-password \
  --region $AWS_REGION | docker login \
  --username AWS \
  --password-stdin \
  $AWS_ACCT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# create a new repo
aws ecr create-repository \
  --repository-name $REPO_NAME \
  --image-scanning-configuration scanOnPush=true \
  --image-tag-mutability MUTABLE # allow tag to be overwritten by new image

ECR_REPO_URI="$AWS_ACCT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME"

# tag the latest docker image the same name on ECR

docker tag \
  $DOCKER_IMAGE_NAME:latest \
  $ECR_REPO_URI:latest

# push to ECR
docker push $ECR_REPO_URI:latest