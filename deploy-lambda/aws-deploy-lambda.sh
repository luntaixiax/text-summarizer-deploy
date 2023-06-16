AWS_REGION=ca-central-1
AWS_ACCT_ID=981401473042
REPO_NAME="cnn-summarizer"
DOCKER_IMAGE_NAME="luntaixia/cnn-summarizer-lambda"
ROLE_NAME="lambda-ex"  # TODO: update this


ECR_REPO_URI="$AWS_ACCT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME"

# create role before move on
aws lambda create-function \
  --function-name $REPO_NAME \  # lambda function name
  --package-type Image \  # the type of lambda function
  --code ImageUri=$ECR_REPO_URI:latest \  # ECR image URI
  --role arn:aws:iam::$AWS_ACCT_ID:role/$ROLE_NAME # IAM ARN number

# update lambda if updated image
aws lambda update-function-code \
	--function-name $REPO_NAME  \
	--region $AWS_REGION \
	--image-uri $ECR_REPO_URI:latest