FROM public.ecr.aws/lambda/python:3.9

ENV HF_HUB_REPO="luntaixia/cnn-summarizer"

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements.txt  .
RUN  python -m pip install --upgrade pip
RUN  yum install gcc -y
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy everything into function code
COPY . ${LAMBDA_TASK_ROOT}

# prepare model
RUN chmod +x prepare_model.py
RUN python3 prepare_model.py

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.api_gateway_lambda_handler" ]