#!/bin/sh
#mlflow ui
#PATH="$HOME/.pyenv/bin:$PATH"
mlflow models serve -h 0.0.0.0 -p ${PORT} --model-uri cnn-summarizer_mlflow_pyfunc --env-manager local