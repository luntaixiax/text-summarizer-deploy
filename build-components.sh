docker build -t luntaixia/cnn-summarizer-mlflow /deploy-mlflow;
docker build -t luntaixia/cnn-summarizer-backend /backend-fastapi;
docker build -t luntaixia/cnn-summarizer-local /deploy-fastapi;
docker build -t luntaixia/cnn-summarizer-monitoring /monitoring;
docker build -t luntaixia/cnn-summarizer-frontend /frontend-streamlit;