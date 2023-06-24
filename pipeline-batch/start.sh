mkdir -p ./dags ./logs ./plugins ./config;
#echo -e "AIRFLOW_UID=$(id -u)" > .env;
docker-compose -f airflow-docker-compose.yaml up airflow-init;
docker-compose -f airflow-docker-compose.yaml up -d;