export VOLUME_MAPPING_PATH=/home/luntaixia/Downloads;
mkdir -p volume-mapping;
export VOLUME_MAPPING_PATH=${PWD}/volume-mapping;

chmod -x start-app.sh;
docker-compose up;