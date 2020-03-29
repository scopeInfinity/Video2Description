#!/bin/bash
set -e
echo "Starting container as development environment!!!"

echo "Stopping any running V2D containers"
bash docker_stop.sh

docker network create --subnet=172.14.0.0/24 v2d_net

echo "Starting backend"
docker container run --name "v2d_backend" -d -e "V2D_CONFIG_FILE=config_docker.json" \
  --net v2d_net --ip 172.14.0.2 \
  --mount type=bind,source="$(pwd)"/src,target=/home/si/v2d/src,readonly \
  --mount type=bind,source="$(pwd)"/uploads,target=/mnt/v2d/uploads/ \
  scopeinfinity/video2description:deploy \
  /bin/bash -i -c 'python -m backend.parser server -s -m /home/si/v2d/models/ResNet_D512L512_G128G64_D1024D0.20BN_BDGRU1024_D0.2L1024DVS_model.dat_4983_loss_2.350_Cider0.355_Blue0.353_Rouge0.571_Meteor0.247_TOTAL_1.558_BEST 2>&1'

echo "Starting web-ui"
docker container run --name "v2d_frontend" -d -p 8080:5000 -e "V2D_CONFIG_FILE=config_docker.json" \
  --net v2d_net --ip 172.14.0.3 \
  --mount type=bind,source="$(pwd)"/src,target=/home/si/v2d/src,readonly \
  --mount type=bind,source="$(pwd)"/uploads,target=/mnt/v2d/uploads/ \
  scopeinfinity/video2description:frontend \
  /bin/bash -c 'python -m frontend.app 2>&1'

echo "V2D running in deattached mode, if not crashed."
