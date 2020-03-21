#!/bin/bash

TIMEOUT_WAIT_FOR_BACKEND=${1:-5}  # in minutes

docker container run scopeinfinity/video2description:latest conda run -n V2D /bin/bash -c 'cd /home/si/v2d/src/ && ./run_tests.sh'
docker container run scopeinfinity/video2description:deploy conda run -n V2D /bin/bash -c 'cd /home/si/v2d/src/ && ./run_tests.sh'

# bash docker_start.sh
# for x in `seq ${TIMEOUT_WAIT_FOR_BACKEND}`;do
#     sleep "1m";
#     curl "http://localhost:8080/model_weights_status" 2> /dev/null | tee /dev/stderr | grep -q '\[SUCCESS\]' && break;
# done 2>&1 || { echo "Backend model_weights_status failed to come to success"; exit 1;}

# Run tests external to docker
# python -m unittest discover tests/
# bash docker_stop.sh