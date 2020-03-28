#!/bin/bash
set -e
trap 'kill $(jobs -p) || echo "No background jobs"' EXIT

TIMEOUT_WAIT_FOR_BACKEND=${1:-5}  # in minutes

echo "[docker][latest] ./run_tests.sh"
docker container run scopeinfinity/video2description:latest conda run -n V2D /bin/bash -c 'cd /home/si/v2d/src/ && ./run_tests.sh'
echo "[docker][deploy] ./run_tests.sh"
docker container run scopeinfinity/video2description:deploy conda run -n V2D /bin/bash -c 'cd /home/si/v2d/src/ && ./run_tests.sh'

bash docker_start.sh
for x in `seq ${TIMEOUT_WAIT_FOR_BACKEND}`;do
    sleep "1m";
    curl "http://localhost:8080/model_weights_status" 2>&1 | tee /dev/stderr | grep -q '\[SUCCESS\]' && break;
done 2>&1 || { echo "Backend model_weights_status failed to come to success"; exit 1; }
echo "Backend model_weights_status: SUCCESS"

bash docker_logs.sh &

# Run tests external to docker
echo "[external] Executing tests on [docker][deploy]"
python -m unittest discover tests/

bash docker_stop.sh
