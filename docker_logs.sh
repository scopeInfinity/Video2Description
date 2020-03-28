#!/bin/bash
trap 'kill $(jobs -p)' EXIT

docker container logs -f v2d &
echo "[docker][logs] Container Log PID: $!"
docker container exec "v2d" tail -f '/var/log/v2d/app.log'
echo "[docker][logs] App Log PID: $!"
