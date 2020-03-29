#!/bin/bash
docker container stop v2d_backend || echo "[docker][backend] Failed to stop container"
docker container rm v2d_backend || echo "[docker][backend] Failed to remove container"
docker container stop v2d_frontend || echo "[docker][frontend] Failed to stop container"
docker container rm v2d_frontend || echo "[docker][frontend] Failed to remove container"
docker network rm v2d_net || echo "[docker][network] Failed to remove network"
docker volume rm v2d_uploads || echo "[docker][storage] Failed to remove volume"