#!/bin/bash
docker pull scopeinfinity/video2description:latest || echo "[docker] Pulling :latest image failed!"
docker pull scopeinfinity/video2description:deploy || echo "[docker] Pulling :deploy image failed!"