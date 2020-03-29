#!/bin/bash
docker container stop v2d || echo "[docker][container] Failed to stop v2d"
docker container rm v2d || echo "[docker][container] Failed to rm v2d"