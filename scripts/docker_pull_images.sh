#!/bin/bash
TAG=${1:-deploy}
docker pull "scopeinfinity/video2description:${TAG}" || echo "[docker] Pulling :${TAG} image failed!"