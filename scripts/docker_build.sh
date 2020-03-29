#!/bin/bash
set -e
bash scripts/docker_pull_images.sh latest
bash scripts/docker_pull_images.sh deploy

docker image build --target v2d \
  -t scopeinfinity/video2description:latest \
  --cache-from scopeinfinity/video2description:latest \
  .

docker image build --target v2d_deploy \
  -t scopeinfinity/video2description:deploy \
  --cache-from scopeinfinity/video2description:latest \
  --cache-from scopeinfinity/video2description:deploy \
  .
