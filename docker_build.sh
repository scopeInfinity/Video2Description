#!/bin/bash
set -e

docker image build \
  -f frontend.Dockerfile \
  -t scopeinfinity/video2description:frontend \
  .

docker image build --target v2d \
  -f backend.Dockerfile \
  -t scopeinfinity/video2description:latest \
  --cache-from scopeinfinity/video2description:latest \
  .

docker image build --target v2d_deploy \
  -f backend.Dockerfile \
  -t scopeinfinity/video2description:deploy \
  --cache-from scopeinfinity/video2description:latest \
  --cache-from scopeinfinity/video2description:deploy \
  .