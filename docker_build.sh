#!/bin/bash
export DOCKER_BUILDKIT=1

docker image build --target v2d \
  -t scopeinfinity/video2description:latest \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --cache-from scopeinfinity/video2description:latest \
  .

docker image build --target v2d_deploy \
  -t scopeinfinity/video2description:deploy \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --cache-from scopeinfinity/video2description:latest \
  --cache-from scopeinfinity/video2description:deploy \
  .
