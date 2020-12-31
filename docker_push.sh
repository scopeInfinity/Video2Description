#!/bin/bash
docker build \
    --target ffmpeg_builder \
    -t scopeinfinity/video2description:ffmpeg_builder \
    -f "backend.Dockerfile" \
    "."

docker build \
    --target glove_builder \
    -t scopeinfinity/video2description:glove_builder \
    -f "backend.Dockerfile" \
    "."

docker push scopeinfinity/video2description:ffmpeg_builder
docker push scopeinfinity/video2description:glove_builder
docker push scopeinfinity/video2description:frontend
docker push scopeinfinity/video2description:deploy