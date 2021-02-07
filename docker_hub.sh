#!/bin/bash
docker $1 scopeinfinity/video2description:frontend
docker $1 scopeinfinity/video2description:ffmpeg_builder
docker $1 scopeinfinity/video2description:glove_builder
docker $1 scopeinfinity/video2description:deploy