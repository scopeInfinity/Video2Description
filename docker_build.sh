#!/bin/bash
remote="scopeinfinity/video2description"
for file_tag in "backend.Dockerfile ffmpeg_builder" "backend.Dockerfile glove_builder" "backend.Dockerfile deploy" "frontend.Dockerfile frontend"; do
	set -- $file_tag
	docker build --target $2 -t $remote:$2 --cache-from $remote:$2 --build-arg BUILDKIT_INLINE_CACHE=1 -f $1 .
done
