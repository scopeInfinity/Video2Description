#!/bin/bash
docker pull scopeinfinity/video2description:latest
docker image build -t scopeinfinity/video2description --cache-from scopeinfinity/video2description:latest .