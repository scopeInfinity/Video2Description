#!/bin/bash
docker container run scopeinfinity/video2description conda run -n V2D /bin/bash -c 'cd /home/si/v2d/src/ && ./run_tests.sh'