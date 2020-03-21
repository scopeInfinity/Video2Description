#!/bin/bash
docker container run scopeinfinity/video2description:latest conda run -n V2D /bin/bash -c 'cd /home/si/v2d/src/ && ./run_tests.sh'
docker container run scopeinfinity/video2description:deploy conda run -n V2D /bin/bash -c 'cd /home/si/v2d/src/ && ./run_tests.sh'
bash docker_start.sh
sleep 30m  # temporary hack
python -m unittest discover tests/
docker container stop v2d
