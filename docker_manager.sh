#!/bin/bash
set -e

REMOTE="scopeinfinity/video2description"

main() {
    action="$1"
    if [[ "$action" == "push" || "$action" == "pull" ]]; then
        docker_execute_each $action
    elif [[ "$action" == "build" ]]; then
        docker_build
    elif [[ "$action" == "test" ]]; then
        docker_test
    else
        echo "Invalid action: $action provided" >&1
        exit 1;
    fi
}

docker_execute_each() {
    action="$1"
    docker $action $REMOTE:frontend
    docker $action $REMOTE:ffmpeg_builder
    docker $action $REMOTE:glove_builder
    docker $action $REMOTE:deploy
}

docker_build() {
    for file_tag in "backend.Dockerfile ffmpeg_builder" "backend.Dockerfile glove_builder" "backend.Dockerfile deploy" "frontend.Dockerfile frontend"; do
        set -- $file_tag
        docker build --target $2 -t $REMOTE:$2 --cache-from $REMOTE:$2 --build-arg BUILDKIT_INLINE_CACHE=1 -f $1 .
    done
}

docker_test() {
    trap 'kill $(jobs -p) || echo "No background jobs"' EXIT

    TIMEOUT_WAIT_FOR_BACKEND=${1:-5}  # in minutes

    echo "[docker][backend] ./run_tests.sh"
    docker container run $REMOTE:deploy conda run -n V2D /bin/bash -c 'cd /home/si/v2d/src/ && ./run_tests.sh'

    docker-compose up --detach
    docker-compose logs -f &

    for x in `seq ${TIMEOUT_WAIT_FOR_BACKEND}`;do
        sleep "1m";
        curl "http://localhost:8080/model_weights_status" 2>&1 | tee /dev/stderr | grep -q '\[SUCCESS\]' && break;
    done 2>&1 || { echo "Backend model_weights_status failed to come to success"; exit 1; }
    echo "Backend model_weights_status: SUCCESS"

    # Run tests external to docker
    echo "[external] Executing tests on [docker][deploy]"
    python -m unittest discover tests/

    docker-compose down
}

main "$@";
exit