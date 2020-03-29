
#!/bin/bash
trap 'kill $(jobs -p)' EXIT

docker container logs -f v2d_backend &
echo "[docker][logs][backend] Log PID: $!"
docker container logs -f v2d_frontend
echo "[docker][logs][frontend] Log PID: $!"
