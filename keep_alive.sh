#!/bin/bash
# Keep streaming something every minute upto X or job get completed whichever first.
TIMEOUT="${1:?}"
EXEC="${2:?}"
shift 2
timeout "${TIMEOUT}m" bash -c 'while true;do echo "Time: $(date)"; sleep 1m;done;' &
TIMER_PID="$!"
(timeout "${TIMEOUT}m" $EXEC "$@";kill $TIMER_PID)
echo "Exiting keep alive"