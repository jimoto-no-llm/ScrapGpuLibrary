#!/bin/bash
# withsamp.sh TAG cmd...: run cmd with the CPU sampler (cpusamp.py) beside it; the samples go to $LOGS/TAG.cpu
TAG=$1; shift
. "$(dirname "$0")/env.sh"
python3 "$(dirname "$0")/cpusamp.py" ${SAMP_SECONDS:-100} > "$LOGS/$TAG.cpu" 2>&1 &
S=$!
"$@"
rc=$?
kill -TERM $S 2>/dev/null; wait $S 2>/dev/null
exit $rc
