#!/bin/bash
# withsched.sh TAG cmd...: cmd with schedsamp.py beside it; the table goes to $LOGS/TAG.sched
TAG=$1; shift
. "$(dirname "$0")/env.sh"
python3 "$(dirname "$0")/schedsamp.py" ${SAMP_SECONDS:-100} > "$LOGS/$TAG.sched" 2>&1 &
S=$!
"$@"
rc=$?
kill -TERM $S 2>/dev/null; wait $S 2>/dev/null
exit $rc
