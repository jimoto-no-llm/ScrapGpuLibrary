#!/bin/bash
# usage: bundle.sh BUNDLE_NAME JOBFILE
# Takes ~/bench/gpu-bench.lock ONCE and runs every line of JOBFILE ("TAG CONFIG cmd...") one after another through go.sh/run.sh
# (RUN_INNER=1 makes run.sh skip its own flock; run.sh still waits for <85 C before each job and kills a job after $TMO s).
# Keep bundles short (a few minutes) because the other agents share the lock.
HERE="$(dirname "$0")"
if [ -z "$BUNDLE_INNER" ]; then export BUNDLE_INNER=1; exec flock "$HOME/bench/gpu-bench.lock" "$0" "$@"; fi
. "$HERE/env.sh"
cd "$SPEC"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  # expand $LB / $MODELS in the job line
  eval "set -- $line"
  RUN_INNER=1 "$HERE/go.sh" "$@"
done < "$2"
