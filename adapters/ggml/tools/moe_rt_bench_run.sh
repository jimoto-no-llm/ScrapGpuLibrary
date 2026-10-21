#!/bin/bash
# Runs one moe_rt_bench command the way the TB250 measurements are taken: under the bench lock, after every card is below 85 C.
#   usage: [env ...] moe_rt_bench_run.sh LOGDIR TAG cmd...
# Writes LOGDIR/TAG.log (header: date, temperatures, environment, command; stderr; footer) and LOGDIR/TAG.out (stdout). Set DRY_RUN=1 to print only.
# TMO (seconds, default 1800) stops the command with SIGINT, then SIGKILL after 20 s (only the child that was started here).
# (the baseline run.sh leaves its `sleep $TMO` behind with the lock descriptor open, so the lock is held for up to TMO seconds after a run)
if [ -z "$MRB_INNER" ] && [ -z "$DRY_RUN" ]; then export MRB_INNER=1; exec flock "${BENCH_LOCK:-$HOME/bench/gpu-bench.lock}" "$0" "$@"; fi
LOGDIR=$1; TAG=$2; shift 2
if [ -n "$DRY_RUN" ]; then echo "DRY_RUN: [$LOGDIR/$TAG] $*"; exit 0; fi
mkdir -p "$LOGDIR"
temps() {
  local t="" x f
  for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input; do [ -r "$f" ] && t="$t amd$(( $(cat $f) / 1000 ))"; done
  x=$(nvidia-smi --query-gpu=name,temperature.gpu --format=csv,noheader 2>/dev/null | sed 's/GeForce //; s/, /=/' | tr '\n' ' ')
  echo "$t $x"
}
maxtemp() { temps | tr ' ' '\n' | sed -E 's/^[A-Za-z0-9_ ]*[=]?//' | grep -E '^[0-9]+$' | sort -n | tail -1; }
for i in $(seq 1 60); do m=$(maxtemp); [ "${m:-0}" -lt 85 ] && break; sleep 15; done
LOG=$LOGDIR/$TAG.log; OUT=$LOGDIR/$TAG.out
{
  echo "# date: $(date +%F_%T)  host: $(hostname)  tag: $TAG"
  echo "# temps at start: $(temps)   (max $(maxtemp) C)"
  echo "# env: $(env | grep -E '^(GGML_|RUSTICL|VK_|LD_LIBRARY|OCL_|MESA_)' | tr '\n' ' ')"
  echo "# cmd: $*"
} > "$LOG"
S=$(date +%s)
"$@" > "$OUT" 2>> "$LOG" &
P=$!
# the watchdog sleeps in 1 s steps: an orphaned `sleep TMO` would keep the inherited lock file descriptor open for TMO seconds after the command ended
( n=0; while [ $n -lt "${TMO:-1800}" ]; do sleep 1; n=$((n+1)); done; kill -INT $P 2>/dev/null; sleep 20; kill -KILL $P 2>/dev/null ) &
W=$!
wait $P; rc=$?
kill $W 2>/dev/null; wait $W 2>/dev/null
{ echo "# rc=$rc  seconds=$(( $(date +%s) - S ))"; echo "# temps at end: $(temps)   (max $(maxtemp) C)"; } >> "$LOG"
exit $rc
