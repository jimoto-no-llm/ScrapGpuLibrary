#!/bin/bash
# usage: run.sh TAG cmd...    (environment from the caller)
# One command under ~/bench/gpu-bench.lock after every card is below 85 C. Writes $LOGS/TAG.log (header with the environment, command, stderr, footer) and
# $LOGS/TAG.out (stdout). TMO seconds (default 1500) stop the command (SIGINT, then SIGKILL after 20 s). The watchdog sleeps in 1 s steps and does not
# inherit the lock descriptor, so no orphan keeps the lock.
if [ -z "$SGLX_RUN_INNER" ]; then export SGLX_RUN_INNER=1; exec flock "$HOME/bench/gpu-bench.lock" "$0" "$@"; fi
. "$(dirname "$0")/env.sh"
TAG=$1; shift
temps() {
  local t="" x f
  for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input; do [ -r "$f" ] && t="$t amd$(( $(cat $f) / 1000 ))"; done
  x=$(nvidia-smi --query-gpu=name,temperature.gpu --format=csv,noheader 2>/dev/null | sed 's/GeForce //; s/, /=/' | tr '\n' ' ')
  echo "$t $x"
}
maxtemp() { temps | tr ' ' '\n' | sed -E 's/^[A-Za-z0-9_ ]*[=]?//' | grep -E '^[0-9]+$' | sort -n | tail -1; }
for i in $(seq 1 60); do m=$(maxtemp); [ "${m:-0}" -lt 85 ] && break; sleep 15; done
LOG=$LOGS/$TAG.log; OUT=$LOGS/$TAG.out
{
  echo "# date: $(date +%F_%T)  host: $(hostname)  tag: $TAG"
  echo "# temps at start: $(temps)   (max $(maxtemp) C)"
  echo "# env: $(env | grep -E '^(GGML_|LLAMA_|RUSTICL|VK_|LD_LIBRARY|OCL_|MESA_)' | tr '\n' ' ')"
  echo "# cmd: $*"
} > "$LOG"
S=$(date +%s)
"$@" > "$OUT" 2>> "$LOG" &
P=$!
( n=0; while [ $n -lt "${TMO:-1500}" ]; do kill -0 $P 2>/dev/null || exit 0; sleep 1; n=$((n+1)); done; kill -INT $P 2>/dev/null; sleep 20; kill -KILL $P 2>/dev/null ) 3<&- &
W=$!
wait $P; rc=$?
kill $W 2>/dev/null; wait $W 2>/dev/null
{ echo "# rc=$rc  seconds=$(( $(date +%s) - S ))"; echo "# temps at end: $(temps)   (max $(maxtemp) C)"; } >> "$LOG"
exit $rc
