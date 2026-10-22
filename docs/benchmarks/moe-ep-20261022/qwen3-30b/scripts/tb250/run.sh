#!/bin/bash
# usage: run.sh TAG cmd...   Same as sglx/run.sh (flock ~/bench/gpu-bench.lock, wait <85 C, watchdog TMO s) plus: perf state at start, memory sampler (max VRAM/GTT AMD, NVIDIA used, system used, process RSS) -> $LOGS/TAG.mem
if [ -z "$Q30_RUN_INNER" ]; then export Q30_RUN_INNER=1; exec flock "$HOME/bench/gpu-bench.lock" "$0" "$@"; fi
. "$(dirname "$0")/env.sh"
TAG=$1; shift
temps() {
  local t="" x f
  for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input; do [ -r "$f" ] && t="$t amd$(( $(cat $f) / 1000 ))"; done
  x=$(nvidia-smi --query-gpu=name,temperature.gpu --format=csv,noheader 2>/dev/null | sed "s/GeForce //; s/, /=/" | tr "\n" " ")
  echo "$t $x"
}
maxtemp() { temps | tr " " "\n" | sed -E "s/^[A-Za-z0-9_ ]*[=]?//" | grep -E "^[0-9]+$" | sort -n | tail -1; }
for i in $(seq 1 60); do m=$(maxtemp); [ "${m:-0}" -lt 85 ] && break; sleep 15; done
LOG=$LOGS/$TAG.log; OUT=$LOGS/$TAG.out
{
  echo "# date: $(date +%F_%T)  host: $(hostname)  tag: $TAG"
  echo "# perf: $(sudo -n /usr/local/sbin/sgl-driver-switch status 2>&1 | grep perf | tr "\n" " ")"
  echo "# temps at start: $(temps)   (max $(maxtemp) C)"
  echo "# env: $(env | grep -E "^(GGML_|RUSTICL|VK_|LD_LIBRARY|OCL_|MESA_)" | tr "\n" " ")"
  echo "# cmd: $*"
} > "$LOG"
S=$(date +%s)
"$@" > "$OUT" 2>> "$LOG" &
P=$!
( n=0; while [ $n -lt "${TMO:-1500}" ]; do kill -0 $P 2>/dev/null || exit 0; sleep 1; n=$((n+1)); done; kill -INT $P 2>/dev/null; sleep 20; kill -KILL $P 2>/dev/null ) 3<&- &
W=$!
if [ -z "$NOMEM" ]; then
( : > "$LOGS/$TAG.mem"
  while kill -0 $P 2>/dev/null; do
    line="t=$(( $(date +%s) - S ))"
    for c in /sys/class/drm/card[0-9]*/device; do
      [ -r $c/mem_info_vram_used ] && line="$line $(basename $(dirname $c)):vram=$(( $(cat $c/mem_info_vram_used)/1048576 )),gtt=$(( $(cat $c/mem_info_gtt_used)/1048576 ))"
    done
    [ -n "$NVSMI" ] && line="$line nv=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr "\n" "/")"
    line="$line sysused=$(awk "/MemTotal/{t=\$2}/MemAvailable/{a=\$2}END{print int((t-a)/1024)}" /proc/meminfo) cached=$(awk "/^Cached/{print int(\$2/1024)}" /proc/meminfo) rss=$(awk "/VmRSS/{print int(\$2/1024)}" /proc/$P/status 2>/dev/null)"
    echo "$line" >> "$LOGS/$TAG.mem"
    sleep 2
  done ) 3<&- &
M=$!
fi
wait $P; rc=$?
kill $W 2>/dev/null; wait $W 2>/dev/null; [ -n "$M" ] && wait $M 2>/dev/null
{ echo "# rc=$rc  seconds=$(( $(date +%s) - S ))"; echo "# temps at end: $(temps)   (max $(maxtemp) C)"; } >> "$LOG"
exit $rc
