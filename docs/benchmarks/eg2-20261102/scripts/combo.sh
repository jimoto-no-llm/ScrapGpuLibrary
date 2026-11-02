#!/bin/bash
# usage: combo.sh NAME DURATION "card:type:L:B ..."     data parallel: the whole model on every listed card, one process per card, all
# measured in the same wall-clock window (a common absolute start time; the slow loads / kernel builds happen before it).
# PINMAP="rx6400-vk=0 default=1" pins the processes to CPU cores (taskset). Output: $LOGS/c_NAME.txt (one WINDOW line per member + the sum) and c_NAME.<card>.err
if [ -z "$EG2_COMBO_INNER" ]; then export EG2_COMBO_INNER=1; exec flock "$HOME/bench/gpu-bench.lock" "$0" "$@"; fi
. "$(dirname "$0")/env.sh"; . "$(dirname "$0")/card_env.sh"
NAME=$1; DUR=$2; MEMBERS=$3
maxtemp() {
  local t="" f x
  for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input; do [ -r "$f" ] && t="$t $(( $(cat $f) / 1000 ))"; done
  x=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | tr '\n' ' ')
  echo "$t $x" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -n | tail -1
}
for i in $(seq 1 60); do m=$(maxtemp); [ "${m:-0}" -lt 85 ] && break; sleep 15; done
T0=$(maxtemp)
START=$(( $(date +%s) + ${LEAD:-90} ))
OUT=$LOGS/c_$NAME.txt
echo "# combo $NAME  members: $MEMBERS  duration ${DUR}s  start $START  temp at start ${T0} C  $(date +%F_%T)" > $OUT
pids=()
for m in $MEMBERS; do
  IFS=: read card t L B <<< "$m"
  cpus=""; if [ -n "$PINMAP" ]; then for kv in $PINMAP; do [ "${kv%%=*}" = "$card" ] && cpus=${kv#*=}; [ "${kv%%=*}" = "default" ] && [ -z "$cpus" ] && cpus=${kv#*=}; done; fi
  ( card_env $card; exec ${cpus:+taskset -c $cpus} $EG2 -m $MODELS/eg2-$t.gguf -ngl 99 --dev $DEVNAME -t ${THREADS:-1} --warm 2 --bench $L $B 1 --duration $DUR --start-at $START > $LOGS/c_$NAME.$card.$t.out 2> $LOGS/c_$NAME.$card.$t.err 3<&- ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
for m in $MEMBERS; do IFS=: read card t L B <<< "$m"; echo "$card $t $(grep -h WINDOW $LOGS/c_$NAME.$card.$t.out || echo 'NO RESULT')" >> $OUT; done
awk '/WINDOW/ { for (i=1;i<=NF;i++) if ($i ~ /^emb_per_s=/) { split($i,a,"="); s+=a[2] } if ($0 ~ /start=/) { for (i=1;i<=NF;i++) if ($i ~ /^start=/) { split($i,b,"="); if (mn==""||b[2]<mn) mn=b[2]; if (b[2]>mx) mx=b[2] } } } END { printf "SUM emb_per_s=%.3f  start_spread_s=%.2f\n", s, mx-mn }' $OUT >> $OUT
echo "# temp at end $(maxtemp) C" >> $OUT
cat $OUT
