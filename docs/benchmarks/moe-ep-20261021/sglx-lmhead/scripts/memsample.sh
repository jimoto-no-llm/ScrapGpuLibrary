#!/bin/bash
# usage: memsample.sh TAG SPLIT [bench args]: bench.sh in the background, VRAM / GTT of the AMD cards every 0.5 s into $LOGS/TAG.mem (max of each)
. "$(dirname "$0")/env.sh"
TAG=$1; shift
"$(dirname "$0")/bench.sh" "$TAG" "$@" > /dev/null 2>&1 &
BP=$!
: > "$LOGS/$TAG.mem"
while kill -0 $BP 2>/dev/null; do
  line=""
  for c in /sys/class/drm/card[0-9]*/device; do
    [ -r $c/mem_info_vram_used ] && line="$line $(basename $(dirname $c)):vram=$(( $(cat $c/mem_info_vram_used)/1048576 )) gtt=$(( $(cat $c/mem_info_gtt_used)/1048576 ))"
  done
  echo "$line" >> "$LOGS/$TAG.mem"
  sleep 0.5
done
wait $BP
sort -u "$LOGS/$TAG.mem" | tail -3
