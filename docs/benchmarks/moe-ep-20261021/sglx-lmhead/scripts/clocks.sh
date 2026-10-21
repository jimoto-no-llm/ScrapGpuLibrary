#!/bin/bash
# usage: clocks.sh TAG SPLIT [bench args]: bench.sh in the background; every 20 ms the current sclk state of the RX 6400 (card1) and WX 2100 (card5) into $LOGS/TAG.clk
. "$(dirname "$0")/env.sh"
TAG=$1; shift
"$(dirname "$0")/bench.sh" "$TAG" "$@" > /dev/null 2>&1 &
BP=$!
: > "$LOGS/$TAG.clk"
while kill -0 $BP 2>/dev/null; do
  a=$(grep '\*' /sys/class/drm/card1/device/pp_dpm_sclk | tr -d '\n'); b=$(grep '\*' /sys/class/drm/card5/device/pp_dpm_sclk | tr -d '\n')
  echo "rx[$a] wx[$b]" >> "$LOGS/$TAG.clk"
  sleep 0.02
done
wait $BP
sort "$LOGS/$TAG.clk" | uniq -c | sort -rn | head -8
