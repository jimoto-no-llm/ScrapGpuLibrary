#!/bin/bash
# usage: final_mg.sh CONFIG TS UB   -r 5, pp512, tg128, pp512+tg128
. "$(dirname "$0")/env.sh"
tag=final_mg_$1_ts${2//,/-}_ub$3
[ -s "$RUNS/$tag.out" ] && [ "$(grep -c avg_ts "$RUNS/$tag.out")" -ge 3 ] && { echo "skip $tag"; exit 0; }
"$(dirname "$0")/mg.sh" "$tag" "$1" "$2" 5 512 128 512,128 "$3"
echo "$tag rc=$?"
