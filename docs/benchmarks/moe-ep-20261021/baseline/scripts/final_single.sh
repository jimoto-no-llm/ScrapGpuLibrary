#!/bin/bash
# final measurements of the single-device bests: -r 5, pp512, tg128, pp512+tg128 (-pg 512,128). usage: final_single.sh BACKEND NCMOE UB
. "$(dirname "$0")/env.sh"
tag=final_$1_ncmoe$2_ub$3
[ -s "$RUNS/$tag.out" ] && [ "$(grep -c avg_ts "$RUNS/$tag.out")" -ge 3 ] && { echo "skip $tag"; exit 0; }
"$(dirname "$0")/bench.sh" "$tag" "$1" "$2" 5 512 128 512,128 "$3"
echo "$tag rc=$?"
