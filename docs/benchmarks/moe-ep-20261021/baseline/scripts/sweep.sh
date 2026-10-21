#!/bin/bash
# usage: sweep.sh BACKEND UB v1 v2 ...   (n-cpu-moe values, in the order given; stops at the first one that does not run)
# -r 2, pp128 + tg64.
. "$(dirname "$0")/env.sh"
BK=$1; UB=$2; shift 2
for n in "$@"; do
  tag=sweep_${BK}_ub${UB}_ncmoe${n}
  if [ -s "$RUNS/$tag.out" ] && grep -q avg_ts "$RUNS/$tag.out"; then echo "skip $tag (done)"; continue; fi
  "$(dirname "$0")/bench.sh" "$tag" "$BK" "$n" 2 128 64 "" "$UB"
  rc=$?
  if [ $rc -ne 0 ] || ! grep -q avg_ts "$RUNS/$tag.out"; then echo "FAILED $tag rc=$rc"; break; fi
  echo "$tag: $(python3 -c "
import json,sys
for l in open('$RUNS/$tag.out'):
    d=json.loads(l); print(d['n_prompt'],d['n_gen'],round(d['avg_ts'],2),end=' | ')")"
done
