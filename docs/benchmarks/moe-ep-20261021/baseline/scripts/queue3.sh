#!/bin/bash
# second exploration: -ub 512 on the official Vulkan builds, finer tensor splits of the two AMD cards
. "$(dirname "$0")/env.sh"
S="$(dirname "$0")"
"$S/sweep.sh" rx-vk 512 12 10 8 7 6 5 4
try() { # TAG CONFIG TS UB
  if [ -s "$RUNS/$1.out" ] && grep -q avg_ts "$RUNS/$1.out"; then echo "skip $1"; return; fi
  "$S/mg.sh" "$1" "$2" "$3" 2 128 64 "" "$4"
  rc=$?
  if [ $rc -ne 0 ] || ! grep -q avg_ts "$RUNS/$1.out"; then echo "FAILED $1 rc=$rc"; return; fi
  echo "$1: $(python3 -c "
import json
for l in open('$RUNS/$1.out'):
    d=json.loads(l); print(d['n_prompt'],d['n_gen'],round(d['avg_ts'],2),end=' | ')")"
}
for ts in 5,1 6,1 7,2 8,1; do try "mgx_amdvk_ts${ts}" amdvk $ts 128; done
for ts in 4,1 5,1; do try "mgx_amdvk_ub512_ts${ts}" amdvk $ts 512; done
echo QUEUE3_DONE
