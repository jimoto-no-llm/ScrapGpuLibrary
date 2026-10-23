#!/bin/bash
# refinement: odd n-cpu-moe values around the memory limit / the optimum, finer -ts and -ub of the two AMD cards
. "$(dirname "$0")/env.sh"
S="$(dirname "$0")"
"$S/sweep.sh" gt730 128 21
"$S/sweep.sh" gt430 128 23
"$S/sweep.sh" gt710 128 19 17
"$S/sweep.sh" wx-cl11 128 19
"$S/sweep.sh" wx-vk 128 21 19 17
"$S/sweep.sh" rx-cl11 128 15 13 11 9 7 5 3
"$S/sweep.sh" rx-vk 128 7 5
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
for ub in 64 256; do try "mgx_amdvk_ub${ub}_ts5,1" amdvk 5,1 $ub; done
for ts in 9,2 11,2 13,3; do try "mgx_amdvk_ts${ts}" amdvk $ts 128; done
for ts in 9,2 11,2; do try "mgx_amdvk_ub512_ts${ts}" amdvk $ts 512; done
echo QUEUE4_DONE
