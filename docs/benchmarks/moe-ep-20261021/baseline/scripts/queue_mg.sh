#!/bin/bash
# exploration of the tensor splits of the multi-GPU configurations (-r 2, pp128 + tg64); ts follows the -dev order of env.sh use_mg
. "$(dirname "$0")/env.sh"
S="$(dirname "$0")"
try() { # TAG CONFIG TS
  if [ -s "$RUNS/$1.out" ] && grep -q avg_ts "$RUNS/$1.out"; then echo "skip $1"; return; fi
  "$S/mg.sh" "$1" "$2" "$3" 2 128 64 "" 128
  rc=$?
  if [ $rc -ne 0 ] || ! grep -q avg_ts "$RUNS/$1.out"; then echo "FAILED $1 rc=$rc"; return; fi
  echo "$1: $(python3 -c "
import json
for l in open('$RUNS/$1.out'):
    d=json.loads(l); print(d['n_prompt'],d['n_gen'],round(d['avg_ts'],2),end=' | ')")"
}
# (b) AMD x2 official Vulkan: Vulkan0 = RX 6400 (4 GiB), Vulkan1 = WX 2100 (2 GiB)
for ts in 1,1 2,1 3,1 4,1 5,2; do try "mgx_amdvk_ts${ts}" amdvk $ts; done
# (a) 5 cards CL11: RPC0 GT 430, GT 730, GT 710, RX 6400, WX 2100 (docs/MULTI_GPU.md: 8,8,18,40,18)
for ts in 8,8,18,40,18 4,4,18,50,24 2,4,16,60,18 0,4,16,60,20; do try "mgx_cl5_ts${ts}" cl5 $ts; done
# 4 cards CL11 without GT 430
for ts in 8,18,40,18 4,12,50,24; do try "mgx_cl4_ts${ts}" cl4 $ts; done
# (c) CL11 (GT 730, GT 710) + official Vulkan (RX 6400, WX 2100)
for ts in 4,8,40,20 2,8,50,20 1,4,60,20 0,4,60,20; do try "mgx_mix4_ts${ts}" mix4 $ts; done
# the same plus GT 430 over the automatic RPC
for ts in 4,4,8,40,20 2,2,8,50,20; do try "mgx_mix5_ts${ts}" mix5 $ts; done
echo QUEUE_MG_DONE
