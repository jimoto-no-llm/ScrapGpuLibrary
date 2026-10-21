#!/bin/bash
# exp22: poll with a short bare C1 tg probe (perf high, REPS=2) every ~60 s until it reads the 44 state (tg<47), then in the same perf-high session the exp21 body (C1/C alternating, REPS=10, 2 rounds). Max N probes (default 45). Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
export PROFILE=0
for s in $(seq 1 ${1:-45}); do
  sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
  REPS=2 ./bench.sh e22_probe${s} "" -p 0 -n 128 >/dev/null 2>&1
  v=$(python3 - logs/e22_probe${s}.out <<P
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d["n_gen"]==128 and d["n_prompt"]==0: print("%.2f"%d["avg_ts"])
P
)
  echo "$(date +%T) probe$s tg=$v"
  if python3 -c "import sys; sys.exit(0 if float('$v')<47 else 1)"; then
    export REPS=10
    for i in 1 2; do ./bench.sh e22_c1_$i ""; ./bench.sh e22_C_$i "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"; done
    REPS=2 ./bench.sh e22_probe_after "" -p 0 -n 128 >/dev/null 2>&1
    echo "$(date +%T) done 44-state set"; break
  fi
  sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1; sleep 45
done
echo FIN
