#!/bin/bash
# exp15: try to provoke the 44 state: before each bare C1 tg run (REPS=2, -p 0 -n 128) apply one trigger; ROUNDS rounds; perf high throughout. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG; . $SG/env.sh
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
export REPS=2 PROFILE=0
tg() { T=$1; $SG/bench.sh cs_$T "" -p 0 -n 128 >/dev/null 2>&1
  python3 - $SG/logs/cs_$T.out $T <<P
import json,sys,time
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d["n_gen"]==128 and d["n_prompt"]==0: print("RESULT",time.strftime("%H:%M:%S"),sys.argv[2],"tg128=%.1f"%d["avg_ts"],flush=True)
P
}
for r in $(seq 1 ${1:-3}); do
  tg e15_r${r}_control
  sleep 120; tg e15_r${r}_idle120
  REPS=2 $SG/bench.sh cs_e15_r${r}_pre_C "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1" -p 0 -n 128 >/dev/null 2>&1; tg e15_r${r}_after_C
  for k in 1 2 3 4 5; do nvidia-smi >/dev/null 2>&1; done; tg e15_r${r}_after_nvsmi
  cat $HOME/moe-ep/models/LFM2.5-1.2B-Instruct-Q8_0.gguf $HOME/moe-ep/models/LFM2.5-8B-A1B-DSpark-Q8_0.gguf > /dev/null; tg e15_r${r}_after_io
  sleep 20; tg e15_r${r}_after_idle20
done
echo FIN
