#!/bin/bash
# exp17: session loop (perf high -> sampled C1 tg run REPS=2, SAMPLE_MS=100 incl. gpu_metrics -> perf auto, 60 s pause) until >=2 runs of each state (tg<47 / tg>=47) are captured or N sessions. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
export REPS=2 PROFILE=0 SAMPLE_MS=${2:-100}
lo=0; hi=0
for s in $(seq 1 ${1:-16}); do
  sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
  for j in 1 2; do
    T=e17_s${s}_$j
    python3 $SG/sampler.py $SG/logs/cs_$T.csv & SP=$!
    $SG/bench.sh cs_$T "" -p 0 -n 128 >/dev/null 2>&1
    kill $SP; wait $SP 2>/dev/null
    v=$(python3 - $SG/logs/cs_$T.out <<P
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d["n_gen"]==128 and d["n_prompt"]==0: print("%.2f"%d["avg_ts"])
P
)
    echo "$(date +%T) $T tg=$v"
    if python3 -c "import sys; sys.exit(0 if float('$v')<47 else 1)"; then lo=$((lo+1)); else hi=$((hi+1)); fi
  done
  sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1
  [ $lo -ge 4 ] && [ $hi -ge 4 ] && break
  sleep 60
done
echo FIN lo=$lo hi=$hi
