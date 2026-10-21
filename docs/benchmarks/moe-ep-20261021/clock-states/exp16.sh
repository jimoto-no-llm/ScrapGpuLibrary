#!/bin/bash
# exp16: wait (up to N sessions) for a bare C1 tg < 47 (the 44 state), then within the same perf-high session: sampled (SAMPLE_MS=20, 100, 300, 1000) vs bare, alternating. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
export REPS=2 PROFILE=0
tg() { T=$1; $SG/bench.sh cs_$T "" -p 0 -n 128 >/dev/null 2>&1
  python3 - $SG/logs/cs_$T.out $T <<P
import json,sys,time
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d["n_gen"]==128 and d["n_prompt"]==0: print("%.2f"%d["avg_ts"])
P
}
stg() { T=$1; SAMPLE_MS=$2 python3 $SG/sampler.py $SG/logs/cs_$T.csv & SP=$!; v=$(tg $T); kill $SP; wait $SP 2>/dev/null; echo $v; }
for s in $(seq 1 ${1:-12}); do
  sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
  v=$(tg e16_s${s}_bare); echo "$(date +%T) s$s bare=$v"
  if python3 -c "import sys; sys.exit(0 if float('$v')<47 else 1)"; then
    echo "44 state caught"
    for ms in 20 100 300 1000; do
      echo "$(date +%T) samp${ms}ms=$(stg e16_s${s}_samp$ms $ms)"; echo "$(date +%T) bare=$(tg e16_s${s}_bare_$ms)"
    done
    break
  fi
  sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1; sleep 60
done
echo FIN
