#!/bin/bash
# exp18 (env PMAX max pause, STOP52 = stop after that many >=47 sessions): N sessions with a random idle pause (5..120 s) before each: snapshot (pre, after the perf-high write, mid-run, post) + bare C1 tg run (REPS=2). Names carry the pause. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
export REPS=2 PROFILE=0
for s in $(seq 1 ${1:-20}); do
  P=$(( (RANDOM % ${PMAX:-116}) + 5 )); sleep $P
  T=e18_s${s}_p$P
  $SG/snap.sh $SG/logs/snap_${T}_0pre.txt
  sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
  $SG/snap.sh $SG/logs/snap_${T}_1afterhigh.txt
  $SG/bench.sh cs_$T "" -p 0 -n 128 >/dev/null 2>&1 &
  BP=$!; sleep 14; $SG/snap.sh $SG/logs/snap_${T}_2mid.txt; wait $BP
  $SG/snap.sh $SG/logs/snap_${T}_3post.txt
  v=$(python3 - $SG/logs/cs_$T.out <<P
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d["n_gen"]==128 and d["n_prompt"]==0: print("%.2f"%d["avg_ts"])
P
)
  echo "$(date +%T) $T pause=$P tg=$v"
  python3 -c "import sys; sys.exit(0 if float('$v')>=47 else 1)" && n52=$((n52+1))
  sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1
  [ -n "$STOP52" ] && [ ${n52:-0} -ge $STOP52 ] && break
done
echo FIN
