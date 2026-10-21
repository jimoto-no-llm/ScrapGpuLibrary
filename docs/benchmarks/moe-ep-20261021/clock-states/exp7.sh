#!/bin/bash
# exp7: sampled C1 (SAMPLE_MS, default 100; REPS=3) N times in a row under perf high, tg per run; to compare the clocks of a 44 run with a 52 run. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
export REPS=3 PROFILE=0 SAMPLE_MS=${3:-100}
for i in $(seq 1 ${1:-8}); do echo "$(date +%T) $($SG/probe.sh ${2:-e7}_$i "")" ; done
echo FIN
