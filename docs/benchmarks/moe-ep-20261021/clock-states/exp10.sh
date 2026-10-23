#!/bin/bash
# exp10: runstat.py (bare C1, REPS=2) N runs under perf high. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
REPS=2 python3 $SG/runstat.py ${2:-e10} ${1:-24}
echo FIN
