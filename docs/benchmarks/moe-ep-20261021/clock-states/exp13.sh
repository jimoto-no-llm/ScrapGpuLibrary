#!/bin/bash
# exp13: N sessions (perf high; runstat.py 2 bare C1 runs with REPS=2; perf auto) to catch both tg states with their host-side cost. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
export REPS=2
for i in $(seq 1 ${1:-14}); do
  sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
  python3 $SG/runstat.py ${2:-e13}_s$i 2
  sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1
done
echo FIN
