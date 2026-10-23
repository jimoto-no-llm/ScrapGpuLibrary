#!/bin/bash
# exp5: sampled C1 (REPS=5) x3 under perf high; used to compare a 44 state with a 50/52 state. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
for i in 1 2 3; do $SG/probe.sh ${1:-e5}_c1_$i ""; done
