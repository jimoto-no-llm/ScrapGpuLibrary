#!/bin/bash
# exp1: perf high, then C1 x4 back to back (sampler running). Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
for i in 1 2 3 4; do $SG/probe.sh e1_c1_$i ""; done
