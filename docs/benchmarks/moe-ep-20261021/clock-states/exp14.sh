#!/bin/bash
# exp14: N sessions (perf high; 2 C1 runs sampled at SAMPLE_MS=1000 with all columns, REPS=2; perf auto): is a 1 s sampler light enough not to disturb, and what do both states look like. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
export REPS=2 SAMPLE_MS=${3:-1000}
for i in $(seq 1 ${1:-14}); do
  sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
  for j in 1 2; do echo "$(date +%T) $($SG/probe.sh ${2:-e14}_s${i}_$j "" | cut -c1-90)"; done
  sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1
done
echo FIN
