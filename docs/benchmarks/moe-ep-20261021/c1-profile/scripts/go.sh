#!/bin/bash
# go.sh TAG IV [--gpu] -- cmd...   (flock, temp wait, sampler)
if [ -z "$GO_INNER" ]; then export GO_INNER=1; exec flock "$HOME/bench/gpu-bench.lock" "$0" "$@"; fi
TAG=$1; IV=$2; shift 2
G=""; [ "$1" = "--gpu" ] && { G="--gpu"; shift; }
shift # --
for i in $(seq 1 60); do m=0; for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input; do t=$(( $(cat $f)/1000 )); [ $t -gt $m ] && m=$t; done; [ $m -lt 85 ] && break; sleep 15; done
D=$HOME/moe-ep/c1p/logs
echo "# $(date +%T) temp_start_max=$m cmd: $*" > $D/$TAG.log
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
python3 $HOME/moe-ep/c1p/scripts/sampler.py $D/$TAG.json $IV $G -- "$@" > $D/$TAG.out 2>> $D/$TAG.log
echo "# rc=$? end $(date +%T)" >> $D/$TAG.log
