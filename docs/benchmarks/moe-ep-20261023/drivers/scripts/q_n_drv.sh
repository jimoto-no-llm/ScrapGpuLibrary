#!/bin/bash
# q_n_drv.sh PREFIX : 30B config N twice (hec/cfg_n.sh), perf pin, logs hec/run/PREFIX_N_<r>. Run under nv390 and nv470 to compare.
. $HOME/moe-ep/hec/env_hec.sh
. $HOME/moe-ep/hec/cfg_n.sh
export HEC_GEN_VERBOSE=1
BENCH="--bench pp512,tg128,pg512:128 -r 3"
SW=/usr/local/sbin/sgl-driver-switch
sudo -n $SW perf pin; trap "sudo -n $SW perf auto" EXIT
for r in 1 2; do run_n $LOGD "$1_N_$r" $GEN -m $MODEL -ub 512 "${N_ARGS[@]}" $BENCH; done
