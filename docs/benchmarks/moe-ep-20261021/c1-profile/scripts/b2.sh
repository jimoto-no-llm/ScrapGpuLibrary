#!/bin/bash
# b2.sh TAG MODELFILE "ENVS" -- llama-bench args  (shim counts + cpu sampling in one go: sampler 0.2s)
TAG=$1; M=$HOME/moe-ep/models/$2; ENVS=$3; shift 4
. $HOME/moe-ep/c1p/scripts/env.sh
exec env CHILD_LD_PRELOAD=$HOME/moe-ep/c1p/vkshim.so VKSHIM_OUT=$HOME/moe-ep/c1p/logs/$TAG.shim.txt $ENVS $HOME/moe-ep/c1p/scripts/go.sh $TAG 0.2 -- $LB/llama-bench -m $M -ngl 99 -o jsonl -p 0 -n 256 -r 3 "$@"
