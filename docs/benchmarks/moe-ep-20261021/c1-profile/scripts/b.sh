#!/bin/bash
# b.sh TAG "ENV1=a ENV2=b" -- llama-bench args... (tg-only, r=3, n=128)
TAG=$1; ENVS=$2; shift 3
. $HOME/moe-ep/c1p/scripts/env.sh
exec env $ENVS $HOME/moe-ep/c1p/scripts/go.sh $TAG 1 -- $LB/llama-bench -m $MODEL -ngl 99 -o jsonl -p 0 -n 128 -r 3 "$@"
