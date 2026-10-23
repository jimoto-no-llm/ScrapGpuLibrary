#!/bin/bash
# gen.sh TAG SPLIT UB : greedy 24 tokens after a ~2 KB prompt (pp of several hundred tokens) 
. "$(dirname "$0")/env.sh"
TAG=$1; SPLIT=$2; UB=$3
if [ -z "$SPLIT" ]; then sglx_env ""; DEVS=Vulkan0,Vulkan1; EXTRA=(-ts 5,1); else sglx_env "$SPLIT"; DEVS=Vulkan0,Vulkan1,SGLX0; EXTRA=(-ts 5,1,0 -ot "token_embd\.weight=SGLX0"); fi
exec ./run.sh $TAG $LB/llama-completion -m $MODEL -dev $DEVS -ngl 99 -fa off -t 2 -b 512 -ub $UB -sm layer "${EXTRA[@]}" -f prompt.txt -n 24 --temp 0 -no-cnv -s 1
