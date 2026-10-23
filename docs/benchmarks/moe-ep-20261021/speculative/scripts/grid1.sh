#!/bin/bash
# grid 1: DSpark draft on the WX 2100 (the device that holds the tied token_embd / LM head with -dev Vulkan0,Vulkan1 -ts 5,1), 3 prompts
. "$(dirname "$0")/env.sh"
cd $SPEC
M=$MODELS/LFM2.5-8B-A1B-Q4_0.gguf
COMMON="-ngl 99 -ngld 99 -sm layer -fa off -t 2 -b 128 -ub 128 -c 2048 -n 192 --temp 0"
for P in explain code chat; do
  # baseline, target only (same device order and split as the spec runs)
  scripts/go.sh base_fwd_$P amdvk $LB/llama-completion -m $M -dev Vulkan0,Vulkan1 -ts 5,1 $COMMON -f prompts/$P.txt --no-display-prompt -no-cnv
  for N in 3 5 7; do
    scripts/go.sh dspark_wx_q8_n${N}_$P amdvk $LB/llama-speculative-simple -m $M -md $MODELS/LFM2.5-8B-A1B-DSpark-Q8_0.gguf --spec-type draft-dspark -dev Vulkan0,Vulkan1 -devd Vulkan1 -ts 5,1 $COMMON -f prompts/$P.txt --spec-draft-n-max $N --spec-draft-n-min 0
  done
done
