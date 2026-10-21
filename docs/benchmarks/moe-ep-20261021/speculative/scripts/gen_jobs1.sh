#!/bin/bash
# generates job files for bundle.sh (grid 1: DSpark draft on the WX 2100 with the target at -dev Vulkan0,Vulkan1 -ts 5,1)
OUT=${1:-/tmp}
M='$MODELS/LFM2.5-8B-A1B-Q4_0.gguf'
DQ8='$MODELS/LFM2.5-8B-A1B-DSpark-Q8_0.gguf'
C="-sm layer -fa off -t 2 -b 128 -ub 128 -c 2048 -n 192 --temp 0"
{
for P in explain code chat; do
  echo "base_fwd_$P amdvk \$LB/llama-completion -m $M -dev Vulkan0,Vulkan1 -ts 5,1 -ngl 99 $C -f prompts/$P.txt --no-display-prompt -no-cnv"
done
for P in explain code chat; do
  echo "dspark_wx_q8_n7_$P amdvk \$LB/llama-speculative-simple -m $M -md $DQ8 --spec-type draft-dspark -dev Vulkan0,Vulkan1 -devd Vulkan1 -ts 5,1 -ngl 99 -ngld 99 $C -f prompts/$P.txt --spec-draft-n-max 7 --spec-draft-n-min 0"
done
} > $OUT/jobs_1a.txt
{
for P in explain code chat; do for N in 3 5; do
  [ "$P$N" = explain3 ] && continue
  echo "dspark_wx_q8_n${N}_$P amdvk \$LB/llama-speculative-simple -m $M -md $DQ8 --spec-type draft-dspark -dev Vulkan0,Vulkan1 -devd Vulkan1 -ts 5,1 -ngl 99 -ngld 99 $C -f prompts/$P.txt --spec-draft-n-max $N --spec-draft-n-min 0"
done; done
} > $OUT/jobs_1b.txt
