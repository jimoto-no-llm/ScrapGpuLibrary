#!/bin/bash
# grid 3c: pp / tg / pg with a ~434 token prompt (prompts/long512.txt) and 128 generated tokens, -b 512 -ub 128 (spec tool needs the prompt inside one batch)
OUT=${1:-/tmp}
M='$MODELS/LFM2.5-8B-A1B-Q4_0.gguf'
DQ='$MODELS/LFM2.5-8B-A1B-DSpark-Q4_K_M.gguf'
C="-sm layer -fa off -t 2 -b 512 -ub 128 -c 2048 -n 128 --temp 0"
{
echo "pg2_base_rev12 amdvk \$LB/llama-completion -m $M -dev Vulkan1,Vulkan0 -ts 1,2 -ngl 99 $C -f prompts/long512.txt --no-display-prompt -no-cnv"
for N in 5 3; do
echo "pg2_dspark_rx_n$N amdvk \$LB/llama-speculative-simple -m $M -md $DQ --spec-type draft-dspark -dev Vulkan1,Vulkan0 -ts 1,2 -ngl 99 -ngld 99 -devd Vulkan0 $C -f prompts/long512.txt --spec-draft-n-max $N --spec-draft-n-min 0"
done
} > $OUT/jobs_3c.txt
