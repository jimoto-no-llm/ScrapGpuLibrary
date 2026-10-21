#!/bin/bash
# grid 3: reversed device order (-dev Vulkan1,Vulkan0 -ts 1,2: RX 6400 holds the LM head and the DSpark draft), draft length sweep
OUT=${1:-/tmp}
M='$MODELS/LFM2.5-8B-A1B-Q4_0.gguf'
DQ='$MODELS/LFM2.5-8B-A1B-DSpark'
C="-sm layer -fa off -t 2 -b 128 -ub 128 -c 2048 -n 192 --temp 0"
SPEC='$LB/llama-speculative-simple'
REV="-dev Vulkan1,Vulkan0 -ts 1,2 -ngl 99 -ngld 99"
{
for P in code chat; do
  echo "base_rev12_$P amdvk \$LB/llama-completion -m $M -dev Vulkan1,Vulkan0 -ts 1,2 -ngl 99 $C -f prompts/$P.txt --no-display-prompt -no-cnv"
done
for P in explain code chat; do for N in 3 7; do
  echo "dspark_rx_rev12_q4km_n${N}_$P amdvk $SPEC -m $M -md ${DQ}-Q4_K_M.gguf --spec-type draft-dspark $REV -devd Vulkan0 $C -f prompts/$P.txt --spec-draft-n-max $N --spec-draft-n-min 0"
done; done
echo "dspark_rx_rev12_q4km_n5_chat amdvk $SPEC -m $M -md ${DQ}-Q4_K_M.gguf --spec-type draft-dspark $REV -devd Vulkan0 $C -f prompts/chat.txt --spec-draft-n-max 5 --spec-draft-n-min 0"
} > $OUT/jobs_3a.txt
