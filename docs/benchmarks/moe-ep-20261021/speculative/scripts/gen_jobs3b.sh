#!/bin/bash
# grid 3b: output identity (llama-server, token ids), and pp / tg / pg numbers with a ~512 token prompt and 128 generated tokens
OUT=${1:-/tmp}
M='$MODELS/LFM2.5-8B-A1B-Q4_0.gguf'
DQ='$MODELS/LFM2.5-8B-A1B-DSpark-Q4_K_M.gguf'
REV="-dev Vulkan1,Vulkan0 -ts 1,2 -ngl 99 -sm layer -fa off -t 2 -b 128 -ub 128 -c 2048"
C="-sm layer -fa off -t 2 -b 128 -ub 128 -c 2048 -n 128 --temp 0"
{
echo "id_server_base amdvk scripts/servertest.sh logs/runs/id_server_base -m $M $REV"
echo "id_server_dspark_n5 amdvk scripts/servertest.sh logs/runs/id_server_dspark_n5 -m $M -md $DQ --spec-type draft-dspark $REV -ngld 99 -devd Vulkan0 --spec-draft-n-max 5 --spec-draft-n-min 0"
echo "pg_base_rev12 amdvk \$LB/llama-completion -m $M -dev Vulkan1,Vulkan0 -ts 1,2 -ngl 99 $C -f prompts/long512.txt --no-display-prompt -no-cnv"
echo "pg_dspark_rx_n5 amdvk \$LB/llama-speculative-simple -m $M -md $DQ --spec-type draft-dspark -dev Vulkan1,Vulkan0 -ts 1,2 -ngl 99 -ngld 99 -devd Vulkan0 $C -f prompts/long512.txt --spec-draft-n-max 5 --spec-draft-n-min 0"
echo "pg_dspark_rx_n3 amdvk \$LB/llama-speculative-simple -m $M -md $DQ --spec-type draft-dspark -dev Vulkan1,Vulkan0 -ts 1,2 -ngl 99 -ngld 99 -devd Vulkan0 $C -f prompts/long512.txt --spec-draft-n-max 3 --spec-draft-n-min 0"
} > $OUT/jobs_3b.txt
