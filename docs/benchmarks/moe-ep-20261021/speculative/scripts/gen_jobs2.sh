#!/bin/bash
# generates job files for bundle.sh (grid 2: draft knobs, failed placements, n-gram self-speculation, reversed device order)
OUT=${1:-/tmp}
M='$MODELS/LFM2.5-8B-A1B-Q4_0.gguf'
DQ='$MODELS/LFM2.5-8B-A1B-DSpark'
C="-sm layer -fa off -t 2 -b 128 -ub 128 -c 2048 -n 192 --temp 0"
SPEC='$LB/llama-speculative-simple'
FWD="-dev Vulkan0,Vulkan1 -ts 5,1 -ngl 99 -ngld 99"
{ # 2a: knobs of the WX draft
for P in explain code; do
  echo "dspark_wx_q4km_n5_$P amdvk $SPEC -m $M -md ${DQ}-Q4_K_M.gguf --spec-type draft-dspark $FWD -devd Vulkan1 $C -f prompts/$P.txt --spec-draft-n-max 5 --spec-draft-n-min 0"
  echo "dspark_wx_f16_n5_$P amdvk $SPEC -m $M -md ${DQ}-F16.gguf --spec-type draft-dspark $FWD -devd Vulkan1 $C -f prompts/$P.txt --spec-draft-n-max 5 --spec-draft-n-min 0"
  echo "dspark_wx_q8_n5_nobs_$P amdvk $SPEC -m $M -md ${DQ}-Q8_0.gguf --spec-type draft-dspark $FWD -devd Vulkan1 $C -f prompts/$P.txt --spec-draft-n-max 5 --spec-draft-n-min 0 --no-spec-draft-backend-sampling"
  echo "dspark_wx_q8_n5_fa_$P amdvk $SPEC -m $M -md ${DQ}-Q8_0.gguf --spec-type draft-dspark $FWD -devd Vulkan1 ${C/-fa off/-fa on} -f prompts/$P.txt --spec-draft-n-max 5 --spec-draft-n-min 0"
  echo "dspark_wx_q8_n7_p03_$P amdvk $SPEC -m $M -md ${DQ}-Q8_0.gguf --spec-type draft-dspark $FWD -devd Vulkan1 $C -f prompts/$P.txt --spec-draft-n-max 7 --spec-draft-n-min 0 --spec-draft-p-min 0.3"
  echo "dspark_wx_q8_n7_p06_$P amdvk $SPEC -m $M -md ${DQ}-Q8_0.gguf --spec-type draft-dspark $FWD -devd Vulkan1 $C -f prompts/$P.txt --spec-draft-n-max 7 --spec-draft-n-min 0 --spec-draft-p-min 0.6"
done
} > $OUT/jobs_2a.txt
{ # 2b: placements that fail (short, -n 32) and the reversed device order
CS="-sm layer -fa off -t 2 -b 128 -ub 128 -c 2048 -n 32 --temp 0"
echo "fail_draft350m_wx amdvk $SPEC -m $M -md \$MODELS/LFM2.5-350M-Q4_0.gguf --spec-type draft-simple $FWD -devd Vulkan1 $CS -f prompts/explain.txt --spec-draft-n-max 5"
echo "fail_dspark_rx amdvk $SPEC -m $M -md ${DQ}-Q8_0.gguf --spec-type draft-dspark $FWD -devd Vulkan0 $CS -f prompts/explain.txt --spec-draft-n-max 5"
echo "fail_dspark_cpu amdvk $SPEC -m $M -md ${DQ}-Q8_0.gguf --spec-type draft-dspark $FWD -devd none $CS -f prompts/explain.txt --spec-draft-n-max 5"
echo "fail_dspark_gt730 mix4 $SPEC -m $M -md ${DQ}-Q8_0.gguf --spec-type draft-dspark $FWD -devd CL11_0 $CS -f prompts/explain.txt --spec-draft-n-max 5"
echo "fail_dspark_gt430rpc mix5 $SPEC -m $M -md ${DQ}-Q8_0.gguf --spec-type draft-dspark $FWD -devd RPC0 $CS -f prompts/explain.txt --spec-draft-n-max 5"
# reversed order: WX first, RX last (RX then holds the LM head); ts 1,2 leaves ~0.6 GB on the RX for the draft
echo "base_rev12_explain amdvk \$LB/llama-completion -m $M -dev Vulkan1,Vulkan0 -ts 1,2 -ngl 99 -sm layer -fa off -t 2 -b 128 -ub 128 -c 2048 -n 192 --temp 0 -f prompts/explain.txt --no-display-prompt -no-cnv"
echo "dspark_rx_rev12_q4km_n5_explain amdvk $SPEC -m $M -md ${DQ}-Q4_K_M.gguf --spec-type draft-dspark -dev Vulkan1,Vulkan0 -ts 1,2 -ngl 99 -ngld 99 -devd Vulkan0 $C -f prompts/explain.txt --spec-draft-n-max 5 --spec-draft-n-min 0"
echo "dspark_rx_rev12_q4km_n5_code amdvk $SPEC -m $M -md ${DQ}-Q4_K_M.gguf --spec-type draft-dspark -dev Vulkan1,Vulkan0 -ts 1,2 -ngl 99 -ngld 99 -devd Vulkan0 $C -f prompts/code.txt --spec-draft-n-max 5 --spec-draft-n-min 0"
} > $OUT/jobs_2b.txt
{ # 2c: n-gram self-speculation (no draft model, CPU-side proposals); lookup n=4 / draft m<=7 so the verify batch stays <= 8 tokens
for P in explain code chat; do
  echo "ngram_simple_n4m7_$P amdvk $SPEC -m $M --spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 7 $FWD $C -f prompts/$P.txt"
  echo "ngram_mapk_n4m7_$P amdvk $SPEC -m $M --spec-type ngram-map-k --spec-ngram-map-k-size-n 4 --spec-ngram-map-k-size-m 7 $FWD $C -f prompts/$P.txt"
  echo "ngram_mod_n3_$P amdvk $SPEC -m $M --spec-type ngram-mod --spec-ngram-mod-n-min 2 --spec-ngram-mod-n-max 7 --spec-ngram-mod-n-match 3 $FWD $C -f prompts/$P.txt"
done
} > $OUT/jobs_2c.txt
