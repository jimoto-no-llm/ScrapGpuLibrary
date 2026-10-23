#!/bin/bash
cd $HOME/moe-ep/c1p
S=scripts/b.sh
BASE="-dev Vulkan0/Vulkan1 -ts 5/1 -sm layer -b 512 -ub 512 -t 2"
# A. output (tied token_embd) placed on RX via -ot
for ts in 5/1 4/1 3/1 2/1; do
  $S ot_$(echo $ts|tr / _) "" -- -dev Vulkan0/Vulkan1 -ts $ts -sm layer -b 512 -ub 512 -t 2 -ot "token_embd\.weight=Vulkan0"
done
# B. knobs on the C1 baseline
$S k_base "" -- $BASE
$S k_reuse_off "LLAMA_GRAPH_REUSE_DISABLE=1" -- $BASE
$S k_fusion_off "GGML_VK_DISABLE_FUSION=1" -- $BASE
$S k_gopt_off "GGML_VK_DISABLE_GRAPH_OPTIMIZE=1" -- $BASE
$S k_multiadd_off "GGML_VK_DISABLE_MULTI_ADD=1" -- $BASE
$S k_async_off "GGML_VK_DISABLE_ASYNC=1" -- $BASE
$S k_serialize "GGML_VK_SERIALIZE_SUBMISSIONS=1" -- $BASE
$S k_maxnodes25 "GGML_VK_MAX_NODES_PER_SUBMIT=25" -- $BASE
$S k_maxnodes1000 "GGML_VK_MAX_NODES_PER_SUBMIT=1000" -- $BASE
$S k_mmvq_off "GGML_VK_DISABLE_MMVQ=1" -- $BASE
$S k_mmvq_force "GGML_VK_FORCE_MMVQ=1" -- $BASE
$S k_intdot_off "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1" -- $BASE
$S k_f16_off "GGML_VK_DISABLE_F16=1" -- $BASE
$S k_descreuse_off "GGML_VK_DISABLE_DESCRIPTOR_REUSE=1" -- $BASE
$S k_transferq "GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1" -- $BASE
$S k_t1 "" -- -dev Vulkan0/Vulkan1 -ts 5/1 -sm layer -b 512 -ub 512 -t 1
$S k_poll0 "" -- $BASE -poll 0
$S k_poll100 "" -- $BASE -poll 100
$S k_nkvo "" -- $BASE -nkvo 1
$S k_ctk_q8 "" -- $BASE -ctk q8_0
$S k_fa1 "" -- $BASE -fa 1
echo Q5DONE
