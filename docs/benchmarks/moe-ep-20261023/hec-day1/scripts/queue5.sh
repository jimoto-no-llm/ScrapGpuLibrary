#!/bin/bash
cd $HOME/moe-ep/hec
B=$HOME/moe-ep/src/llama-build/bin; M=$HOME/moe-ep/models/Qwen3-30B-A3B-Q4_0.gguf
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
R="bash src/hec_run.sh logs"
V="--model $M --vk-lib $B/libggml-vulkan.so --vk-dev Vulkan0 --iters 200"
TMO=900 $R spin_mem0 bin/hec_bench spin $V --busy 0 --mem
TMO=900 $R spin_mem1 bin/hec_bench spin $V --busy 1 --mem
echo QUEUE5 DONE > logs/queue5.done
