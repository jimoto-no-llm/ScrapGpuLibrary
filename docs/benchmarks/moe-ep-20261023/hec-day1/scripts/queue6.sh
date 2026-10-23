#!/bin/bash
cd $HOME/moe-ep/hec
B=$HOME/moe-ep/src/llama-build/bin; M=$HOME/moe-ep/models/Qwen3-30B-A3B-Q4_0.gguf
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
R="bash src/hec_run.sh logs"
V="--model $M --cpu-lib $B/libggml-cpu.so --vk-lib $B/libggml-vulkan.so --vk-dev Vulkan0 --iters 200 --pairs 8:0,0:8,6:2,4:4,2:6,1:7"
TMO=900 $R conc_vk0_t1_pin bin/hec_bench conc $V --threads 1 --pin
echo QUEUE6 DONE > logs/queue6.done
