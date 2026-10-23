#!/bin/bash
cd $HOME/moe-ep/hec
B=$HOME/moe-ep/src/llama-build/bin; M=$HOME/moe-ep/models/Qwen3-30B-A3B-Q4_0.gguf
D=$B/libggml-cpu.so
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
R="bash src/hec_run.sh logs"
V="--model $M --cpu-lib $D --vk-lib $B/libggml-vulkan.so"
TMO=1200 $R conc_vk0_t2_a bin/hec_bench conc $V --vk-dev Vulkan0 --threads 2 --iters 200
TMO=1200 $R conc_vk0_t1_a bin/hec_bench conc $V --vk-dev Vulkan0 --threads 1 --iters 200
TMO=1200 $R conc_vk0_t2_poll0 bin/hec_bench conc $V --vk-dev Vulkan0 --threads 2 --poll 0 --iters 200
TMO=1200 $R conc_vk1_t2 bin/hec_bench conc $V --vk-dev Vulkan1 --threads 2 --iters 200
TMO=1200 $R conc_vk0_t2_b bin/hec_bench conc $V --vk-dev Vulkan0 --threads 2 --iters 200
echo QUEUE3B DONE > logs/queue3b.done
