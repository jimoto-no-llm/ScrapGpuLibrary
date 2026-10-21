LB=$HOME/moe-ep/src/llama-build/bin
MODEL=$HOME/moe-ep/models/LFM2.5-8B-A1B-Q4_0.gguf
C1="$LB/llama-bench -m $MODEL -dev Vulkan0/Vulkan1 -ngl 99 -fa 0 -b 512 -ub 512 -ts 5/1 -sm layer -ncmoe 0 -o jsonl -p 0"
